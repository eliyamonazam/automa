"""Read-only local web server for the web UI (D78).

Serves the pages of :mod:`automa.webui.render` with the standard library's
``http.server``:

* binds only to a loopback IP address (``127.0.0.1`` by default); anything else is refused;
* answers only ``GET`` and ``HEAD``; every other method gets ``405``;
* checks the ``Host`` header against the bound address (DNS-rebinding defence): a page
  is only served to ``127.0.0.1:<port>``, ``localhost:<port>`` (or ``[::1]:<port>``);
* sends strict security headers on every response (CSP incl. ``frame-ancestors``,
  ``X-Frame-Options: DENY``, ``nosniff``, ``no-store``, ``no-referrer``);
* reads spending from the Ledger opened with ``read_only=True`` per request; a missing
  ledger means zero spending and is never created.

By default nothing here writes a file, creates a directory, decides an approval or calls
a model. Only with ``actions`` (``automa web --allow-actions``, D123) does it also accept
``POST`` to the approve/reject paths, checked and recorded by :mod:`automa.webui.actions`,
and — when a planner is given (D124) — to the new-idea paths of :mod:`automa.webui.ideas`
(``/new``, ``/projects/<pid>/clarify``), which call the Executive through the guarded
gateway — and, when a run preflight is given (D125), to ``/projects/<pid>/start``,
which starts ``automa run`` as a background child process (:mod:`automa.webui.runs`).
Every POST passes the same Host, same-origin and CSRF checks first.
"""

from __future__ import annotations

import ipaddress
import socket
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, quote, unquote, urlsplit

from automa.budget.ledger import CostLedger, LedgerError
from automa.project.store import ProjectStore, ProjectStoreError, validate_project_id
from automa.tools.approvals import ApprovalError
from automa.webui.actions import (
    MAX_FORM_BYTES,
    ActionRefused,
    ActionSettings,
    ActionTarget,
    allowed_origins,
    check_same_origin,
    check_token,
    decide_from_form,
    new_csrf_token,
    parse_action_path,
    parse_form,
)
from automa.webui.ideas import (
    ANSWER_FIELDS,
    IDEA_FIELDS,
    MAX_IDEA_FORM_BYTES,
    NEW_PATH,
    IdeaFlow,
    IdeaPlanner,
    is_idea_path,
    parse_clarify_path,
)
from automa.webui.render import (
    ACTION_CONTENT_SECURITY_POLICY,
    CONTENT_SECURITY_POLICY,
    ActionForms,
    RunPanel,
    page,
    render_clarify,
    render_new_idea,
    render_overview,
    render_project,
    text,
    visible,
)
from automa.webui.runs import (
    START_FIELDS,
    STOP_FIELDS,
    BackgroundRuns,
    RunPreflight,
    parse_start_path,
    parse_stop_path,
)
from automa.webui.views import build_overview, build_project_view

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
ALLOWED_METHODS = ("GET", "HEAD")
ACTION_METHODS = ("GET", "HEAD", "POST")
HEADER_CSP = CONTENT_SECURITY_POLICY + "; frame-ancestors 'none'"
ACTION_HEADER_CSP = ACTION_CONTENT_SECURITY_POLICY + "; frame-ancestors 'none'"
# Headers that must appear at most once on a POST (a second copy could hide the first).
_SINGLE_HEADERS = frozenset({"host", "origin", "referer", "sec-fetch-site", "content-type"})
_PROJECT_PREFIX = "/projects/"

SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("Content-Security-Policy", HEADER_CSP),
    ("X-Frame-Options", "DENY"),
    ("X-Content-Type-Options", "nosniff"),
    ("Cache-Control", "no-store"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
)
# Action mode: forms may post to 'self' and the browser must send a real Origin (D123).
ACTION_SECURITY_HEADERS: tuple[tuple[str, str], ...] = tuple(
    (name, ACTION_HEADER_CSP)
    if name == "Content-Security-Policy"
    else (name, "same-origin")
    if name == "Referrer-Policy"
    else (name, value)
    for name, value in SECURITY_HEADERS
)


class WebServerError(ValueError):
    """Invalid server settings (non-loopback host, bad port)."""


def validate_bind_host(host: str) -> str:
    """``host`` if it is a loopback IP literal, else ``WebServerError``.

    Host names (even ``localhost``) are refused: they are resolved at bind time and
    could point anywhere.
    """
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        raise WebServerError(
            f"refusing to listen on {host!r}: only a loopback IP address such as 127.0.0.1"
        ) from None
    if not addr.is_loopback:
        raise WebServerError(f"refusing to listen on {host!r}: not a loopback address")
    return str(addr)


def validate_port(port: int) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise WebServerError(f"invalid port: {port!r}")
    return port


def allowed_hosts(host: str, port: int) -> frozenset[str]:
    """Exact ``Host`` header values accepted for a server bound to ``host:port``."""
    addr = ipaddress.ip_address(host)
    literal = f"[{addr}]" if addr.version == 6 else str(addr)
    names = {literal, "localhost"}
    return frozenset(f"{n}:{port}" for n in names)


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    headers: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def with_headers(self, *extra: tuple[str, str]) -> Response:
        return Response(self.status, self.body, (*self.headers, *extra))


def _html(
    status: int,
    document: str,
    extra: tuple[tuple[str, str], ...] = (),
    *,
    security: tuple[tuple[str, str], ...] = SECURITY_HEADERS,
) -> Response:
    headers = (("Content-Type", "text/html; charset=utf-8"), *security, *extra)
    return Response(status=status, body=document.encode("utf-8"), headers=headers)


def _error_page(
    status: int,
    title: str,
    detail: str = "",
    *,
    security: tuple[tuple[str, str], ...] = SECURITY_HEADERS,
) -> Response:
    body = f"<h1>{text(title)}</h1>"
    if detail:
        body += f'<p class="err">{text(detail)}</p>'
    body += '<p><a href="/">← all projects</a></p>'
    actions = security is not SECURITY_HEADERS
    return _html(status, page(title, body, actions=actions), security=security)


class WebApp:
    """Request → response logic, independent of sockets (unit-testable)."""

    def __init__(
        self,
        workspace: Path,
        ledger_path: Path,
        allowed: frozenset[str],
        *,
        actions: ActionSettings | None = None,
        planner: IdeaPlanner | None = None,
        runs: RunPreflight | BackgroundRuns | None = None,
    ):
        self._store = ProjectStore.for_workspace(workspace)
        # Background runs only in action mode (D125); a read-only server never runs.
        if actions is None or runs is None:
            self._runs: BackgroundRuns | None = None
        elif isinstance(runs, BackgroundRuns):
            self._runs = runs
        else:
            self._runs = BackgroundRuns(self._store, runs)
        # New ideas only in action mode (D124); a read-only server never plans.
        self._ideas = IdeaFlow(self._store, planner) if actions is not None and planner else None
        self._ledger_path = ledger_path
        self._allowed = allowed
        self._actions = actions
        self._methods = ACTION_METHODS if actions is not None else ALLOWED_METHODS
        self._security = ACTION_SECURITY_HEADERS if actions is not None else SECURITY_HEADERS
        self._decide_lock = threading.Lock()

    @property
    def actions_enabled(self) -> bool:
        return self._actions is not None

    @property
    def ideas_enabled(self) -> bool:
        return self._ideas is not None

    @property
    def runs_enabled(self) -> bool:
        return self._runs is not None

    def max_body_bytes(self, target: str) -> int:
        """The largest POST body accepted for ``target`` (idea forms are larger)."""
        if self._ideas is not None and is_idea_path(urlsplit(target).path):
            return MAX_IDEA_FORM_BYTES
        return MAX_FORM_BYTES

    def error(self, status: int, title: str, detail: str = "") -> Response:
        """An error page with this app's security headers."""
        return _error_page(status, title, detail, security=self._security)

    def _spending(self) -> tuple[dict[str, int], str | None]:
        """Spending per project from a read-only Ledger; never creates one."""
        if not self._ledger_path.exists() and not self._ledger_path.is_symlink():
            return {}, None
        try:
            with CostLedger(self._ledger_path, read_only=True) as ledger:
                spent: dict[str, int] = {}
                for e in ledger.entries():
                    if e.project_id:
                        spent[e.project_id] = spent.get(e.project_id, 0) + e.cost_nusd
                return spent, None
        except LedgerError as e:
            return {}, f"ledger: {e}"

    def handle(
        self,
        method: str,
        target: str,
        host: str | None,
        headers: Sequence[tuple[str, str]] = (),
        body: bytes = b"",
    ) -> Response:
        if method not in self._methods:
            detail = (
                "Only the forms of these pages are accepted."
                if self._actions is not None
                else "This interface is read-only."
            )
            refused = self.error(405, "Method not allowed", detail)
            return refused.with_headers(("Allow", ", ".join(self._methods)))
        if host is None or host.strip().lower() not in self._allowed:
            return self.error(
                421, "Unexpected Host header", "Open the address printed by automa web."
            )
        if method == "POST":
            return self._post(target, headers, body)
        # HEAD gets the same status and headers; the handler omits the body.
        return self._route(target)

    def _post(self, target: str, headers: Sequence[tuple[str, str]], body: bytes) -> Response:
        if self._actions is None:  # POST is only allowed in action mode (never reached)
            return self.error(405, "Method not allowed", "This interface is read-only.")
        parts = urlsplit(target)
        if self._ideas is not None and not parts.query and is_idea_path(parts.path):
            return self._post_idea(parts.path, headers, body)
        start_pid = parse_start_path(parts.path) if not parts.query else None
        if self._runs is not None and start_pid is not None:
            return self._post_start(start_pid, headers, body)
        stop_pid = parse_stop_path(parts.path) if not parts.query else None
        if self._runs is not None and stop_pid is not None:
            return self._post_stop(stop_pid, headers, body)
        action = parse_action_path(parts.path) if not parts.query else None
        if action is None:
            return self.error(404, "Not found")
        lowered = self._single_headers(headers)
        if isinstance(lowered, Response):
            return lowered
        return self._decide(action, lowered, body)

    def _redirect(self, location: str) -> Response:
        return _html(
            303,
            page("Done", f'<p><a href="{text(location)}">continue</a></p>', actions=True),
            (("Location", location),),
            security=self._security,
        )

    def _post_idea(self, path: str, headers: Sequence[tuple[str, str]], body: bytes) -> Response:
        """``POST /new`` or ``/projects/<pid>/clarify`` (D124), after the same checks."""
        if self._actions is None or self._ideas is None:  # never reached
            return self.error(404, "Not found")
        lowered = self._single_headers(headers)
        if isinstance(lowered, Response):
            return lowered
        pid = parse_clarify_path(path)
        fields = IDEA_FIELDS if pid is None else ANSWER_FIELDS
        try:
            check_same_origin(lowered, self._actions)
            form = parse_form(lowered, body, fields=fields, max_bytes=MAX_IDEA_FORM_BYTES)
            check_token(form, self._actions)
            location = self._ideas.start(form) if pid is None else self._ideas.answer(pid, form)
        except ActionRefused as e:
            return self.error(e.status, "Refused", e.message)
        return self._redirect(location)

    def _post_start(self, pid: str, headers: Sequence[tuple[str, str]], body: bytes) -> Response:
        """``POST /projects/<pid>/start`` (D125), after the same checks."""
        if self._actions is None or self._runs is None:  # never reached
            return self.error(404, "Not found")
        lowered = self._single_headers(headers)
        if isinstance(lowered, Response):
            return lowered
        try:
            check_same_origin(lowered, self._actions)
            form = parse_form(lowered, body, fields=START_FIELDS)
            check_token(form, self._actions)
            self._runs.start(pid, form)
        except ActionRefused as e:
            return self.error(e.status, "Refused", e.message)
        return self._redirect(f"/projects/{quote(pid, safe='')}")

    def _post_stop(self, pid: str, headers: Sequence[tuple[str, str]], body: bytes) -> Response:
        """``POST /projects/<pid>/stop`` (D126), after the same checks."""
        if self._actions is None or self._runs is None:  # never reached
            return self.error(404, "Not found")
        lowered = self._single_headers(headers)
        if isinstance(lowered, Response):
            return lowered
        try:
            check_same_origin(lowered, self._actions)
            form = parse_form(lowered, body, fields=STOP_FIELDS)
            check_token(form, self._actions)
            self._runs.stop(pid, form)
        except ActionRefused as e:
            return self.error(e.status, "Refused", e.message)
        return self._redirect(f"/projects/{quote(pid, safe='')}")

    def _run_panel(self, pid: str) -> RunPanel | None:
        """The start button and last background run of a project page (D125)."""
        if self._runs is None:
            return None
        state = self._runs.state(pid)
        can_start = self._runs.can_start(pid)
        needs = self._runs.needs_confirm(pid) if can_start else False
        if state is None:
            return RunPanel(can_start=can_start, needs_confirm=needs)
        return RunPanel(
            can_start=can_start,
            needs_confirm=needs,
            can_stop=self._runs.can_stop(pid),
            state=state.summary,
            running=state.running,
            started_at=state.started_at,
            log_name=state.log_path.name,
            tail=state.tail,
        )

    def _single_headers(self, headers: Sequence[tuple[str, str]]) -> dict[str, str] | Response:
        lowered: dict[str, str] = {}
        for name, value in headers:
            key = name.lower()
            if key in lowered and key in _SINGLE_HEADERS:
                return self.error(400, "Refused", f"duplicate {key} header")
            lowered.setdefault(key, value)
        return lowered

    def _decide(self, action: ActionTarget, lowered: dict[str, str], body: bytes) -> Response:
        if self._actions is None:  # never reached
            return self.error(405, "Method not allowed", "This interface is read-only.")
        try:
            check_same_origin(lowered, self._actions)
            form = parse_form(lowered, body)
            check_token(form, self._actions)
            with self._decide_lock:
                decided = decide_from_form(self._store, action, form)
        except ActionRefused as e:
            return self.error(e.status, "Refused", e.message)
        location = (
            f"/projects/{quote(action.project_id, safe='')}?decided={quote(decided.id, safe='')}"
        )
        done = _html(
            303,
            page("Recorded", f'<p><a href="{text(location)}">continue</a></p>', actions=True),
            (("Location", location),),
            security=self._security,
        )
        return done

    def _idea_lines(self, pid: str, planned: bool) -> tuple[str | None, bool]:
        """Cost range of a planned project and whether questions wait (D124)."""
        if self._ideas is None:
            return None, False
        if planned:
            return self._ideas.planner.cost_range(pid), False
        try:
            self._ideas.pending_questions(pid)
        except ActionRefused:
            return None, False
        return None, True

    def _notice(self, pid: str, query: str) -> str | None:
        """One line about a decision made from this page (``?decided=<id>``)."""
        try:
            pairs = parse_qsl(query, max_num_fields=1)
        except ValueError:
            return None
        if len(pairs) != 1 or pairs[0][0] != "decided":
            return None
        try:
            req = self._store.approvals(pid).get(pairs[0][1])
        except (ApprovalError, ProjectStoreError):
            return None
        if not req.decided:
            return None
        if req.status.value == "approved":
            return f"{req.id}: approved; the call runs at most once, on the next: automa run {pid}"
        return f"{req.id}: rejected; the model will be told on the next: automa run {pid}"

    def _route(self, target: str) -> Response:
        parts = urlsplit(target)
        path = parts.path
        actions = self._actions is not None
        if path == "/":
            spent, err = self._spending()
            view = build_overview(self._store, spent)
            if err:
                view = view.model_copy(update={"errors": (*view.errors, err)})
            document = render_overview(view, actions=actions, ideas=self._ideas is not None)
            return _html(200, document, security=self._security)
        if self._ideas is not None and self._actions is not None:
            forms = ActionForms(self._actions.csrf_token)
            if path == NEW_PATH:
                return _html(200, render_new_idea(forms), security=self._security)
            clarify_pid = parse_clarify_path(path)
            if clarify_pid is not None:
                try:
                    questions = self._ideas.pending_questions(clarify_pid)
                    goal = self._store.load(clarify_pid).goal
                except ActionRefused as e:
                    return self.error(e.status, "Not available", e.message)
                except ProjectStoreError as e:
                    return self.error(500, "Project unreadable", str(e))
                document = render_clarify(clarify_pid, goal, questions, forms)
                return _html(200, document, security=self._security)
        if path.startswith(_PROJECT_PREFIX):
            raw = unquote(path[len(_PROJECT_PREFIX) :])
            try:
                pid = validate_project_id(raw)
            except ProjectStoreError:
                return self.error(404, "Not found")
            if pid not in self._store.list_ids():
                return self.error(404, "Not found", f"no project {pid}")
            spent, err = self._spending()
            try:
                pview = build_project_view(self._store, pid, spent.get(pid, 0))
            except (ProjectStoreError, ApprovalError) as e:
                return self.error(500, "Project unreadable", str(e))
            if err:
                pview = pview.model_copy(update={"errors": (*pview.errors, err)})
            forms = ActionForms(self._actions.csrf_token) if self._actions is not None else None
            notice = self._notice(pid, parts.query) if actions and parts.query else None
            cost_range, waiting = self._idea_lines(pid, pview.plan_version is not None)
            document = render_project(
                pview,
                forms,
                notice,
                cost_range=cost_range,
                questions_waiting=waiting,
                run=self._run_panel(pid),
            )
            return _html(200, document, security=self._security)
        return self.error(404, "Not found")


def make_handler(
    app: WebApp, log: Callable[[str], None] | None = None
) -> type[BaseHTTPRequestHandler]:
    """A request-handler class serving ``app``."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "automa-web"
        sys_version = ""

        def _read_body(self) -> bytes | Response:
            """The body of an action-mode POST, or the refusal to send instead."""
            if self.headers.get("Transfer-Encoding") is not None:
                return app.error(400, "Refused", "chunked bodies are not accepted")
            lengths = self.headers.get_all("Content-Length") or []
            if len(lengths) != 1 or not lengths[0].strip().isdigit():
                return app.error(411, "Length required")
            length = int(lengths[0].strip())
            if length > app.max_body_bytes(self.path):
                return app.error(413, "Refused", "form is too large")
            return self.rfile.read(length) if length else b""

        def _serve(self) -> None:
            response: Response | None = None
            body = b""
            if self.command == "POST" and app.actions_enabled:
                read = self._read_body()
                if isinstance(read, Response):
                    response = read
                else:
                    body = read
            if response is None:
                response = app.handle(
                    self.command,
                    self.path,
                    self.headers.get("Host"),
                    list(self.headers.items()),
                    body,
                )
            self.send_response(response.status)
            for name, value in response.headers:
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(response.body)
            self.close_connection = True

        def __getattr__(self, name: str) -> Callable[[], None]:
            # Every ``do_<METHOD>`` goes through the same path; non-GET/HEAD get 405.
            if name.startswith("do_"):
                return self._serve
            raise AttributeError(name)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            if log is not None:
                log(visible(f"{self.address_string()} {format % args}"))

    return Handler


def create_server(
    workspace: Path,
    ledger_path: Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    log: Callable[[str], None] | None = None,
    allow_actions: bool = False,
    planner: IdeaPlanner | None = None,
    runs: RunPreflight | None = None,
) -> ThreadingHTTPServer:
    """A bound (not yet serving) server. ``port=0`` picks a free port (tests).

    ``allow_actions`` (D123) enables approve/reject forms with a fresh CSRF token for
    this server's lifetime; the default is the read-only server of D78. ``planner``
    (only used with ``allow_actions``, D124) enables the "New idea" pages; ``runs``
    (only with ``allow_actions``, D125) enables the "Start run" button."""
    host = validate_bind_host(host)
    port = validate_port(port)

    family = socket.AF_INET6 if ipaddress.ip_address(host).version == 6 else socket.AF_INET

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        address_family = family

    # Bind first so the Host check uses the real port (port=0 picks a free one).
    server = Server((host, port), BaseHTTPRequestHandler)
    hosts = allowed_hosts(host, int(server.server_address[1]))
    actions = (
        ActionSettings(csrf_token=new_csrf_token(), origins=allowed_origins(hosts))
        if allow_actions
        else None
    )
    app = WebApp(workspace, ledger_path, hosts, actions=actions, planner=planner, runs=runs)
    server.RequestHandlerClass = make_handler(app, log)
    return server


def server_url(server: ThreadingHTTPServer) -> str:
    """The address to open in a browser (accepted by the ``Host`` check)."""
    host, port = str(server.server_address[0]), int(server.server_address[1])
    shown = f"[{host}]" if ":" in host else host
    return f"http://{shown}:{port}/"


def stderr_log(line: str) -> None:
    print(line, file=sys.stderr)
