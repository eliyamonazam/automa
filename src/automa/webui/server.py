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
``POST`` to the approve/reject paths, checked and recorded by :mod:`automa.webui.actions`.
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
    allowed_origins,
    check_same_origin,
    check_token,
    decide_from_form,
    new_csrf_token,
    parse_action_path,
    parse_form,
)
from automa.webui.render import (
    ACTION_CONTENT_SECURITY_POLICY,
    CONTENT_SECURITY_POLICY,
    ActionForms,
    page,
    render_overview,
    render_project,
    text,
    visible,
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
    ):
        self._store = ProjectStore.for_workspace(workspace)
        self._ledger_path = ledger_path
        self._allowed = allowed
        self._actions = actions
        self._methods = ACTION_METHODS if actions is not None else ALLOWED_METHODS
        self._security = ACTION_SECURITY_HEADERS if actions is not None else SECURITY_HEADERS
        self._decide_lock = threading.Lock()

    @property
    def actions_enabled(self) -> bool:
        return self._actions is not None

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
                "Only approve/reject forms are accepted."
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
        action = parse_action_path(parts.path) if not parts.query else None
        if action is None:
            return self.error(404, "Not found")
        lowered: dict[str, str] = {}
        for name, value in headers:
            key = name.lower()
            if key in lowered and key in _SINGLE_HEADERS:
                return self.error(400, "Refused", f"duplicate {key} header")
            lowered.setdefault(key, value)
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
            return _html(200, render_overview(view, actions=actions), security=self._security)
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
            return _html(200, render_project(pview, forms, notice), security=self._security)
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
            if length > MAX_FORM_BYTES:
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
) -> ThreadingHTTPServer:
    """A bound (not yet serving) server. ``port=0`` picks a free port (tests).

    ``allow_actions`` (D123) enables approve/reject forms with a fresh CSRF token for
    this server's lifetime; the default is the read-only server of D78."""
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
    app = WebApp(workspace, ledger_path, hosts, actions=actions)
    server.RequestHandlerClass = make_handler(app, log)
    return server


def server_url(server: ThreadingHTTPServer) -> str:
    """The address to open in a browser (accepted by the ``Host`` check)."""
    host, port = str(server.server_address[0]), int(server.server_address[1])
    shown = f"[{host}]" if ":" in host else host
    return f"http://{shown}:{port}/"


def stderr_log(line: str) -> None:
    print(line, file=sys.stderr)
