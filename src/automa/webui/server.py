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

``/projects/<pid>/files…`` (D128) browses the project's ``files/`` read-only through
:mod:`automa.webui.files`; only a file view of a checked image gets an ``img-src``,
and only for that project's ``raw`` path.
"""

from __future__ import annotations

import ipaddress
import math
import socket
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, quote, unquote, urlsplit

from automa.budget.ledger import CostLedger, LedgerEntry, LedgerError, nusd_to_usd
from automa.finance.anomaly import AnomalySettings
from automa.l10n import DEFAULT_TIMEZONE, LANGUAGES, Language, resolve_timezone
from automa.project.store import (
    ProjectStatus,
    ProjectStore,
    ProjectStoreError,
    validate_project_id,
)
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
from automa.webui.files import (
    FILE_CONTENT_SECURITY_POLICY,
    FilesRefused,
    FilesRoute,
    build_listing,
    content_disposition,
    download_bytes,
    image_bytes,
    image_source,
    parse_files_path,
    parse_path_query,
    read_file,
)
from automa.webui.i18n import t
from automa.webui.ideas import (
    ANSWER_FIELDS,
    IDEA_FIELDS,
    MAX_IDEA_FORM_BYTES,
    NEW_PATH,
    IdeaFlow,
    IdeaPlanner,
    is_idea_path,
    parse_clarify_path,
    parse_thinking_path,
    project_path,
)
from automa.webui.preview import PREVIEW_HEADERS, parse_preview_path, preview_response
from automa.webui.progress import ProgressView, build_progress
from automa.webui.refusal import Refusal, refusal_key
from automa.webui.render import (
    ACTION_CONTENT_SECURITY_POLICY,
    CONTENT_SECURITY_POLICY,
    ActionForms,
    RunPanel,
    page,
    page_csp,
    render_clarify,
    render_file,
    render_files,
    render_new_idea,
    render_overview,
    render_project,
    render_thinking,
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
from automa.webui.views import ProjectView, build_overview, build_project_view

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


def _replace_csp(security: tuple[tuple[str, str], ...], csp: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, csp) if name == "Content-Security-Policy" else (name, value)
        for name, value in security
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


def _error_detail(
    lang: Language,
    detail: str,
    code: str | None,
    params: Mapping[str, str | int],
    lead: str | None,
) -> str:
    """The detail paragraph(s) of an error page (D147).

    English shows the message exactly as before. Persian shows the translated
    sentence of ``code``; a message without a code (an underlying error) gets a
    Persian lead and stays visible, left-to-right, as a technical detail."""
    if code is not None:
        sentence = t(refusal_key(code), lang, **params)
        return f'<p class="err">{text(sentence)}</p>'
    if not detail:
        return ""
    if lang == "en":
        return f'<p class="err">{text(detail)}</p>'
    lead_text = t(refusal_key(lead) if lead else "error.lead.generic", lang)
    return (
        f'<p class="err">{text(lead_text)}</p>'
        f'<p class="muted">{text(t("error.technical", lang))} '
        f'<code dir="ltr">{text(detail)}</code></p>'
    )


def _error_page(
    status: int,
    title_key: str,
    detail: str = "",
    *,
    security: tuple[tuple[str, str], ...] = SECURITY_HEADERS,
    lang: Language = "en",
    code: str | None = None,
    params: Mapping[str, str | int] | None = None,
    lead: str | None = None,
) -> Response:
    title = t(title_key, lang)
    body = f"<h1>{text(title)}</h1>"
    body += _error_detail(lang, detail, code, params or {}, lead)
    body += f'<p><a href="/">{text(t("error.back", lang))}</a></p>'
    actions = security is not SECURITY_HEADERS
    return _html(status, page(title, body, actions=actions, lang=lang), security=security)


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
        anomaly: AnomalySettings | None = None,
        toman_per_usd: float | None = None,
        background_ideas: bool = False,
        language: Language = "en",
        timezone: str = DEFAULT_TIMEZONE,
        clock: Callable[[], datetime] | None = None,
    ):
        self._store = ProjectStore.for_workspace(workspace)
        if language not in LANGUAGES:
            raise ValueError(f"unknown language: {language!r}")
        # Display only (D140): the language and local time zone of translated pages.
        self._lang: Language = language
        self._tz = resolve_timezone(timezone)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._anomaly = anomaly
        if toman_per_usd is not None and not (math.isfinite(toman_per_usd) and toman_per_usd > 0):
            raise ValueError(f"toman_per_usd must be a positive number, got {toman_per_usd!r}")
        self._rate = toman_per_usd  # display only (D133): never used by the guard
        # Background runs only in action mode (D125); a read-only server never runs.
        if actions is None or runs is None:
            self._runs: BackgroundRuns | None = None
        elif isinstance(runs, BackgroundRuns):
            self._runs = runs
        else:
            self._runs = BackgroundRuns(self._store, runs)
        # New ideas only in action mode (D124); a read-only server never plans.
        self._ideas = (
            IdeaFlow(self._store, planner, background=background_ideas)
            if actions is not None and planner
            else None
        )
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

    def error(
        self,
        status: int,
        title_key: str,
        detail: str = "",
        *,
        code: str | None = None,
        params: Mapping[str, str | int] | None = None,
        lead: str | None = None,
    ) -> Response:
        """An error page with this app's security headers, in this app's language.

        ``title_key`` is an ``error.title.*`` key; ``code`` (a ``refusal.*`` code)
        translates the detail, else ``detail`` is shown as is (technical in ``fa``)."""
        return _error_page(
            status,
            title_key,
            detail,
            security=self._security,
            lang=self._lang,
            code=code,
            params=params,
            lead=lead,
        )

    def refused(self, e: Refusal, title_key: str = "error.title.refused") -> Response:
        """The error page of a :class:`Refusal` (its translated or technical detail)."""
        return self.error(e.status, title_key, e.message, code=e.code, params=e.params, lead=e.lead)

    def _entries(self) -> tuple[list[LedgerEntry], str | None]:
        """All Ledger entries, read-only; never creates a Ledger."""
        if not self._ledger_path.exists() and not self._ledger_path.is_symlink():
            return [], None
        try:
            with CostLedger(self._ledger_path, read_only=True) as ledger:
                return list(ledger.entries()), None
        except LedgerError as e:
            return [], f"ledger: {e}"

    def _totals(self) -> tuple[float, float] | None:
        """Spent today and this month (local calendar, D139); ``None`` if unreadable."""
        entries, err = self._entries()
        if err:
            return None
        now = self._clock().astimezone(self._tz)
        today = month = 0
        for e in entries:
            local = e.ts.astimezone(self._tz)
            if (local.year, local.month) == (now.year, now.month):
                month += e.cost_nusd
                if local.date() == now.date():
                    today += e.cost_nusd
        return nusd_to_usd(today), nusd_to_usd(month)

    def _spending(self) -> tuple[dict[str, int], str | None]:
        """Spending per project from a read-only Ledger; never creates one."""
        entries, err = self._entries()
        spent: dict[str, int] = {}
        for e in entries:
            if e.project_id:
                spent[e.project_id] = spent.get(e.project_id, 0) + e.cost_nusd
        return spent, err

    def _progress(self, view: ProjectView, entries: list[LedgerEntry]) -> ProgressView:
        """Progress of a project page (D127): refreshes while the project or its
        background run is running and no approval waits."""
        running = view.status == ProjectStatus.RUNNING.value
        if self._runs is not None:
            state = self._runs.state(view.id)
            running = running or (state is not None and state.running)
        return build_progress(
            self._store,
            view.id,
            [t.state for t in view.tasks],
            entries,
            running=running,
            pending_approvals=len(view.pending),
            anomaly=self._anomaly,
        )

    def handle(
        self,
        method: str,
        target: str,
        host: str | None,
        headers: Sequence[tuple[str, str]] = (),
        body: bytes = b"",
    ) -> Response:
        if method not in self._methods:
            code = "forms_only" if self._actions is not None else "read_only"
            refused = self.error(405, "error.title.method_not_allowed", code=code)
            return refused.with_headers(("Allow", ", ".join(self._methods)))
        if host is None or host.strip().lower() not in self._allowed:
            return self.error(421, "error.title.unexpected_host", code="open_printed_address")
        if method == "POST":
            return self._post(target, headers, body)
        # HEAD gets the same status and headers; the handler omits the body.
        return self._route(target, host.strip().lower())

    def _post(self, target: str, headers: Sequence[tuple[str, str]], body: bytes) -> Response:
        if self._actions is None:  # POST is only allowed in action mode (never reached)
            return self.error(405, "error.title.method_not_allowed", code="read_only")
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
            return self.error(404, "error.title.not_found")
        lowered = self._single_headers(headers)
        if isinstance(lowered, Response):
            return lowered
        return self._decide(action, lowered, body)

    def _redirect(self, location: str, title_key: str = "redirect.done") -> Response:
        link = f'<p><a href="{text(location)}">{text(t("redirect.continue", self._lang))}</a></p>'
        return _html(
            303,
            page(t(title_key, self._lang), link, actions=True, lang=self._lang),
            (("Location", location),),
            security=self._security,
        )

    def _post_idea(self, path: str, headers: Sequence[tuple[str, str]], body: bytes) -> Response:
        """``POST /new`` or ``/projects/<pid>/clarify`` (D124), after the same checks."""
        if self._actions is None or self._ideas is None:  # never reached
            return self.error(404, "error.title.not_found")
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
            return self.refused(e)
        return self._redirect(location)

    def _post_start(self, pid: str, headers: Sequence[tuple[str, str]], body: bytes) -> Response:
        """``POST /projects/<pid>/start`` (D125), after the same checks."""
        if self._actions is None or self._runs is None:  # never reached
            return self.error(404, "error.title.not_found")
        lowered = self._single_headers(headers)
        if isinstance(lowered, Response):
            return lowered
        try:
            check_same_origin(lowered, self._actions)
            form = parse_form(lowered, body, fields=START_FIELDS)
            check_token(form, self._actions)
            self._runs.start(pid, form)
        except ActionRefused as e:
            return self.refused(e)
        return self._redirect(f"/projects/{quote(pid, safe='')}")

    def _post_stop(self, pid: str, headers: Sequence[tuple[str, str]], body: bytes) -> Response:
        """``POST /projects/<pid>/stop`` (D126), after the same checks."""
        if self._actions is None or self._runs is None:  # never reached
            return self.error(404, "error.title.not_found")
        lowered = self._single_headers(headers)
        if isinstance(lowered, Response):
            return lowered
        try:
            check_same_origin(lowered, self._actions)
            form = parse_form(lowered, body, fields=STOP_FIELDS)
            check_token(form, self._actions)
            self._runs.stop(pid, form)
        except ActionRefused as e:
            return self.refused(e)
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
            exit_code=state.exit_code,
            stopping=state.stop_requested,
        )

    def _single_headers(self, headers: Sequence[tuple[str, str]]) -> dict[str, str] | Response:
        lowered: dict[str, str] = {}
        for name, value in headers:
            key = name.lower()
            if key in lowered and key in _SINGLE_HEADERS:
                return self.error(
                    400, "error.title.refused", code="duplicate_header", params={"name": key}
                )
            lowered.setdefault(key, value)
        return lowered

    def _decide(self, action: ActionTarget, lowered: dict[str, str], body: bytes) -> Response:
        if self._actions is None:  # never reached
            return self.error(405, "error.title.method_not_allowed", code="read_only")
        try:
            check_same_origin(lowered, self._actions)
            form = parse_form(lowered, body)
            check_token(form, self._actions)
            with self._decide_lock:
                decided = decide_from_form(self._store, action, form)
        except ActionRefused as e:
            return self.refused(e)
        location = (
            f"/projects/{quote(action.project_id, safe='')}?decided={quote(decided.id, safe='')}"
        )
        return self._redirect(location, "redirect.recorded")

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
            return t("decided.approved", self._lang, id=req.id)
        return t("decided.rejected", self._lang, id=req.id)

    def _files(self, pid: str, route: FilesRoute, query: str, host: str) -> Response:
        """The read-only browser of ``files/`` (D128)."""
        actions = self._actions is not None
        try:
            path = parse_path_query(query, allow_root=route is FilesRoute.LIST)
            if route is FilesRoute.LIST:
                listing = build_listing(self._store, pid, path)
                document = render_files(listing, actions=actions, lang=self._lang)
                return _html(200, document, security=self._security)
            if route is FilesRoute.VIEW:
                content = read_file(self._store, pid, path)
                img_src = image_source(host, pid) if content.kind == "image" else None
                document = render_file(content, actions=actions, img_src=img_src, lang=self._lang)
                security = self._security
                if img_src is not None:
                    csp = page_csp(actions=actions, img_src=img_src) + "; frame-ancestors 'none'"
                    security = _replace_csp(security, csp)
                return _html(200, document, security=security)
            if route is FilesRoute.RAW:
                data, mime = image_bytes(self._store, pid, path)
                disposition = content_disposition(path, attachment=False)
            else:
                data, mime = download_bytes(self._store, pid, path), "application/octet-stream"
                disposition = content_disposition(path, attachment=True)
        except FilesRefused as e:
            return self.refused(e, "error.title.not_available")
        headers = (
            ("Content-Type", mime),
            ("Content-Disposition", disposition),
            *_replace_csp(self._security, FILE_CONTENT_SECURITY_POLICY),
        )
        return Response(200, data, headers)

    def _thinking(self, pid: str) -> Response:
        """The waiting page of a background Executive step (D137)."""
        assert self._ideas is not None  # noqa: S101 - only routed in idea mode
        job = self._ideas.job(pid)
        if job is None:
            return self._redirect(project_path(pid))
        if job.done and job.next_path is not None:
            return self._redirect(job.next_path)
        if job.done:
            return self.error(409, "error.title.executive_failed", job.error or "unknown error")
        elapsed = int(time.monotonic() - job.started)
        document = render_thinking(pid, job.phase, elapsed, self._lang)
        return _html(200, document, security=self._security)

    def _preview(self, pid: str, path: str) -> Response:
        """A project's HTML result, isolated in an opaque sandbox (D136)."""
        try:
            data, mime, csp = preview_response(self._store, pid, path)
        except FilesRefused as e:
            return self.refused(e, "error.title.not_available")
        headers = (
            ("Content-Type", mime),
            ("Content-Security-Policy", csp),
            *PREVIEW_HEADERS,
        )
        return Response(200, data, headers)

    def _route(self, target: str, host: str = "") -> Response:
        parts = urlsplit(target)
        path = parts.path
        actions = self._actions is not None
        files_target = parse_files_path(path)
        if files_target is not None:
            return self._files(*files_target, parts.query, host)
        preview_target = parse_preview_path(path)
        if preview_target is not None:
            return self._preview(*preview_target)
        if path == "/":
            spent, err = self._spending()
            try:
                show_tests = ("tests", "1") in parse_qsl(parts.query, max_num_fields=4)
            except ValueError:
                show_tests = False
            view = build_overview(self._store, spent, include_tests=show_tests)
            if err:
                view = view.model_copy(update={"errors": (*view.errors, err)})
            document = render_overview(
                view,
                actions=actions,
                ideas=self._ideas is not None,
                rate=self._rate,
                totals=self._totals(),
                lang=self._lang,
                tz=self._tz,
                now=self._clock(),
            )
            return _html(200, document, security=self._security)
        if self._ideas is not None and self._actions is not None:
            forms = ActionForms(self._actions.csrf_token)
            thinking_pid = parse_thinking_path(path)
            if thinking_pid is not None:
                return self._thinking(thinking_pid)
            if path == NEW_PATH:
                return _html(200, render_new_idea(forms, self._lang), security=self._security)
            clarify_pid = parse_clarify_path(path)
            if clarify_pid is not None:
                try:
                    questions = self._ideas.pending_questions(clarify_pid)
                    goal = self._store.load(clarify_pid).goal
                except ActionRefused as e:
                    return self.refused(e, "error.title.not_available")
                except ProjectStoreError as e:
                    return self.error(
                        500, "error.title.project_unreadable", str(e), lead="lead.store"
                    )
                document = render_clarify(clarify_pid, goal, questions, forms, self._lang)
                return _html(200, document, security=self._security)
        if path.startswith(_PROJECT_PREFIX):
            raw = unquote(path[len(_PROJECT_PREFIX) :])
            try:
                pid = validate_project_id(raw)
            except ProjectStoreError:
                return self.error(404, "error.title.not_found")
            if pid not in self._store.list_ids():
                return self.error(
                    404, "error.title.not_found", code="no_project", params={"id": pid}
                )
            entries, err = self._entries()
            own = [e for e in entries if e.project_id == pid]
            try:
                pview = build_project_view(
                    self._store,
                    pid,
                    sum(e.cost_nusd for e in own),
                    planning_nusd=sum(e.cost_nusd for e in own if e.task_id is None),
                    lang=self._lang,
                )
            except (ProjectStoreError, ApprovalError) as e:
                return self.error(500, "error.title.project_unreadable", str(e), lead="lead.store")
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
                progress=self._progress(pview, own),
                rate=self._rate,
                task_costs=_task_costs(own),
                lang=self._lang,
                tz=self._tz,
                now=self._clock(),
            )
            return _html(200, document, security=self._security)
        return self.error(404, "error.title.not_found")


def _task_costs(entries: Sequence[LedgerEntry]) -> dict[str, float]:
    """Cost of each task from its ledger entries (D139)."""
    totals: dict[str, int] = {}
    for e in entries:
        if e.task_id is not None:
            totals[e.task_id] = totals.get(e.task_id, 0) + e.cost_nusd
    return {k: nusd_to_usd(v) for k, v in totals.items()}


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
                return app.error(400, "error.title.refused", code="chunked")
            lengths = self.headers.get_all("Content-Length") or []
            if len(lengths) != 1 or not lengths[0].strip().isdigit():
                return app.error(411, "error.title.length_required")
            length = int(lengths[0].strip())
            if length > app.max_body_bytes(self.path):
                return app.error(413, "error.title.refused", code="form_too_large")
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
    anomaly: AnomalySettings | None = None,
    toman_per_usd: float | None = None,
    language: Language = "en",
    timezone: str = DEFAULT_TIMEZONE,
) -> ThreadingHTTPServer:
    """A bound (not yet serving) server. ``port=0`` picks a free port (tests).

    ``allow_actions`` (D123) enables approve/reject forms with a fresh CSRF token for
    this server's lifetime; the default is the read-only server of D78. ``planner``
    (only used with ``allow_actions``, D124) enables the "New idea" pages; ``runs``
    (only with ``allow_actions``, D125) enables the "Start run" button; ``anomaly``
    is the unusual-spending threshold of the progress panel (``[finance]``, D132);
    ``toman_per_usd`` shows every amount first in rounded toman (D133); ``language``
    and ``timezone`` are ``[display]`` (translated pages and local times, D140)."""
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
    app = WebApp(
        workspace,
        ledger_path,
        hosts,
        actions=actions,
        planner=planner,
        runs=runs,
        anomaly=anomaly,
        toman_per_usd=toman_per_usd,
        language=language,
        timezone=timezone,
        background_ideas=True,  # the browser waits on a "thinking" page (D137)
    )
    server.RequestHandlerClass = make_handler(app, log)
    return server


def server_url(server: ThreadingHTTPServer) -> str:
    """The address to open in a browser (accepted by the ``Host`` check)."""
    host, port = str(server.server_address[0]), int(server.server_address[1])
    shown = f"[{host}]" if ":" in host else host
    return f"http://{shown}:{port}/"


def stderr_log(line: str) -> None:
    print(line, file=sys.stderr)
