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

Nothing here writes a file, creates a directory, decides an approval or calls a model.
"""

from __future__ import annotations

import ipaddress
import socket
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from automa.budget.ledger import CostLedger, LedgerError
from automa.project.store import ProjectStore, ProjectStoreError, validate_project_id
from automa.tools.approvals import ApprovalError
from automa.webui.render import (
    CONTENT_SECURITY_POLICY,
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
HEADER_CSP = CONTENT_SECURITY_POLICY + "; frame-ancestors 'none'"
_PROJECT_PREFIX = "/projects/"

SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("Content-Security-Policy", HEADER_CSP),
    ("X-Frame-Options", "DENY"),
    ("X-Content-Type-Options", "nosniff"),
    ("Cache-Control", "no-store"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
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


def _html(status: int, document: str, extra: tuple[tuple[str, str], ...] = ()) -> Response:
    headers = (("Content-Type", "text/html; charset=utf-8"), *SECURITY_HEADERS, *extra)
    return Response(status=status, body=document.encode("utf-8"), headers=headers)


def _error_page(status: int, title: str, detail: str = "") -> Response:
    body = f"<h1>{text(title)}</h1>"
    if detail:
        body += f'<p class="err">{text(detail)}</p>'
    body += '<p><a href="/">← all projects</a></p>'
    return _html(status, page(title, body))


class WebApp:
    """Request → response logic, independent of sockets (unit-testable)."""

    def __init__(self, workspace: Path, ledger_path: Path, allowed: frozenset[str]):
        self._store = ProjectStore.for_workspace(workspace)
        self._ledger_path = ledger_path
        self._allowed = allowed

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

    def handle(self, method: str, target: str, host: str | None) -> Response:
        if method not in ALLOWED_METHODS:
            refused = _error_page(405, "Method not allowed", "This interface is read-only.")
            return refused.with_headers(("Allow", ", ".join(ALLOWED_METHODS)))
        if host is None or host.strip().lower() not in self._allowed:
            return _error_page(
                421, "Unexpected Host header", "Open the address printed by automa web."
            )
        # HEAD gets the same status and headers; the handler omits the body.
        return self._route(target)

    def _route(self, target: str) -> Response:
        path = urlsplit(target).path
        if path == "/":
            spent, err = self._spending()
            view = build_overview(self._store, spent)
            if err:
                view = view.model_copy(update={"errors": (*view.errors, err)})
            return _html(200, render_overview(view))
        if path.startswith(_PROJECT_PREFIX):
            raw = unquote(path[len(_PROJECT_PREFIX) :])
            try:
                pid = validate_project_id(raw)
            except ProjectStoreError:
                return _error_page(404, "Not found")
            if pid not in self._store.list_ids():
                return _error_page(404, "Not found", f"no project {pid}")
            spent, err = self._spending()
            try:
                pview = build_project_view(self._store, pid, spent.get(pid, 0))
            except (ProjectStoreError, ApprovalError) as e:
                return _error_page(500, "Project unreadable", str(e))
            if err:
                pview = pview.model_copy(update={"errors": (*pview.errors, err)})
            return _html(200, render_project(pview))
        return _error_page(404, "Not found")


def make_handler(
    app: WebApp, log: Callable[[str], None] | None = None
) -> type[BaseHTTPRequestHandler]:
    """A request-handler class serving ``app``."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "automa-web"
        sys_version = ""

        def _serve(self) -> None:
            response = app.handle(self.command, self.path, self.headers.get("Host"))
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
) -> ThreadingHTTPServer:
    """A bound (not yet serving) server. ``port=0`` picks a free port (tests)."""
    host = validate_bind_host(host)
    port = validate_port(port)

    family = socket.AF_INET6 if ipaddress.ip_address(host).version == 6 else socket.AF_INET

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        address_family = family

    # Bind first so the Host check uses the real port (port=0 picks a free one).
    server = Server((host, port), BaseHTTPRequestHandler)
    app = WebApp(workspace, ledger_path, allowed_hosts(host, int(server.server_address[1])))
    server.RequestHandlerClass = make_handler(app, log)
    return server


def server_url(server: ThreadingHTTPServer) -> str:
    """The address to open in a browser (accepted by the ``Host`` check)."""
    host, port = str(server.server_address[0]), int(server.server_address[1])
    shown = f"[{host}]" if ":" in host else host
    return f"http://{shown}:{port}/"


def stderr_log(line: str) -> None:
    print(line, file=sys.stderr)
