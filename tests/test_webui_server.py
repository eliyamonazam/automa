"""Tests for the read-only web server (``automa.webui.server``, D78).

Only a loopback socket inside the test process; no model, no external network.
"""

from __future__ import annotations

import http.client
import socket
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.budget import CostLedger
from automa.project import ProjectStore
from automa.tools.base import RiskLevel
from automa.webui import server as ws
from test_ledger import FixedClock, make_pair

CLOCK = FixedClock(datetime(2026, 11, 15, 12, 0, tzinfo=UTC))
ALL_INTERFACES = "0.0.0." + "0"  # refused by the server; built so linters do not flag it
LEDGER = "ledger.sqlite3"


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else b"<dir>"
        for p in sorted(root.rglob("*"))
    }


def _app(workspace: Path, port: int = 8765) -> ws.WebApp:
    return ws.WebApp(workspace, workspace / LEDGER, ws.allowed_hosts("127.0.0.1", port))


def _headers(r: ws.Response) -> dict[str, str]:
    return dict(r.headers)


GOOD_HOST = "127.0.0.1:8765"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("write <b>a</b> poem", project_id="p1")
    store.approvals("p1").request(
        tool="publish",
        risk=RiskLevel.EXTERNAL,
        arguments={"text": "<script>alert(1)</script>"},
        reason="needs approval",
        task_id="a",
    )
    with CostLedger(tmp_path / LEDGER, clock=CLOCK) as led:
        led.record(*make_pair(0.25, project="p1"))
        led.record(*make_pair(0.5, project="p1"))
    return tmp_path


# -- settings ----------------------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1"])
def test_loopback_hosts_are_accepted(host: str) -> None:
    assert ws.validate_bind_host(host)


@pytest.mark.parametrize(
    "host",
    [ALL_INTERFACES, "::", "192.168.1.5", "10.0.0.1", "8.8.8.8", "localhost", "", "example.com"],
)
def test_non_loopback_hosts_are_refused(host: str) -> None:
    with pytest.raises(ws.WebServerError):
        ws.validate_bind_host(host)


@pytest.mark.parametrize("port", [-1, 65536, True, "80"])
def test_invalid_ports_are_refused(port: object) -> None:
    with pytest.raises(ws.WebServerError):
        ws.validate_port(port)  # type: ignore[arg-type]


def test_allowed_hosts() -> None:
    assert ws.allowed_hosts("127.0.0.1", 9000) == {"127.0.0.1:9000", "localhost:9000"}
    assert ws.allowed_hosts("::1", 9000) == {"[::1]:9000", "localhost:9000"}


def test_create_server_refuses_non_loopback_before_binding(tmp_path: Path) -> None:
    with pytest.raises(ws.WebServerError):
        ws.create_server(tmp_path, tmp_path / LEDGER, host=ALL_INTERFACES, port=0)
    assert list(tmp_path.iterdir()) == []


# -- request handling (no socket) --------------------------------------------------------


def test_overview_page_with_ledger_spending(workspace: Path) -> None:
    r = _app(workspace).handle("GET", "/", GOOD_HOST)
    assert r.status == 200
    body = r.body.decode()
    assert 'href="/projects/p1"' in body and "$0.75" in body
    assert "1 request(s) to approve" in body and 'href="/projects/p1#approvals"' in body
    assert "<b>a</b>" not in body and "&lt;b&gt;a&lt;/b&gt;" in body


def test_project_page(workspace: Path) -> None:
    r = _app(workspace).handle("GET", "/projects/p1", GOOD_HOST)
    assert r.status == 200
    body = r.body.decode()
    assert "$0.75" in body and "Pending approvals (1)" in body
    assert "<script" not in body and "&lt;script&gt;" in body


def test_query_string_is_ignored(workspace: Path) -> None:
    assert _app(workspace).handle("GET", "/?x=1", GOOD_HOST).status == 200


@pytest.mark.parametrize(
    "target",
    [
        "/projects/",
        "/projects/missing",
        "/projects/..%2F..%2Fetc",
        "/projects/P1",
        "/projects/p1/extra",
        "/../ledger.sqlite3",
        "/ledger.sqlite3",
        "/favicon.ico",
        "/projects/p1%00",
    ],
)
def test_unknown_paths_are_404(workspace: Path, target: str) -> None:
    r = _app(workspace).handle("GET", target, GOOD_HOST)
    assert r.status == 404
    assert "<script" not in r.body.decode()


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "OPTIONS", "TRACE", "FOO"])
def test_only_get_and_head(workspace: Path, method: str) -> None:
    before = _tree(workspace)
    r = _app(workspace).handle(method, "/projects/p1", GOOD_HOST)
    assert r.status == 405
    assert _headers(r)["Allow"] == "GET, HEAD"
    assert _tree(workspace) == before


@pytest.mark.parametrize(
    "host",
    [
        None,
        "",
        "evil.example:8765",
        "127.0.0.1",
        "127.0.0.1:80",
        "localhost",
        "127.0.0.1:8766",
        "attacker.test",
        "127.0.0.1:8765.evil",
        "0.0.0.0:8765",
    ],
)
def test_wrong_host_header_is_refused(workspace: Path, host: str | None) -> None:
    r = _app(workspace).handle("GET", "/", host)
    assert r.status == 421
    assert "p1" not in r.body.decode()


@pytest.mark.parametrize("host", ["127.0.0.1:8765", "localhost:8765", "LOCALHOST:8765"])
def test_good_host_headers(workspace: Path, host: str) -> None:
    assert _app(workspace).handle("GET", "/", host).status == 200


@pytest.mark.parametrize(
    ("method", "target", "host"),
    [
        ("GET", "/", GOOD_HOST),
        ("GET", "/nope", GOOD_HOST),
        ("POST", "/", GOOD_HOST),
        ("GET", "/", "evil:1"),
        ("GET", "/projects/p1", GOOD_HOST),
    ],
)
def test_security_headers_on_every_response(
    workspace: Path, method: str, target: str, host: str
) -> None:
    h = _headers(_app(workspace).handle(method, target, host))
    assert h["X-Frame-Options"] == "DENY"
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["Cache-Control"] == "no-store"
    assert h["Referrer-Policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in h["Content-Security-Policy"]
    assert "default-src 'none'" in h["Content-Security-Policy"]
    assert "script-src" not in h["Content-Security-Policy"]
    assert h["Content-Type"] == "text/html; charset=utf-8"


def test_head_keeps_status_and_headers(workspace: Path) -> None:
    app = _app(workspace)
    get, head = app.handle("GET", "/", GOOD_HOST), app.handle("HEAD", "/", GOOD_HOST)
    assert head.status == get.status and head.headers == get.headers


def test_no_ledger_means_zero_and_none_is_created(tmp_path: Path) -> None:
    ProjectStore.for_workspace(tmp_path).create("goal", project_id="p1")
    before = _tree(tmp_path)
    app = _app(tmp_path)
    for target in ("/", "/projects/p1"):
        r = app.handle("GET", target, GOOD_HOST)
        assert r.status == 200 and "$0.00" in r.body.decode()
    assert not (tmp_path / LEDGER).exists()
    assert _tree(tmp_path) == before


def test_empty_workspace_creates_nothing(tmp_path: Path) -> None:
    r = _app(tmp_path).handle("GET", "/", GOOD_HOST)
    assert r.status == 200 and "No projects yet" in r.body.decode()
    assert list(tmp_path.iterdir()) == []


def test_serving_does_not_change_any_file(workspace: Path) -> None:
    before = _tree(workspace)
    app = _app(workspace)
    for target in ("/", "/projects/p1", "/projects/nope", "/x"):
        for method in ("GET", "HEAD", "POST"):
            app.handle(method, target, GOOD_HOST)
    assert _tree(workspace) == before


def test_corrupt_ledger_is_an_error_line(tmp_path: Path) -> None:
    ProjectStore.for_workspace(tmp_path).create("goal", project_id="p1")
    (tmp_path / LEDGER).write_bytes(b"not a database at all, sorry" * 10)
    before = _tree(tmp_path)
    for target in ("/", "/projects/p1"):
        r = _app(tmp_path).handle("GET", target, GOOD_HOST)
        assert r.status == 200
        assert "Unreadable data" in r.body.decode() and "ledger" in r.body.decode()
    assert _tree(tmp_path) == before


def test_symlinked_ledger_is_refused(
    workspace: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    other = tmp_path_factory.mktemp("other") / "real.sqlite3"
    (workspace / LEDGER).rename(other)
    (workspace / LEDGER).symlink_to(other)
    r = _app(workspace).handle("GET", "/", GOOD_HOST)
    assert r.status == 200
    body = r.body.decode()
    assert "symlink" in body and "$0.75" not in body


def test_corrupt_project_page_is_500_without_crash(workspace: Path) -> None:
    (workspace / "projects" / "p1" / "project.json").write_text("{broken", encoding="utf-8")
    r = _app(workspace).handle("GET", "/projects/p1", GOOD_HOST)
    assert r.status == 500
    assert _app(workspace).handle("GET", "/", GOOD_HOST).status == 200


def test_approval_page_has_no_decision_controls(workspace: Path) -> None:
    body = _app(workspace).handle("GET", "/projects/p1", GOOD_HOST).body.decode().lower()
    assert "<form" not in body and "<button" not in body and "<script" not in body
    assert "decide in the terminal" in body and "automa approvals" not in body


# -- real loopback socket ----------------------------------------------------------------


@pytest.fixture
def running(workspace: Path) -> Iterator[tuple[int, list[str]]]:
    logs: list[str] = []
    server = ws.create_server(workspace, workspace / LEDGER, port=0, log=logs.append)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1]), logs
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(
    port: int, method: str, path: str, host: str | None = None
) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": host if host is not None else f"127.0.0.1:{port}"}
    conn.putrequest(method, path, skip_host=True)
    for k, v in headers.items():
        conn.putheader(k, v)
    conn.endheaders()
    resp = conn.getresponse()
    resp.read_body = resp.read()  # type: ignore[attr-defined]
    conn.close()
    return resp


def test_server_binds_loopback_and_serves(running: tuple[int, list[str]]) -> None:
    port, logs = running
    r = _request(port, "GET", "/projects/p1")
    assert r.status == 200
    assert r.getheader("X-Frame-Options") == "DENY"
    assert r.getheader("Server", "").startswith("automa-web")
    assert "Python" not in r.getheader("Server", "")
    body = r.read_body  # type: ignore[attr-defined]
    assert int(r.getheader("Content-Length", "0")) == len(body)
    assert b"Pending approvals (1)" in body
    assert logs  # requests are logged (to stderr in the CLI)


def test_server_head_has_no_body(running: tuple[int, list[str]]) -> None:
    port, _ = running
    get = _request(port, "GET", "/")
    head = _request(port, "HEAD", "/")
    assert head.status == 200
    assert head.read_body == b""  # type: ignore[attr-defined]
    assert head.getheader("Content-Length") == get.getheader("Content-Length")


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "BREW"])
def test_server_refuses_other_methods(running: tuple[int, list[str]], method: str) -> None:
    port, _ = running
    r = _request(port, method, "/")
    assert r.status == 405 and r.getheader("Allow") == "GET, HEAD"


def test_server_checks_host_header(running: tuple[int, list[str]]) -> None:
    port, _ = running
    assert _request(port, "GET", "/", host="rebound.example").status == 421
    assert _request(port, "GET", "/", host=f"localhost:{port}").status == 200


def test_server_log_lines_are_made_visible(running: tuple[int, list[str]]) -> None:
    port, logs = running
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(b"GET /\x1b[31m\xe2\x80\xae HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n\r\n" % port)
        assert sock.recv(65536).startswith(b"HTTP/1.")
    assert logs
    assert all("\x1b" not in line and "\u202e" not in line for line in logs)


# -- CLI -------------------------------------------------------------------------


def _config(tmp_path: Path) -> Path:
    cfg = tmp_path / "automa.toml"
    cfg.write_text(f'workspace = "{tmp_path / "ws"}"\n', encoding="utf-8")
    return cfg


def test_cli_refuses_non_loopback_host(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli.app, ["web", "--host", ALL_INTERFACES, "--config", str(_config(tmp_path))]
    )
    assert result.exit_code == cli.EXIT_USER
    assert "loopback" in result.output
    assert not (tmp_path / "ws").exists()


def test_cli_refuses_bad_port(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli.app, ["web", "--port", "0", "--config", str(_config(tmp_path))])
    assert result.exit_code != 0


def test_cli_starts_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeServer:
        server_address = ("127.0.0.1", 9123)

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            seen["closed"] = True

    def fake_create(workspace: Path, ledger: Path, **kw: object) -> FakeServer:
        seen.update(workspace=workspace, ledger=ledger, **kw)
        return FakeServer()

    monkeypatch.setattr(ws, "create_server", fake_create)
    result = CliRunner().invoke(
        cli.app, ["web", "--port", "9123", "--config", str(_config(tmp_path))]
    )
    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:9123/" in result.output and "read-only" in result.output
    assert seen["host"] == "127.0.0.1" and seen["port"] == 9123 and seen["closed"]
    assert seen["ledger"] == tmp_path / "ws" / LEDGER
    assert not (tmp_path / "ws").exists()


def test_cli_reports_port_in_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*a: object, **kw: object) -> None:
        raise OSError("address already in use")

    monkeypatch.setattr(ws, "create_server", fail)
    result = CliRunner().invoke(cli.app, ["web", "--config", str(_config(tmp_path))])
    assert result.exit_code == cli.EXIT_USER
    assert "address already in use" in result.output


def test_module_has_no_write_or_model_calls() -> None:
    source = Path(ws.__file__).read_text(encoding="utf-8")
    for needle in (
        "write_text",
        "write_bytes",
        "mkdir",
        "open(",
        "automa.llm",
        "approve(",
        "reject(",
        ".decide(",
    ):
        assert needle not in source, needle
