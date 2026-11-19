"""Tests for the web UI's action mode (``automa web --allow-actions``, D123).

Approve/reject from the browser: same-origin proof, CSRF token, fingerprint check and
explicit confirmation. Only loopback sockets inside the test process; no model calls.
"""

from __future__ import annotations

import http.client
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote, urlencode

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.project import ProjectStore
from automa.tools.approval_decisions import DECIDED_EVENT, record_decision
from automa.tools.approvals import MAX_NOTE_CHARS, ApprovalRequest, ApprovalStatus
from automa.tools.base import RiskLevel
from automa.webui import actions as wa
from automa.webui import server as ws
from automa.webui.render import ACTION_CONTENT_SECURITY_POLICY, CONTENT_SECURITY_POLICY

LEDGER = "ledger.sqlite3"
PORT = 8765
HOST = f"127.0.0.1:{PORT}"
ORIGIN = f"http://{HOST}"
TOKEN = "t" * 43
FORM = "application/x-www-form-urlencoded"


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else b"<dir>"
        for p in sorted(root.rglob("*"))
    }


def _settings(token: str = TOKEN) -> wa.ActionSettings:
    hosts = ws.allowed_hosts("127.0.0.1", PORT)
    return wa.ActionSettings(csrf_token=token, origins=wa.allowed_origins(hosts))


def _app(workspace: Path, token: str = TOKEN) -> ws.WebApp:
    hosts = ws.allowed_hosts("127.0.0.1", PORT)
    return ws.WebApp(workspace, workspace / LEDGER, hosts, actions=_settings(token))


def _readonly(workspace: Path) -> ws.WebApp:
    return ws.WebApp(workspace, workspace / LEDGER, ws.allowed_hosts("127.0.0.1", PORT))


class Env:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.store = ProjectStore.for_workspace(root)
        self.store.create("goal with <b>html</b>", project_id="p1")
        self.req = self.store.approvals("p1").request(
            tool="publish",
            risk=RiskLevel.EXTERNAL,
            arguments={"text": '<script>alert(1)</script> "quoted" ‮'},
            reason="needs approval",
            task_id="a",
        )
        self.app = _app(root)

    def path(self, action: str = "approve", aid: str | None = None, pid: str = "p1") -> str:
        return f"/projects/{pid}/approvals/{aid or self.req.id}/{action}"

    def form(self, **overrides: str | None) -> bytes:
        fields: dict[str, str | None] = {
            "csrf": TOKEN,
            "fingerprint": self.req.fingerprint,
            "confirm": "yes",
        }
        fields.update(overrides)
        return urlencode({k: v for k, v in fields.items() if v is not None}).encode()

    def post(
        self,
        action: str = "approve",
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        *,
        target: str | None = None,
        host: str | None = HOST,
        app: ws.WebApp | None = None,
    ) -> ws.Response:
        hdrs = {"Origin": ORIGIN, "Content-Type": FORM, "Sec-Fetch-Site": "same-origin"}
        if headers is not None:
            hdrs.update(headers)
        pairs = [(k, v) for k, v in hdrs.items() if v is not None]
        return (app or self.app).handle(
            "POST",
            target or self.path(action),
            host,
            pairs,
            self.form() if body is None else body,
        )

    def get_req(self) -> ApprovalRequest:
        return self.store.approvals("p1").get(self.req.id)

    def events(self) -> list[dict[str, object]]:
        return [e.data for e in self.store.events("p1").read() if e.type == DECIDED_EVENT]


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def _h(r: ws.Response) -> dict[str, str]:
    return dict(r.headers)


# -- shared decision -------------------------------------------------------------------


def test_record_decision_keeps_cli_event_shape(env: Env) -> None:
    decided = record_decision(env.store, "p1", env.req.id, approve=False, note="  ")
    assert decided.status is ApprovalStatus.REJECTED and decided.note is None
    assert env.events() == [
        {
            "approval": env.req.id,
            "task": "a",
            "tool": "publish",
            "status": "rejected",
            "note": False,
        }
    ]


def test_record_decision_marks_web(env: Env) -> None:
    record_decision(env.store, "p1", env.req.id, approve=True, note=" ok ", via="web")
    assert env.get_req().note == "ok"
    assert env.events()[0]["via"] == "web" and env.events()[0]["status"] == "approved"


# -- default stays read-only -----------------------------------------------------------


def test_default_server_refuses_post_even_with_a_valid_form(env: Env) -> None:
    before = _tree(env.root)
    r = env.post(app=_readonly(env.root))
    assert r.status == 405 and _h(r)["Allow"] == "GET, HEAD"
    assert _tree(env.root) == before


def test_default_page_has_no_forms(env: Env) -> None:
    body = _readonly(env.root).handle("GET", "/projects/p1", HOST).body.decode()
    assert "<form" not in body and TOKEN not in body
    assert "form-action 'none'" in body


# -- action-mode pages -------------------------------------------------------------------


def test_action_page_has_forms_token_and_fingerprint(env: Env) -> None:
    r = env.app.handle("GET", "/projects/p1", HOST)
    body = r.body.decode()
    assert r.status == 200
    assert body.count("<form") == 2 and "<script" not in body
    assert f'name="csrf" value="{TOKEN}"' in body
    assert f'name="fingerprint" value="{env.req.fingerprint}"' in body
    assert f'action="{env.path("approve")}"' in body and f'action="{env.path("reject")}"' in body
    assert 'name="confirm" value="yes" required' in body
    # the arguments stay escaped, bidi made visible
    assert "<script>alert" not in body and "&lt;script&gt;" in body and "\\u202e" in body
    assert "‮" not in body


def test_action_page_security_headers(env: Env) -> None:
    for target in ("/", "/projects/p1", "/nope"):
        h = _h(env.app.handle("GET", target, HOST))
        csp = h["Content-Security-Policy"]
        assert "form-action 'self'" in csp and "frame-ancestors 'none'" in csp
        assert "default-src 'none'" in csp and "script-src" not in csp
        assert h["Referrer-Policy"] == "same-origin"
        assert h["X-Frame-Options"] == "DENY" and h["Cache-Control"] == "no-store"
        assert "Access-Control-Allow-Origin" not in h


def test_action_csp_only_changes_form_action() -> None:
    assert ACTION_CONTENT_SECURITY_POLICY == CONTENT_SECURITY_POLICY.replace(
        "form-action 'none'", "form-action 'self'"
    )


def test_overview_says_actions_enabled(env: Env) -> None:
    body = env.app.handle("GET", "/", HOST).body.decode()
    assert "Actions enabled" in body and "<form" not in body


# -- successful decisions ----------------------------------------------------------------


def test_approve_records_once_and_redirects(env: Env) -> None:
    r = env.post(body=env.form(note="  fine once  "))
    assert r.status == 303
    loc = _h(r)["Location"]
    assert loc == f"/projects/p1?decided={env.req.id}"
    decided = env.get_req()
    assert decided.status is ApprovalStatus.APPROVED and decided.note == "fine once"
    assert env.events() == [
        {
            "approval": env.req.id,
            "task": "a",
            "tool": "publish",
            "status": "approved",
            "note": True,
            "via": "web",
        }
    ]
    page = env.app.handle("GET", loc, HOST).body.decode()
    assert "approved; the call runs at most once" in page and "Pending approvals (0)" in page


def test_reject_needs_no_confirmation(env: Env) -> None:
    r = env.post("reject", body=env.form(confirm=None, note="not yet"))
    assert r.status == 303
    assert env.get_req().status is ApprovalStatus.REJECTED
    page = env.app.handle("GET", _h(r)["Location"], HOST).body.decode()
    assert "rejected; the model is told when the run continues" in page


def test_referer_is_accepted_when_origin_is_missing(env: Env) -> None:
    r = env.post(headers={"Origin": None, "Referer": f"{ORIGIN}/projects/p1"})  # type: ignore[dict-item]
    assert r.status == 303


@pytest.mark.parametrize("origin", ["http://localhost:8765", "HTTP://LOCALHOST:8765"])
def test_localhost_origin_is_accepted(env: Env, origin: str) -> None:
    assert env.post(headers={"Origin": origin}).status == 303


def test_missing_sec_fetch_site_is_tolerated(env: Env) -> None:
    assert env.post(headers={"Sec-Fetch-Site": None}).status == 303  # type: ignore[dict-item]


def test_notice_ignores_bad_queries(env: Env) -> None:
    for q in ("?decided=nope", "?decided=../x", "?x=1", f"?decided={env.req.id}&decided=y"):
        body = env.app.handle("GET", "/projects/p1" + q, HOST).body.decode()
        assert "the call runs at most once" not in body and r"rejected;" not in body
    # still pending: no notice
    body = env.app.handle("GET", f"/projects/p1?decided={env.req.id}", HOST).body.decode()
    assert "approved;" not in body


# -- refusals (nothing changes) --------------------------------------------------------


def _refused(env: Env, r: ws.Response, status: int) -> None:
    assert r.status == status, r.body.decode()
    assert env.get_req().status is ApprovalStatus.PENDING
    assert env.events() == []
    assert "Location" not in _h(r)


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Origin": "null"},
        {"Origin": "http://127.0.0.1:8766"},
        {"Origin": "https://127.0.0.1:8765"},
        {"Origin": "http://127.0.0.1:8765.evil"},
        {"Origin": None, "Referer": None},
        {"Origin": None, "Referer": "http://evil.example/projects/p1"},
        {"Origin": None, "Referer": "http://user@127.0.0.1:8765/"},
        {"Origin": None, "Referer": "not a url"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
        {"Sec-Fetch-Site": "none"},
    ],
)
def test_cross_origin_posts_are_refused(env: Env, headers: dict[str, str]) -> None:
    before = _tree(env.root)
    _refused(env, env.post(headers=headers), 403)
    assert _tree(env.root) == before


@pytest.mark.parametrize("token", [None, "", "x", TOKEN + "x", TOKEN[:-1], "T" * 43])
def test_wrong_or_missing_token_is_refused(env: Env, token: str | None) -> None:
    before = _tree(env.root)
    _refused(env, env.post(body=env.form(csrf=token)), 403)
    assert _tree(env.root) == before


def test_token_of_another_session_is_refused(env: Env) -> None:
    other = _app(env.root, token="o" * 43)
    _refused(env, env.post(app=other), 403)


@pytest.mark.parametrize(
    "ctype", [None, "text/plain", "application/json", "multipart/form-data; boundary=x"]
)
def test_only_urlencoded_forms(env: Env, ctype: str | None) -> None:
    _refused(env, env.post(headers={"Content-Type": ctype}), 415)  # type: ignore[dict-item]


def test_urlencoded_with_charset_is_fine(env: Env) -> None:
    assert env.post(headers={"Content-Type": FORM + "; charset=UTF-8"}).status == 303


@pytest.mark.parametrize(
    "body",
    [
        b"csrf=%ff%fe",
        b"\xff=1",
        b"csrf=" + TOKEN.encode() + b"&csrf=" + TOKEN.encode(),
        b"csrf=" + TOKEN.encode() + b"&evil=1",
        b"&&&&&",
        b"a=1&b=2&c=3&d=4&e=5",
    ],
)
def test_malformed_forms_are_refused(env: Env, body: bytes) -> None:
    _refused(env, env.post(body=body), 400)


def test_oversized_form_is_refused(env: Env) -> None:
    _refused(env, env.post(body=env.form(note="x" * wa.MAX_FORM_BYTES)), 413)


def test_too_long_note_is_refused(env: Env) -> None:
    _refused(env, env.post(body=env.form(note="n" * (MAX_NOTE_CHARS + 1))), 400)


def test_approve_needs_confirmation(env: Env) -> None:
    _refused(env, env.post(body=env.form(confirm=None)), 400)
    _refused(env, env.post(body=env.form(confirm="on")), 400)


def test_fingerprint_must_match_the_shown_request(env: Env) -> None:
    _refused(env, env.post(body=env.form(fingerprint="0" * 64)), 409)
    _refused(env, env.post(body=env.form(fingerprint=None)), 409)


def test_decided_request_cannot_be_decided_again(env: Env) -> None:
    assert env.post().status == 303
    r = env.post("reject", body=env.form(confirm=None))
    assert r.status == 409
    assert env.get_req().status is ApprovalStatus.APPROVED and len(env.events()) == 1


def test_fingerprint_of_one_request_cannot_approve_another(env: Env) -> None:
    other = env.store.approvals("p1").request(
        tool="publish", risk=RiskLevel.EXTERNAL, arguments={"text": "other"}, reason="r"
    )
    r = env.post(target=env.path(aid=other.id))  # form carries env.req's fingerprint
    assert r.status == 409
    assert env.store.approvals("p1").get(other.id).status is ApprovalStatus.PENDING


@pytest.mark.parametrize(
    "target",
    [
        "/projects/p1/approvals/apr-000000000000/approve",
        "/projects/nope/approvals/apr-000000000000/approve",
        "/projects/..%2f/approvals/x/approve",
        "/projects/p1/approvals/../../x/approve",
        "/projects/p1/approvals/x/delete",
        "/projects/p1",
        "/",
        "/projects/p1/approvals/{id}/approve?x=1",
        "/projects/p1/approvals/{id}/approve/",
    ],
)
def test_unknown_action_targets_are_404(env: Env, target: str) -> None:
    before = _tree(env.root)
    r = env.post(target=target.replace("{id}", env.req.id))
    _refused(env, r, 404)
    assert _tree(env.root) == before


def test_wrong_host_is_refused_before_anything(env: Env) -> None:
    for host in (None, "evil.example:8765", "127.0.0.1"):
        _refused(env, env.post(host=host), 421)


def test_duplicate_origin_header_is_refused(env: Env) -> None:
    pairs = [
        ("Origin", ORIGIN),
        ("Origin", "http://evil.example"),
        ("Content-Type", FORM),
    ]
    r = env.app.handle("POST", env.path(), HOST, pairs, env.form())
    _refused(env, r, 400)


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH", "OPTIONS"])
def test_other_methods_still_405(env: Env, method: str) -> None:
    r = env.app.handle(method, env.path(), HOST, [("Origin", ORIGIN)], env.form())
    _refused(env, r, 405)
    assert _h(r)["Allow"] == "GET, HEAD, POST"


def test_concurrent_posts_decide_once(env: Env) -> None:
    results: list[int] = []
    barrier = threading.Barrier(8)

    def worker(action: str) -> None:
        barrier.wait()
        body = env.form() if action == "approve" else env.form(confirm=None)
        results.append(env.post(action, body=body).status)

    threads = [
        threading.Thread(target=worker, args=("approve" if i % 2 else "reject",)) for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert sorted(results).count(303) == 1 and results.count(409) == 7
    assert len(env.events()) == 1


# -- unit pieces ---------------------------------------------------------------------


def test_tokens_are_random_and_long() -> None:
    tokens = {wa.new_csrf_token() for _ in range(20)}
    assert len(tokens) == 20 and all(len(t) >= 43 for t in tokens)


def test_allowed_origins_follow_hosts() -> None:
    assert wa.allowed_origins(ws.allowed_hosts("127.0.0.1", 9000)) == {
        "http://127.0.0.1:9000",
        "http://localhost:9000",
    }
    assert "http://[::1]:9000" in wa.allowed_origins(ws.allowed_hosts("::1", 9000))


def test_parse_action_path() -> None:
    t = wa.parse_action_path("/projects/p1/approvals/apr-abc/reject")
    assert t == wa.ActionTarget(project_id="p1", approval_id="apr-abc", approve=False)
    assert wa.parse_action_path("/projects/p1/approvals/apr-abc/approve/x") is None
    assert wa.parse_action_path("/projects/../approvals/a/approve") is None


def test_actions_module_decides_only_through_record_decision() -> None:
    source = Path(wa.__file__).read_text(encoding="utf-8")
    assert ".decide(" not in source and "record_decision(" in source
    for needle in ("write_text", "write_bytes", "open(", "automa.llm", "Access-Control"):
        assert needle not in source, needle
    server_source = Path(ws.__file__).read_text(encoding="utf-8")
    assert ".decide(" not in server_source and "Access-Control" not in server_source


# -- real loopback socket ------------------------------------------------------------


@pytest.fixture
def running(env: Env) -> Iterator[int]:
    server = ws.create_server(env.root, env.root / LEDGER, port=0, allow_actions=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _raw(
    port: int, method: str, path: str, headers: dict[str, str], body: bytes = b""
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    for k, v in headers.items():
        conn.putheader(k, v)
    conn.endheaders()
    if body:
        conn.send(body)
    resp = conn.getresponse()
    data = resp.read()
    hdrs = {k: v for k, v in resp.getheaders()}
    conn.close()
    return resp.status, hdrs, data


def _token(port: int) -> str:
    _, _, body = _raw(port, "GET", "/projects/p1", {"Host": f"127.0.0.1:{port}"})
    m = re.search(r'name="csrf" value="([^"]+)"', body.decode())
    assert m is not None
    return m.group(1)


def test_socket_full_approve_flow(env: Env, running: int) -> None:
    port = running
    token = _token(port)
    body = urlencode(
        {"csrf": token, "fingerprint": env.req.fingerprint, "confirm": "yes", "note": "سلام"}
    ).encode()
    headers = {
        "Host": f"127.0.0.1:{port}",
        "Origin": f"http://127.0.0.1:{port}",
        "Content-Type": FORM,
        "Content-Length": str(len(body)),
    }
    path = f"/projects/p1/approvals/{quote(env.req.id)}/approve"
    status, hdrs, _ = _raw(port, "POST", path, headers, body)
    assert status == 303 and hdrs["Location"].startswith("/projects/p1?decided=")
    assert env.get_req().status is ApprovalStatus.APPROVED and env.get_req().note == "سلام"


def test_socket_tokens_differ_per_server(env: Env, running: int) -> None:
    other = ws.create_server(env.root, env.root / LEDGER, port=0, allow_actions=True)
    thread = threading.Thread(target=other.serve_forever, daemon=True)
    thread.start()
    try:
        assert _token(running) != _token(int(other.server_address[1]))
    finally:
        other.shutdown()
        other.server_close()
        thread.join(timeout=5)


def test_socket_body_framing_is_strict(env: Env, running: int) -> None:
    port = running
    base = {
        "Host": f"127.0.0.1:{port}",
        "Origin": f"http://127.0.0.1:{port}",
        "Content-Type": FORM,
    }
    path = f"/projects/p1/approvals/{env.req.id}/approve"
    assert _raw(port, "POST", path, base)[0] == 411
    chunked = {**base, "Transfer-Encoding": "chunked"}
    assert _raw(port, "POST", path, chunked, b"0\r\n\r\n")[0] == 400
    huge = {**base, "Content-Length": str(wa.MAX_FORM_BYTES + 1)}
    assert _raw(port, "POST", path, huge)[0] == 413
    bad = {**base, "Content-Length": "-1"}
    assert _raw(port, "POST", path, bad)[0] == 411
    assert env.get_req().status is ApprovalStatus.PENDING


def test_socket_cross_origin_post_is_refused(env: Env, running: int) -> None:
    port = running
    body = urlencode(
        {"csrf": _token(port), "fingerprint": env.req.fingerprint, "confirm": "yes"}
    ).encode()
    headers = {
        "Host": f"127.0.0.1:{port}",
        "Origin": "http://evil.example",
        "Content-Type": FORM,
        "Content-Length": str(len(body)),
    }
    status, _, _ = _raw(port, "POST", f"/projects/p1/approvals/{env.req.id}/approve", headers, body)
    assert status == 403 and env.get_req().status is ApprovalStatus.PENDING


def test_socket_read_only_server_does_not_read_post_body(env: Env) -> None:
    server = ws.create_server(env.root, env.root / LEDGER, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        status, hdrs, _ = _raw(
            port,
            "POST",
            f"/projects/p1/approvals/{env.req.id}/approve",
            {"Host": f"127.0.0.1:{port}", "Content-Length": "5"},
            b"a=b&c",
        )
        assert status == 405 and hdrs["Allow"] == "GET, HEAD"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# -- CLI ---------------------------------------------------------------------------------


def _config(tmp_path: Path) -> Path:
    cfg = tmp_path / "automa.toml"
    cfg.write_text(f'workspace = "{tmp_path / "ws"}"\n', encoding="utf-8")
    return cfg


class _FakeServer:
    server_address = ("127.0.0.1", 9123)

    def serve_forever(self) -> None:
        raise KeyboardInterrupt

    def server_close(self) -> None:
        pass


@pytest.mark.parametrize("flag", [True, False])
def test_cli_allow_actions_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: bool
) -> None:
    seen: dict[str, object] = {}

    def fake_create(workspace: Path, ledger: Path, **kw: object) -> _FakeServer:
        seen.update(kw)
        return _FakeServer()

    monkeypatch.setattr(ws, "create_server", fake_create)
    args = ["web", "--config", str(_config(tmp_path))] + (["--allow-actions"] if flag else [])
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert seen["allow_actions"] is flag
    if flag:
        assert "actions enabled" in result.output and "read-only" not in result.output
    else:
        assert "read-only" in result.output and "actions enabled" not in result.output
