# ruff: noqa: S106 - test passwords
"""Tests for the web UI login (M15, D181): login page, session cookie, per-session CSRF,
logout, session end and the login limiter.

Only in-process calls and loopback sockets; no network, no model calls. Accounts are
made with the smallest allowed scrypt cost so the tests stay fast.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import pytest

from automa.project import ProjectStore
from automa.tools.approvals import ApprovalStatus
from automa.tools.base import RiskLevel
from automa.users import LOCKOUT, MAX_FAILED_ATTEMPTS, UserRole, UserStore
from automa.webui import actions as wa
from automa.webui import auth as wauth
from automa.webui import server as ws
from automa.webui.upload import MAX_UPLOAD_BYTES

LEDGER = "ledger.sqlite3"
PORT = 8765
HOST = f"127.0.0.1:{PORT}"
ORIGIN = f"http://{HOST}"
SERVER_TOKEN = "s" * 43
LOGIN_TOKEN = "l" * 43
FORM = "application/x-www-form-urlencoded"
N = 2**14
PW = "correct horse 1"
PW_KID = "another-pass 2"


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


class Env:
    def __init__(self, root: Path, *, actions: bool = True, lang: str = "en") -> None:
        self.root = root
        self.clock = Clock()
        self.store = ProjectStore.for_workspace(root)
        self.store.create("secret family goal <b>x</b>", project_id="p1")
        self.req = self.store.approvals("p1").request(
            tool="publish",
            risk=RiskLevel.EXTERNAL,
            arguments={"text": "hi"},
            reason="needs approval",
            task_id="a",
        )
        self.users = UserStore(root, now=self.clock, scrypt_n=N)
        self.users.add("eli", PW, UserRole.OWNER)
        self.users.add("dad", PW_KID, UserRole.MEMBER)
        hosts = ws.allowed_hosts("127.0.0.1", PORT)
        origins = wa.allowed_origins(hosts)
        self.auth = wauth.AuthSettings(
            users=self.users,
            origins=origins,
            login_token=LOGIN_TOKEN,
            sessions=wauth.SessionStore(now=self.clock),
            limiter=wauth.LoginLimiter(now=self.clock),
            now=self.clock,
        )
        settings = wa.ActionSettings(csrf_token=SERVER_TOKEN, origins=origins) if actions else None
        self.app = ws.WebApp(
            root,
            root / LEDGER,
            hosts,
            actions=settings,
            auth=self.auth,
            language=lang,  # type: ignore[arg-type]
        )

    # -- requests ---------------------------------------------------------------------

    def get(self, target: str = "/", cookie: str | None = None) -> ws.Response:
        headers = [("Cookie", f"automa_session={cookie}")] if cookie else []
        return self.app.handle("GET", target, HOST, headers)

    def post(
        self,
        target: str,
        fields: dict[str, str],
        *,
        cookie: str | None = None,
        headers: dict[str, str | None] | None = None,
        client: str = "10.0.0.1",
        raw: bytes | None = None,
    ) -> ws.Response:
        hdrs: dict[str, str | None] = {
            "Origin": ORIGIN,
            "Content-Type": FORM,
            "Sec-Fetch-Site": "same-origin",
        }
        if cookie:
            hdrs["Cookie"] = f"automa_session={cookie}"
        hdrs.update(headers or {})
        pairs = [(k, v) for k, v in hdrs.items() if v is not None]
        body = urlencode(fields).encode() if raw is None else raw
        return self.app.handle("POST", target, HOST, pairs, body, client=client)

    def login(
        self, name: str = "eli", password: str = PW, *, client: str = "10.0.0.1", **kw: object
    ) -> ws.Response:
        fields = {"csrf": LOGIN_TOKEN, "name": name, "password": password}
        return self.post("/login", fields, client=client, **kw)  # type: ignore[arg-type]

    def signed_in(self, name: str = "eli", password: str = PW) -> str:
        r = self.login(name, password)
        assert r.status == 303, r.body.decode()
        return cookie_of(r)

    def csrf_of(self, cookie: str) -> str:
        body = self.get("/", cookie).body.decode()
        m = re.search(r'action="/logout"><input type="hidden" name="csrf" value="([^"]+)"', body)
        assert m is not None
        return m.group(1)


def h(r: ws.Response) -> dict[str, str]:
    return dict(r.headers)


def cookie_of(r: ws.Response) -> str:
    value = h(r)["Set-Cookie"]
    m = re.match(r"automa_session=([A-Za-z0-9_-]{43});", value)
    assert m is not None, value
    return m.group(1)


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


# -- without accounts nothing changes ---------------------------------------------------


def test_no_accounts_keeps_the_old_behaviour(tmp_path: Path) -> None:
    ProjectStore.for_workspace(tmp_path).create("goal", project_id="p1")
    app = ws.WebApp(tmp_path, tmp_path / LEDGER, ws.allowed_hosts("127.0.0.1", PORT))
    r = app.handle("GET", "/", HOST)
    assert r.status == 200 and "Set-Cookie" not in h(r)
    assert app.handle("GET", "/projects/p1", HOST).status == 200
    assert app.handle("GET", "/login", HOST).status == 404
    post = app.handle("POST", "/login", HOST, [("Origin", ORIGIN)], b"")
    assert post.status == 405  # read-only stays GET/HEAD only
    assert not app.accepts_post("/login")
    assert not (tmp_path / "users.json").exists()


def test_empty_users_file_means_no_login(tmp_path: Path) -> None:
    path = tmp_path / "users.json"
    path.write_text('{"schema_version": 1, "users": []}\n')
    os.chmod(path, 0o600)
    app = ws.WebApp(tmp_path, tmp_path / LEDGER, ws.allowed_hosts("127.0.0.1", PORT))
    assert app.handle("GET", "/", HOST).status == 200


def test_default_auth_reads_the_workspace_users(tmp_path: Path) -> None:
    UserStore(tmp_path, scrypt_n=N).add("eli", PW, UserRole.OWNER)
    app = ws.WebApp(tmp_path, tmp_path / LEDGER, ws.allowed_hosts("127.0.0.1", PORT))
    r = app.handle("GET", "/", HOST)
    assert r.status == 303 and h(r)["Location"] == "/login"


# -- everything needs a session ---------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "/",
        "/projects/p1",
        "/projects/p1/files",
        "/projects/p1/download.zip",
        "/new",
        "/nope",
        "/projects/p1/preview/index.html",
    ],
)
def test_pages_redirect_to_login_without_a_session(env: Env, target: str) -> None:
    for method in ("GET", "HEAD"):
        r = env.app.handle(method, target, HOST)
        assert r.status == 303 and h(r)["Location"] == "/login"
        body = r.body.decode().split("</style>", 1)[1]
        assert "secret family goal" not in body and "p1" not in body
        assert 'class="month"' not in body and "newidea" not in body
        assert "<nav>" not in body


def test_static_assets_need_no_session(env: Env) -> None:
    assert env.get("/static/vazirmatn.woff2").status == 200


def test_post_without_session_is_refused_and_changes_nothing(env: Env) -> None:
    fields = {"csrf": SERVER_TOKEN, "fingerprint": env.req.fingerprint, "confirm": "yes"}
    r = env.post(f"/projects/p1/approvals/{env.req.id}/approve", fields)
    assert r.status == 403 and "sign in first" in r.body.decode()
    assert env.store.approvals("p1").get(env.req.id).status is ApprovalStatus.PENDING


def test_host_check_still_comes_first(env: Env) -> None:
    r = env.app.handle("GET", "/login", "evil.example:8765")
    assert r.status == 421


# -- the login page ---------------------------------------------------------------------


def test_login_page(env: Env) -> None:
    r = env.get("/login")
    body = r.body.decode()
    assert r.status == 200
    assert '<form class="card login" method="post" action="/login">' in body
    assert f'name="csrf" value="{LOGIN_TOKEN}"' in body
    assert 'type="password" name="password"' in body and 'autocomplete="current-password"' in body
    assert "<script" not in body and "secret family goal" not in body and "<nav>" not in body
    csp = h(r)["Content-Security-Policy"]
    assert "form-action 'self'" in csp and "frame-ancestors 'none'" in csp
    assert h(r)["Referrer-Policy"] == "same-origin"  # so the browser sends a real Origin
    assert h(r)["Cache-Control"] == "no-store"


def test_login_page_in_persian(tmp_path: Path) -> None:
    env = Env(tmp_path, lang="fa")
    body = env.get("/login").body.decode()
    assert 'dir="rtl"' in body and "ورود به Automa" in body and "نام کاربری" in body
    bad = env.login(password="wrong password").body.decode()
    assert "نام کاربری یا رمز درست نیست." in bad


def test_login_page_redirects_when_signed_in(env: Env) -> None:
    cookie = env.signed_in()
    r = env.get("/login", cookie)
    assert r.status == 303 and h(r)["Location"] == "/"


# -- signing in -------------------------------------------------------------------------


def test_successful_login_sets_a_strict_cookie(env: Env) -> None:
    r = env.login()
    assert r.status == 303 and h(r)["Location"] == "/"
    value = h(r)["Set-Cookie"]
    for flag in ("HttpOnly", "Secure", "SameSite=Strict", "Path=/", "Max-Age=2592000"):
        assert flag in value
    assert "Domain" not in value
    cookie = cookie_of(r)
    page = env.get("/", cookie)
    body = page.body.decode()
    assert page.status == 200 and "secret family goal" in body
    assert "Signed in as eli" in body and 'action="/logout"' in body
    assert h(page)["Referrer-Policy"] == "same-origin"


def test_login_name_ignores_case_and_spaces(env: Env) -> None:
    assert env.login(" Eli ").status == 303


def test_wrong_password_and_unknown_name_look_the_same(env: Env) -> None:
    wrong = env.login(password="wrong password")
    unknown = env.login("nobody", password="wrong password")
    for r in (wrong, unknown):
        assert r.status == 403 and "Set-Cookie" not in h(r)
        assert "Wrong user name or password." in r.body.decode()
    assert "wrong password" not in wrong.body.decode()  # the password is never echoed
    assert 'value="eli"' in wrong.body.decode()


def test_typed_name_is_escaped(env: Env) -> None:
    r = env.login('"><script>x</script>', password="whatever pw")
    body = r.body.decode()
    assert "<script>x" not in body and "&lt;script&gt;" in body


def test_disabled_account(env: Env) -> None:
    env.users.disable("dad")
    ok_pw = env.login("dad", PW_KID)
    assert ok_pw.status == 403 and "disabled" in ok_pw.body.decode()
    wrong = env.login("dad", "wrong password")
    assert "Wrong user name or password." in wrong.body.decode()


def test_account_lock_after_wrong_passwords(env: Env) -> None:
    for i in range(MAX_FAILED_ATTEMPTS - 1):
        assert env.login(password=f"wrong pass {i}", client=f"10.1.0.{i}").status == 403
    locked = env.login(password="wrong again", client="10.2.0.1")
    assert locked.status == 429 and "locked" in locked.body.decode()
    assert h(locked)["Retry-After"] == str(int(LOCKOUT.total_seconds()))
    # even the right password is refused while locked
    assert env.login(client="10.3.0.1").status == 429
    env.clock.advance(LOCKOUT + timedelta(seconds=1))
    assert env.login(client="10.3.0.1").status == 303


def test_client_limit_after_many_failures(env: Env) -> None:
    for i in range(wauth.MAX_CLIENT_FAILURES):
        r = env.login(f"user{i}", "wrong password", client="10.9.9.9")
        assert r.status == 403
    slow = env.login(client="10.9.9.9")  # even the right password waits
    assert slow.status == 429 and "Too many failed sign-ins" in slow.body.decode()
    assert int(h(slow)["Retry-After"]) > 0
    assert env.login(client="10.9.9.10").status == 303  # another client is not affected
    env.clock.advance(wauth.CLIENT_WINDOW)
    assert env.login(client="10.9.9.9").status == 303


@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({"Origin": "http://evil.example"}, 403),
        ({"Origin": "null"}, 403),
        ({"Origin": None}, 403),
        ({"Sec-Fetch-Site": "cross-site"}, 403),
        ({"Content-Type": "text/plain"}, 415),
    ],
)
def test_login_post_needs_same_origin_and_a_form(
    env: Env, headers: dict[str, str | None], status: int
) -> None:
    r = env.login(headers=headers)
    assert r.status == status and "Set-Cookie" not in h(r)


def test_login_post_needs_the_login_token(env: Env) -> None:
    for token in ("", SERVER_TOKEN, "x" * 43):
        fields = {"csrf": token, "name": "eli", "password": PW}
        r = env.post("/login", fields)
        assert r.status == 403 and "Set-Cookie" not in h(r)


def test_login_post_refuses_extra_or_repeated_fields(env: Env) -> None:
    extra = urlencode({"csrf": LOGIN_TOKEN, "name": "eli", "password": PW, "role": "owner"})
    assert env.post("/login", {}, raw=extra.encode()).status == 400
    twice = f"csrf={LOGIN_TOKEN}&name=eli&name=dad&password=x"
    assert env.post("/login", {}, raw=twice.encode()).status == 400


def test_login_refuses_duplicate_origin_headers(env: Env) -> None:
    body = urlencode({"csrf": LOGIN_TOKEN, "name": "eli", "password": PW}).encode()
    pairs = [("Origin", ORIGIN), ("Origin", "http://evil.example"), ("Content-Type", FORM)]
    assert env.app.handle("POST", "/login", HOST, pairs, body).status == 400


def test_login_replaces_any_old_session(env: Env) -> None:
    first = env.signed_in()
    r = env.login(cookie=first)
    second = cookie_of(r)
    assert second != first
    assert env.get("/", first).status == 303  # no fixation: the old token is gone
    assert env.get("/", second).status == 200


def test_planted_cookie_never_becomes_a_session(env: Env) -> None:
    planted = "A" * 43
    r = env.login(cookie=planted)
    assert cookie_of(r) != planted
    assert env.get("/", planted).status == 303


# -- the session ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cookie_header",
    [
        "automa_session=" + "B" * 43,
        "automa_session=short",
        "automa_session=" + "!" * 43,
        "other=1",
        "",
    ],
)
def test_forged_or_malformed_cookies_are_no_session(env: Env, cookie_header: str) -> None:
    r = env.app.handle("GET", "/", HOST, [("Cookie", cookie_header)])
    assert r.status == 303


def test_duplicate_session_cookies_are_ambiguous(env: Env) -> None:
    cookie = env.signed_in()
    header = f"automa_session={cookie}; automa_session={'C' * 43}"
    assert env.app.handle("GET", "/", HOST, [("Cookie", header)]).status == 303
    both = [("Cookie", f"automa_session={cookie}"), ("Cookie", f"automa_session={cookie}")]
    assert env.app.handle("GET", "/", HOST, both).status == 303
    other = [("Cookie", f"theme=dark; automa_session={cookie}")]
    assert env.app.handle("GET", "/", HOST, other).status == 200


def test_idle_session_expires(env: Env) -> None:
    cookie = env.signed_in()
    env.clock.advance(wauth.SESSION_IDLE - timedelta(minutes=1))
    assert env.get("/", cookie).status == 200  # use restarts the idle time
    env.clock.advance(wauth.SESSION_IDLE - timedelta(minutes=1))
    assert env.get("/", cookie).status == 200
    env.clock.advance(wauth.SESSION_IDLE)
    assert env.get("/", cookie).status == 303


def test_session_has_an_absolute_limit(env: Env) -> None:
    cookie = env.signed_in()
    step = wauth.SESSION_IDLE - timedelta(hours=1)
    used = timedelta(0)
    while used + step < wauth.SESSION_MAX:
        env.clock.advance(step)
        used += step
        assert env.get("/", cookie).status == 200
    env.clock.advance(wauth.SESSION_MAX - used)
    assert env.get("/", cookie).status == 303


def test_disabling_the_account_ends_its_sessions(env: Env) -> None:
    cookie = env.signed_in("dad", PW_KID)
    env.users.disable("dad")
    assert env.get("/", cookie).status == 303
    env.users.enable("dad")
    assert env.get("/", cookie).status == 303  # the session was dropped, sign in again


def test_new_password_ends_old_sessions(env: Env) -> None:
    cookie = env.signed_in()
    env.clock.advance(timedelta(seconds=1))
    env.users.set_password("eli", "brand new pass 3")
    assert env.get("/", cookie).status == 303
    assert env.login(password=PW).status == 403
    assert env.login(password="brand new pass 3").status == 303


def test_role_change_ends_the_session(env: Env) -> None:
    cookie = env.signed_in("dad", PW_KID)
    path = env.root / "users.json"
    data = json.loads(path.read_text())
    for u in data["users"]:
        if u["name"] == "dad":
            u["role"] = "owner"
    path.write_text(json.dumps(data))
    assert env.get("/", cookie).status == 303


def test_damaged_users_file_fails_closed(env: Env) -> None:
    cookie = env.signed_in()
    (env.root / "users.json").write_text("{broken")
    for target in ("/", "/login", "/projects/p1"):
        r = env.get(target, cookie)
        assert r.status == 503 and "secret family goal" not in r.body.decode()
    assert env.login().status == 503
    assert env.app.accepts_post("/login")


def test_open_users_file_permissions_fail_closed(env: Env) -> None:
    os.chmod(env.root / "users.json", 0o644)
    r = env.get("/")
    assert r.status == 503 and "chmod 600" in r.body.decode()


# -- per-session CSRF -------------------------------------------------------------------


def test_forms_carry_the_session_token_not_the_server_token(env: Env) -> None:
    cookie = env.signed_in()
    csrf = env.csrf_of(cookie)
    body = env.get("/projects/p1", cookie).body.decode()
    assert SERVER_TOKEN not in body and LOGIN_TOKEN not in body
    assert body.count(f'name="csrf" value="{csrf}"') >= 3  # approve, reject, sign out


def test_approve_with_session_token(env: Env) -> None:
    cookie = env.signed_in()
    target = f"/projects/p1/approvals/{env.req.id}/approve"
    fields = {"fingerprint": env.req.fingerprint, "confirm": "yes"}
    server = env.post(target, {**fields, "csrf": SERVER_TOKEN}, cookie=cookie)
    assert server.status == 403
    assert env.store.approvals("p1").get(env.req.id).status is ApprovalStatus.PENDING
    ok = env.post(target, {**fields, "csrf": env.csrf_of(cookie)}, cookie=cookie)
    assert ok.status == 303
    assert env.store.approvals("p1").get(env.req.id).status is ApprovalStatus.APPROVED


def test_session_tokens_are_not_interchangeable(env: Env) -> None:
    eli = env.signed_in()
    dad = env.signed_in("dad", PW_KID)
    assert env.csrf_of(eli) != env.csrf_of(dad)
    target = f"/projects/p1/approvals/{env.req.id}/reject"
    r = env.post(target, {"csrf": env.csrf_of(eli), "fingerprint": env.req.fingerprint}, cookie=dad)
    assert r.status == 403


def test_post_with_session_still_needs_same_origin(env: Env) -> None:
    cookie = env.signed_in()
    target = f"/projects/p1/approvals/{env.req.id}/reject"
    fields = {"csrf": env.csrf_of(cookie), "fingerprint": env.req.fingerprint}
    r = env.post(target, fields, cookie=cookie, headers={"Origin": "http://evil.example"})
    assert r.status == 403


def test_upload_size_needs_a_session(env: Env) -> None:
    assert env.app.max_body_bytes("/new") == wa.MAX_FORM_BYTES
    forged = [("Cookie", "automa_session=" + "D" * 43)]
    assert env.app.max_body_bytes("/new", forged) == wa.MAX_FORM_BYTES
    cookie = env.signed_in()
    upload_path = "/projects/p1/requests/req-0123456789ab/answer"
    signed = [("Cookie", f"automa_session={cookie}")]
    assert env.app.max_body_bytes(upload_path, signed) == MAX_UPLOAD_BYTES


# -- signing out ------------------------------------------------------------------------


def test_logout_ends_the_session(env: Env) -> None:
    cookie = env.signed_in()
    r = env.post("/logout", {"csrf": env.csrf_of(cookie)}, cookie=cookie)
    assert r.status == 303 and h(r)["Location"] == "/login"
    assert h(r)["Set-Cookie"].startswith("automa_session=; Max-Age=0;")
    assert env.get("/", cookie).status == 303


def test_logout_needs_the_session_token_and_origin(env: Env) -> None:
    cookie = env.signed_in()
    assert env.post("/logout", {"csrf": LOGIN_TOKEN}, cookie=cookie).status == 403
    cross = env.post(
        "/logout",
        {"csrf": env.csrf_of(cookie)},
        cookie=cookie,
        headers={"Origin": "http://evil.example"},
    )
    assert cross.status == 403
    assert env.get("/", cookie).status == 200  # still signed in


def test_logout_without_session_just_goes_to_login(env: Env) -> None:
    r = env.post("/logout", {"csrf": "x"})
    assert r.status == 303 and h(r)["Location"] == "/login"


# -- read-only server with accounts -----------------------------------------------------


def test_read_only_server_accepts_only_login_and_logout_posts(tmp_path: Path) -> None:
    env = Env(tmp_path, actions=False)
    assert env.app.accepts_post("/login") and env.app.accepts_post("/logout")
    assert not env.app.accepts_post("/new")
    cookie = env.signed_in()
    page = env.get("/projects/p1", cookie)
    body = page.body.decode()
    assert page.status == 200 and "approvals/" not in body  # no approve forms
    assert "form-action 'self'" in h(page)["Content-Security-Policy"]
    assert "form-action 'self'" in body  # the meta CSP lets "Sign out" post too
    fields = {"csrf": env.csrf_of(cookie), "fingerprint": env.req.fingerprint}
    other = env.post(f"/projects/p1/approvals/{env.req.id}/reject", fields, cookie=cookie)
    assert other.status == 405
    assert env.post("/logout", {"csrf": env.csrf_of(cookie)}, cookie=cookie).status == 303


# -- pure helpers -----------------------------------------------------------------------


def test_cookie_helpers() -> None:
    token = wauth.new_token()
    assert len(token) == 43
    assert wauth.cookie_token([("Cookie", f"a=1; automa_session={token}")]) == token
    assert wauth.cookie_token([("Cookie", "automa_session=" + "x" * 9000)]) is None
    with pytest.raises(ValueError):
        wauth.session_cookie("bad;token")
    assert "Max-Age=0" in wauth.clear_cookie()


def test_session_store_is_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(wauth, "MAX_SESSIONS", 3)
    clock = Clock()
    users = UserStore(tmp_path, now=clock, scrypt_n=N)
    user = users.add("eli", PW, UserRole.OWNER)
    sessions = wauth.SessionStore(now=clock)
    tokens = [sessions.create(user)[0] for _ in range(4)]
    assert len(sessions) == 3
    assert sessions.get(tokens[0]) is None and sessions.get(tokens[3]) is not None
    assert sessions.peek(tokens[1]) is not None


def test_limiter_global_cap() -> None:
    clock = Clock()
    limiter = wauth.LoginLimiter(now=clock)
    for i in range(wauth.MAX_ATTEMPTS_PER_MINUTE):
        assert limiter.retry_after(f"c{i}") is None
    wait = limiter.retry_after("someone-else")
    assert wait is not None and 1 <= wait <= 61
    clock.advance(timedelta(minutes=1))
    assert limiter.retry_after("someone-else") is None


def test_minutes_left_rounds_up() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert wauth.minutes_left(now + timedelta(seconds=61), now) == 2
    assert wauth.minutes_left(now, now) == 1
    assert wauth.minutes_left(None, now) == 1


# -- real sockets -----------------------------------------------------------------------


@pytest.fixture
def running(tmp_path: Path) -> Iterator[tuple[ws.ThreadingHTTPServer, Path]]:
    ProjectStore.for_workspace(tmp_path).create("socket goal", project_id="p1")
    UserStore(tmp_path, scrypt_n=N).add("eli", PW, UserRole.OWNER)
    server = ws.create_server(tmp_path, tmp_path / LEDGER, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, tmp_path
    finally:
        server.shutdown()
        server.server_close()


def test_socket_login_flow_on_read_only_server(
    running: tuple[ws.ThreadingHTTPServer, Path],
) -> None:
    server, _ = running
    port = int(server.server_address[1])
    host = f"127.0.0.1:{port}"

    def request(
        method: str, path: str, body: bytes = b"", headers: dict[str, str] | None = None
    ) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(method, path, body=body, headers={"Host": host, **(headers or {})})
        return conn.getresponse()

    first = request("GET", "/")
    assert first.status == 303 and first.getheader("Location") == "/login"
    page = request("GET", "/login").read().decode()
    token = re.search(r'name="csrf" value="([^"]+)"', page)
    assert token is not None
    body = urlencode({"csrf": token.group(1), "name": "eli", "password": PW}).encode()
    resp = request(
        "POST",
        "/login",
        body,
        {"Origin": f"http://{host}", "Content-Type": FORM, "Content-Length": str(len(body))},
    )
    assert resp.status == 303
    set_cookie = resp.getheader("Set-Cookie") or ""
    cookie = set_cookie.split(";", 1)[0]
    home = request("GET", "/", headers={"Cookie": cookie})
    assert home.status == 200 and "socket goal" in home.read().decode()
    # an unauthenticated large body is refused before it is read
    big = request(
        "POST",
        "/login",
        b"",
        {"Origin": f"http://{host}", "Content-Type": FORM, "Content-Length": "999999"},
    )
    assert big.status == 413
