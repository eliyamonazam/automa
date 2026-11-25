# ruff: noqa: S106 - test passwords
"""Tests for public mode of the web UI (M15, D187): ``automa web --public --host-name``.

In-process calls and a loopback socket only: no tunnel, no real network, no model calls.
"""

from __future__ import annotations

import http.client
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlencode

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.project import ProjectStore
from automa.users import UserRole, UserStore
from automa.webui import actions as wa
from automa.webui import auth as wauth
from automa.webui import server as ws
from automa.webui.files import image_source
from automa.webui.render import page_csp

NAME = "automa.example.com"
PORT = 8765
LOCAL = f"127.0.0.1:{PORT}"
PUBLIC_ORIGIN = f"https://{NAME}"
LOGIN_TOKEN = "l" * 43
SERVER_TOKEN = "s" * 43
FORM = "application/x-www-form-urlencoded"
PW = "correct horse 1"
N = 2**14


def hosts() -> frozenset[str]:
    return ws.allowed_hosts("127.0.0.1", PORT) | {NAME}


def make_app(root: Path, *, users: bool = True, actions: bool = True) -> ws.WebApp:
    store = UserStore(root, scrypt_n=N)
    if users:
        store.add("eli", PW, UserRole.OWNER)
    origins = ws.browser_origins(hosts(), NAME)
    auth = wauth.AuthSettings(users=store, origins=origins, login_token=LOGIN_TOKEN)
    settings = wa.ActionSettings(csrf_token=SERVER_TOKEN, origins=origins) if actions else None
    return ws.WebApp(
        root, root / "ledger.sqlite3", hosts(), actions=settings, auth=auth, public_host=NAME
    )


def login(
    app: ws.WebApp,
    *,
    origin: str = PUBLIC_ORIGIN,
    host: str = NAME,
    password: str = PW,
    extra: list[tuple[str, str]] | None = None,
    client: str = "127.0.0.1",
) -> ws.Response:
    body = urlencode({"csrf": LOGIN_TOKEN, "name": "eli", "password": password}).encode()
    headers = [
        ("Origin", origin),
        ("Content-Type", FORM),
        ("Sec-Fetch-Site", "same-origin"),
        *(extra or []),
    ]
    return app.handle("POST", "/login", host, headers, body, client=client)


# -- host name ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("automa.example.com", "automa.example.com"),
        ("  Automa.Example.COM. ", "automa.example.com"),
        ("a-b.c-d.ir", "a-b.c-d.ir"),
    ],
)
def test_valid_public_host_names(raw: str, expected: str) -> None:
    assert ws.validate_public_host_name(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "automa",
        "localhost",
        "a.localhost",
        "https://automa.example.com",
        "automa.example.com:443",
        "automa.example.com/x",
        "*.example.com",
        "1.2.3.4",
        "-bad.example.com",
        "bad-.example.com",
        "a..example.com",
        "user@automa.example.com",
        "automa .example.com",
        "x" * 64 + ".com",
    ],
)
def test_invalid_public_host_names_are_refused(raw: str) -> None:
    with pytest.raises(ws.WebServerError):
        ws.validate_public_host_name(raw)


def test_public_origin_is_https_only() -> None:
    origins = ws.browser_origins(hosts(), NAME)
    assert PUBLIC_ORIGIN in origins
    assert f"http://{NAME}" not in origins
    assert f"http://{LOCAL}" in origins
    assert ws.browser_origins(hosts() - {NAME}) == wa.allowed_origins(hosts() - {NAME})


def test_app_refuses_public_name_outside_allowed_hosts(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ws.WebApp(
            tmp_path,
            tmp_path / "ledger.sqlite3",
            ws.allowed_hosts("127.0.0.1", PORT),
            public_host=NAME,
        )


# -- requests -------------------------------------------------------------------------


def test_public_name_needs_sign_in_and_gets_hsts(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    r = app.handle("GET", "/", NAME)
    assert r.status == 303 and dict(r.headers)["Location"] == "/login"
    assert ("Strict-Transport-Security", "max-age=31536000") in r.headers
    # The loopback address works too, with the same login and HSTS (ignored on http).
    local = app.handle("GET", "/", LOCAL)
    assert local.status == 303
    assert ws.HSTS_HEADER in local.headers


def test_other_hosts_are_still_refused(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    for host in ("evil.example.com", f"{NAME}:8765", f"{NAME}.evil.com", None):
        assert app.handle("GET", "/", host).status == 421


def test_login_through_the_tunnel_with_https_origin(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    r = login(app)
    assert r.status == 303, r.body.decode()
    cookie = dict(r.headers)["Set-Cookie"]
    assert "Secure" in cookie and "HttpOnly" in cookie
    token = re.match(r"automa_session=([A-Za-z0-9_-]{43});", cookie)
    assert token is not None
    page = app.handle("GET", "/", NAME, [("Cookie", f"automa_session={token.group(1)}")])
    assert page.status == 200


@pytest.mark.parametrize(
    "origin", [f"http://{NAME}", "https://evil.example.com", f"https://{NAME}:8443"]
)
def test_login_from_wrong_origin_is_refused(tmp_path: Path, origin: str) -> None:
    app = make_app(tmp_path)
    assert login(app, origin=origin).status == 403


def test_https_referer_is_accepted_when_origin_is_missing() -> None:
    settings = wa.ActionSettings(csrf_token=SERVER_TOKEN, origins=ws.browser_origins(hosts(), NAME))
    wa.check_same_origin({"referer": f"https://{NAME}/login"}, settings)
    with pytest.raises(wa.ActionRefused):
        wa.check_same_origin({"referer": f"http://{NAME}/login"}, settings)


def test_no_account_refuses_everything_in_public_mode(tmp_path: Path) -> None:
    app = make_app(tmp_path, users=False)
    for host in (NAME, LOCAL):
        r = app.handle("GET", "/", host)
        assert r.status == 503
        assert ws.HSTS_HEADER in r.headers
        assert "automa users add" in r.body.decode()
    assert app.handle("GET", "/projects/p1", NAME).status == 503


def test_no_account_page_is_persian(tmp_path: Path) -> None:
    store = UserStore(tmp_path, scrypt_n=N)
    origins = ws.browser_origins(hosts(), NAME)
    auth = wauth.AuthSettings(users=store, origins=origins)
    app = ws.WebApp(
        tmp_path, tmp_path / "l.sqlite3", hosts(), auth=auth, public_host=NAME, language="fa"
    )
    body = app.handle("GET", "/", NAME).body.decode()
    assert "هنوز هیچ حسابی تعریف نشده" in body


def test_without_public_mode_no_hsts_and_no_account_is_open(tmp_path: Path) -> None:
    app = ws.WebApp(tmp_path, tmp_path / "l.sqlite3", ws.allowed_hosts("127.0.0.1", PORT))
    r = app.handle("GET", "/", LOCAL)
    assert r.status == 200
    assert all(name != "Strict-Transport-Security" for name, _ in r.headers)
    assert app.handle("GET", "/", NAME).status == 421


# -- login limiter through the tunnel ---------------------------------------------------


def test_one_strangers_failures_do_not_lock_out_others(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    # Distinct per-account lockout (D180) would hide the limiter: use unknown names.
    body = urlencode({"csrf": LOGIN_TOKEN, "name": "nobody", "password": "wrong pass 1"}).encode()
    base = [("Origin", PUBLIC_ORIGIN), ("Content-Type", FORM), ("Sec-Fetch-Site", "same-origin")]
    for _ in range(wauth.MAX_CLIENT_FAILURES):
        r = app.handle(
            "POST",
            "/login",
            NAME,
            [*base, ("CF-Connecting-IP", "203.0.113.9")],
            body,
            client="127.0.0.1",
        )
        assert r.status in (200, 401, 403), r.status
    blocked = app.handle(
        "POST",
        "/login",
        NAME,
        [*base, ("CF-Connecting-IP", "203.0.113.9")],
        body,
        client="127.0.0.1",
    )
    assert blocked.status == 429
    # Another visitor through the same tunnel (same socket address) can still sign in.
    ok = login(app, extra=[("CF-Connecting-IP", "198.51.100.7")])
    assert ok.status == 303


def test_tunnel_client_header_rules() -> None:
    tc = ws.WebApp._tunnel_client
    assert tc([("CF-Connecting-IP", "203.0.113.9")], "127.0.0.1") == "tunnel:203.0.113.9"
    assert tc([("cf-connecting-ip", " 2001:db8::1 ")], "127.0.0.1") == "tunnel:2001:db8::1"
    assert tc([], "127.0.0.1") == "127.0.0.1"
    assert tc([("CF-Connecting-IP", "not-an-ip")], "127.0.0.1") == "127.0.0.1"
    two = [("CF-Connecting-IP", "203.0.113.9"), ("CF-Connecting-IP", "198.51.100.7")]
    assert tc(two, "127.0.0.1") == "127.0.0.1"


def test_tunnel_header_is_ignored_on_the_loopback_host(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    seen: list[str] = []
    original = app._auth.limiter.retry_after

    def spy(client: str) -> int | None:
        seen.append(client)
        return original(client)

    app._auth.limiter.retry_after = spy  # type: ignore[method-assign]
    login(app, host=LOCAL, origin=f"http://{LOCAL}", extra=[("CF-Connecting-IP", "203.0.113.9")])
    login(app, extra=[("CF-Connecting-IP", "203.0.113.9")])
    assert seen == ["127.0.0.1", "tunnel:203.0.113.9"]


# -- images through the tunnel -----------------------------------------------------------


def test_image_source_uses_https_for_the_public_name() -> None:
    src = image_source(NAME, "p1", "https")
    assert src == f"https://{NAME}/projects/p1/files/raw"
    assert f"img-src {src}" in page_csp(img_src=src)
    with pytest.raises(ValueError):
        image_source(NAME, "p1", "javascript")
    with pytest.raises(ValueError):
        page_csp(img_src="ftp://x/y")


def test_image_file_view_through_the_tunnel(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    ProjectStore.for_workspace(tmp_path).create("goal", project_id="p1")
    files = ProjectStore.for_workspace(tmp_path).files_dir("p1")
    files.mkdir(parents=True, exist_ok=True)
    # Smallest valid PNG header + IHDR is enough for the "image" kind check.
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
    )
    (files / "a.png").write_bytes(png)
    r = login(app)
    token = re.match(r"automa_session=([A-Za-z0-9_-]{43});", dict(r.headers)["Set-Cookie"])
    assert token is not None
    cookie = [("Cookie", f"automa_session={token.group(1)}")]
    page = app.handle("GET", "/projects/p1/files/view?path=a.png", NAME, cookie)
    assert page.status == 200, page.body.decode()
    csp = dict(page.headers)["Content-Security-Policy"]
    assert f"img-src https://{NAME}/projects/p1/files/raw" in csp


# -- server and CLI -----------------------------------------------------------------------


def test_create_server_refuses_public_without_an_account(tmp_path: Path) -> None:
    with pytest.raises(ws.WebServerError, match="automa users add"):
        ws.create_server(tmp_path, tmp_path / "l.sqlite3", port=0, public_host=NAME)


def test_create_server_allows_public_with_an_active_account(tmp_path: Path) -> None:
    store = UserStore(tmp_path, scrypt_n=N)
    store.add("eli", PW, UserRole.OWNER)
    store.add("kid", PW, UserRole.MEMBER)
    store.disable("kid")
    server = ws.create_server(tmp_path, tmp_path / "l.sqlite3", port=0, public_host=NAME)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


def test_create_server_refuses_bad_public_name(tmp_path: Path) -> None:
    UserStore(tmp_path, scrypt_n=N).add("eli", PW, UserRole.OWNER)
    with pytest.raises(ws.WebServerError, match="invalid public host name"):
        ws.create_server(tmp_path, tmp_path / "l.sqlite3", port=0, public_host="https://x.com")


@pytest.fixture
def live(tmp_path: Path) -> Iterator[int]:
    UserStore(tmp_path, scrypt_n=N).add("eli", PW, UserRole.OWNER)
    server = ws.create_server(
        tmp_path, tmp_path / "l.sqlite3", port=0, public_host=NAME, allow_actions=True
    )
    assert server.server_address[0] == "127.0.0.1"  # still loopback only
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


def test_live_server_answers_the_public_name(live: int) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", live, timeout=5)
    conn.request("GET", "/", headers={"Host": NAME})
    r = conn.getresponse()
    r.read()
    assert r.status == 303
    assert r.getheader("Strict-Transport-Security") == "max-age=31536000"
    conn.close()
    conn = http.client.HTTPConnection("127.0.0.1", live, timeout=5)
    conn.request("GET", "/", headers={"Host": "evil.example.com"})
    r = conn.getresponse()
    r.read()
    assert r.status == 421
    conn.close()


def _config(tmp_path: Path) -> Path:
    cfg = tmp_path / "automa.toml"
    cfg.write_text(f'workspace = "{tmp_path / "ws"}"\n', encoding="utf-8")
    return cfg


@pytest.mark.parametrize("args", [["--public"], ["--host-name", NAME]])
def test_cli_public_and_host_name_go_together(tmp_path: Path, args: list[str]) -> None:
    result = CliRunner().invoke(cli.app, ["web", *args, "--config", str(_config(tmp_path))])
    assert result.exit_code == cli.EXIT_USER
    assert "--public and --host-name go together" in result.output


def test_cli_public_without_account_fails_clearly(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli.app,
        [
            "web",
            "--public",
            "--host-name",
            NAME,
            "--port",
            "8799",
            "--config",
            str(_config(tmp_path)),
        ],
    )
    assert result.exit_code == cli.EXIT_USER
    assert "automa users add" in result.output


def test_cli_public_passes_the_name_and_prints_https(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    class FakeServer:
        server_address = ("127.0.0.1", 9123)

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            seen["closed"] = True

    def fake_create(workspace: Path, ledger: Path, **kw: object) -> FakeServer:
        seen.update(kw)
        return FakeServer()

    monkeypatch.setattr(ws, "create_server", fake_create)
    result = CliRunner().invoke(
        cli.app,
        [
            "web",
            "--public",
            "--host-name",
            "Automa.Example.com",
            "--config",
            str(_config(tmp_path)),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["public_host"] == "Automa.Example.com" and seen["closed"]
    assert f"https://{NAME}/" in result.output
    assert seen["host"] == "127.0.0.1"


def test_cli_without_public_passes_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeServer:
        server_address = ("127.0.0.1", 9123)

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            pass

    def fake_create(workspace: Path, ledger: Path, **kw: object) -> FakeServer:
        seen.update(kw)
        return FakeServer()

    monkeypatch.setattr(ws, "create_server", fake_create)
    result = CliRunner().invoke(cli.app, ["web", "--config", str(_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert seen["public_host"] is None
    assert "public address" not in result.output
