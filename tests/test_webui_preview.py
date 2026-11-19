"""Opening a project's HTML result in an isolated preview (D136).

A browser test on the dev box confirmed: the page's inline script, relative CSS and
image work; its origin is opaque (``null``), ``document.cookie`` throws and
``fetch('/')`` is blocked. These tests pin the headers and routing that make it so.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from automa.project import ProjectStore
from automa.webui import preview as wp
from automa.webui import server as ws

HOST = "127.0.0.1:8765"


def _png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    raw = zlib.compress(b"\x00" + b"\x00\x00\x00" * 2 + b"\x00" + b"\x00\x00\x00" * 2)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("goal", project_id="p1")
    files = store.files_dir("p1")
    files.mkdir(exist_ok=True)
    (files / "index.html").write_text("<p>hi</p><script>1</script>", encoding="utf-8")
    (files / "style.css").write_text("body{}", encoding="utf-8")
    (files / "app.js").write_text("1", encoding="utf-8")
    (files / "images").mkdir()
    (files / "images" / "hero.png").write_bytes(_png())
    (files / "fake.png").write_bytes(b"<svg onload=alert(1)>")
    (files / "notes.txt").write_text("x", encoding="utf-8")
    (files / "link.html").symlink_to(tmp_path / "secret.html")
    (tmp_path / "secret.html").write_text("SECRET", encoding="utf-8")
    return tmp_path


def _app(workspace: Path, *, actions: bool = False) -> ws.WebApp:
    hosts = ws.allowed_hosts("127.0.0.1", 8765)
    settings = None
    if actions:
        from automa.webui.actions import ActionSettings, allowed_origins

        settings = ActionSettings(csrf_token="t" * 32, origins=allowed_origins(hosts))
    return ws.WebApp(workspace, workspace / "ledger.sqlite3", hosts, actions=settings)


def _get(app: ws.WebApp, target: str) -> ws.Response:
    return app.handle("GET", target, HOST)


@pytest.mark.parametrize("actions", [False, True])
def test_html_is_served_in_an_opaque_sandbox(workspace: Path, actions: bool) -> None:
    r = _get(_app(workspace, actions=actions), "/projects/p1/preview/index.html")
    assert r.status == 200 and b"<p>hi</p>" in r.body
    h = dict(r.headers)
    csp = h["Content-Security-Policy"]
    assert csp.startswith("sandbox allow-scripts")
    assert "allow-same-origin" not in csp and "allow-forms" not in csp
    assert "allow-top-navigation" not in csp and "allow-popups" not in csp
    assert "connect-src 'none'" in csp and "form-action 'none'" in csp
    assert "frame-ancestors 'none'" in csp and "base-uri 'none'" in csp
    assert "http" not in csp  # no external source of any kind
    assert h["X-Frame-Options"] == "DENY" and h["X-Content-Type-Options"] == "nosniff"
    assert h["Content-Type"].startswith("text/html")


def test_assets_are_served_as_non_documents(workspace: Path) -> None:
    app = _app(workspace)
    for path, mime in (
        ("style.css", "text/css"),
        ("app.js", "text/javascript"),
        ("images/hero.png", "image/png"),
    ):
        r = _get(app, f"/projects/p1/preview/{path}")
        assert r.status == 200, path
        h = dict(r.headers)
        assert h["Content-Type"].startswith(mime)
        assert h["Content-Security-Policy"] == wp.PREVIEW_ASSET_CSP


@pytest.mark.parametrize(
    ("target", "status"),
    [
        ("/projects/p1/preview/notes.txt", 415),
        ("/projects/p1/preview/fake.png", 415),
        ("/projects/p1/preview/link.html", 404),
        ("/projects/p1/preview/missing.html", 404),
        ("/projects/p1/preview/../../secret.html", 404),
        ("/projects/p1/preview/%2e%2e/%2e%2e/secret.html", 404),
        ("/projects/p1/preview/", 404),
        ("/projects/nope/preview/index.html", 404),
        ("/projects/..%2fp1/preview/index.html", 404),
    ],
)
def test_refused_paths(workspace: Path, target: str, status: int) -> None:
    r = _get(_app(workspace), target)
    assert r.status == status
    assert b"SECRET" not in r.body


def test_parse_preview_path() -> None:
    assert wp.parse_preview_path("/projects/p1/preview/a/b.html") == ("p1", "a/b.html")
    assert wp.parse_preview_path("/projects/p1/preview/") is None
    assert wp.parse_preview_path("/projects/p1/files/view") is None
    assert wp.parse_preview_path("/projects/p1/preview/../x.html") is None


def test_listing_and_file_view_offer_open(workspace: Path) -> None:
    app = _app(workspace)
    listing = _get(app, "/projects/p1/files").body.decode()
    assert "/projects/p1/preview/index.html" in listing
    assert 'rel="noopener noreferrer"' in listing
    assert "/projects/p1/preview/style.css" not in listing
    view = _get(app, "/projects/p1/files/view?path=index.html").body.decode()
    assert "/projects/p1/preview/index.html" in view
