"""Tests for the read-only ``files/`` browser of the web UI (D128).

Only temporary directories and a loopback socket; no model, no external network.
"""

from __future__ import annotations

import http.client
import os
import struct
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from automa.project import ProjectStore
from automa.tools.workspace import FileTooLarge, Workspace, WorkspaceError
from automa.webui import files as wf
from automa.webui import render
from automa.webui import server as ws

HOST = "127.0.0.1:8765"
LEDGER = "ledger.sqlite3"


def _png(width: int = 2, height: int = 3) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
        + b"\x00" * 8
    )


def _jpeg(width: int = 4, height: int = 5) -> bytes:
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + sof + b"\xff\xd9"


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() and not p.is_symlink() else b"-"
        for p in sorted(root.rglob("*"))
    }


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("goal", project_id="p1")
    store.create("other", project_id="p2")
    files = store.files_dir("p1")
    files.mkdir(exist_ok=True)
    (files / "notes.txt").write_text("سلام <script>alert(1)</script>\u202e", encoding="utf-8")
    (files / "sub").mkdir()
    (files / "sub" / "deep.md").write_text("# deep", encoding="utf-8")
    (files / "pic.png").write_bytes(_png())
    (files / "photo.JPG").write_bytes(_jpeg())
    (files / "fake.png").write_bytes(b"<svg onload=alert(1)>")
    (files / "blob.bin").write_bytes(b"\x00\x01\x02\xff")
    (files / "link").symlink_to("/etc/passwd")
    (files / "dirlink").symlink_to(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    other = store.files_dir("p2")
    other.mkdir(exist_ok=True)
    (other / "p2.txt").write_text("p2 data", encoding="utf-8")
    return tmp_path


def _app(workspace: Path, *, actions: bool = False) -> ws.WebApp:
    hosts = ws.allowed_hosts("127.0.0.1", 8765)
    settings = None
    if actions:
        from automa.webui.actions import ActionSettings, allowed_origins

        settings = ActionSettings(csrf_token="t" * 32, origins=allowed_origins(hosts))
    return ws.WebApp(workspace, workspace / LEDGER, hosts, actions=settings)


def _get(app: ws.WebApp, target: str, host: str = HOST) -> ws.Response:
    return app.handle("GET", target, host)


def _h(r: ws.Response) -> dict[str, str]:
    return dict(r.headers)


# -- Workspace additions -----------------------------------------------------------


def test_read_bytes_reads_and_refuses_large(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"12345")
    w = Workspace(tmp_path)
    assert w.read_bytes("a", max_bytes=5) == (b"12345", False)
    assert w.read_bytes("a", max_bytes=3, truncate=True) == (b"123", True)
    with pytest.raises(FileTooLarge):
        w.read_bytes("a", max_bytes=4)


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_read_bytes_rejects_bad_limit(tmp_path: Path, bad: object) -> None:
    (tmp_path / "a").write_bytes(b"x")
    with pytest.raises(WorkspaceError):
        Workspace(tmp_path).read_bytes("a", max_bytes=bad)  # type: ignore[arg-type]


def test_read_bytes_refuses_symlink_and_dirs(tmp_path: Path) -> None:
    (tmp_path / "real").write_text("x")
    (tmp_path / "ln").symlink_to(tmp_path / "real")
    (tmp_path / "d").mkdir()
    (tmp_path / "dl").symlink_to(tmp_path / "d")
    (tmp_path / "d" / "f").write_text("y")
    w = Workspace(tmp_path)
    for path in ("ln", "d", "dl/f", "../x", "/etc/passwd", "missing"):
        with pytest.raises(WorkspaceError):
            w.read_bytes(path, max_bytes=100)


def test_file_size(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"abc")
    (tmp_path / "ln").symlink_to(tmp_path / "a")
    w = Workspace(tmp_path)
    assert w.file_size("a") == 3
    with pytest.raises(WorkspaceError):
        w.file_size("ln")
    with pytest.raises(WorkspaceError):
        w.file_size("nope")


# -- path parsing --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/projects/p1/files", ("p1", wf.FilesRoute.LIST)),
        ("/projects/p1/files/view", ("p1", wf.FilesRoute.VIEW)),
        ("/projects/p1/files/raw", ("p1", wf.FilesRoute.RAW)),
        ("/projects/p1/files/download", ("p1", wf.FilesRoute.DOWNLOAD)),
        ("/projects/p1/files/", None),
        ("/projects/p1/files/other", None),
        ("/projects/p1/files/view/x", None),
        ("/projects/P1/files", None),
        ("/projects/../files", None),
        ("/projects/p1", None),
        ("/projects/p1/filesx", None),
        ("/files", None),
        ("/projects/%2e%2e/files", None),
    ],
)
def test_parse_files_path(path: str, expected: object) -> None:
    assert wf.parse_files_path(path) == expected


@pytest.mark.parametrize(
    ("query", "allow_root", "expected"),
    [
        ("", True, ""),
        ("path=", True, ""),
        ("path=.", True, ""),
        ("path=a.txt", False, "a.txt"),
        ("path=sub%2Fdeep.md", False, "sub/deep.md"),
        ("path=sub/./deep.md", False, "sub/deep.md"),
        ("path=%D8%B3.txt", False, "س.txt"),
    ],
)
def test_parse_path_query_ok(query: str, allow_root: bool, expected: str) -> None:
    assert wf.parse_path_query(query, allow_root=allow_root) == expected


@pytest.mark.parametrize(
    "query",
    [
        "",
        "path=",
        "path=../x",
        "path=sub/../../x",
        "path=/etc/passwd",
        "path=a%00b",
        "path=a&path=b",
        "path=a&x=1",
        "x=a",
        "path",
        "path=%ff",
        "path=" + "a" * 5000,
    ],
)
def test_parse_path_query_refused(query: str) -> None:
    with pytest.raises(wf.FilesRefused):
        wf.parse_path_query(query, allow_root=False)


def test_files_href_round_trip() -> None:
    href = wf.files_href("p1", wf.FilesRoute.VIEW, "a b/س&x=1.txt")
    assert href.startswith("/projects/p1/files/view?path=")
    query = href.split("?", 1)[1]
    assert wf.parse_path_query(query, allow_root=False) == "a b/س&x=1.txt"
    assert wf.files_href("p1", wf.FilesRoute.LIST) == "/projects/p1/files"


def test_image_source_is_exact_raw_path() -> None:
    assert wf.image_source(HOST, "p1") == "http://127.0.0.1:8765/projects/p1/files/raw"


@pytest.mark.parametrize("bad", ["https://x", "http://a b", "http://a;b", "http://a'b", "x"])
def test_page_csp_refuses_bad_image_source(bad: str) -> None:
    with pytest.raises(ValueError):
        render.page_csp(img_src=bad)


def test_page_csp_only_changes_img_src() -> None:
    csp = render.page_csp(img_src="http://127.0.0.1:1/projects/p1/files/raw")
    assert "img-src http://127.0.0.1:1/projects/p1/files/raw;" in csp
    assert "default-src 'none'" in csp and "form-action 'none'" in csp
    assert "script-src" not in csp


# -- content helpers -----------------------------------------------------------------


def test_decode_text() -> None:
    assert wf.decode_text("سلام".encode(), False) == "سلام"
    assert wf.decode_text("سلام".encode()[:-1], True) == "سلا"
    assert wf.decode_text("سلام".encode()[:-1], False) is None
    assert wf.decode_text(b"\xff\xfe", False) is None
    assert wf.decode_text(b"a\x00b", False) is None


def test_checked_image() -> None:
    assert wf.checked_image("a.png", _png(7, 9)) == ("png", 7, 9)
    assert wf.checked_image("a.jpeg", _jpeg(4, 5)) == ("jpeg", 4, 5)
    assert wf.checked_image("a.png", _jpeg()) is None  # extension and signature disagree
    assert wf.checked_image("a.jpg", _png()) is None
    assert wf.checked_image("a.gif", _png()) is None
    assert wf.checked_image("a.png", b"<svg/>") is None


@pytest.mark.parametrize(
    ("path", "ascii_name"),
    [
        ("a.txt", "a.txt"),
        ("sub/report.md", "report.md"),
        ("سلام.txt", "____.txt"),
        ('a"b\r\n.txt', "a_b__.txt"),
        ("...", "download"),
        (".env", "env"),
    ],
)
def test_ascii_filename(path: str, ascii_name: str) -> None:
    assert wf.ascii_filename(path) == ascii_name


def test_content_disposition_is_safe() -> None:
    value = wf.content_disposition('x/"a;\r\nb".txt', attachment=True)
    assert value.startswith('attachment; filename="')
    assert "\r" not in value and "\n" not in value
    assert value.count('"') == 2
    assert "filename*=UTF-8''%22a%3B%0D%0Ab%22.txt" in value
    assert wf.content_disposition("i.png", attachment=False).startswith("inline;")


# -- listing -------------------------------------------------------------------------


def test_listing_root(workspace: Path) -> None:
    store = ProjectStore.for_workspace(workspace)
    listing = wf.build_listing(store, "p1")
    kinds = {e.name: e.kind for e in listing.entries}
    assert kinds["notes.txt"] == "file"
    assert kinds["sub"] == "dir"
    assert kinds["link"] == "symlink"
    assert kinds["dirlink"] == "symlink"
    assert listing.parent is None and listing.dir == ""
    assert [e.name for e in listing.entries] == sorted(kinds)


def test_listing_subdir(workspace: Path) -> None:
    store = ProjectStore.for_workspace(workspace)
    listing = wf.build_listing(store, "p1", "sub")
    assert [e.path for e in listing.entries] == ["sub/deep.md"]
    assert listing.parent == ""


@pytest.mark.parametrize("directory", ["dirlink", "link", "notes.txt", "nope"])
def test_listing_refuses_non_dirs(workspace: Path, directory: str) -> None:
    store = ProjectStore.for_workspace(workspace)
    with pytest.raises(wf.FilesRefused) as e:
        wf.build_listing(store, "p1", directory)
    assert e.value.status == 404


def test_listing_missing_files_dir(workspace: Path) -> None:
    store = ProjectStore.for_workspace(workspace)
    store.create("g", project_id="p3")
    root = store.files_dir("p3")
    if root.exists():
        root.rmdir()
    listing = wf.build_listing(store, "p3")
    assert listing.missing and listing.entries == ()
    with pytest.raises(wf.FilesRefused):
        wf.build_listing(store, "p3", "sub")
    assert not root.exists()  # never created


def test_listing_unknown_project(workspace: Path) -> None:
    store = ProjectStore.for_workspace(workspace)
    with pytest.raises(wf.FilesRefused) as e:
        wf.build_listing(store, "zz")
    assert e.value.status == 404


def test_files_root_symlink_is_refused(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("g", project_id="p1")
    root = store.files_dir("p1")
    if root.exists():
        root.rmdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "x.txt").write_text("outside")
    root.symlink_to(elsewhere)
    with pytest.raises(wf.FilesRefused) as e:
        wf.build_listing(store, "p1")
    assert e.value.status == 500
    with pytest.raises(wf.FilesRefused):
        wf.read_file(store, "p1", "x.txt")


def test_listing_is_capped(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wf, "MAX_LIST_ENTRIES", 2)
    store = ProjectStore.for_workspace(workspace)
    listing = wf.build_listing(store, "p1")
    assert len(listing.entries) == 2 and listing.total > 2
    assert "Showing 2 of" in render.render_files(listing)


# -- reading ------------------------------------------------------------------------


def test_read_text_file(workspace: Path) -> None:
    store = ProjectStore.for_workspace(workspace)
    content = wf.read_file(store, "p1", "notes.txt")
    assert content.kind == "text" and content.text is not None
    assert content.text.startswith("سلام") and not content.truncated


def test_read_text_truncated(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wf, "MAX_TEXT_BYTES", 5)
    store = ProjectStore.for_workspace(workspace)
    content = wf.read_file(store, "p1", "notes.txt")
    assert content.kind == "text" and content.truncated
    assert content.text == "سل"  # the cut character is dropped


def test_read_image_and_fake_image(workspace: Path) -> None:
    store = ProjectStore.for_workspace(workspace)
    png = wf.read_file(store, "p1", "pic.png")
    assert (png.kind, png.image_format, png.width, png.height) == ("image", "png", 2, 3)
    jpg = wf.read_file(store, "p1", "photo.JPG")
    assert (jpg.kind, jpg.image_format) == ("image", "jpeg")
    fake = wf.read_file(store, "p1", "fake.png")
    assert fake.kind == "binary" and fake.note and "not that image type" in fake.note


def test_large_image_is_not_shown(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wf, "MAX_IMAGE_BYTES", 10)
    store = ProjectStore.for_workspace(workspace)
    content = wf.read_file(store, "p1", "pic.png")
    assert content.kind == "binary" and content.note and "larger" in content.note
    with pytest.raises(wf.FilesRefused) as e:
        wf.image_bytes(store, "p1", "pic.png")
    assert e.value.status == 413


def test_binary_file(workspace: Path) -> None:
    store = ProjectStore.for_workspace(workspace)
    assert wf.read_file(store, "p1", "blob.bin").kind == "binary"


@pytest.mark.parametrize("path", ["link", "dirlink/secret.txt", "sub", "missing.txt"])
def test_read_refuses_symlinks_dirs_missing(workspace: Path, path: str) -> None:
    store = ProjectStore.for_workspace(workspace)
    for fn in (wf.read_file, wf.download_bytes, wf.image_bytes):
        with pytest.raises(wf.FilesRefused):
            fn(store, "p1", path)


def test_image_bytes(workspace: Path) -> None:
    store = ProjectStore.for_workspace(workspace)
    assert wf.image_bytes(store, "p1", "pic.png") == (_png(), "image/png")
    assert wf.image_bytes(store, "p1", "photo.JPG")[1] == "image/jpeg"
    for path in ("fake.png", "notes.txt", "blob.bin"):
        with pytest.raises(wf.FilesRefused) as e:
            wf.image_bytes(store, "p1", path)
        assert e.value.status == 415


def test_download_cap(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wf, "MAX_DOWNLOAD_BYTES", 3)
    store = ProjectStore.for_workspace(workspace)
    with pytest.raises(wf.FilesRefused) as e:
        wf.download_bytes(store, "p1", "blob.bin")
    assert e.value.status == 413


# -- rendering -----------------------------------------------------------------------


def test_render_files_escapes_and_does_not_link_symlinks(workspace: Path) -> None:
    store = ProjectStore.for_workspace(workspace)
    files = store.files_dir("p1")
    (files / "<b>x<i>\u202e.txt").write_text("x")
    doc = render.render_files(wf.build_listing(store, "p1"))
    assert "<b>x<i>" not in doc and "&lt;b&gt;x&lt;i&gt;\\u202e.txt" in doc
    assert "symlink, not followed" in doc
    assert "path=link" not in doc and "path=dirlink" not in doc
    assert "<script" not in doc and "<img" not in doc


def test_render_text_view_escapes(workspace: Path) -> None:
    store = ProjectStore.for_workspace(workspace)
    doc = render.render_file(wf.read_file(store, "p1", "notes.txt"))
    assert "<script>" not in doc and "&lt;script&gt;" in doc
    assert "\\u202e" in doc and "\u202e" not in doc
    assert '<pre dir="auto" class="prose">' in doc and "img-src 'none'" in doc


def test_prose_files_read_in_the_page_font_and_code_stays_monospace(workspace: Path) -> None:
    store = ProjectStore.for_workspace(workspace)
    md = render.render_file(wf.read_file(store, "p1", "sub/deep.md"))
    assert '<pre dir="auto" class="prose">' in md and "pre.prose{" in md
    (store.files_dir("p1") / "main.py").write_text("print('hi')\n", encoding="utf-8")
    code = render.render_file(wf.read_file(store, "p1", "main.py"))
    assert '<pre dir="auto">' in code and 'class="prose"' not in code


def test_render_image_without_source_shows_no_image(workspace: Path) -> None:
    store = ProjectStore.for_workspace(workspace)
    doc = render.render_file(wf.read_file(store, "p1", "pic.png"))
    assert "<img" not in doc and "img-src 'none'" in doc


def test_project_page_links_files(workspace: Path) -> None:
    r = _get(_app(workspace), "/projects/p1")
    assert r.status == 200 and b'href="/projects/p1/files"' in r.body


# -- server --------------------------------------------------------------------------


@pytest.mark.parametrize("actions", [False, True])
def test_server_listing_and_text(workspace: Path, actions: bool) -> None:
    app = _app(workspace, actions=actions)
    r = _get(app, "/projects/p1/files")
    assert r.status == 200 and b"notes.txt" in r.body
    assert "img-src 'none'" in _h(r)["Content-Security-Policy"]
    r = _get(app, "/projects/p1/files/view?path=notes.txt")
    assert r.status == 200 and b"&lt;script&gt;" in r.body
    assert _h(r)["X-Content-Type-Options"] == "nosniff"


def test_server_image_view_csp_is_scoped(workspace: Path) -> None:
    app = _app(workspace)
    r = _get(app, "/projects/p1/files/view?path=pic.png")
    assert r.status == 200
    csp = _h(r)["Content-Security-Policy"]
    assert "img-src http://127.0.0.1:8765/projects/p1/files/raw;" in csp
    assert "frame-ancestors 'none'" in csp and "script-src" not in csp
    assert b'<img src="/projects/p1/files/raw?path=pic.png"' in r.body
    assert b"img-src http://127.0.0.1:8765/projects/p1/files/raw;" in r.body  # meta too
    r = _get(app, "/projects/p1/files/view?path=pic.png", host="localhost:8765")
    assert (
        "img-src http://localhost:8765/projects/p1/files/raw;" in _h(r)["Content-Security-Policy"]
    )


def test_server_fake_image_gets_no_img_src(workspace: Path) -> None:
    r = _get(_app(workspace), "/projects/p1/files/view?path=fake.png")
    assert r.status == 200 and b"<img" not in r.body
    assert "img-src 'none'" in _h(r)["Content-Security-Policy"]


def test_server_raw_image(workspace: Path) -> None:
    r = _get(_app(workspace), "/projects/p1/files/raw?path=pic.png")
    h = _h(r)
    assert r.status == 200 and r.body == _png()
    assert h["Content-Type"] == "image/png"
    assert h["Content-Disposition"].startswith("inline;")
    assert h["Content-Security-Policy"] == wf.FILE_CONTENT_SECURITY_POLICY
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["Cross-Origin-Resource-Policy"] == "same-origin"


@pytest.mark.parametrize("path", ["fake.png", "notes.txt", "link", "../project.json"])
def test_server_raw_refuses(workspace: Path, path: str) -> None:
    r = _get(_app(workspace), f"/projects/p1/files/raw?path={path}")
    assert r.status in (400, 404, 415)
    assert _h(r)["Content-Type"].startswith("text/html")


def test_server_download(workspace: Path) -> None:
    r = _get(_app(workspace), "/projects/p1/files/download?path=notes.txt")
    h = _h(r)
    assert r.status == 200
    assert r.body == (workspace / "projects" / "p1" / "files" / "notes.txt").read_bytes()
    assert h["Content-Type"] == "application/octet-stream"
    assert h["Content-Disposition"].startswith('attachment; filename="notes.txt"')
    assert "sandbox" in h["Content-Security-Policy"]


def test_server_download_binary_and_subdir(workspace: Path) -> None:
    app = _app(workspace)
    assert _get(app, "/projects/p1/files/download?path=blob.bin").body == b"\x00\x01\x02\xff"
    assert _get(app, "/projects/p1/files/download?path=sub%2Fdeep.md").body == b"# deep"


@pytest.mark.parametrize(
    "target",
    [
        "/projects/p1/files/view?path=../project.json",
        "/projects/p1/files/download?path=../../secret.txt",
        "/projects/p1/files/download?path=%2Fetc%2Fpasswd",
        "/projects/p1/files/download?path=link",
        "/projects/p1/files/download?path=dirlink/secret.txt",
        "/projects/p1/files/view?path=dirlink%2Fsecret.txt",
        "/projects/p1/files?path=dirlink",
        "/projects/p1/files?path=..",
        "/projects/p1/files/download?path=..%2Fp2%2Ffiles%2Fp2.txt",
        "/projects/p1/files/download",
        "/projects/p1/files/download?path=a&path=b",
        "/projects/zz/files",
        "/projects/zz/files/download?path=p2.txt",
    ],
)
def test_server_never_leaves_files_dir(workspace: Path, target: str) -> None:
    r = _get(_app(workspace), target)
    assert r.status in (400, 404)
    assert b"TOP SECRET" not in r.body and b"p2 data" not in r.body
    assert b"root:" not in r.body
    assert _h(r)["Content-Type"].startswith("text/html")


def test_server_files_never_writes(workspace: Path) -> None:
    before = _tree(workspace)
    app = _app(workspace)
    for target in (
        "/projects/p1/files",
        "/projects/p1/files?path=sub",
        "/projects/p1/files/view?path=notes.txt",
        "/projects/p1/files/view?path=pic.png",
        "/projects/p1/files/raw?path=pic.png",
        "/projects/p1/files/download?path=blob.bin",
        "/projects/p1/files/view?path=../x",
    ):
        _get(app, target)
    assert _tree(workspace) == before
    assert not (workspace / LEDGER).exists()


def test_server_post_to_files_is_refused(workspace: Path) -> None:
    assert _app(workspace).handle("POST", "/projects/p1/files", HOST).status == 405
    r = _app(workspace, actions=True).handle("POST", "/projects/p1/files/download?path=a", HOST)
    assert r.status in (400, 403, 404)


def test_server_wrong_host_is_refused(workspace: Path) -> None:
    r = _get(_app(workspace), "/projects/p1/files/download?path=notes.txt", host="evil:8765")
    assert r.status == 421 and "سلام".encode() not in r.body


def test_server_swapped_symlink_after_listing(workspace: Path) -> None:
    app = _app(workspace)
    files = workspace / "projects" / "p1" / "files"
    assert _get(app, "/projects/p1/files/view?path=sub%2Fdeep.md").status == 200
    (files / "sub" / "deep.md").unlink()
    (files / "sub").rmdir()
    os.symlink(workspace, files / "sub")
    r = _get(app, "/projects/p1/files/download?path=sub%2Fsecret.txt")
    assert r.status == 404 and b"TOP SECRET" not in r.body


# -- real socket ---------------------------------------------------------------------


@pytest.fixture
def served(workspace: Path) -> Iterator[tuple[str, int]]:
    server = ws.create_server(workspace, workspace / LEDGER, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "127.0.0.1", int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


def test_socket_download_and_head(served: tuple[str, int]) -> None:
    host, port = served
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", "/projects/p1/files/download?path=blob.bin")
    resp = conn.getresponse()
    assert resp.status == 200 and resp.read() == b"\x00\x01\x02\xff"
    assert resp.getheader("Content-Disposition", "").startswith("attachment;")
    assert resp.getheader("Content-Length") == "4"
    conn.close()
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("HEAD", "/projects/p1/files/raw?path=pic.png")
    resp = conn.getresponse()
    assert resp.status == 200 and resp.read() == b""
    assert resp.getheader("Content-Type") == "image/png"
    conn.close()


def test_socket_image_view_matches_port(served: tuple[str, int]) -> None:
    host, port = served
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", "/projects/p1/files/view?path=pic.png")
    resp = conn.getresponse()
    resp.read()
    csp = resp.getheader("Content-Security-Policy", "")
    assert f"img-src http://127.0.0.1:{port}/projects/p1/files/raw;" in csp
    conn.close()
