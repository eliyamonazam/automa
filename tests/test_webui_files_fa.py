"""Tests for the Persian file browser, decision notice and request target lines (D146).

Temporary directories only; no model, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from automa.project import ProjectStore
from automa.tools.approval_decisions import record_decision
from automa.tools.base import RiskLevel
from automa.webui import render as rd
from automa.webui import server as ws
from automa.webui.files import FileContent, FileListing
from automa.webui.files import FileEntryView as FileEntry
from automa.webui.views import approval_view, build_project_view
from test_webui_actions import PORT, _settings
from test_webui_files import _png
from test_webui_i18n import LATIN_WORD, _strip_data

HOST = f"127.0.0.1:{PORT}"
PID = "p-20261120-101010-abcdef"


def _interface(doc: str, *data: str) -> str:
    """Interface text only: without recorded data (ids, file names, file text)."""
    doc = re.sub(r"<title>.*?</title>", "", doc, flags=re.S)
    doc = re.sub(r"<pre[^>]*>.*?</pre>", "", doc, flags=re.S)
    doc = re.sub(r'<(a|span|h1|li) [^>]*dir="auto"[^>]*>.*?</\1>', "", doc, flags=re.S)
    for d in (PID, *data):
        doc = doc.replace(d, "")
    return _strip_data(doc)


def _listing(**kw: object) -> FileListing:
    data: dict[str, object] = {
        "project_id": PID,
        "dir": "src",
        "parent": "",
        "entries": (
            FileEntry(name="lib", path="src/lib", kind="dir", size=0),
            FileEntry(name="index.html", path="src/index.html", kind="file", size=2048),
            FileEntry(name="notes.txt", path="src/notes.txt", kind="file", size=12),
            FileEntry(name="big.bin", path="src/big.bin", kind="file", size=3 * 1024 * 1024),
            FileEntry(name="ln", path="src/ln", kind="symlink", size=0),
            FileEntry(name="fifo", path="src/fifo", kind="other", size=0),
        ),
        "total": 9,
    }
    data.update(kw)
    return FileListing(**data)  # type: ignore[arg-type]


# -- file list -----------------------------------------------------------------------


def test_persian_file_list_has_no_english_interface() -> None:
    doc = rd.render_files(_listing(), actions=True, lang="fa")
    assert '<html lang="fa" dir="rtl">' in doc
    assert "فایل‌های پروژهٔ" in doc and "↑ پوشهٔ بالاتر" in doc
    assert "<th>نام</th><th>اندازه</th>" in doc
    assert "۲٫۰ کیلوبایت" in doc and "۱۲ بایت" in doc and "۳٫۰ مگابایت" in doc
    assert "دانلود" in doc and "▶ باز کردن" in doc
    assert "میان‌بر؛ دنبالش نمی‌کنیم" in doc and "فایل معمولی نیست" in doc
    assert "۶ مورد از ۹ مورد نشون داده شده." in doc
    assert ">فایل‌ها</a>" in doc  # breadcrumb root
    assert LATIN_WORD.findall(_interface(doc)) == []


@pytest.mark.parametrize(
    ("kw", "expected"),
    [({"missing": True}, "هنوز فایلی ساخته نشده."), ({"entries": ()}, "این پوشه خالیه.")],
)
def test_persian_missing_and_empty_directory(kw: dict[str, object], expected: str) -> None:
    doc = rd.render_files(_listing(**kw), lang="fa")
    assert expected in doc
    assert LATIN_WORD.findall(_interface(doc)) == []


def test_english_file_list_is_unchanged() -> None:
    doc = rd.render_files(_listing())
    assert '<html lang="en">' in doc
    assert f"Files of {PID}" in doc and "↑ up" in doc and "<th>Name</th><th>Size</th>" in doc
    assert "2.0 KiB" in doc and "12 B" in doc and "3.0 MiB" in doc
    assert "(symlink, not followed)" in doc and "(not a regular file)" in doc
    assert "Showing 6 of 9 entries." in doc


def test_file_names_are_not_translated_and_stay_escaped() -> None:
    entries = (FileEntry(name="<b>x</b>.txt", path="<b>x</b>.txt", kind="file", size=1),)
    doc = rd.render_files(_listing(entries=entries, dir="", parent=None), lang="fa")
    assert "<b>x</b>" not in doc and "&lt;b&gt;x&lt;/b&gt;.txt" in doc


# -- file view -----------------------------------------------------------------------


def _content(**kw: object) -> FileContent:
    data: dict[str, object] = {"project_id": PID, "path": "src/index.html", "kind": "text"}
    data.update(kw)
    data.setdefault("size", 5000)
    return FileContent(**data)  # type: ignore[arg-type]


def test_persian_text_file_view() -> None:
    doc = rd.render_file(_content(text="Hello <world>", truncated=True), lang="fa")
    assert '<html lang="fa" dir="rtl">' in doc
    assert "۴٫۹ کیلوبایت" in doc and "دانلود" in doc
    assert "بدون اینترنت و بدون دسترسی به Automa" in doc
    assert "فقط ۱۳ بایت اول نشون داده شده" in doc
    assert '<pre dir="auto">Hello &lt;world&gt;</pre>' in doc
    assert LATIN_WORD.findall(_interface(doc)) == []


@pytest.mark.parametrize(
    ("code", "fa"),
    [
        ("too_large", "عکس بزرگ‌تر از ۱۰ مگابایته"),
        ("not_image", "پسوند فایل می‌گه عکسه"),
        ("not_text", "متن سادهٔ قابل‌خوندن نیست"),
        (None, "فایل غیرمتنی"),
    ],
)
def test_persian_not_shown_notes(code: str | None, fa: str) -> None:
    content = _content(path="a.bin", kind="binary", note="english note", note_code=code)
    doc = rd.render_file(content, lang="fa")
    assert fa in doc and "اینجا نشون داده نمی‌شه" in doc and "english note" not in doc
    assert LATIN_WORD.findall(_interface(doc)) == []


def test_english_notes_are_unchanged() -> None:
    for code, note in [
        ("too_large", "image larger than 10 MiB: download it"),
        ("not_image", "the extension says image but the content is not that image type"),
        ("not_text", "not a UTF-8 text file"),
    ]:
        doc = rd.render_file(_content(path="a.bin", kind="binary", note=note, note_code=code))
        assert f"{note}: not shown here." in doc
    doc = rd.render_file(_content(path="a.bin", kind="binary"))
    assert "binary file: not shown here." in doc


def test_persian_image_view_localizes_dimensions() -> None:
    content = _content(path="pic.png", kind="image", image_format="PNG", width=2, height=3)
    doc = rd.render_file(content, img_src="http://127.0.0.1:8765", lang="fa")
    assert "PNG ۲×۳" in doc and 'width="2" height="3"' in doc


def test_read_file_sets_note_codes(tmp_path: Path) -> None:
    from automa.webui.files import read_file

    store = ProjectStore.for_workspace(tmp_path)
    store.create("g", project_id="p1")
    files = store.files_dir("p1")
    files.mkdir(exist_ok=True)
    (files / "fake.png").write_bytes(b"<svg>")
    (files / "blob.bin").write_bytes(b"\x00\xff\xfe")
    (files / "ok.png").write_bytes(_png())
    assert read_file(store, "p1", "fake.png").note_code == "not_image"
    assert read_file(store, "p1", "blob.bin").note_code == "not_text"
    assert read_file(store, "p1", "ok.png").note_code is None


# -- the server ----------------------------------------------------------------------


def _fa_app(root: Path) -> ws.WebApp:
    hosts = ws.allowed_hosts("127.0.0.1", PORT)
    return ws.WebApp(root, root / "ledger.sqlite3", hosts, actions=_settings(), language="fa")


def _store(root: Path) -> ProjectStore:
    store = ProjectStore.for_workspace(root)
    store.create("goal", project_id="p1")
    files = store.files_dir("p1")
    files.mkdir(exist_ok=True)
    (files / "notes.txt").write_text("سلام", encoding="utf-8")
    return store


def test_server_file_pages_are_persian(tmp_path: Path) -> None:
    _store(tmp_path)
    app = _fa_app(tmp_path)
    listing = app.handle("GET", "/projects/p1/files", HOST).body.decode()
    assert '<html lang="fa" dir="rtl">' in listing and "فایل‌های پروژهٔ p1" in listing
    view = app.handle("GET", "/projects/p1/files/view?path=notes.txt", HOST).body.decode()
    assert '<html lang="fa" dir="rtl">' in view and "۸ بایت" in view


@pytest.mark.parametrize(
    ("approve", "expected"),
    [(True, "تأیید شد؛ وقتی اجرا ادامه پیدا کنه"), (False, "رد شد؛ وقتی اجرا ادامه پیدا کنه")],
)
def test_server_decision_notice_is_persian(tmp_path: Path, approve: bool, expected: str) -> None:
    store = _store(tmp_path)
    req = store.approvals("p1").request(
        tool="publish", risk=RiskLevel.EXTERNAL, arguments={"t": 1}, reason="r", task_id="a"
    )
    record_decision(store, "p1", req.id, approve=approve, note=None, via="web")
    body = _fa_app(tmp_path).handle("GET", f"/projects/p1?decided={req.id}", HOST).body.decode()
    assert f"{req.id}: {expected}" in body
    assert "approved;" not in body and "rejected;" not in body


# -- request target lines ------------------------------------------------------------


def _req(store: ProjectStore, tool: str, **args: object):  # type: ignore[no-untyped-def]
    return store.approvals("p1").request(
        tool=tool, risk=RiskLevel.EXTERNAL, arguments=args, reason="r", task_id="a"
    )


def test_persian_fetch_target_lines(tmp_path: Path) -> None:
    store = _store(tmp_path)
    req = _req(store, "fetch_url", url="https://example.com:8443/a/b?x=1&y=2")
    lines = approval_view(req, "fa").request_lines
    assert lines == (
        "میزبان: example.com",
        "درگاه (پورت): 8443",
        "مسیر: /a/b",
        "پارامترهای نشانی: ۲",
    )
    assert approval_view(req).request_lines[:2] == ("host: example.com", "port: 8443")
    bad = approval_view(_req(store, "fetch_url", url="ftp://example.com/"), "fa")
    assert bad.request_lines[0].startswith("نشانی نامعتبر (")
    assert bad.request_lines[0].endswith("؛ ابزار ردش می‌کنه")


def test_persian_github_target_lines(tmp_path: Path) -> None:
    store = _store(tmp_path)
    req = _req(store, "github_read", kind="issue", owner="octo", repo="hello", number=3)
    lines = approval_view(req, "fa").request_lines
    assert lines[0] == "نشانی API: https://api.github.com/repos/octo/hello/issues/3"
    assert lines[1:] == ("نوع: issue", "مخزن: octo/hello")
    bad = approval_view(_req(store, "github_read", kind="nope"), "fa")
    assert bad.request_lines[0].startswith("درخواست نامعتبر (")


def test_project_view_passes_the_language(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _req(store, "fetch_url", url="https://example.com/")
    assert build_project_view(store, "p1", 0, lang="fa").pending[0].request_lines[0] == (
        "میزبان: example.com"
    )
    assert build_project_view(store, "p1", 0).pending[0].request_lines[0] == "host: example.com"


def test_persian_project_page_shows_persian_target_lines(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _req(store, "fetch_url", url="https://example.com/x")
    body = _fa_app(tmp_path).handle("GET", "/projects/p1", HOST).body.decode()
    assert '<li dir="auto">میزبان: example.com</li>' in body
    assert "host: example.com" not in body
