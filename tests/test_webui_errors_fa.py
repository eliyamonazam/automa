"""Tests for translated error pages and refusals (D147).

Temporary directories only; no model, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from automa.project import ProjectStore
from automa.webui import i18n
from automa.webui import server as ws
from automa.webui.actions import ActionRefused
from automa.webui.files import FilesRefused
from automa.webui.ideas import IdeaError, _refusal
from automa.webui.refusal import Refusal, refusal_key
from test_webui_actions import FORM, ORIGIN, PORT, TOKEN, _settings
from test_webui_i18n import LATIN_WORD, _strip_data

HOST = f"127.0.0.1:{PORT}"
SRC = Path(__file__).resolve().parents[1] / "src" / "automa" / "webui"


def _app(root: Path, lang: i18n.Language = "fa", *, actions: bool = True) -> ws.WebApp:
    hosts = ws.allowed_hosts("127.0.0.1", PORT)
    return ws.WebApp(
        root,
        root / "ledger.sqlite3",
        hosts,
        actions=_settings() if actions else None,
        language=lang,
    )


def _body(r: ws.Response) -> str:
    return r.body.decode("utf-8")


def _interface(doc: str) -> str:
    """Interface text only: technical details (``<code dir="ltr">``) and data removed."""
    doc = re.sub(r"<title>.*?</title>", "", doc, flags=re.S)
    doc = re.sub(r"<code[^>]*>.*?</code>", "", doc, flags=re.S)
    return _strip_data(doc)


def _project(root: Path) -> str:
    return ProjectStore.for_workspace(root).create("یک صفحهٔ ساده").id


def _post(app: ws.WebApp, target: str, body: bytes, **extra: str) -> ws.Response:
    headers = [("Content-Type", FORM), ("Origin", ORIGIN), *extra.items()]
    return app.handle("POST", target, HOST, headers, body)


# -- the Refusal type ----------------------------------------------------------------


def test_coded_refusal_message_is_the_english_template() -> None:
    e = ActionRefused.coded(409, "already_decided", id="r-1", status="approved")
    assert e.message == "request r-1 is already approved"
    assert e.status == 409 and e.code == "already_decided"
    assert dict(e.params) == {"id": "r-1", "status": "approved"}
    assert str(e) == e.message


def test_refusal_rejects_unknown_code_and_lead_and_both() -> None:
    with pytest.raises(ValueError):
        Refusal(400, "x", code="nope")
    with pytest.raises(ValueError):
        Refusal(400, "x", lead="nope")
    with pytest.raises(ValueError):
        Refusal(400, "x", code="malformed_form", lead="lead.store")
    with pytest.raises(ValueError):
        ActionRefused.coded(400, "note_too_long")  # missing {max}


def test_refusal_params_are_read_only() -> None:
    e = FilesRefused.coded(404, "no_project", id="p-1")
    with pytest.raises(TypeError):
        e.params["id"] = "p-2"  # type: ignore[index]


def test_every_code_used_in_the_source_exists_and_no_code_is_unused() -> None:
    source = "\n".join(p.read_text("utf-8") for p in SRC.glob("*.py"))
    used = set(re.findall(r'\.coded\(\s*\d+,\s*"([a-z_]+)"', source))
    used |= set(re.findall(r'(?<![_a-z])code="([a-z_]+)"', source))
    used |= {"forms_only", "read_only"}  # chosen by a conditional in server.handle
    leads = set(re.findall(r'lead="([a-z_.]+)"', source)) | {
        f"idea.{kind}" for kind in ("config", "budget", "planning", "provider")
    }
    for code in used | leads:
        assert refusal_key(code) in i18n.MESSAGES, code
    defined = {k.removeprefix("refusal.") for k in i18n.MESSAGES if k.startswith("refusal.")}
    assert defined == used | leads


def test_idea_refusal_keeps_its_english_message_and_gets_a_lead() -> None:
    e = _refusal(IdeaError("budget", "daily cap reached"), None)
    assert e.message == "daily cap reached (nothing was created)"
    assert e.status == 409 and e.lead == "idea.budget" and e.code is None


# -- English pages are unchanged -----------------------------------------------------


def test_english_error_page_is_unchanged(tmp_path: Path) -> None:
    app = _app(tmp_path, "en")
    r = app.handle("GET", "/projects/p-20261120-101010-abcdef", HOST)
    doc = _body(r)
    assert r.status == 404
    assert (
        '<h1>Not found</h1><p class="err">no project p-20261120-101010-abcdef</p>'
        '<p><a href="/">← all projects</a></p>'
    ) in doc
    assert '<html lang="en">' in doc


def test_english_technical_error_shows_the_message_only(tmp_path: Path) -> None:
    r = _app(tmp_path, "en").error(
        500, "error.title.project_unreadable", "bad json", lead="lead.store"
    )
    doc = _body(r)
    assert '<h1>Project unreadable</h1><p class="err">bad json</p>' in doc
    assert "Technical" not in doc


# -- Persian pages -------------------------------------------------------------------


def _assert_persian(doc: str) -> None:
    assert '<html lang="fa" dir="rtl">' in doc
    assert "→ همهٔ پروژه‌ها" in doc
    assert LATIN_WORD.findall(_interface(doc)) == [], _interface(doc)


def test_persian_not_found_pages(tmp_path: Path) -> None:
    app = _app(tmp_path)
    r = app.handle("GET", "/nowhere", HOST)
    assert r.status == 404 and "<h1>پیدا نشد</h1>" in _body(r)
    _assert_persian(_body(r))
    pid = "p-20261120-101010-abcdef"
    r = app.handle("GET", f"/projects/{pid}", HOST)
    assert f"پروژهٔ {pid} وجود نداره" in _body(r)
    _assert_persian(_body(r).replace(pid, ""))


@pytest.mark.parametrize(("actions", "expected"), [(False, "فقط برای دیدنه"), (True, "فقط فرم")])
def test_persian_method_not_allowed(tmp_path: Path, actions: bool, expected: str) -> None:
    r = _app(tmp_path, actions=actions).handle("PUT", "/", HOST)
    assert r.status == 405 and "<h1>این کار مجاز نیست</h1>" in _body(r) and expected in _body(r)
    _assert_persian(_body(r))


def test_persian_wrong_host(tmp_path: Path) -> None:
    r = _app(tmp_path).handle("GET", "/", "evil.example:80")
    assert r.status == 421 and "automa web" in _body(r)
    _assert_persian(_body(r).replace("automa web", ""))


def test_persian_refused_form_token(tmp_path: Path) -> None:
    pid = _project(tmp_path)
    r = _post(_app(tmp_path), f"/projects/{pid}/approvals/r-1/approve", b"csrf=wrong")
    assert r.status == 403
    assert "<h1>انجام نشد</h1>" in _body(r) and "کد امنیتی فرم درست نیست" in _body(r)
    _assert_persian(_body(r))


def test_persian_refused_other_origin(tmp_path: Path) -> None:
    pid = _project(tmp_path)
    headers = [("Content-Type", FORM), ("Origin", "http://evil.example")]
    r = _app(tmp_path).handle("POST", f"/projects/{pid}/approvals/r-1/approve", HOST, headers, b"")
    assert r.status == 403 and "غیر از همین رابط" in _body(r)
    _assert_persian(_body(r))


def test_persian_duplicate_header(tmp_path: Path) -> None:
    pid = _project(tmp_path)
    headers = [("Content-Type", FORM), ("Origin", ORIGIN), ("Origin", ORIGIN)]
    r = _app(tmp_path).handle("POST", f"/projects/{pid}/approvals/r-1/reject", HOST, headers, b"")
    assert r.status == 400 and "سرآیند تکراری" in _body(r)
    _assert_persian(_body(r).replace("(origin)", ""))


def test_persian_start_without_plan_is_refused_with_a_sentence(tmp_path: Path) -> None:
    pid = _project(tmp_path)
    hosts = ws.allowed_hosts("127.0.0.1", PORT)

    class _Preflight:
        def needs_confirm(self, project_id: str) -> bool:
            return False

        def command(self, project_id: str, *, yes: bool) -> list[str]:
            return ["true"]

    app = ws.WebApp(
        tmp_path,
        tmp_path / "ledger.sqlite3",
        hosts,
        actions=_settings(),
        runs=_Preflight(),  # type: ignore[arg-type]
        language="fa",
    )
    r = _post(app, f"/projects/{pid}/start", f"csrf={TOKEN}".encode())
    assert r.status == 409
    assert f"پروژهٔ {pid} هنوز برنامه نداره" in _body(r)
    _assert_persian(_body(r).replace(pid, ""))


def test_persian_files_pages(tmp_path: Path) -> None:
    pid = _project(tmp_path)
    app = _app(tmp_path)
    r = app.handle("GET", f"/projects/{pid}/files/view?path=a.txt", HOST)
    assert r.status == 404 and "<h1>در دسترس نیست</h1>" in _body(r)
    assert "فایل خونده نشد." in _body(r) and '<code dir="ltr">' in _body(r)
    _assert_persian(_body(r))
    ProjectStore.for_workspace(tmp_path).files_dir(pid).rmdir()
    r = app.handle("GET", f"/projects/{pid}/files/view?path=a.txt", HOST)
    assert r.status == 404 and "این پروژه هنوز فایلی نداره" in _body(r)
    _assert_persian(_body(r))
    r = app.handle("GET", f"/projects/{pid}/files/view", HOST)
    assert r.status == 400 and "نشانی فایل مشخص نشده" in _body(r)
    _assert_persian(_body(r))


def test_persian_preview_of_an_unsupported_type(tmp_path: Path) -> None:
    pid = _project(tmp_path)
    files = ProjectStore.for_workspace(tmp_path).files_dir(pid)
    files.mkdir(parents=True, exist_ok=True)
    (files / "data.bin").write_bytes(b"\x00\x01")
    r = _app(tmp_path).handle("GET", f"/projects/{pid}/preview/data.bin", HOST)
    assert r.status == 415 and "پیش‌نمایش دارن" in _body(r)
    _assert_persian(_body(r).replace("HTML", "").replace("CSS", ""))


def test_persian_technical_error_has_a_lead_and_ltr_detail(tmp_path: Path) -> None:
    r = _app(tmp_path).error(
        500, "error.title.project_unreadable", "project.json: bad <json>", lead="lead.store"
    )
    doc = _body(r)
    assert "<h1>پروژه خونده نشد</h1>" in doc
    assert "داده‌های پروژه خونده یا نوشته نشد." in doc
    assert 'جزئیات فنی: <code dir="ltr">project.json: bad &lt;json&gt;</code>' in doc
    _assert_persian(doc)


def test_persian_technical_error_without_lead_uses_the_generic_one(tmp_path: Path) -> None:
    r = _app(tmp_path).error(409, "error.title.executive_failed", "unknown error")
    doc = _body(r)
    assert "<h1>برنامه‌ریزی تموم نشد</h1>" in doc and "مشکلی پیش اومد" in doc
    _assert_persian(doc)


def test_persian_unreadable_project_page(tmp_path: Path) -> None:
    pid = _project(tmp_path)
    (ProjectStore.for_workspace(tmp_path).project_dir(pid) / "project.json").write_text("{")
    r = _app(tmp_path).handle("GET", f"/projects/{pid}", HOST)
    assert r.status == 500 and "<h1>پروژه خونده نشد</h1>" in _body(r)
    _assert_persian(_body(r))


@pytest.mark.parametrize("kind", ["config", "budget", "planning", "provider"])
def test_persian_idea_refusal_leads(tmp_path: Path, kind: str) -> None:
    e = _refusal(IdeaError(kind, "gateway said no"), "p-1")
    doc = _body(_app(tmp_path).refused(e))
    assert i18n.t(refusal_key(f"idea.{kind}"), "fa") in doc
    assert '<code dir="ltr">gateway said no (project p-1)</code>' in doc
    _assert_persian(doc.replace("automa.toml", ""))


def test_persian_redirect_page(tmp_path: Path) -> None:
    app = _app(tmp_path)
    r = app._redirect("/projects/x", "redirect.recorded")
    assert r.status == 303 and ("Location", "/projects/x") in r.headers
    assert "<title>ثبت شد</title>" in _body(r) and ">ادامه</a>" in _body(r)


def test_english_redirect_page_is_unchanged(tmp_path: Path) -> None:
    r = _app(tmp_path, "en")._redirect("/x")
    assert "<title>Done</title>" in _body(r) and '<a href="/x">continue</a>' in _body(r)


def test_persian_error_keeps_security_headers(tmp_path: Path) -> None:
    app = _app(tmp_path)
    r = app.handle("GET", "/nowhere", HOST)
    names = {n for n, _ in r.headers}
    assert {"Content-Security-Policy", "X-Content-Type-Options", "Cache-Control"} <= names
