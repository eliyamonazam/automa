# ruff: noqa: S106 - test passwords
"""Tests for the family guide (M15, D189): ``GET /guide`` and ``docs/FAMILY-GUIDE.md``.

The guide names only real UI labels (filled from the pages' own message tables), the
Markdown file is exactly what the module generates, the page needs a login like every
other page, and every full top bar links to it.

Only in-process calls; no network, no model calls.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from automa.webui import guide as g
from automa.webui import i18n
from automa.webui import server as ws
from test_webui_isolation import IsoEnv
from test_webui_login import HOST, LEDGER, PW, PW_KID

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "FAMILY-GUIDE.md"


def _body(r: ws.Response) -> str:
    return r.body.decode("utf-8", "replace")


def _main(r: ws.Response) -> str:
    m = re.search(r'<main class="wrap">(.*)</main>', _body(r), re.S)
    assert m is not None
    return m.group(1)


class FaEnv(IsoEnv):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.app = ws.WebApp(
            root,
            root / LEDGER,
            ws.allowed_hosts("127.0.0.1", 8765),
            actions=self.app._base_actions,
            auth=self.auth,
            language="fa",
        )


@pytest.fixture
def env(tmp_path: Path) -> IsoEnv:
    return IsoEnv(tmp_path)


# -- the markdown file ---------------------------------------------------------------------


def test_doc_file_is_generated_from_the_module() -> None:
    assert DOC.read_text(encoding="utf-8") == g.guide_markdown("fa"), (
        "docs/FAMILY-GUIDE.md is stale: python -m automa.webui.guide > docs/FAMILY-GUIDE.md"
    )


@pytest.mark.parametrize("lang", ["fa", "en"])
def test_markdown_has_every_section_and_no_placeholder(lang: str) -> None:
    md = g.guide_markdown(lang)  # type: ignore[arg-type]
    for s in g.SECTIONS:
        assert f"## {g.fill_text(s.title, lang)}" in md  # type: ignore[arg-type]
    assert "{" not in md and "}" not in md
    # Ordered lists use ASCII digits so every Markdown viewer numbers them.
    assert re.search(r"^1\. ", md, re.M)


def test_doc_covers_what_the_roadmap_asks() -> None:
    md = DOC.read_text(encoding="utf-8")
    for needle in (
        "صفحهٔ اصلی",  # saving on the phone
        "«ورود»",
        "«ایدهٔ جدید»",
        "TSETMC",
        "بورس:",
        "وب:",
        "«فایل‌های کمکی (اختیاری)»",
        "«فرستادن جواب‌ها و ساختن برنامه»",
        "«بازهٔ هزینه»",
        "«شروع»",
        "«⬇ دانلود همهٔ فایل‌ها (یک فایل فشرده)»",
        "وقتی چیزی گیر کرد",
    ):
        assert needle in md, needle


# -- the labels are the real ones ----------------------------------------------------------


def test_labels_come_from_the_ui_tables() -> None:
    assert g.label("ui", "run.start.button", "fa") == i18n.MESSAGES["run.start.button"]["fa"]
    assert g.label("ui", "nav.new", "fa") == "ایدهٔ جدید"
    assert g.label("ui", "project.cost_range", "fa") == "بازهٔ هزینه"  # no trailing colon
    assert g.label("ui", "overview.new_idea", "en") == "New idea"  # no "+ "
    with pytest.raises(KeyError):
        g.label("ui", "no.such.key", "fa")


def test_every_reference_resolves_in_both_languages() -> None:
    texts = [g.TITLE, g.LEAD, *g.EXAMPLES]
    for s in g.SECTIONS:
        texts += [s.title, *s.steps, *([s.tip] if s.tip else [])]
    for item in texts:
        for lang in ("fa", "en"):
            filled = g.fill_text(item, lang)  # type: ignore[arg-type]
            assert "{" not in filled and "}" not in filled, filled


def test_numbers_follow_the_upload_limits() -> None:
    from automa.ideafiles import MAX_IDEA_FILE_BYTES, MAX_IDEA_FILES

    md = g.guide_markdown("en")
    assert f"Up to {MAX_IDEA_FILES} files, each up to {MAX_IDEA_FILE_BYTES // 2**20} MB" in md


def test_the_ask_box_is_ticked_as_the_guide_says() -> None:
    from automa.webui.render import ActionForms, render_new_idea

    html = render_new_idea(ActionForms(csrf_token="t" * 32), lang="fa")
    label = i18n.MESSAGES["idea.ask"]["fa"]
    assert label in html
    assert re.search(r'<input type="checkbox" name="ask" value="[^"]*" checked>', html)


def test_check_rejects_bad_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = (
        {"fa": "x {ui:run.start.button}", "en": "x"},  # labels differ
        {"fa": "x {ui:no.such}", "en": "x {ui:no.such}"},  # unknown key
        {"fa": "x {oops}", "en": "x {oops}"},  # unknown placeholder
        {"fa": "x {max_files}", "en": "x"},  # numbers differ
        {"fa": "only persian"},  # one language
    )
    for item in bad:
        monkeypatch.setattr(g, "_ALL_TEXTS", (item,))
        with pytest.raises((ValueError, KeyError)):
            g._check()


# -- the page ------------------------------------------------------------------------------


def test_page_needs_login_when_accounts_exist(env: IsoEnv) -> None:
    r = env.get("/guide")
    assert r.status == 303 and dict(r.headers)["Location"] == "/login"


def test_member_and_owner_see_the_guide(env: IsoEnv) -> None:
    for name, pw in (("kid", "kid-pass-123"), ("dad", PW_KID), ("eli", PW)):
        r = env.get("/guide", env.signed_in(name, pw))
        assert r.status == 200, name
        main = _main(r)
        assert "Automa guide" in main
        for s in g.SECTIONS:
            assert f'id="{s.id}"' in main and f'href="#{s.id}"' in main
        assert '<b class="ui">“Start run”</b>' in main


def test_page_without_accounts_on_loopback(tmp_path: Path) -> None:
    app = ws.WebApp(tmp_path, tmp_path / LEDGER, ws.allowed_hosts("127.0.0.1", 8765))
    r = app.handle("GET", "/guide", HOST)
    assert r.status == 200 and "Automa guide" in _body(r)


def test_page_is_fixed_text_with_no_script_or_outside_source(env: IsoEnv) -> None:
    body = _body(env.get("/guide", env.signed_in("eli", PW)))
    assert "<script" not in body
    assert not re.search(r'(src|href)="(https?:)?//', body)
    assert "{" not in _main(env.get("/guide", env.signed_in("eli", PW)))


def test_head_and_post(env: IsoEnv) -> None:
    eli = env.signed_in("eli", PW)
    head = env.app.handle("HEAD", "/guide", HOST, [("Cookie", f"automa_session={eli}")])
    assert head.status == 200
    post = env.post("/guide", {"csrf": env.csrf_of(eli)}, cookie=eli)
    assert post.status in (404, 405)


def test_top_bar_links_to_the_guide(env: IsoEnv) -> None:
    body = _body(env.get("/", env.signed_in("kid", "kid-pass-123")))
    assert '<a class="guide" href="/guide"' in body
    # Not on the login page (bare bar): nothing there before signing in.
    assert 'href="/guide"' not in _body(env.get("/login"))


def test_persian_page(tmp_path: Path) -> None:
    env = FaEnv(tmp_path)
    r = env.get("/guide", env.signed_in("kid", "kid-pass-123"))
    body = _body(r)
    assert 'lang="fa" dir="rtl"' in body
    main = _main(r)
    assert "راهنمای Automa" in main and '<b class="ui">«شروع»</b>' in main
    assert "۱۰ فایل" in main  # Persian digits
    assert ">راهنما<" in body  # the top-bar link
    for english in ("Sign in", "Start", "Guide", "Download"):
        assert english not in main, english


def test_unknown_language_is_refused() -> None:
    with pytest.raises(ValueError):
        g.render_guide("de")  # type: ignore[arg-type]
