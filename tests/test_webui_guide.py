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
from automa.webui import static as st
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


# -- the sample pictures (D190) -------------------------------------------------------------

SHOTS = ROOT / "src" / "automa" / "webui" / "static" / "guide"
SCRIPT = ROOT / "scripts" / "guide_screenshots.py"


def _csp(r: ws.Response) -> str:
    return dict(r.headers)["Content-Security-Policy"]


def test_every_picture_belongs_to_a_section_and_is_an_asset() -> None:
    ids = [s.id for s in g.SECTIONS]
    assert set(g.IMAGES) <= set(ids)
    assert set(g.IMAGES) == set(st.GUIDE_IMAGES)
    for name in g.IMAGES:
        path = st.guide_image_path(name)
        assert path == f"/static/guide/{name}.png" and path in st.ASSETS
    with pytest.raises(ValueError):
        st.guide_image_path("../OFL")


@pytest.mark.parametrize("name", st.GUIDE_IMAGES)
def test_picture_is_a_small_phone_png(name: str) -> None:
    data = (SHOTS / f"{name}.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert 5_000 < len(data) < 150_000, len(data)  # small enough for the repo and a phone
    width, height = st.guide_image_size(name)
    assert width == 390  # a phone screen
    assert 200 <= height <= 844


def test_no_stray_file_in_the_pictures_folder() -> None:
    assert sorted(p.name for p in SHOTS.iterdir()) == sorted(f"{n}.png" for n in st.GUIDE_IMAGES)


def test_size_of_a_file_that_is_not_png(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(st, "_read", lambda name: b"GIF89a" + b"\0" * 40)
    with pytest.raises(ValueError):
        st.guide_image_size("login")


def test_check_rejects_a_picture_without_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(g, "IMAGES", {**g.IMAGES, "files": g._s("فایل", "File")})
    with pytest.raises(ValueError, match="pictures"):
        g._check()
    monkeypatch.setattr(g, "IMAGES", {k: v for k, v in g.IMAGES.items() if k != "login"})
    with pytest.raises(ValueError, match="pictures"):
        g._check()


def test_page_shows_each_picture_with_caption_and_size(env: IsoEnv) -> None:
    main = _main(env.get("/guide", env.signed_in("kid", "kid-pass-123")))
    for name, alt in g.IMAGES.items():
        width, height = st.guide_image_size(name)
        tag = (
            f'<img src="/static/guide/{name}.png" alt="{alt["en"]}" '
            f'width="{width}" height="{height}" loading="lazy">'
        )
        assert tag in main, name
        section = main.split(f'id="{name}"', 1)[1].split("</section>", 1)[0]
        assert tag in section  # under its own section
        assert f"{alt['en']} · sample picture</figcaption>" in section
    assert main.count("<img ") == len(g.IMAGES)


def test_persian_captions_say_sample(tmp_path: Path) -> None:
    env = FaEnv(tmp_path)
    main = _main(env.get("/guide", env.signed_in("kid", "kid-pass-123")))
    assert main.count("· تصویر نمونه</figcaption>") == len(g.IMAGES)
    assert 'alt="صفحهٔ ورود"' in main


def test_only_the_guide_may_load_own_pictures(env: IsoEnv) -> None:
    kid = env.signed_in("kid", "kid-pass-123")
    guide = env.get("/guide", kid)
    assert "img-src 'self'" in _csp(guide) and "img-src 'none'" not in _csp(guide)
    assert "frame-ancestors 'none'" in _csp(guide)
    assert "img-src 'self'" in _body(guide)  # the page's own meta CSP too
    for other in ("/", "/new", "/spending", "/login"):
        r = env.get(other, kid if other != "/login" else None)
        assert "img-src 'self'" not in _csp(r), other
        assert "img-src 'self'" not in _body(r), other


def test_page_csp_self_images() -> None:
    from automa.webui.render import page_csp

    assert "img-src 'self'" in page_csp(self_images=True)
    assert "form-action 'self'" in page_csp(actions=True, self_images=True)
    assert "img-src 'none'" in page_csp()
    with pytest.raises(ValueError):
        page_csp(self_images=True, img_src="http://127.0.0.1:8765/x")


@pytest.mark.parametrize("name", st.GUIDE_IMAGES)
def test_pictures_are_served_as_fixed_assets(env: IsoEnv, name: str) -> None:
    r = env.get(f"/static/guide/{name}.png")
    assert r.status == 200
    headers = dict(r.headers)
    assert headers["Content-Type"] == "image/png"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert r.body == (SHOTS / f"{name}.png").read_bytes()


@pytest.mark.parametrize(
    "path",
    [
        "/static/guide/",
        "/static/guide/LOGIN.png",
        "/static/guide/login.PNG",
        "/static/guide/../OFL.txt",
        "/static/guide/%6cogin.png",
        "/static/guide/login.png/",
        "/static/guide//login.png",
        "/static/guide/other.png",
    ],
)
def test_other_paths_under_guide_are_not_found(env: IsoEnv, path: str) -> None:
    r = env.get(path)
    assert r.status in (400, 404), path
    assert dict(r.headers).get("Content-Type", "").startswith("text/html")


def test_markdown_shows_the_same_pictures() -> None:
    doc = g.guide_markdown("fa")
    for name, alt in g.IMAGES.items():
        src = f"../src/automa/webui/static/guide/{name}.png"
        assert (DOC.parent / src).resolve() == (SHOTS / f"{name}.png").resolve()
        assert f'<img src="{src}" alt="{alt["fa"]}" width="260">' in doc
        assert f"<sub>{alt['fa']} · تصویر نمونه</sub>" in doc
        section = doc.split(
            f"## {g.fill_text(next(s for s in g.SECTIONS if s.id == name).title, 'fa')}"
        )[1]
        assert src in section.split("\n## ", 1)[0]


def test_picture_texts_are_safe_in_html_attributes() -> None:
    for alt in g.IMAGES.values():
        for value in alt.values():
            assert not set(value) & set('"<>&'), value


def test_screenshot_script_matches_the_guide_and_stays_offline() -> None:
    src = SCRIPT.read_text(encoding="utf-8")
    assert "from automa.webui.guide import EXAMPLES, IMAGES" in src
    for name in g.IMAGES:
        assert f'"{name}":' in src, name
    # Stand-ins only: no provider, no gateway, no real run, no outside page.
    for banned in ("automa.providers", "Gateway", "subprocess", "http://", "https://"):
        assert banned not in src, banned
    assert "tempfile.TemporaryDirectory()" in src
