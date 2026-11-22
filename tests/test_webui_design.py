"""Shared design system of the web UI (M16, D161): the local Vazirmatn font served
from ``/static/`` under ``font-src 'self'``, the same top bar on every page ("+ New
idea" and this month's spending) and inline SVG status icons instead of emoji.

Projects and Ledger entries are written into tmp_path; no network, no model calls.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from automa.budget import CostLedger
from automa.project import ProjectStore
from automa.webui import render as rd
from automa.webui import server as ws
from automa.webui import static as st
from automa.webui import status as stt
from test_ledger import FixedClock, make_pair
from test_webui_ideas import HOST, Env

NOW = datetime(2026, 11, 22, 9, 0, tzinfo=UTC)
EMOJI = re.compile("[☀-➿\U0001f300-\U0001faff]")
LOCAL = "127.0.0.1:8765"


def _app(workspace: Path, **kw: object) -> ws.WebApp:
    return ws.WebApp(
        workspace, workspace / "ledger.sqlite3", ws.allowed_hosts("127.0.0.1", 8765), **kw
    )


def _spend(workspace: Path, usd: float, when: datetime = NOW) -> None:
    with CostLedger(workspace / "ledger.sqlite3", clock=FixedClock(when)) as led:
        led.record(*make_pair(usd, project="p1"))


def _topbar(doc: str) -> str:
    m = re.search(r'<header class="topbar">.*?</header>', doc, re.S)
    assert m, doc[:400]
    return m.group(0)


# -- the font file and its route ---------------------------------------------------


def test_font_and_licence_ship_inside_the_package() -> None:
    data, mime = st.static_asset(st.FONT_PATH) or (b"", "")
    assert data[:4] == b"wOF2" and mime == "font/woff2" and len(data) > 50_000
    licence, lmime = st.static_asset(st.LICENSE_PATH) or (b"", "")
    assert b"SIL Open Font License" in licence and b"Vazirmatn" in licence
    assert lmime == "text/plain; charset=utf-8"


def test_server_serves_the_font_with_strict_cacheable_headers(tmp_path: Path) -> None:
    r = _app(tmp_path).handle("GET", st.FONT_PATH, LOCAL)
    headers = dict(r.headers)
    assert r.status == 200 and r.body[:4] == b"wOF2"
    assert headers["Content-Type"] == "font/woff2"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert headers["Cache-Control"].startswith("public")
    assert "Set-Cookie" not in headers


def test_head_of_the_font_has_the_same_status_and_type(tmp_path: Path) -> None:
    r = _app(tmp_path).handle("HEAD", st.FONT_PATH, LOCAL)
    assert r.status == 200 and dict(r.headers)["Content-Type"] == "font/woff2"


def test_licence_is_served_as_plain_text(tmp_path: Path) -> None:
    r = _app(tmp_path).handle("GET", st.LICENSE_PATH, LOCAL)
    assert r.status == 200 and b"Open Font License" in r.body
    assert dict(r.headers)["Content-Type"] == "text/plain; charset=utf-8"


@pytest.mark.parametrize(
    "path",
    [
        "/static/",
        "/static/Vazirmatn.woff2",
        "/static/vazirmatn.woff2/",
        "/static/../pyproject.toml",
        "/static/%2e%2e/server.py",
        "/static/..%2fserver.py",
        "/static/status.py",
        "/static/static.py",
        "/static//etc/passwd",
        "/static/vazirmatn.woff2%00",
        "/static/OFL.TXT",
    ],
)
def test_only_the_listed_assets_are_served(tmp_path: Path, path: str) -> None:
    r = _app(tmp_path).handle("GET", path, LOCAL)
    assert r.status == 404 and b"wOF2" not in r.body
    assert b"Copyright" not in r.body


def test_query_does_not_change_the_asset(tmp_path: Path) -> None:
    r = _app(tmp_path).handle("GET", st.FONT_PATH + "?v=1", LOCAL)
    assert r.status == 200 and r.body[:4] == b"wOF2"


def test_static_route_still_checks_the_host(tmp_path: Path) -> None:
    r = _app(tmp_path).handle("GET", st.FONT_PATH, "evil.example")
    assert r.status == 421 and b"wOF2" not in r.body


def test_post_to_static_is_never_an_asset(tmp_path: Path) -> None:
    env = Env(tmp_path)
    r = env.post(st.FONT_PATH, {"csrf": "x"})
    assert r.status != 200 and b"wOF2" not in r.body
    r = _app(tmp_path).handle("POST", st.FONT_PATH, LOCAL)  # read-only: no POST at all
    assert r.status == 405


def test_static_does_not_read_the_ledger(tmp_path: Path) -> None:
    (tmp_path / "ledger.sqlite3").write_bytes(b"not a database")
    r = _app(tmp_path).handle("GET", st.FONT_PATH, LOCAL)
    assert r.status == 200


# -- the page CSP and the @font-face ------------------------------------------------


@pytest.mark.parametrize("csp", [rd.CONTENT_SECURITY_POLICY, rd.ACTION_CONTENT_SECURITY_POLICY])
def test_csp_allows_fonts_from_this_server_only(csp: str) -> None:
    assert "font-src 'self';" in csp and csp.count("font-src") == 1
    assert "default-src 'none'" in csp and "img-src 'none'" in csp
    assert "http" not in csp and "*" not in csp and "data:" not in csp


def test_header_csp_matches_the_page_csp() -> None:
    assert ws.HEADER_CSP.startswith(rd.CONTENT_SECURITY_POLICY)
    assert "font-src 'self'" in ws.ACTION_HEADER_CSP


def test_image_source_does_not_touch_the_font_rule() -> None:
    csp = rd.page_csp(img_src="http://127.0.0.1:8765/projects/p1/files/raw")
    assert "font-src 'self'" in csp and "img-src http://127.0.0.1:8765/" in csp


def test_font_face_points_only_at_the_local_file() -> None:
    assert st.FONT_FACE.count("url(") == 1 and "url(/static/vazirmatn.woff2)" in st.FONT_FACE
    assert "//" not in st.FONT_FACE and "<" not in st.FONT_FACE and "@import" not in st.FONT_FACE
    assert 'font-family:"Vazirmatn"' in st.FONT_FACE and "font-display:swap" in st.FONT_FACE
    assert st.FONT_FACE.count("{") == st.FONT_FACE.count("}") == 1


def test_every_page_carries_the_font_face_before_the_style() -> None:
    doc = rd.page("t", "<p>x</p>", lang="fa")
    assert f"<style>{st.FONT_FACE}" in doc
    assert doc.index(st.FONT_FACE) < doc.index("font-family:" + rd.FONT_STACK)
    assert rd.FONT_STACK.startswith('"Vazirmatn"')


# -- the same top bar on every page ---------------------------------------------------


def test_top_bar_defaults_to_the_plain_bar() -> None:
    bar = _topbar(rd.page("t", ""))
    assert "newidea" not in bar and 'class="month"' not in bar
    assert 'href="/"' in bar and "Automa" in bar


def test_top_bar_context_sets_and_resets() -> None:
    with rd.top_bar(rd.TopBar(new_idea=True, month_usd=1.5)):
        inner = _topbar(rd.page("t", ""))
        with rd.top_bar(rd.TopBar()):
            nested = _topbar(rd.page("t", ""))
        again = _topbar(rd.page("t", ""))
    after = _topbar(rd.page("t", ""))
    assert "newidea" in inner and "$1.50" in inner
    assert "newidea" not in nested and "$1.50" not in nested
    assert again == inner
    assert "newidea" not in after and 'class="month"' not in after


def test_top_bar_resets_after_an_error() -> None:
    with pytest.raises(RuntimeError), rd.top_bar(rd.TopBar(new_idea=True)):
        raise RuntimeError
    assert "newidea" not in _topbar(rd.page("t", ""))


def test_month_spending_in_toman_first_with_dollars_in_the_tooltip() -> None:
    with rd.top_bar(rd.TopBar(month_usd=3.0, rate=322_000)):
        bar = _topbar(rd.page("t", "", lang="fa"))
    assert "<strong>حدود ۹۶۶ هزار تومان</strong>" in bar
    assert 'title="این ماه: حدود ۹۶۶ هزار تومان (۳٫۰۰ دلار)"' in bar
    assert '<span class="lbl">این ماه:</span>' in bar


@pytest.mark.parametrize("month", [None, 0.0])
def test_nothing_spent_or_unreadable_hides_the_month(month: float | None) -> None:
    with rd.top_bar(rd.TopBar(month_usd=month, rate=322_000)):
        bar = _topbar(rd.page("t", "", lang="fa"))
    assert 'class="month"' not in bar


def _every_page(env: Env) -> dict[str, str]:
    env.store.create("A small tool", project_id="p1")
    pages = {
        "home": "/",
        "project": "/projects/p1",
        "files": "/projects/p1/files",
        "new": "/new",
        "missing": "/projects/nope",
        "unknown": "/nowhere",
    }
    out = {}
    for name, path in pages.items():
        r = env.get(path)
        assert r.status in (200, 404), (name, r.status)
        out[name] = r.body.decode()
    return out


def test_action_mode_shows_new_idea_and_month_on_every_page(tmp_path: Path) -> None:
    _spend(tmp_path, 0.5, datetime.now(UTC))
    env = Env(tmp_path, language="fa")
    for name, doc in _every_page(env).items():
        bar = _topbar(doc)
        assert 'href="/new"' in bar and "ایدهٔ جدید" in bar, name
        assert 'class="month"' in bar and "۰٫۵۰ دلار" in bar, name


def test_read_only_mode_has_no_new_idea_but_keeps_the_month(tmp_path: Path) -> None:
    _spend(tmp_path, 0.5, datetime.now(UTC))
    env = Env(tmp_path, actions=False)
    for name, doc in _every_page(env).items():
        bar = _topbar(doc)
        assert "newidea" not in bar and 'href="/new"' not in bar, name
        assert "$0.50" in bar, name


def test_action_mode_without_a_planner_has_no_new_idea(tmp_path: Path) -> None:
    env = Env(tmp_path, planner=False)
    env.store.create("x", project_id="p1")
    assert "newidea" not in _topbar(env.get("/projects/p1").body.decode())


def test_last_months_spending_is_not_this_months(tmp_path: Path) -> None:
    _spend(tmp_path, 2.0, datetime(2026, 9, 1, tzinfo=UTC))
    app = _app(tmp_path, clock=lambda: NOW)
    assert 'class="month"' not in _topbar(app.handle("GET", "/", LOCAL).body.decode())


def test_unreadable_ledger_hides_the_month_but_pages_still_render(tmp_path: Path) -> None:
    (tmp_path / "ledger.sqlite3").write_bytes(b"not a database")
    ProjectStore.for_workspace(tmp_path).create("x", project_id="p1")
    app = _app(tmp_path)
    for path in ("/", "/projects/p1"):
        r = app.handle("GET", path, LOCAL)
        assert r.status == 200 and 'class="month"' not in _topbar(r.body.decode())


def test_redirect_after_a_post_has_the_same_bar(tmp_path: Path) -> None:
    env = Env(tmp_path)
    r = env.new("Build a site")
    assert r.status == 303 and "newidea" in _topbar(r.body.decode())


def test_request_does_not_leak_its_bar(tmp_path: Path) -> None:
    _spend(tmp_path, 0.5, datetime.now(UTC))
    Env(tmp_path).get("/")
    bar = _topbar(rd.page("t", ""))
    assert "newidea" not in bar and 'class="month"' not in bar


def test_home_page_reads_the_ledger_once_for_bar_and_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _spend(tmp_path, 0.5, datetime.now(UTC))
    app = _app(tmp_path)
    calls = 0
    real = ws.WebApp._totals

    def counted(self: ws.WebApp) -> tuple[float, float] | None:
        nonlocal calls
        calls += 1
        return real(self)

    monkeypatch.setattr(ws.WebApp, "_totals", counted)
    doc = app.handle("GET", "/", LOCAL).body.decode()
    assert calls == 1 and "$0.50" in doc


# -- SVG status icons instead of emoji ------------------------------------------------


def test_every_status_look_has_an_svg_icon() -> None:
    names = {i for i, _ in (*stt.PROJECT_LOOK.values(), *stt.TASK_LOOK.values())}
    assert names | {stt.UNKNOWN_ICON} <= set(stt.STATUS_ICONS)


@pytest.mark.parametrize("name", sorted(stt.STATUS_ICONS))
def test_status_icons_are_fixed_inert_markup(name: str) -> None:
    svg = stt.STATUS_ICONS[name]
    assert svg.startswith('<svg class="icon"') and svg.endswith("</svg>")
    assert 'aria-hidden="true"' in svg and 'focusable="false"' in svg
    for bad in ("url(", "href", "<script", "style=", "<image", "<foreignObject"):
        assert bad not in svg
    assert not re.search(r"\son[a-z]+=", svg)  # no event handlers
    assert not EMOJI.search(svg)


def test_chip_html_draws_the_svg_and_keeps_the_label() -> None:
    html = rd._chip_html(stt.project_chip("done", "fa"), "ps-done")
    assert stt.STATUS_ICONS["check"] in html and "تموم شد" in html
    unknown = rd._chip_html(stt.project_chip("<b>x", "fa"), "ps-x")
    assert stt.STATUS_ICONS["dot"] in unknown and "&lt;b&gt;x" in unknown


def test_project_page_status_labels_have_no_emoji(tmp_path: Path) -> None:
    from automa.plan import Plan
    from test_webui import _task

    store = ProjectStore.for_workspace(tmp_path)
    store.create("یک ابزار", project_id="p1")
    store.save_plan("p1", Plan(goal="یک ابزار", tasks=[_task("a"), _task("b", ["a"])]))
    doc = _app(tmp_path, language="fa").handle("GET", "/projects/p1", LOCAL).body.decode()
    badges = re.findall(r'<span class="badge[^"]*"[^>]*>.*?</span> [^<]*</span>', doc)
    assert badges
    for badge in badges:
        assert "<svg" in badge and not EMOJI.search(badge), badge


def test_env_host_is_the_local_one() -> None:
    assert HOST == LOCAL


def test_narrow_screens_keep_the_bar_on_one_line() -> None:
    # 640px and below: the project link (the brand already goes home) and the month
    # label are hidden; 430px and below: "+ New idea" keeps its label for screen
    # readers but shows only the plus.
    narrow = rd._STYLE[rd._STYLE.index("@media (max-width:640px)") :]
    assert ".topbar nav a.projects{display:none}" in narrow
    assert ".topbar .month .lbl{display:none}" in narrow
    tiny = rd._STYLE[rd._STYLE.index("@media (max-width:430px)") :]
    assert ".topbar .newidea .lbl{position:absolute" in tiny and "clip-path:inset(50%)" in tiny
    with rd.top_bar(rd.TopBar(new_idea=True)):
        bar = _topbar(rd.page("t", "", lang="fa"))
    assert 'title="ایدهٔ جدید"' in bar and '<span class="lbl">ایدهٔ جدید</span>' in bar
