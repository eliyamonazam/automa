"""Tests for the clean, Persian-friendly look of the web UI (D129).

Pure rendering: projects, plans and approvals are written into tmp_path by hand;
no provider, no network, no model calls.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from automa.plan import Plan
from automa.project import ProjectStore
from automa.webui import render as rd
from automa.webui.files import build_listing
from automa.webui.views import build_overview, build_project_view
from test_webui import _request, _task

PERSIAN = "یک شعر کوتاه درباره‌ی باران بنویس"
STYLE = rd._STYLE
FORM_TOKEN = "t" * 32  # a fake per-session CSRF token for rendering


@pytest.fixture
def store(tmp_path: Path) -> ProjectStore:
    s = ProjectStore.for_workspace(tmp_path)
    s.create(PERSIAN, project_id="p1")
    s.save_plan(
        "p1",
        Plan(goal=PERSIAN, tasks=[_task("a", title="نوشتن پیش‌نویس"), _task("b", ["a"])]),
    )
    return s


def _css_block(css: str, selector: str) -> str:
    m = re.search(re.escape(selector) + r"\{([^}]*)\}", css)
    assert m, selector
    return m.group(1)


def _vars(block: str) -> set[str]:
    return set(re.findall(r"(--[a-z]+):", block))


# -- the stylesheet ------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["url(", "@import", "@font-face", "expression(", "<", "//"])
def test_style_loads_nothing_and_cannot_break_out(bad: str) -> None:
    assert bad not in STYLE


def test_style_braces_are_balanced() -> None:
    depth = 0
    for ch in STYLE:
        depth += {"{": 1, "}": -1}.get(ch, 0)
        assert depth >= 0
    assert depth == 0


@pytest.mark.parametrize("face", ['"Vazirmatn"', '"Segoe UI"', "Tahoma", '"Noto Sans Arabic"'])
def test_font_stack_has_persian_capable_local_faces(face: str) -> None:
    assert face in rd.FONT_STACK
    assert "font-family:" + rd.FONT_STACK in STYLE


def test_font_stack_ends_with_generic_family() -> None:
    assert rd.FONT_STACK.endswith("sans-serif")
    assert rd.MONO_STACK.endswith("monospace")


def test_persian_faces_come_before_the_system_face() -> None:
    assert rd.FONT_STACK.index("Vazirmatn") < rd.FONT_STACK.index("system-ui")
    assert rd.FONT_STACK.index("Tahoma") < rd.FONT_STACK.index("system-ui")


def test_dark_theme_redefines_every_colour_variable() -> None:
    light = _vars(_css_block(STYLE, ":root"))
    dark_part = STYLE[STYLE.index("@media (prefers-color-scheme: dark){:root{") :]
    dark = _vars(_css_block(dark_part, ":root"))
    assert light and light == dark
    assert "color-scheme:light dark" in _css_block(STYLE, ":root")


def test_colours_come_from_variables() -> None:
    body = _css_block(STYLE, "\nbody")
    assert "color:var(--fg)" in body and "background:var(--bg)" in body
    assert "line-height:1.7" in body


def test_mixed_direction_text_is_isolated_per_line() -> None:
    assert "unicode-bidi:plaintext" in _css_block(STYLE, "[dir=auto],pre")


def test_card_and_grid_styles_exist() -> None:
    assert "border-radius" in _css_block(STYLE, ".card")
    assert "grid-template-columns" in _css_block(STYLE, ".cards")
    assert "overflow-x:auto" in _css_block(STYLE, ".scroll")


def test_task_borders_use_logical_side() -> None:
    assert "border-inline-start" in _css_block(STYLE, ".task")
    assert "border-left" not in STYLE and "border-right" not in STYLE


def test_page_declares_colour_scheme_and_keeps_csp() -> None:
    doc = rd.page("t", "b")
    assert '<meta name="color-scheme" content="light dark">' in doc
    assert rd.CONTENT_SECURITY_POLICY in doc
    assert "<script" not in doc and "<link" not in doc and 'src="' not in doc


def test_style_is_not_mistaken_for_an_action_word() -> None:
    # Read-only pages must not mention approving (test_webui relies on this).
    assert "approve" not in STYLE.lower()


# -- overview cards -------------------------------------------------------------------


def test_overview_shows_projects_as_cards(store: ProjectStore) -> None:
    doc = rd.render_overview(build_overview(store, {}))
    assert '<div class="cards"><article class="card project">' in doc
    assert f'<p class="goal" dir="auto">{PERSIAN}</p>' in doc
    assert "<table" not in doc
    assert 'href="/projects/p1"' in doc and "cost: $" in doc and "approvals: 0" in doc


def test_overview_card_marks_pending_approvals(store: ProjectStore) -> None:
    _request(store, "p1")
    doc = rd.render_overview(build_overview(store, {}))
    assert '<span class="badge warn">1 pending</span>' in doc


def test_overview_card_escapes_goal(tmp_path: Path) -> None:
    s = ProjectStore.for_workspace(tmp_path)
    s.create('<b onclick="x">سلام</b>‮', project_id="p2")
    doc = rd.render_overview(build_overview(s, {}))
    assert "<b onclick" not in doc and "&lt;b onclick" in doc
    assert "‮" not in doc and "\\u202e" in doc


def test_overview_with_no_projects_has_no_card_grid(tmp_path: Path) -> None:
    doc = rd.render_overview(build_overview(ProjectStore.for_workspace(tmp_path), {}))
    assert "No projects yet." in doc and 'class="cards"' not in doc


# -- project page cards ---------------------------------------------------------------


def test_project_header_is_a_card_with_persian_goal(store: ProjectStore) -> None:
    doc = rd.render_project(build_project_view(store, "p1", 0))
    assert '<section class="card"><h1>p1</h1>' in doc
    assert f'<p class="goal" dir="auto">{PERSIAN}</p>' in doc


def test_tasks_are_cards_with_persian_title(store: ProjectStore) -> None:
    doc = rd.render_project(build_project_view(store, "p1", 0))
    assert doc.count('<article class="card task t-none">') == 2
    assert '<p class="goal" dir="auto">نوشتن پیش‌نویس</p>' in doc
    assert "After: a" in doc and "Role: writer" in doc and "Attempts: 0" in doc


def test_task_card_title_is_escaped(tmp_path: Path) -> None:
    s = ProjectStore.for_workspace(tmp_path)
    s.create("g", project_id="p1")
    s.save_plan("p1", Plan(goal="g", tasks=[_task("a", title="<i>x</i>⁦")]))
    doc = rd.render_project(build_project_view(s, "p1", 0))
    assert "<i>x" not in doc and "&lt;i&gt;x&lt;/i&gt;\\u2066" in doc


@pytest.mark.parametrize("state", [None, "weird", '" onmouseover="x', "passed passed"])
def test_task_card_class_is_fixed_for_unknown_states(state: str | None) -> None:
    assert rd._state_class(state) == "none"


def test_sections_and_cards_are_balanced(store: ProjectStore) -> None:
    _request(store, "p1")
    doc = rd.render_project(build_project_view(store, "p1", 0))
    for tag in ("section", "article", "div"):
        assert doc.count(f"<{tag}") == doc.count(f"</{tag}>"), tag


def test_approval_is_a_card_with_ltr_arguments_and_auto_reason(store: ProjectStore) -> None:
    _request(store, "p1", text="سلام دنیا")
    doc = rd.render_project(build_project_view(store, "p1", 0))
    assert '<article class="card"><h3>' in doc
    assert '<pre dir="ltr">' in doc and "سلام دنیا" in doc
    assert '<p>reason: <span dir="auto">' in doc


def test_fetch_target_lines_have_auto_direction(store: ProjectStore) -> None:
    _request(store, "p1", tool="fetch_url", url="https://example.com/")
    doc = rd.render_project(build_project_view(store, "p1", 0))
    assert '<li dir="auto">' in doc


def test_error_list_is_a_card_with_auto_direction() -> None:
    out = rd._errors(("خطا در خواندن",))
    assert out.startswith('<section class="card err">')
    assert '<li dir="auto">خطا در خواندن</li>' in out


def test_run_panel_is_a_card_and_log_tail_has_auto_direction() -> None:
    panel = rd.RunPanel(can_start=False, needs_confirm=False, state="running", tail=("سلام",))
    out = rd._run_panel("p1", panel, rd.ActionForms(FORM_TOKEN))
    assert out.startswith('<section class="card"><h2>Run</h2>') and out.endswith("</section>")
    assert '<pre dir="auto">سلام</pre>' in out


def test_new_idea_and_clarify_forms_are_cards() -> None:
    forms = rd.ActionForms(FORM_TOKEN)
    assert '<form class="card" method="post" action="/new">' in rd.render_new_idea(forms)
    doc = rd.render_clarify("p1", PERSIAN, ["چه لحنی؟"], forms)
    assert '<form class="card" method="post" action="/projects/p1/clarify">' in doc
    assert '<p dir="auto"><label>1. چه لحنی؟' in doc


def test_files_table_is_in_a_scrolling_card(store: ProjectStore) -> None:
    files = store.project_dir("p1") / "files"
    files.mkdir(exist_ok=True)
    (files / "شعر.txt").write_text("باران", encoding="utf-8")
    doc = rd.render_files(build_listing(store, "p1"))
    assert '<div class="card scroll"><table>' in doc
    assert 'dir="auto">شعر.txt</a>' in doc


def test_rendering_keeps_files_untouched(store: ProjectStore, tmp_path: Path) -> None:
    before = sorted((p, p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file())
    rd.render_overview(build_overview(store, {}))
    rd.render_project(build_project_view(store, "p1", 0))
    after = sorted((p, p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file())
    assert before == after


def test_labels_never_decide_the_direction_of_recorded_text(store: ProjectStore) -> None:
    # A leading English label would make a dir="auto" block left-to-right.
    _request(store, "p1")
    pages = [
        rd.render_project(build_project_view(store, "p1", 0)),
        rd.render_clarify("p1", PERSIAN, ["q"], rd.ActionForms(FORM_TOKEN)),
    ]
    for doc in pages:
        for m in re.finditer(r'dir="auto">([^<]*)<', doc):
            assert not m.group(1).startswith(("Goal", "reason", "Last result")), m.group(0)
        assert f'<p class="goal" dir="auto">{PERSIAN}</p>' in doc
