"""Web UI home page (M16, D160): delivery-centred project cards, "waiting for your
answer" first, step progress, a friendly empty state with examples that pre-fill the
idea form, and "+ New idea" in the top bar.

Projects come from the real store (and the real orchestrator with a scripted
FakeProvider for a delivered product). No network, no real model calls.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from automa.project import ProjectStatus
from automa.webui import cards as wc
from automa.webui import render as rd
from automa.webui import server as ws
from automa.webui.cards import CardState, card_state
from automa.webui.i18n import WELCOME_EXAMPLE_KEYS, t
from automa.webui.views import Overview, ProjectRow, build_overview
from test_orchestrator import Rig, write
from test_webui_delivery import _plan
from test_webui_ideas import EVIL, TOKEN
from test_webui_ideas import Env as IdeaEnv

T0 = datetime(2026, 11, 22, 9, 0, tzinfo=UTC)


def _row(**kw: object) -> ProjectRow:
    data: dict[str, object] = {
        "id": "p1",
        "status": "paused",
        "title": "A small tool",
        "goal_preview": "A small tool",
        "created_at": T0,
        "cost_usd": 0.0,
        "pending_approvals": 0,
    }
    data.update(kw)
    return ProjectRow.model_validate(data)


# -- the card state (pure) ------------------------------------------------------


@pytest.mark.parametrize(
    "status,delivery,kw,expected",
    [
        ("done", "ready", {}, CardState.READY),
        ("failed", "partial", {}, CardState.PARTIAL),
        ("done", "partial", {}, CardState.PARTIAL),  # a main file is missing
        ("done", None, {}, CardState.PARTIAL),  # no report: never shown as ready
        ("failed", None, {}, CardState.PARTIAL),
        ("running", "in_progress", {}, CardState.WORKING),
        ("planning", None, {}, CardState.WORKING),
        ("paused", "in_progress", {}, CardState.PAUSED),
        ("paused", "in_progress", {"running": True}, CardState.WORKING),
        ("new", None, {}, CardState.TO_START),
        ("paused", "not_started", {}, CardState.TO_START),
        ("weird", None, {}, CardState.UNKNOWN),
        # anything waiting for the user comes first
        ("paused", "in_progress", {"open_asks": 1}, CardState.WAITING),
        ("running", "in_progress", {"pending_approvals": 2}, CardState.WAITING),
        ("new", None, {"questions_waiting": True}, CardState.WAITING),
        ("done", "ready", {"open_asks": 1}, CardState.WAITING),
    ],
)
def test_card_state(status: str, delivery: str | None, kw: dict, expected: CardState) -> None:
    assert card_state(status, delivery, **kw) is expected


def test_card_texts_exist_in_both_languages() -> None:
    for state in CardState:
        for lang in ("fa", "en"):
            if state is CardState.UNKNOWN:
                assert wc.card_label(state, lang, "x<y") == "x<y"  # escaped by the caller
                assert wc.card_hint(state, lang) == ""
            else:
                assert wc.card_label(state, lang) and wc.card_hint(state, lang)
            assert state in wc.ICONS


def test_card_text_parameters_are_checked() -> None:
    assert wc.ct("card.steps", "fa", done=2, total=3) == "۲ از ۳ مرحله"
    assert wc.ct("card.steps", "en", done=2, total=3) == "2 of 3 steps"
    with pytest.raises(ValueError):
        wc.ct("card.steps", "fa", done=1)
    with pytest.raises(TypeError):
        wc.ct("card.steps", "fa", done="<b>", total=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        wc.ct("card.steps", "fa", done=True, total=1)
    with pytest.raises(ValueError):
        wc.ct("card.ready", "de")  # type: ignore[arg-type]


def test_icons_are_fixed_inline_svg() -> None:
    for icon in (*wc.ICONS.values(), wc.EMPTY_ART):
        assert icon.startswith("<svg ") and icon.endswith("</svg>")
        assert 'aria-hidden="true"' in icon
        assert "http" not in icon.replace('xmlns="http://www.w3.org/2000/svg"', "")
        assert (
            "<script" not in icon
            and "href" not in icon
            and "on" not in re.findall(r"\s(\w+)=", icon)
        )


# -- rendering ------------------------------------------------------------------


def test_waiting_projects_come_first_with_an_answer_button() -> None:
    rows = (
        _row(id="p-done", status="done", delivery="ready", tasks_total=2, tasks_passed=2),
        _row(id="p-ask", title="Needs a file", open_asks=1),
        _row(id="p-q", status="new", title="New idea", questions_waiting=True),
    )
    doc = rd.render_overview(Overview(projects=rows), actions=True, ideas=True, lang="fa")
    waiting = doc.index('<section class="waiting">')
    assert waiting < doc.index('<div class="cards">')
    section = doc[waiting : doc.index("</section>", waiting)]
    assert "منتظر جواب تو" in section and section.count("جواب بده") == 2
    assert 'href="/projects/p-ask#ask"' in section and "۱ سؤال وسط کار" in section
    assert 'href="/projects/p-q/clarify"' in section and "چند سؤال قبل از برنامه" in section
    assert "p-done" not in section


def test_questions_answer_goes_to_the_project_page_without_the_idea_form() -> None:
    row = _row(id="p-q", status="new", questions_waiting=True)
    doc = rd.render_overview(Overview(projects=(row,)))
    assert '<a class="button" href="/projects/p-q">' in doc and "/clarify" not in doc


def test_no_waiting_section_when_nothing_waits() -> None:
    doc = rd.render_overview(Overview(projects=(_row(),)))
    assert 'class="waiting"' not in doc


def test_card_shows_state_progress_and_toman() -> None:
    row = _row(status="done", delivery="ready", tasks_total=3, tasks_passed=3, cost_usd=0.45)
    doc = rd.render_overview(Overview(projects=(row,)), rate=322_000, lang="fa")
    assert '<article class="card project cs-ready">' in doc
    assert "</svg>آماده</span>" in doc
    assert "تموم شد و همهٔ فایل‌های اصلی محصول بررسی شدن." in doc
    assert '<span style="width:100%"></span>' in doc and "۳ از ۳ مرحله" in doc
    assert 'aria-label="۳ از ۳ مرحله"' in doc
    assert "هزینه: حدود ۱۴۵ هزار تومان (۰٫۴۵ دلار)" in doc


def test_partial_progress_is_rounded() -> None:
    row = _row(status="failed", delivery="partial", tasks_total=3, tasks_passed=1)
    doc = rd.render_overview(Overview(projects=(row,)))
    assert "cs-partial" in doc and "Incomplete</span>" in doc
    assert '<span style="width:33%"></span>' in doc and "1 of 3 steps" in doc


def test_card_without_plan_says_so() -> None:
    doc = rd.render_overview(Overview(projects=(_row(status="new"),)), lang="fa")
    assert "هنوز برنامه نداره" in doc and "mini-bar" not in doc.split("<main")[1]


def test_goal_preview_only_when_it_adds_something() -> None:
    same = rd.render_overview(Overview(projects=(_row(),)))
    assert '<p class="goal muted"' not in same
    more = rd.render_overview(Overview(projects=(_row(goal_preview="A small tool\nfor me"),)))
    assert '<p class="goal muted" dir="auto">A small tool\nfor me</p>' in more


def test_recorded_text_on_cards_is_escaped() -> None:
    row = _row(title=EVIL, goal_preview=EVIL + " more", status='"><x>', open_asks=1)
    doc = rd.render_overview(Overview(projects=(row,)), actions=True, ideas=True)
    body = doc.split("<main")[1]
    assert "<script>" not in body and "<x>" not in body and "‮" not in body
    assert "&lt;script&gt;" in body and "\\u202e" in body


def test_unknown_status_card_shows_the_recorded_status() -> None:
    doc = rd.render_overview(Overview(projects=(_row(status="odd"),)))
    assert '<article class="card project cs-unknown">' in doc and "</svg>odd</span>" in doc


def test_new_idea_in_the_top_bar_only_when_ideas_are_on() -> None:
    on = rd.render_overview(Overview(projects=(_row(),)), actions=True, ideas=True)
    assert '<header class="topbar">' in on and 'class="button small newidea" href="/new"' in on
    for actions, ideas in ((False, True), (True, False), (False, False)):
        off = rd.render_overview(Overview(projects=(_row(),)), actions=actions, ideas=ideas)
        assert 'href="/new"' not in off


def test_home_page_keeps_the_strict_csp() -> None:
    doc = rd.render_overview(Overview(projects=(_row(open_asks=1),)), actions=True, ideas=True)
    assert rd.ACTION_CONTENT_SECURITY_POLICY in doc and "img-src 'none'" in doc
    assert "<script" not in doc and "<img" not in doc and "https://" not in doc


# -- empty state ----------------------------------------------------------------


def test_empty_state_with_examples() -> None:
    doc = rd.render_overview(Overview(), actions=True, ideas=True, lang="fa")
    assert '<div class="card empty"><svg class="empty-art"' in doc
    assert "بیا اولین ایده‌ات رو بسازیم" in doc
    for i, key in enumerate(WELCOME_EXAMPLE_KEYS, start=1):
        assert f'href="/new?example={i}">{rd.text(t(key, "fa"))}</a>' in doc


def test_empty_state_read_only_has_no_links_to_the_form() -> None:
    doc = rd.render_overview(Overview(), lang="fa")
    assert 'class="card empty"' in doc and "هنوز پروژه‌ای نیست" in doc
    assert "/new" not in doc


def test_example_href_range() -> None:
    assert rd.example_href(1) == "/new?example=1"
    for bad in (0, len(WELCOME_EXAMPLE_KEYS) + 1, -1):
        with pytest.raises(ValueError):
            rd.example_href(bad)


@pytest.mark.parametrize(
    "query,index",
    [
        ("example=1", 1),
        ("example=3", 3),
        ("", None),
        ("example=0", None),
        ("example=4", None),
        ("example=01", 1),
        ("example=999", None),
        ("example=١", None),  # non-ASCII digit
        ("example=1x", None),
        ("example=-1", None),
        ("example=1&example=2", None),
        ("example=1&x=2", None),
        ("goal=hello", None),
        ("example=%3Cb%3E", None),
    ],
)
def test_example_goal(query: str, index: int | None) -> None:
    got = ws.example_goal(query, "en")
    assert got == ("" if index is None else t(WELCOME_EXAMPLE_KEYS[index - 1], "en"))


def test_served_new_idea_form_is_prefilled_from_an_example(tmp_path: Path) -> None:
    env = IdeaEnv(tmp_path, language="fa")
    body = env.get("/new?example=2").body.decode()
    expected = rd.text(t(WELCOME_EXAMPLE_KEYS[1], "fa"))
    assert f">{expected}</textarea>" in body
    plain = env.get("/new?goal=%3Cscript%3E").body.decode()
    assert "></textarea>" in plain and "script&gt;" not in plain


def test_served_empty_home_page_links_examples(tmp_path: Path) -> None:
    env = IdeaEnv(tmp_path, language="fa")
    body = env.get("/").body.decode()
    assert 'href="/new?example=1"' in body and 'class="button small newidea"' in body


def test_served_home_page_without_ideas_has_no_new_idea(tmp_path: Path) -> None:
    env = IdeaEnv(tmp_path, planner=False)
    env.store.create("x", project_id="p1")
    assert 'href="/new"' not in env.get("/").body.decode()


# -- build_overview from the real store ------------------------------------------


def test_overview_reads_open_questions_and_idea_questions(tmp_path: Path) -> None:
    env = IdeaEnv(tmp_path)
    ask = env.store.create("Needs a logo", project_id="p-ask").id
    env.store.set_status(ask, ProjectStatus.PAUSED)
    env.store.user_requests(ask).request("build", "Upload the logo", why="header", kind="file")
    idea = env.store.create("New idea", project_id="p-idea").id
    env.store.save_questions(idea, ["Who is it for?"])
    plain = env.store.create("Nothing waits", project_id="p-plain").id
    rows = {r.id: r for r in build_overview(env.store, {}).projects}
    assert rows[ask].open_asks == 1 and rows[ask].card is CardState.WAITING
    assert rows[idea].questions_waiting and rows[idea].card is CardState.WAITING
    assert rows[plain].card is CardState.TO_START and not rows[plain].questions_waiting
    body = env.get("/").body.decode()
    assert 'href="/projects/p-ask#ask"' in body and 'href="/projects/p-idea/clarify"' in body


def test_answered_questions_no_longer_wait(tmp_path: Path) -> None:
    env = IdeaEnv(tmp_path)
    r = env.new("Build a site")
    pid = dict(r.headers)["Location"].split("/")[2]
    env.post(f"/projects/{pid}/clarify", {"csrf": TOKEN, "a0": "me", "a1": "x"})
    row = next(r for r in build_overview(env.store, {}).projects if r.id == pid)
    assert env.store.has_plan(pid)
    assert not row.questions_waiting and row.card is not CardState.WAITING


def test_answered_ask_no_longer_waits(tmp_path: Path) -> None:
    env = IdeaEnv(tmp_path)
    pid = env.store.create("x", project_id="p1").id
    env.store.set_status(pid, ProjectStatus.PAUSED)
    req = env.store.user_requests(pid).request("build", "Name?", why="title", kind="text")
    env.store.user_requests(pid).answer(req.id, "Automa", via="cli")
    row = build_overview(env.store, {}).projects[0]
    assert row.open_asks == 0 and row.card is CardState.PAUSED


def test_unreadable_questions_become_an_error_line(tmp_path: Path) -> None:
    env = IdeaEnv(tmp_path)
    pid = env.store.create("x", project_id="p1").id
    requests = env.store.project_dir(pid) / "requests"
    requests.mkdir(parents=True, exist_ok=True)
    (requests / "req-0123456789ab.json").write_text("{", encoding="utf-8")
    view = build_overview(env.store, {})
    assert view.projects[0].open_asks == 0
    assert any("questions unreadable" in e for e in view.errors)


def test_delivered_project_card_is_ready_with_full_progress(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [*write("a"), *write("b")], _plan())
    rig.run(max_attempts=2)
    view = build_overview(rig.store, {})
    row = view.projects[0]
    assert row.delivery == "ready" and row.card is CardState.READY
    assert (row.tasks_total, row.tasks_passed) == (2, 2)
    doc = rd.render_overview(view, lang="fa")
    assert "cs-ready" in doc and "۲ از ۲ مرحله" in doc


def test_failed_project_card_is_incomplete(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [*write("a", "bad"), *write("a", "bad")], _plan())
    rig.run(max_attempts=2)
    row = build_overview(rig.store, {}).projects[0]
    assert row.delivery == "partial" and row.card is CardState.PARTIAL
    assert row.tasks_passed == 0 and row.tasks_total == 2
