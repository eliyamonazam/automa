"""Human status chips: icon, short word, plain sentence and a fixed colour tone."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from automa.plan.models import TaskStatus
from automa.project.store import ProjectStatus
from automa.webui import render as rd
from automa.webui import status as stt
from automa.webui.views import Overview, ProjectRow, ProjectView, TaskRow

T0 = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
TONES = {"ok", "busy", "paused", "bad", "info", "none"}


def _row(status: str) -> ProjectRow:
    return ProjectRow(
        id="p1",
        status=status,
        title="t",
        goal_preview="g",
        created_at=T0,
        cost_usd=0.0,
        pending_approvals=0,
    )


def _project(status: str = "running", states: tuple[str | None, ...] = ()) -> ProjectView:
    tasks = tuple(
        TaskRow(order=i + 1, id=f"t{i}", title=f"Task {i}", role="developer", state=s)
        for i, s in enumerate(states)
    )
    return ProjectView(
        id="p1",
        goal="g",
        status=status,
        created_at=T0,
        updated_at=T0,
        cost_usd=0,
        plan_version=1,
        tasks=tasks,
    )


def test_tables_cover_every_recorded_value() -> None:
    assert set(stt.PROJECT_LOOK) == {s.value for s in ProjectStatus}
    assert set(stt.TASK_LOOK) == {s.value for s in TaskStatus} | {"none"}
    assert set(stt.PROJECT_LOOK) == set(rd._PROJECT_STATUSES)
    assert set(stt.TASK_LOOK) - {"none"} == set(rd._STATE_CLASSES)


@pytest.mark.parametrize("lang", ["en", "fa"])
def test_every_known_value_has_icon_word_and_sentence(lang: str) -> None:
    for status in stt.PROJECT_LOOK:
        chip = stt.project_chip(status, lang)  # type: ignore[arg-type]
        assert chip.icon and chip.label and chip.hint and chip.tone in TONES - {"none"}
    for state in [*stt.TASK_LOOK, None]:
        key = None if state in (None, "none") else state
        chip = stt.task_chip(key, lang)  # type: ignore[arg-type]
        assert chip.icon and chip.label and chip.hint and chip.tone in TONES - {"none"}


def test_the_four_main_states_read_as_asked() -> None:
    assert stt.project_chip("done", "fa") == stt.Chip(
        "✅", "تموم شد", "همهٔ مرحله‌ها انجام شد.", "ok"
    )
    running = stt.project_chip("running", "fa")
    paused = stt.project_chip("paused", "fa")
    failed = stt.project_chip("failed", "fa")
    assert (running.icon, running.label, running.tone) == ("⏳", "در حال کار", "busy")
    assert (paused.icon, paused.label, paused.tone) == ("⏸️", "متوقف شد", "paused")
    assert (failed.icon, failed.label, failed.tone) == ("❌", "ناموفق", "bad")
    assert stt.task_chip("passed", "fa").icon == "✅"
    assert stt.task_chip("running", "fa").label == "در حال کار"
    assert stt.task_chip("failed", "en").label == "failed"


def test_project_and_task_tones_agree() -> None:
    for status, task_state in [("running", "running"), ("done", "passed"), ("failed", "failed")]:
        assert stt.project_chip(status, "fa").tone == stt.task_chip(task_state, "fa").tone


def test_unknown_value_is_shown_as_recorded_without_guessing() -> None:
    chip = stt.project_chip("archived", "fa")
    assert chip == stt.Chip(stt.UNKNOWN_ICON, "archived", "", "none")
    chip = stt.task_chip("weird", "fa")
    assert (chip.label, chip.hint, chip.tone) == ("weird", "", "none")


def test_unknown_language_is_refused() -> None:
    with pytest.raises(ValueError):
        stt.project_chip("done", "de")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        stt.task_chip("passed", "de")  # type: ignore[arg-type]


def test_messages_have_both_languages_and_no_fields() -> None:
    for key, by_lang in stt.MESSAGES.items():
        assert set(by_lang) == {"en", "fa"}, key
        assert "{" not in by_lang["en"] and "{" not in by_lang["fa"], key


def test_home_card_shows_chip_and_sentence() -> None:
    doc = rd.render_overview(Overview(projects=(_row("paused"),)), lang="fa")
    assert '<article class="card project cs-paused">' in doc
    assert "</svg>متوقف</span>" in doc
    assert '<p class="hint" dir="auto">وسط کار ایستاده؛ می‌شه ادامه داد.</p>' in doc


def test_project_page_hero_and_task_cards_use_chips() -> None:
    doc = rd.render_project(_project("running", ("passed", "running", None, "blocked")), lang="fa")
    assert 'class="badge ps-running tone-busy"' in doc
    assert 'class="badge st-passed tone-ok"' in doc
    assert 'class="badge st-running tone-busy"' in doc
    assert 'class="badge st-none tone-info"' in doc
    assert 'class="badge st-blocked tone-bad"' in doc
    assert '<p class="hint">انجام شد و بررسی رو قبول شد.</p>' in doc
    assert '<p class="hint">هنوز نوبتش نرسیده.</p>' in doc


def test_unknown_status_is_escaped_and_has_no_tooltip() -> None:
    doc = rd.render_overview(Overview(projects=(_row('"><x>'),)), lang="fa")
    assert "<x>" not in doc and "&quot;&gt;&lt;x&gt;" in doc
    assert '<article class="card project cs-unknown">' in doc
    assert '<p class="hint"' not in doc  # an unknown status gets no guessed sentence
    doc = rd.render_project(_project('"><x>', ('"><y>',)), lang="en")
    assert "<x>" not in doc and "<y>" not in doc
    assert 'class="badge st-none tone-none">' in doc


def test_paused_tone_has_its_own_colour() -> None:
    assert ".badge.tone-paused{" in rd._STYLE
