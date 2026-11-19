"""The project journey: steps, percentage and the "now" sentence (D141)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from automa.webui import journey as jy
from automa.webui.views import ApprovalView, ProjectView, TaskRow

T0 = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)


def _view(status: str = "planning", *, plan: bool = True, states=(), pending: int = 0):
    tasks = tuple(
        TaskRow(order=i + 1, id=f"t{i}", title=f"Task {i}", role="developer", state=s)
        for i, s in enumerate(states)
    )
    approvals = tuple(
        ApprovalView(
            id=f"a{i}",
            task_id=None,
            tool="fetch_url",
            risk="external",
            reason="r",
            created_at=T0,
            arguments={},
        )
        for i in range(pending)
    )
    return ProjectView(
        id="p1",
        goal="g",
        status=status,
        created_at=T0,
        updated_at=T0,
        cost_usd=0,
        plan_version=1 if plan else None,
        tasks=tasks,
        pending=approvals,
    )


def _states(j: jy.Journey) -> list[str]:
    return [s.state for s in j.steps]


def test_planning_without_plan() -> None:
    j = jy.build_journey(_view(plan=False), lang="en")
    assert [s.name for s in j.steps] == ["Idea", "Questions", "Plan", "Result"]
    assert _states(j) == ["done", "done", "now", "todo"]
    assert j.now == "Making the plan…" and j.tone == "busy"


def test_questions_waiting() -> None:
    j = jy.build_journey(_view(plan=False), lang="fa", questions_waiting=True)
    assert _states(j) == ["done", "now", "todo", "todo"]
    assert j.now == "منتظر جواب تو به چند تا سؤاله." and j.tone == "busy"


def test_ready_plan_lists_each_task() -> None:
    j = jy.build_journey(_view(states=(None, None)), lang="en")
    assert [s.name for s in j.steps][3:5] == ["Task 0", "Task 1"]
    assert _states(j) == ["done", "done", "done", "todo", "todo", "todo"]
    assert j.percent == 50 and j.tone == "info" and "Start" in j.now


def test_running_marks_the_first_open_task_now() -> None:
    j = jy.build_journey(_view("running", states=("passed", None, None)), running=True)
    assert _states(j)[3:6] == ["done", "now", "todo"]
    assert j.now == "Working on step 2 of 3: Task 1" and j.tone == "busy"


def test_persian_digits_in_sentence() -> None:
    j = jy.build_journey(_view("running", states=("running",)), lang="fa", running=True)
    assert "مرحلهٔ ۱ از ۱" in j.now


def test_done_is_100_percent() -> None:
    j = jy.build_journey(_view("done", states=("passed",)), lang="en")
    assert j.percent == 100 and j.tone == "ok" and _states(j)[-1] == "done"


def test_never_100_before_done() -> None:
    j = jy.build_journey(_view("running", states=("passed", "passed")), lang="en")
    assert j.percent < 100


def test_failed_names_the_task() -> None:
    j = jy.build_journey(_view("failed", states=("passed", "failed", "blocked")), lang="en")
    assert j.tone == "bad" and "Task 1" in j.now
    assert _states(j)[3:6] == ["done", "fail", "fail"]


def test_pending_approval_comes_first() -> None:
    j = jy.build_journey(_view("running", states=("running",), pending=2), running=True)
    assert j.tone == "busy" and "2 request" in j.now


def test_paused() -> None:
    j = jy.build_journey(_view("paused", states=("passed", None)), lang="en")
    assert j.now.startswith("Paused after 1 of 2") and j.tone == "info"


def test_messages_have_both_languages_and_same_fields() -> None:
    for key, by_lang in jy.MESSAGES.items():
        assert set(by_lang) == {"en", "fa"}, key


def test_bad_params_raise() -> None:
    with pytest.raises(ValueError):
        jy.jt("now.working", "en", step=1)
    with pytest.raises(ValueError):
        jy.jt("step.idea", "xx")  # type: ignore[arg-type]


# --- rendering (D141) ------------------------------------------------------------------

from automa.webui import progress as pg  # noqa: E402
from automa.webui import render as rd  # noqa: E402


def test_project_page_shows_hero_now_bar_and_steps() -> None:
    view = _view("running", states=("passed", None)).model_copy(
        update={"goal": "Build a <tool>\nmore"}
    )
    doc = rd.render_project(view, lang="fa")
    assert '<html lang="fa" dir="rtl">' in doc
    assert '<header class="hero"><h1 dir="auto">Build a &lt;tool&gt;' in doc
    assert '<span class="badge ps-running">در حال کار</span>' in doc
    assert '<p class="now info" dir="auto">' in doc
    assert 'class="fill" style="width:' in doc and "٪ انجام شده" in doc
    assert doc.count('<li class="s-done">') == 4  # idea, questions, plan, task 0
    assert '<li class="s-todo"><span class="dot">۵</span>' in doc
    assert "<title>Automa — Build a &lt;tool&gt;</title>" in doc
    assert "→ همهٔ پروژه‌ها" in doc


def test_running_run_marks_step_now() -> None:
    from automa.webui.render import RunPanel

    view = _view("running", states=(None,))
    run = RunPanel(can_start=False, needs_confirm=False, running=True, state="running")
    doc = rd.render_project(view, run=run)
    assert '<li class="s-now"><span class="dot">4</span>' in doc
    assert "Working on step 1 of 1" in doc


def test_unknown_status_class_is_fixed() -> None:
    doc = rd.render_project(_view('"><x>'))
    assert 'class="badge ps-none"' in doc and "<x>" not in doc


def _cost(spent: float, verdict: str) -> pg.CostProgress:
    return pg.CostProgress(
        estimated_at=T0,
        spent_usd=spent,
        other_usd=0,
        low_usd=0.04,
        high_usd=0.14,
        worst_usd=1.0,
        verdict=verdict,
    )


def test_meter_turns_red_above_the_range() -> None:
    doc = rd.render_project(_view(), progress=pg.ProgressView(cost=_cost(0.43, "above")))
    assert 'low="0.140000" high="0.140000" optimum="0" max="0.430000" value="0.430000"' in doc


def test_cost_card_in_persian() -> None:
    view = _view().model_copy(update={"cost_usd": 0.5, "planning_usd": 0.1})
    doc = rd.render_project(
        view, progress=pg.ProgressView(cost=_cost(0.43, "above")), rate=323_000, lang="fa"
    )
    assert "این اجرا از بازهٔ پیش‌بینی بیشتر خرج کرد:" in doc
    assert "برنامه‌ریزی: " in doc and "انجام کارها: " in doc
    assert "تومان" in doc and "This run" not in doc


def test_events_are_folded_into_technical_details() -> None:
    ev = (pg.EventLine(ts=T0, type="plan.saved", summary=""),)
    doc = rd.render_project(_view(), progress=pg.ProgressView(events=ev))
    assert doc.index("<details><summary>Technical details</summary>") < doc.index("plan.saved")
