"""Tests for the task spending brake (``automa.budget.brake``, D153).

Unit tests drive a real ``BudgetGuard`` over an in-memory ``CostLedger``; the
end-to-end tests run the real ``automa run`` stack with a scripted ``FakeProvider``.
No network, no real model calls.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from automa import cli
from automa.budget import (
    BRAKE_CONTINUED,
    BRAKE_STOP,
    BrakeError,
    BudgetError,
    BudgetExceeded,
    BudgetGuard,
    BudgetScope,
    CostLedger,
    TaskBrake,
    TaskBrakeStop,
    brake_continues,
    brake_steps,
    cap_to_nusd,
    continue_task,
    pending_brake,
    usd_to_nusd,
)
from automa.config import BudgetLimits, Tier
from automa.events import Event, MemoryEventSink
from automa.llm import (
    CallContext,
    CallEstimate,
    ChatResponse,
    LLMResult,
    Message,
    ToolCall,
    Usage,
)
from automa.llm.types import FinishReason
from automa.llm.types import Role as MsgRole
from automa.orchestrator import StopReason
from automa.project import ProjectStatus
from automa.webui import stops as sp
from test_e2e import E2E, PID, TOML, build_attempt, design_attempt, docs_attempt, judgment

LIMITS = BudgetLimits(project_usd=1.0, daily_usd=2.0, per_call_usd=0.5)
T0 = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
STEP = usd_to_nusd(0.10)


@pytest.fixture
def ledger() -> Iterator[CostLedger]:
    with CostLedger(":memory:", clock=lambda: T0) as led:
        yield led


def est(
    max_cost: float = 0.01, project: str | None = "p1", task: str | None = "t1"
) -> CallEstimate:
    return CallEstimate(
        tier=Tier.SMALL,
        model="m-small",
        max_input_tokens=100,
        max_output_tokens=50,
        max_cost_usd=max_cost,
        context=CallContext(project_id=project, task_id=task),
    )


def res(cost: float) -> LLMResult:
    return LLMResult(
        tier=Tier.SMALL,
        response=ChatResponse(
            model="m-small", message=Message.assistant("ok"), usage=Usage(input_tokens=1)
        ),
        cost_usd=cost,
    )


def spend(guard: TaskBrake | BudgetGuard, cost: float, **ctx: Any) -> None:
    e = est(cost, **ctx)
    guard.authorize(e)
    guard.record(e, res(cost))


def make(
    ledger: CostLedger,
    *,
    limits: BudgetLimits = LIMITS,
    steps: dict[str, int] | None = None,
    continues: dict[str, int] | None = None,
    sink: MemoryEventSink | None = None,
) -> tuple[TaskBrake, BudgetGuard]:
    inner = BudgetGuard(limits, ledger)
    brake = TaskBrake(
        inner,
        ledger,
        "p1",
        {"t1": STEP} if steps is None else steps,
        default_step_nusd=usd_to_nusd(0.05),
        continues=continues,
        events=sink,
    )
    return brake, inner


# -- brake_steps --------------------------------------------------------------------


def test_steps_are_factor_times_high_with_a_minimum() -> None:
    limits = BudgetLimits(task_brake_min_usd=0.15, task_brake_factor=3.0)
    steps = brake_steps(["a", "b", "c"], limits, {"a": usd_to_nusd(0.40), "b": usd_to_nusd(0.01)})
    assert steps["a"] == usd_to_nusd(1.20)
    assert steps["b"] == cap_to_nusd(0.15)  # 3 x 0.01 is below the minimum
    assert steps["c"] == cap_to_nusd(0.15)  # no estimate: the minimum


def test_steps_without_any_estimate_are_the_minimum() -> None:
    limits = BudgetLimits(task_brake_min_usd=0.2)
    assert brake_steps(["a"], limits) == {"a": cap_to_nusd(0.2)}


def test_steps_round_down() -> None:
    limits = BudgetLimits(task_brake_min_usd=1e-9, task_brake_factor=1.5)
    assert brake_steps(["a"], limits, {"a": 3})["a"] == 4


def test_default_brake_settings() -> None:
    limits = BudgetLimits()
    assert limits.task_brake_min_usd == 0.15 and limits.task_brake_factor == 3.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task_brake_min_usd": 0},
        {"task_brake_min_usd": -1},
        {"task_brake_min_usd": float("inf")},
        {"task_brake_factor": 0.5},
        {"task_brake_factor": 1000},
        {"task_brake_factor": float("nan")},
    ],
)
def test_brake_settings_are_validated(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BudgetLimits(**kwargs)  # type: ignore[arg-type]


# -- TaskBrake ----------------------------------------------------------------------


def test_brake_refuses_once_the_task_cost_more_than_its_step(ledger: CostLedger) -> None:
    brake, inner = make(ledger)
    spend(brake, 0.06)
    spend(brake, 0.04)  # exactly the step: not more, still allowed afterwards
    spend(brake, 0.01)  # now 0.11 > 0.10
    with pytest.raises(TaskBrakeStop) as info:
        brake.authorize(est())
    stop = info.value
    assert isinstance(stop, BudgetError)  # every caller already stops on it
    assert stop.task_id == "t1" and stop.steps == 1
    assert stop.spent_nusd == usd_to_nusd(0.11) and stop.limit_nusd == STEP
    assert "far more than estimated" in str(stop)
    # Refused before the inner guard: nothing reserved, nothing latched.
    assert inner.pending_nusd() == 0
    assert inner.stops() == []


def test_brake_keeps_refusing_without_a_continue(ledger: CostLedger) -> None:
    brake, _ = make(ledger)
    spend(brake, 0.2)
    for _ in range(3):
        with pytest.raises(TaskBrakeStop):
            brake.authorize(est())
    assert ledger.spent_nusd(project_id="p1") == usd_to_nusd(0.2)


def test_other_tasks_projects_and_untasked_calls_are_not_braked(ledger: CostLedger) -> None:
    brake, _ = make(ledger)
    spend(brake, 0.2)
    spend(brake, 0.04, task="t2")  # its own (default) step
    spend(brake, 0.01, task=None)  # e.g. the Executive
    spend(brake, 0.01, project="p2")  # another project: only the guard applies
    with pytest.raises(TaskBrakeStop):
        brake.authorize(est())


def test_default_step_for_unknown_tasks(ledger: CostLedger) -> None:
    brake, _ = make(ledger)
    spend(brake, 0.06, task="new")
    with pytest.raises(TaskBrakeStop) as info:
        brake.authorize(est(task="new"))
    assert info.value.step_nusd == usd_to_nusd(0.05)


def test_continues_allow_one_more_step_each(ledger: CostLedger) -> None:
    brake, _ = make(ledger, continues={"t1": 1})
    assert brake.limit_nusd("t1") == 2 * STEP and brake.allowed_steps("t1") == 2
    spend(brake, 0.15)
    spend(brake, 0.05)  # 0.20 == two steps
    spend(brake, 0.01)
    with pytest.raises(TaskBrakeStop) as info:
        brake.authorize(est())
    assert info.value.steps == 2


def test_budget_guard_still_applies_under_the_brake(ledger: CostLedger) -> None:
    limits = BudgetLimits(project_usd=0.1, daily_usd=2.0, per_call_usd=0.05)
    brake, inner = make(ledger, limits=limits, steps={"t1": usd_to_nusd(10.0)})
    with pytest.raises(BudgetExceeded) as per_call:
        brake.authorize(est(0.06))
    assert per_call.value.scope is BudgetScope.PER_CALL
    spend(brake, 0.05)
    spend(brake, 0.04)
    with pytest.raises(BudgetExceeded) as project:
        brake.authorize(est(0.02))
    assert project.value.scope is BudgetScope.PROJECT
    assert not isinstance(project.value, TaskBrakeStop)
    assert inner.is_stopped(BudgetScope.PROJECT, "p1")


def test_record_and_release_go_to_the_inner_guard(ledger: CostLedger) -> None:
    brake, inner = make(ledger)
    e = est(0.02)
    brake.authorize(e)
    assert inner.pending_nusd() == usd_to_nusd(0.02)
    brake.release(e)
    assert inner.pending_nusd() == 0
    with pytest.raises(BudgetError):
        brake.record(est(0.03), res(0.03))  # never authorized: the guard refuses


def test_brake_stop_event_once_per_step(ledger: CostLedger) -> None:
    sink = MemoryEventSink()
    brake, _ = make(ledger, sink=sink)
    spend(brake, 0.2)
    for _ in range(2):
        with pytest.raises(TaskBrakeStop):
            brake.authorize(est())
    stops = [e for e in sink.events if e.type == BRAKE_STOP]
    assert len(stops) == 1
    data = stops[0].data
    assert data["task"] == "t1" and data["steps"] == 1
    assert data["spent_nusd"] == usd_to_nusd(0.2) and data["step_nusd"] == STEP


def test_failing_event_sink_still_refuses(ledger: CostLedger) -> None:
    class Broken:
        def emit(self, event: Event) -> None:
            raise OSError("disk full")

    brake = TaskBrake(
        BudgetGuard(LIMITS, ledger),
        ledger,
        "p1",
        {"t1": STEP},
        default_step_nusd=STEP,
        events=Broken(),
    )
    spend(brake, 0.2)
    with pytest.raises(TaskBrakeStop):
        brake.authorize(est())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"project_id": " "}, "project id"),
        ({"steps": {"t1": 0}}, "strictly positive"),
        ({"default_step_nusd": 0}, "strictly positive"),
        ({"continues": {"t1": -1}}, "negative"),
    ],
)
def test_brake_rejects_bad_settings(
    ledger: CostLedger, kwargs: dict[str, Any], message: str
) -> None:
    args: dict[str, Any] = {
        "project_id": "p1",
        "steps": {"t1": STEP},
        "default_step_nusd": STEP,
        "continues": None,
    } | kwargs
    with pytest.raises(ValueError, match=message):
        TaskBrake(
            BudgetGuard(LIMITS, ledger),
            ledger,
            args["project_id"],
            args["steps"],
            default_step_nusd=args["default_step_nusd"],
            continues=args["continues"],
        )


def test_ledger_spent_by_task(ledger: CostLedger) -> None:
    guard = BudgetGuard(LIMITS, ledger)
    spend(guard, 0.01)
    spend(guard, 0.02, task="t2")
    spend(guard, 0.04, project="p2")
    assert ledger.spent_nusd(project_id="p1", task_id="t1") == usd_to_nusd(0.01)
    assert ledger.spent_nusd(project_id="p1", task_id="t2") == usd_to_nusd(0.02)
    assert ledger.spent_nusd(task_id="t1") == usd_to_nusd(0.05)


# -- continue_task ------------------------------------------------------------------


def _event(type_: str, task: str = "t1", **data: object) -> Event:
    return Event(ts=T0, type=type_, project_id="p1", data={"task": task, **data})


def test_continue_needs_a_pending_stop() -> None:
    sink = MemoryEventSink()
    with pytest.raises(BrakeError):
        continue_task([], sink, "p1", "t1", via="cli")
    events = [_event(BRAKE_STOP, step_nusd=STEP, spent_nusd=2 * STEP)]
    with pytest.raises(BrakeError):
        continue_task(events, sink, "p1", "t2", via="cli")  # another task
    first = continue_task(events, sink, "p1", "t1", via="cli", now=T0)
    assert first.type == BRAKE_CONTINUED and first.data["steps"] == 2
    assert first.data["via"] == "cli" and first.data["step_nusd"] == STEP
    events.append(first)
    with pytest.raises(BrakeError):  # not ahead of time: one stop, one continue
        continue_task(events, sink, "p1", "t1", via="cli")
    events.append(_event(BRAKE_STOP, steps=2))
    second = continue_task(events, sink, "p1", "t1", via="web")
    assert second.data["steps"] == 3
    assert brake_continues([*events, second]) == {"t1": 2}
    assert len(sink.events) == 2


def test_pending_brake_ignores_other_events() -> None:
    events = [
        _event(BRAKE_STOP),
        Event(ts=T0, type="orchestrator.paused", project_id="p1", data={"task": "t1"}),
    ]
    assert pending_brake(events, "t1") is events[0]
    assert brake_continues([_event(BRAKE_CONTINUED, task=None)]) == {}  # type: ignore[arg-type]


# -- web UI sentence ----------------------------------------------------------------


def test_web_stop_sentence_for_the_brake() -> None:
    pause = Event(
        ts=T0,
        type="orchestrator.paused",
        project_id="p1",
        data={"reason": "task_brake", "task": "t1"},
    )
    start = Event(ts=T0, type="orchestrator.started", project_id="p1", data={})
    stop = sp.latest_stop([start, pause])
    assert stop is not None and stop.kind == "task_brake"
    line = sp.describe_stop(stop, lang="fa", tz=UTC, now=T0)
    assert "ترمز خرج" in line.text and "automa brake continue" in line.text
    assert line.resume is False and line.tone == "bad"


# -- end to end: automa run + automa brake continue ---------------------------------


def costly(*items: tuple[str, dict[str, Any]], input_tokens: int = 200_000) -> ChatResponse:
    """A worker step whose real usage is far larger than usual ($0.20 at $1/Mtok)."""
    tcs = tuple(
        ToolCall(id=f"c{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(items)
    )
    return ChatResponse(
        model="fake",
        message=Message(role=MsgRole.ASSISTANT, content=None, tool_calls=tcs),
        usage=Usage(input_tokens=input_tokens, output_tokens=40),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def _brake_config(e2e: E2E, *, minimum: float = 0.5) -> None:
    # The architect task runs on fake-large ($4/Mtok input): one costly step is
    # $0.80, more than one $0.50 step (the minimum is above 1 x its range) but not
    # more than two.
    text = TOML.format(ws=e2e.ws.as_posix(), project=5.0, per_call=0.25)
    text += f"task_brake_min_usd = {minimum}\ntask_brake_factor = 1.0\n"
    e2e.config.write_text(text, encoding="utf-8")


@pytest.fixture
def e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> E2E:
    return E2E(tmp_path, monkeypatch)


def test_e2e_brake_pauses_continue_and_finish(e2e: E2E) -> None:
    e2e.new_project()
    _brake_config(e2e)
    note = {"path": "notes.md", "content": "x"}
    e2e.script(costly(("write_file", note)), *design_attempt())
    r = e2e.cli("run", PID, "--yes")
    assert r.exit_code == cli.EXIT_PAUSED, r.output
    assert "spending brake:" in r.output
    assert "paused at task design" in r.output and "task brake" in r.output
    assert f"automa brake continue {PID} design" in r.output
    # 2 planning calls + the one costly worker step; nothing after the brake.
    assert len(e2e.provider.requests) == 3
    assert e2e.store.load(PID).status is ProjectStatus.PAUSED
    events = e2e.store.events(PID).read()
    paused = [e for e in events if e.type == "orchestrator.paused"]
    assert paused[-1].data["reason"] == StopReason.TASK_BRAKE.value
    assert [e.data["task"] for e in events if e.type == BRAKE_STOP] == ["design"]
    spent = e2e.ledger_spent_nusd()

    # Running again without a decision: refused before any provider call.
    r = e2e.cli("run", PID, "--yes")
    assert r.exit_code == cli.EXIT_PAUSED, r.output
    assert len(e2e.provider.requests) == 3 and e2e.ledger_spent_nusd() == spent

    # Unknown task or a task that was not braked: refused, nothing recorded.
    for task in ("nope", "build"):
        r = e2e.cli("brake", "continue", PID, task)
        assert r.exit_code == cli.EXIT_USER, r.output
        assert "not stopped by the spending brake" in r.output

    r = e2e.cli("brake", "continue", PID, "design")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "one more step of $0.5000 allowed (now 2 step(s))" in r.output
    r = e2e.cli("brake", "continue", PID, "design")
    assert r.exit_code == cli.EXIT_USER, r.output  # one step per stop
    continued = [e for e in e2e.store.events(PID).read() if e.type == BRAKE_CONTINUED]
    assert len(continued) == 1 and continued[0].data["via"] == "cli"

    # With one more step the project completes.
    e2e.script(*build_attempt(), judgment(True), *docs_attempt())
    r = e2e.cli("run", PID, "--yes")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert e2e.status_json()["status"] == ProjectStatus.DONE.value
    assert e2e.provider.remaining == 0


def test_e2e_cheap_project_never_meets_the_brake(e2e: E2E) -> None:
    e2e.new_project()
    e2e.script(*design_attempt(), *build_attempt(), judgment(True), *docs_attempt())
    r = e2e.cli("run", PID, "--yes")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert BRAKE_STOP not in e2e.event_types()


def test_e2e_brake_continue_unknown_project(e2e: E2E) -> None:
    r = e2e.cli("brake", "continue", "missing", "design")
    assert r.exit_code == cli.EXIT_USER, r.output


def test_e2e_project_cap_still_wins_over_a_continue(e2e: E2E) -> None:
    e2e.new_project()
    _brake_config(e2e)
    e2e.script(costly(("write_file", {"path": "n.md", "content": "x"})), *design_attempt())
    assert e2e.cli("run", PID, "--yes").exit_code == cli.EXIT_PAUSED
    assert e2e.cli("brake", "continue", PID, "design").exit_code == cli.EXIT_OK
    # A project cap below what was already spent: the BudgetGuard stops the run.
    e2e.write_config(project=e2e.ledger_spent_nusd() / 1e9 + 0.001)
    mark = len(e2e.provider.requests)
    r = e2e.cli("run", PID, "--yes")
    assert r.exit_code == cli.EXIT_BUDGET, r.output
    assert len(e2e.provider.requests) == mark
