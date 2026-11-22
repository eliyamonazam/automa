"""The whole-run spending brake (D172).

Unit tests drive a real ``BudgetGuard`` over an in-memory ``CostLedger``; the
end-to-end tests run the real ``automa run`` stack with a scripted ``FakeProvider``.
No network, no real model calls.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from automa import cli
from automa.budget import (
    BRAKE_CONTINUED,
    BRAKE_STOP,
    RUN_KEY,
    BudgetError,
    BudgetExceeded,
    BudgetGuard,
    CostLedger,
    RunBrakeStop,
    TaskBrake,
    TaskBrakeStop,
    continue_task,
    pending_brake,
    usd_to_nusd,
)
from automa.budget.brake import decline_task, run_brake_limit
from automa.config import BudgetLimits
from automa.events import Event, MemoryEventSink
from automa.orchestrator import StopReason
from automa.plan.models import TASK_ID_PATTERN
from automa.project import ProjectStatus
from automa.webui import stops as sp
from test_e2e import E2E, PID, TOML, build_attempt, design_attempt, docs_attempt, judgment
from test_task_brake import LIMITS, STEP, T0, costly, est, ledger, spend  # noqa: F401

RUN = usd_to_nusd(0.10)


def make(
    led: CostLedger,
    *,
    limits: BudgetLimits = LIMITS,
    baseline: int = 0,
    waiting: bool = False,
    sink: MemoryEventSink | None = None,
    task_step: int = usd_to_nusd(10),
) -> TaskBrake:
    return TaskBrake(
        BudgetGuard(limits, led),
        led,
        "p1",
        {"t1": task_step, "t2": task_step},
        default_step_nusd=task_step,
        events=sink,
        run_limit_nusd=RUN,
        run_baseline_nusd=baseline,
        run_waiting=waiting,
    )


# -- settings -------------------------------------------------------------------------


def test_run_limit_is_factor_times_high_with_a_minimum() -> None:
    limits = BudgetLimits(run_brake_min_usd=0.30, run_brake_factor=2.0)
    assert run_brake_limit(limits, usd_to_nusd(1.0)) == usd_to_nusd(2.0)
    assert run_brake_limit(limits, usd_to_nusd(0.01)) == usd_to_nusd(0.30)
    assert run_brake_limit(limits, None) == usd_to_nusd(0.30)  # never off
    assert run_brake_limit(BudgetLimits(run_brake_factor=1.5), 3) == usd_to_nusd(0.30)


def test_default_run_brake_settings() -> None:
    lim = BudgetLimits()
    assert (lim.run_brake_min_usd, lim.run_brake_factor) == (0.30, 2.0)


@pytest.mark.parametrize(
    "kwargs",
    [{"run_brake_min_usd": 0}, {"run_brake_factor": 0.5}, {"run_brake_factor": 101}],
)
def test_run_brake_settings_are_validated(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BudgetLimits(**kwargs)


def test_run_key_is_never_a_task_id() -> None:
    import re

    assert re.match(TASK_ID_PATTERN, RUN_KEY) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_limit_nusd": 0},
        {"run_limit_nusd": 5, "run_baseline_nusd": -1},
        {"run_waiting": True},
    ],
)
def test_bad_run_settings_are_refused(ledger: CostLedger, kwargs: dict[str, object]) -> None:  # noqa: F811
    with pytest.raises(ValueError):
        TaskBrake(BudgetGuard(LIMITS, ledger), ledger, "p1", {}, default_step_nusd=1, **kwargs)  # type: ignore[arg-type]


# -- the brake ------------------------------------------------------------------------


def test_run_brake_refuses_once_the_run_cost_more(ledger: CostLedger) -> None:  # noqa: F811
    sink = MemoryEventSink()
    brake = make(ledger, sink=sink)
    spend(brake, 0.06, task="t1")
    spend(brake, 0.04, task="t2")  # exactly the limit: still allowed
    spend(brake, 0.01, task="t2")  # now more than the limit
    before = ledger.spent_nusd()
    for task in ("t1", "t2", None):  # every call of the project, with or without a task
        with pytest.raises(RunBrakeStop) as info:
            brake.authorize(est(0.01, task=task))
        assert isinstance(info.value, BudgetError)  # every caller already stops on it
        assert not isinstance(info.value, TaskBrakeStop)
    assert ledger.spent_nusd() == before  # nothing reserved or recorded after the brake
    stops = [e for e in sink.events if e.type == BRAKE_STOP]
    assert len(stops) == 1  # announced once
    assert stops[0].data["task"] == RUN_KEY and stops[0].data["scope"] == "run"
    assert stops[0].data["limit_nusd"] == RUN
    assert stops[0].data["spent_nusd"] == usd_to_nusd(0.11)


def test_run_brake_counts_only_this_run(ledger: CostLedger) -> None:  # noqa: F811
    spend(BudgetGuard(LIMITS, ledger), 0.50, task="t1")  # an earlier run
    brake = make(ledger, baseline=ledger.spent_nusd(project_id="p1"))
    spend(brake, 0.09, task="t1")
    spend(brake, 0.01, task="t1")


def test_other_projects_are_not_braked(ledger: CostLedger) -> None:  # noqa: F811
    brake = make(ledger)
    spend(brake, 0.2, project="p2")
    spend(brake, 0.2, project="p2")
    spend(brake, 0.05, task="t1")


def test_waiting_run_brake_refuses_the_first_call(ledger: CostLedger) -> None:  # noqa: F811
    sink = MemoryEventSink()
    brake = make(ledger, waiting=True, sink=sink)
    with pytest.raises(RunBrakeStop) as info:
        brake.authorize(est(0.001, task="t1"))
    assert info.value.waiting
    assert ledger.spent_nusd() == 0 and not sink.events  # no new stop recorded
    spend(brake, 0.01, project="p2")  # another project still runs


def test_task_brake_comes_first(ledger: CostLedger) -> None:  # noqa: F811
    brake = make(ledger, task_step=usd_to_nusd(0.05))
    spend(brake, 0.06, task="t1")
    with pytest.raises(TaskBrakeStop):
        brake.authorize(est(0.01, task="t1"))
    spend(brake, 0.05, task="t2")  # t2 within its step, the run now over its brake
    with pytest.raises(RunBrakeStop):
        brake.authorize(est(0.01, task="t2"))


def test_budget_guard_still_applies_under_the_run_brake(ledger: CostLedger) -> None:  # noqa: F811
    tight = BudgetLimits(project_usd=0.05, daily_usd=2.0, per_call_usd=0.05)
    brake = make(ledger, limits=tight)
    spend(brake, 0.04, task="t1")
    with pytest.raises(BudgetExceeded):
        brake.authorize(est(0.02, task="t1"))  # the run brake allows it, the cap does not


def _ev(type_: str, n: int, **data: object) -> Event:
    return Event(ts=T0 + timedelta(seconds=n), type=type_, project_id="p1", data=data)


def test_run_continue_and_decline_use_the_run_key() -> None:
    events = [_ev(BRAKE_STOP, 0, task=RUN_KEY, scope="run", spent_nusd=5, limit_nusd=3)]
    assert pending_brake(events, RUN_KEY) is not None
    sink = MemoryEventSink()
    decline_task(events, sink, "p1", RUN_KEY, via="web", now=T0)
    events += sink.events
    assert pending_brake(events, RUN_KEY) is not None  # declining allows nothing
    continue_task(events, sink, "p1", RUN_KEY, via="web", now=T0)
    events.append(sink.events[-1])
    assert events[-1].type == BRAKE_CONTINUED
    assert pending_brake(events, RUN_KEY) is None


# -- the web's stop sentence ----------------------------------------------------------


def test_web_stop_for_the_run_brake() -> None:
    from zoneinfo import ZoneInfo

    stop_ev = _ev(BRAKE_STOP, 1, task=RUN_KEY, scope="run", spent_nusd=900, limit_nusd=400)
    events = [
        _ev("orchestrator.started", 0),
        stop_ev,
        _ev("orchestrator.paused", 2, reason="run_brake", task="t1"),
    ]
    info = sp.latest_stop(events)
    assert info is not None and info.kind == "run_brake"
    assert (info.task, info.spent_nusd, info.limit_nusd) == (RUN_KEY, 900, 400)
    assert info.awaiting and not info.declined
    tz = ZoneInfo("UTC")
    line = sp.describe_stop(info, lang="fa", tz=tz, now=T0)
    assert "این اجرا" in line.text and not line.resume
    # A run started while waiting is refused at once without a new stop event.
    events += [
        _ev("orchestrator.started", 3),
        _ev("orchestrator.paused", 4, reason="run_brake", task="t1"),
    ]
    info = sp.latest_stop(events)
    assert info is not None and info.spent_nusd == 900 and info.awaiting
    events.append(_ev(BRAKE_CONTINUED, 5, task=RUN_KEY, via="cli"))
    info = sp.latest_stop(events)
    assert info is not None and not info.awaiting
    line = sp.describe_stop(info, lang="en", tz=tz, now=T0)
    assert line.resume and "going on is allowed" in line.text


# -- end to end -----------------------------------------------------------------------


def _run_brake_config(e2e: E2E) -> None:
    # The task brake is far away; the run brake is 1 x the run's range (or $0.50),
    # which one costly architect step ($0.80) passes.
    text = TOML.format(ws=e2e.ws.as_posix(), project=5.0, per_call=0.25)
    text += (
        "task_brake_min_usd = 50\ntask_brake_factor = 1.0\n"
        "run_brake_min_usd = 0.5\nrun_brake_factor = 1.0\n"
    )
    e2e.config.write_text(text, encoding="utf-8")


@pytest.fixture
def e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> E2E:
    return E2E(tmp_path, monkeypatch)


def test_e2e_run_brake_pauses_continue_and_finish(e2e: E2E) -> None:
    e2e.new_project()
    _run_brake_config(e2e)
    note = {"path": "notes.md", "content": "x"}
    e2e.script(costly(("write_file", note)), *design_attempt())
    r = e2e.cli("run", PID, "--yes")
    assert r.exit_code == cli.EXIT_PAUSED, r.output
    # 1 x the high end of the run estimate ($0.7192) is above the $0.50 minimum.
    assert "this run pauses once it costs more than $0.7192" in r.output
    assert "status: paused (run_brake)" in r.output
    assert f"automa brake continue {PID} --run" in r.output
    # 2 planning calls + the one costly worker step; nothing after the brake.
    assert len(e2e.provider.requests) == 3
    assert e2e.store.load(PID).status is ProjectStatus.PAUSED
    events = e2e.store.events(PID).read()
    paused = [e for e in events if e.type == "orchestrator.paused"]
    assert paused[-1].data["reason"] == StopReason.RUN_BRAKE.value
    assert [e.data["task"] for e in events if e.type == BRAKE_STOP] == [RUN_KEY]
    spent = e2e.ledger_spent_nusd()

    # Running again without a decision: refused before any provider call.
    r = e2e.cli("run", PID, "--yes")
    assert r.exit_code == cli.EXIT_PAUSED, r.output
    assert len(e2e.provider.requests) == 3 and e2e.ledger_spent_nusd() == spent

    # A task id and --run together, or neither: refused.
    assert e2e.cli("brake", "continue", PID, "design", "--run").exit_code == cli.EXIT_USER
    assert e2e.cli("brake", "continue", PID).exit_code == cli.EXIT_USER
    r = e2e.cli("brake", "continue", PID, "design")  # the task was not braked
    assert r.exit_code == cli.EXIT_USER and "not stopped by the spending brake" in r.output

    r = e2e.cli("brake", "continue", PID, "--run")
    assert r.exit_code == cli.EXIT_OK, r.output
    r = e2e.cli("brake", "continue", PID, "--run")
    assert r.exit_code == cli.EXIT_USER, r.output  # one decision per stop
    continued = [e for e in e2e.store.events(PID).read() if e.type == BRAKE_CONTINUED]
    assert len(continued) == 1 and continued[0].data["task"] == RUN_KEY

    # The next run starts with a fresh allowance and the project completes.
    e2e.script(*build_attempt(), judgment(True), *docs_attempt())
    r = e2e.cli("run", PID, "--yes")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert e2e.status_json()["status"] == ProjectStatus.DONE.value
    assert e2e.provider.remaining == 0


def test_e2e_project_cap_still_wins_after_a_run_continue(e2e: E2E) -> None:
    e2e.new_project()
    _run_brake_config(e2e)
    e2e.script(costly(("write_file", {"path": "n.md", "content": "x"})), *design_attempt())
    assert e2e.cli("run", PID, "--yes").exit_code == cli.EXIT_PAUSED
    assert e2e.cli("brake", "continue", PID, "--run").exit_code == cli.EXIT_OK
    e2e.write_config(project=e2e.ledger_spent_nusd() / 1e9 + 0.001)
    mark = len(e2e.provider.requests)
    r = e2e.cli("run", PID, "--yes")
    assert r.exit_code == cli.EXIT_BUDGET, r.output
    assert len(e2e.provider.requests) == mark
