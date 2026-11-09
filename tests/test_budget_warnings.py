"""Tests for budget warning (50% / 80%) and stop events. FakeProvider only, no network."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from automa.budget import (
    WARNING_THRESHOLDS_PCT,
    BudgetError,
    BudgetExceeded,
    BudgetGuard,
    BudgetScope,
    CostLedger,
    cap_to_nusd,
)
from automa.config import AutomaConfig, BudgetLimits, Tier
from automa.events import Event, JsonlEventLog, MemoryEventSink
from automa.llm import (
    CallContext,
    CallEstimate,
    ChatResponse,
    FakeProvider,
    LLMGateway,
    LLMResult,
    Message,
    Usage,
)

LIMITS = BudgetLimits(project_usd=1.0, daily_usd=2.0, per_call_usd=0.5)


class FixedClock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now


class FailingSink:
    def __init__(self) -> None:
        self.attempts = 0

    def emit(self, event: Event) -> None:
        self.attempts += 1
        raise OSError("disk full")


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(datetime(2026, 11, 9, 12, 0, tzinfo=UTC))


@pytest.fixture
def ledger(clock: FixedClock) -> Iterator[CostLedger]:
    with CostLedger(":memory:", clock=clock) as led:
        yield led


@pytest.fixture
def sink() -> MemoryEventSink:
    return MemoryEventSink()


@pytest.fixture
def guard(ledger: CostLedger, sink: MemoryEventSink) -> BudgetGuard:
    return BudgetGuard(LIMITS, ledger, events=sink)


def est(max_cost: float, project: str = "p1") -> CallEstimate:
    return CallEstimate(
        tier=Tier.SMALL,
        model="m-small",
        max_input_tokens=10,
        max_output_tokens=10,
        max_cost_usd=max_cost,
        context=CallContext(project_id=project, task_id="t1"),
    )


def res(cost: float) -> LLMResult:
    return LLMResult(
        tier=Tier.SMALL,
        response=ChatResponse(
            model="m-small", message=Message.assistant("ok"), usage=Usage(input_tokens=1)
        ),
        cost_usd=cost,
    )


def spend(guard: BudgetGuard, cost: float, project: str = "p1") -> None:
    e = est(cost, project)
    guard.authorize(e)
    guard.record(e, res(cost))


def warnings(sink: MemoryEventSink) -> list[tuple[str, str, int]]:
    return [
        (e.data["scope"], e.data["key"], e.data["threshold_pct"])
        for e in sink.of_type("budget.warning")
    ]


def test_thresholds_are_50_and_80() -> None:
    assert WARNING_THRESHOLDS_PCT == (50, 80)


def test_no_warning_below_50(guard: BudgetGuard, sink: MemoryEventSink) -> None:
    spend(guard, 0.4)
    spend(guard, 0.099999999)  # 0.499999999 of a $1 project cap
    assert sink.events == [] and guard.events() == []


def test_project_warnings_at_exact_thresholds(guard: BudgetGuard, sink: MemoryEventSink) -> None:
    spend(guard, 0.5)  # exactly 50% of project
    assert warnings(sink) == [("project", "p1", 50)]
    spend(guard, 0.3)  # exactly 80% of project, 40% of daily
    assert warnings(sink) == [("project", "p1", 50), ("project", "p1", 80)]
    ev = sink.of_type("budget.warning")[-1]
    assert ev.project_id == "p1"
    assert ev.data["limit_nusd"] == cap_to_nusd(1.0)
    assert ev.data["spent_nusd"] == cap_to_nusd(0.8)
    assert ev.data["spent_usd"] == pytest.approx(0.8)
    assert ev.ts == datetime(2026, 11, 9, 12, 0, tzinfo=UTC)


def test_single_jump_emits_both_in_order(guard: BudgetGuard, sink: MemoryEventSink) -> None:
    spend(guard, 0.45)
    spend(guard, 0.4)  # 0.85 of project in one step
    assert warnings(sink) == [("project", "p1", 50), ("project", "p1", 80)]


def test_each_warning_emitted_once(guard: BudgetGuard, sink: MemoryEventSink) -> None:
    for _ in range(9):
        spend(guard, 0.1)
    assert warnings(sink) == [("project", "p1", 50), ("project", "p1", 80)]
    assert sink.of_type("budget.stop") == []


def test_warnings_do_not_block_spending(guard: BudgetGuard, ledger: CostLedger) -> None:
    spend(guard, 0.5)
    spend(guard, 0.3)
    spend(guard, 0.2)  # allowed up to exactly the cap
    assert ledger.project_cost("p1") == pytest.approx(1.0)


def test_daily_warning_spans_projects(guard: BudgetGuard, sink: MemoryEventSink) -> None:
    spend(guard, 0.45, "p1")
    spend(guard, 0.45, "p2")
    spend(guard, 0.1, "p3")  # daily now exactly $1 = 50% of $2
    daily = [w for w in warnings(sink) if w[0] == "daily"]
    assert daily == [("daily", "2026-11-09", 50)]
    assert sink.of_type("budget.warning")[-1].project_id == "p3"  # attributed to trigger


def test_daily_warnings_reset_next_utc_day(
    guard: BudgetGuard, sink: MemoryEventSink, clock: FixedClock
) -> None:
    for p in ("a", "b"):
        spend(guard, 0.5, p)
    clock.now += timedelta(days=1)
    for p in ("c", "d"):
        spend(guard, 0.5, p)
    daily = [w for w in warnings(sink) if w[0] == "daily"]
    assert daily == [("daily", "2026-11-09", 50), ("daily", "2026-11-10", 50)]


def test_projects_warn_independently(guard: BudgetGuard, sink: MemoryEventSink) -> None:
    spend(guard, 0.5, "p1")
    spend(guard, 0.5, "p2")
    project = [w for w in warnings(sink) if w[0] == "project"]
    assert project == [("project", "p1", 50), ("project", "p2", 50)]


def test_prior_ledger_spending_warns_on_next_record(
    tmp_path: Path, clock: FixedClock, sink: MemoryEventSink
) -> None:
    db = tmp_path / "ledger.sqlite3"
    with CostLedger(db, clock=clock) as led:
        spend(BudgetGuard(LIMITS, led), 0.45)
    with CostLedger(db, clock=clock) as led:
        g = BudgetGuard(LIMITS, led, events=sink)
        spend(g, 0.4)
    assert warnings(sink) == [("project", "p1", 50), ("project", "p1", 80)]


def test_stop_event_when_actual_spending_reaches_cap(
    guard: BudgetGuard, sink: MemoryEventSink
) -> None:
    spend(guard, 0.5)
    spend(guard, 0.5)
    stops = sink.of_type("budget.stop")
    assert [(e.data["scope"], e.data["key"]) for e in stops] == [("project", "p1")]
    assert guard.is_stopped(BudgetScope.PROJECT, "p1")
    with pytest.raises(BudgetExceeded):
        guard.authorize(est(0.01))
    assert len(sink.of_type("budget.stop")) == 1  # repeated refusals do not spam


def test_stop_event_on_refused_authorization(guard: BudgetGuard, sink: MemoryEventSink) -> None:
    spend(guard, 0.4)
    spend(guard, 0.4)
    with pytest.raises(BudgetExceeded) as info:
        guard.authorize(est(0.3))
    stop = sink.of_type("budget.stop")
    assert len(stop) == 1
    assert stop[0].data["scope"] == "project" and stop[0].data["requested_nusd"] == (
        info.value.requested_nusd
    )
    with pytest.raises(BudgetExceeded) as again:
        guard.authorize(est(0.01))
    assert again.value.latched and len(sink.of_type("budget.stop")) == 1


def test_per_call_refusal_emits_nothing(guard: BudgetGuard, sink: MemoryEventSink) -> None:
    with pytest.raises(BudgetExceeded):
        guard.authorize(est(0.6))
    assert sink.events == []


def test_events_kept_in_memory_without_sink(ledger: CostLedger) -> None:
    g = BudgetGuard(LIMITS, ledger)
    spend(g, 0.5)
    assert [e.type for e in g.events()] == ["budget.warning"]


def test_failing_sink_fails_closed_but_keeps_accounting(ledger: CostLedger) -> None:
    failing = FailingSink()
    g = BudgetGuard(LIMITS, ledger, events=failing)
    e = est(0.5)
    g.authorize(e)
    with pytest.raises(BudgetError, match="event sink failed") as info:
        g.record(e, res(1.0))  # overrun: reaches 50%, 80% and the stop at once
    assert isinstance(info.value.__cause__, OSError)
    assert failing.attempts == 4  # project 50%, 80%, daily 50%, stop: all attempted
    assert ledger.project_cost("p1") == pytest.approx(1.0)  # money still recorded
    assert g.pending_nusd() == 0  # reservation freed
    assert g.is_stopped(BudgetScope.PROJECT, "p1")  # latch still in effect
    with pytest.raises(BudgetExceeded):
        g.authorize(est(0.01))


def test_failing_sink_on_refusal_still_refuses(ledger: CostLedger) -> None:
    g = BudgetGuard(LIMITS, ledger, events=FailingSink())
    spend(g, 0.45)
    with pytest.raises(BudgetError):
        spend(g, 0.45)  # warnings fail to deliver
    with pytest.raises(BudgetExceeded) as info:
        g.authorize(est(0.2))
    assert isinstance(info.value.__cause__, BudgetError)
    assert g.is_stopped(BudgetScope.PROJECT, "p1")


def test_unauthorized_record_with_events_still_raises(
    guard: BudgetGuard, ledger: CostLedger, sink: MemoryEventSink
) -> None:
    with pytest.raises(BudgetError, match="never authorized"):
        guard.record(est(0.6), res(0.6))
    assert ledger.project_cost("p1") == pytest.approx(0.6)
    assert warnings(sink) == [("project", "p1", 50)]


def test_unauthorized_record_with_failing_sink_reports_bypass(ledger: CostLedger) -> None:
    g = BudgetGuard(LIMITS, ledger, events=FailingSink())
    with pytest.raises(BudgetError, match="never authorized"):
        g.record(est(0.6), res(0.6))
    assert ledger.project_cost("p1") == pytest.approx(0.6)


def test_gateway_writes_warnings_to_jsonl(tmp_path: Path, ledger: CostLedger) -> None:
    config = AutomaConfig.model_validate(
        {
            "tiers": {"small": "m-small"},
            "prices": {"m-small": {"input_per_mtok": 1000.0, "output_per_mtok": 1000.0}},
            "limits": {"project_usd": 1.0, "daily_usd": 2.0, "per_call_usd": 0.5},
        }
    )
    usage = Usage(input_tokens=100, output_tokens=100)  # $0.2 actual per call
    fake = FakeProvider(
        [ChatResponse(model="m-small", message=Message.assistant("x"), usage=usage)] * 4
    )
    log = JsonlEventLog(tmp_path / "events.jsonl")
    gw = LLMGateway(config, fake, guard=BudgetGuard(config.limits, ledger, events=log))
    ctx = CallContext(project_id="p1", task_id="t1")
    for _ in range(4):
        gw.complete([Message.user("hi")], tier=Tier.SMALL, max_tokens=200, context=ctx)
    got = [(e.type, e.data["scope"], e.data.get("threshold_pct")) for e in log.read()]
    assert got == [("budget.warning", "project", 50), ("budget.warning", "project", 80)]
