"""Tests for BudgetGuard. No network: FakeProvider only, SQLite in memory/tmp_path."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from automa.budget import (
    BudgetError,
    BudgetExceeded,
    BudgetGuard,
    BudgetScope,
    CostLedger,
    cap_to_nusd,
    usd_to_nusd,
)
from automa.config import AutomaConfig, BudgetLimits, Tier
from automa.llm import (
    CallContext,
    CallEstimate,
    ChatResponse,
    FakeProvider,
    LLMGateway,
    LLMResult,
    Message,
    ProviderError,
    Usage,
)

LIMITS = BudgetLimits(project_usd=1.0, daily_usd=2.0, per_call_usd=0.5)


class FixedClock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(datetime(2026, 11, 9, 12, 0, tzinfo=UTC))


@pytest.fixture
def ledger(clock: FixedClock) -> Iterator[CostLedger]:
    with CostLedger(":memory:", clock=clock) as led:
        yield led


@pytest.fixture
def guard(ledger: CostLedger) -> BudgetGuard:
    return BudgetGuard(LIMITS, ledger)


def est(max_cost: float, project: str | None = "p1", task: str = "t1") -> CallEstimate:
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


def spend(guard: BudgetGuard, cost: float, project: str = "p1") -> None:
    e = est(cost, project)
    guard.authorize(e)
    guard.record(e, res(cost))


# --- basic allow / record ---------------------------------------------------


def test_allows_and_records(guard: BudgetGuard, ledger: CostLedger) -> None:
    e = est(0.3)
    guard.authorize(e)
    assert guard.pending_nusd() == usd_to_nusd(0.3)
    entry = guard.record(e, res(0.1))
    assert entry.cost_nusd == usd_to_nusd(0.1)
    assert guard.pending_nusd() == 0
    assert ledger.project_cost("p1") == pytest.approx(0.1)
    assert guard.stops() == []


def test_cap_to_nusd_rounds_down() -> None:
    assert cap_to_nusd(1.0) == 1_000_000_000
    assert cap_to_nusd(1e-9 * 1.5) == 1
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(BudgetError):
            cap_to_nusd(bad)


# --- per-call cap -------------------------------------------------------------


def test_per_call_cap_refuses_without_latch(guard: BudgetGuard, ledger: CostLedger) -> None:
    with pytest.raises(BudgetExceeded) as info:
        guard.authorize(est(0.5000001))
    err = info.value
    assert err.scope is BudgetScope.PER_CALL and not err.latched
    assert err.limit_usd == pytest.approx(0.5)
    assert guard.pending_nusd() == 0 and ledger.entries() == []
    guard.authorize(est(0.5))  # exactly at the cap is fine, and nothing was latched
    assert guard.stops() == []


# --- project cap --------------------------------------------------------------


def test_project_cap_exact_boundary_allowed(guard: BudgetGuard) -> None:
    spend(guard, 0.5)
    spend(guard, 0.5)  # spent == cap exactly
    assert guard.is_stopped(BudgetScope.PROJECT, "p1")  # nothing more can fit


def test_project_cap_refuses_and_latches(guard: BudgetGuard) -> None:
    spend(guard, 0.4)
    spend(guard, 0.4)
    with pytest.raises(BudgetExceeded) as info:
        guard.authorize(est(0.3))
    err = info.value
    assert err.scope is BudgetScope.PROJECT and err.key == "p1" and not err.latched
    assert err.spent_usd == pytest.approx(0.8) and err.requested_usd == pytest.approx(0.3)
    assert "User approval required" in str(err)
    # Hard stop: even a call that would fit is now refused.
    with pytest.raises(BudgetExceeded) as again:
        guard.authorize(est(0.01))
    assert again.value.latched and again.value.scope is BudgetScope.PROJECT
    assert guard.is_stopped(BudgetScope.PROJECT, "p1")


def test_project_stop_does_not_block_other_projects(guard: BudgetGuard) -> None:
    spend(guard, 0.5, "p1")
    spend(guard, 0.5, "p1")
    with pytest.raises(BudgetExceeded):
        guard.authorize(est(0.1, "p1"))
    guard.authorize(est(0.1, "p2"))


def test_missing_project_is_refused(guard: BudgetGuard) -> None:
    with pytest.raises(BudgetError, match="project_id"):
        guard.authorize(est(0.01, project=None))
    with pytest.raises(BudgetError, match="project_id"):
        guard.authorize(est(0.01, project=""))
    assert guard.pending_nusd() == 0


# --- daily cap ----------------------------------------------------------------


def test_daily_cap_spans_projects_and_latches(guard: BudgetGuard) -> None:
    spend(guard, 0.5, "a")
    spend(guard, 0.5, "b")
    spend(guard, 0.5, "c")
    spend(guard, 0.4, "d")
    with pytest.raises(BudgetExceeded) as info:
        guard.authorize(est(0.2, "e"))
    assert info.value.scope is BudgetScope.DAILY and info.value.key == "2026-11-09"
    with pytest.raises(BudgetExceeded) as again:
        guard.authorize(est(0.01, "f"))
    assert again.value.latched and again.value.scope is BudgetScope.DAILY


def test_daily_cap_resets_on_new_utc_day(guard: BudgetGuard, clock: FixedClock) -> None:
    for p in "abcd":
        spend(guard, 0.5, p)
    with pytest.raises(BudgetExceeded):
        guard.authorize(est(0.01, "z"))
    clock.now += timedelta(days=1)
    guard.authorize(est(0.01, "z"))  # new UTC day, new daily budget
    assert guard.is_stopped(BudgetScope.DAILY, "2026-11-09")


# --- reservations ---------------------------------------------------------------


def test_in_flight_reservations_count(guard: BudgetGuard) -> None:
    guard.authorize(est(0.5))
    guard.authorize(est(0.5))
    with pytest.raises(BudgetExceeded) as info:
        guard.authorize(est(0.01))  # nothing spent yet, but 1.0 is reserved
    assert info.value.scope is BudgetScope.PROJECT
    assert info.value.spent_usd == pytest.approx(1.0)


def test_release_frees_reservation(guard: BudgetGuard) -> None:
    e = est(0.5)
    guard.authorize(e)
    guard.release(e)
    assert guard.pending_nusd() == 0
    with pytest.raises(BudgetError, match="never authorized"):
        guard.release(e)


def test_unauthorized_record_is_still_persisted_then_raises(
    guard: BudgetGuard, ledger: CostLedger
) -> None:
    with pytest.raises(BudgetError, match="never authorized"):
        guard.record(est(0.2), res(0.2))
    assert ledger.project_cost("p1") == pytest.approx(0.2)  # money spent is never lost


def test_overrun_is_recorded_and_latches(guard: BudgetGuard, ledger: CostLedger) -> None:
    e = est(0.1)
    guard.authorize(e)
    guard.record(e, res(1.5))  # actual cost far above estimate (e.g. gateway-reported)
    assert ledger.project_cost("p1") == pytest.approx(1.5)
    assert guard.is_stopped(BudgetScope.PROJECT, "p1")
    with pytest.raises(BudgetExceeded):
        guard.authorize(est(0.01))


def test_prior_spending_in_ledger_counts(tmp_path: Path, clock: FixedClock) -> None:
    path = tmp_path / "ledger.sqlite3"
    with CostLedger(path, clock=clock) as led:
        g = BudgetGuard(LIMITS, led)
        spend(g, 0.5)
        spend(g, 0.4)
    with CostLedger(path, clock=clock) as led:
        g = BudgetGuard(LIMITS, led)  # fresh process: no in-memory state
        with pytest.raises(BudgetExceeded) as info:
            g.authorize(est(0.2))
        assert info.value.spent_usd == pytest.approx(0.9)


def test_concurrent_authorizations_never_exceed_cap(guard: BudgetGuard) -> None:
    accepted: list[CallEstimate] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        e = est(0.07, task=f"t{i}")
        try:
            guard.authorize(e)
        except BudgetExceeded:
            return
        with lock:
            accepted.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(accepted) == 14  # floor(1.0 / 0.07)
    assert guard.pending_nusd() <= cap_to_nusd(LIMITS.project_usd)


# --- integration with the gateway ---------------------------------------------


@pytest.fixture
def config() -> AutomaConfig:
    return AutomaConfig.model_validate(
        {
            "tiers": {"small": "m-small"},
            "prices": {"m-small": {"input_per_mtok": 1000.0, "output_per_mtok": 1000.0}},
            "limits": {"project_usd": 1.0, "daily_usd": 2.0, "per_call_usd": 0.5},
        }
    )


CTX = CallContext(project_id="p1", task_id="t1")


def test_gateway_blocks_before_provider(config: AutomaConfig, ledger: CostLedger) -> None:
    fake = FakeProvider(["never"])
    gw = LLMGateway(config, fake, guard=BudgetGuard(config.limits, ledger))
    with pytest.raises(BudgetExceeded) as info:
        # 1000 output tokens at $1000/Mtok = $1 worst case > $0.5 per call.
        gw.complete([Message.user("hi")], tier=Tier.SMALL, max_tokens=1000, context=CTX)
    assert info.value.scope is BudgetScope.PER_CALL
    assert fake.requests == [] and ledger.entries() == []


def test_gateway_spends_until_project_stop(config: AutomaConfig, ledger: CostLedger) -> None:
    usage = Usage(input_tokens=100, output_tokens=100)  # $0.2 actual
    fake = FakeProvider(
        [ChatResponse(model="m-small", message=Message.assistant("x"), usage=usage)] * 10
    )
    gw = LLMGateway(config, fake, guard=BudgetGuard(config.limits, ledger))
    calls = 0
    with pytest.raises(BudgetExceeded):
        for _ in range(10):
            gw.complete([Message.user("hi")], tier=Tier.SMALL, max_tokens=200, context=CTX)
            calls += 1
    assert ledger.project_cost("p1") <= config.limits.project_usd
    assert len(fake.requests) == calls == len(ledger.entries())


def test_gateway_provider_failure_releases(config: AutomaConfig, ledger: CostLedger) -> None:
    guard = BudgetGuard(config.limits, ledger)
    gw = LLMGateway(config, FakeProvider([ProviderError("down"), "ok"]), guard=guard)
    with pytest.raises(ProviderError):
        gw.complete([Message.user("hi")], tier=Tier.SMALL, max_tokens=100, context=CTX)
    assert guard.pending_nusd() == 0 and ledger.entries() == []
    gw.complete([Message.user("hi")], tier=Tier.SMALL, max_tokens=100, context=CTX)
    assert len(ledger.entries()) == 1 and guard.pending_nusd() == 0
