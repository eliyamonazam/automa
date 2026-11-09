"""Tests for CostLedger. No network: FakeProvider only, SQLite in tmp_path/memory."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from automa.budget import CostLedger, LedgerEntry, LedgerError, nusd_to_usd, usd_to_nusd
from automa.config import AutomaConfig, Tier
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


def make_pair(
    cost: float,
    *,
    project: str | None = "p1",
    task: str | None = "t1",
    estimate: float | None = None,
    reported: float | None = None,
    tier: Tier = Tier.SMALL,
) -> tuple[CallEstimate, LLMResult]:
    est = CallEstimate(
        tier=tier,
        model="m-small",
        max_input_tokens=100,
        max_output_tokens=50,
        max_cost_usd=estimate if estimate is not None else cost * 2,
        context=CallContext(project_id=project, task_id=task),
    )
    res = LLMResult(
        tier=tier,
        response=ChatResponse(
            model="vendor/m-small-2026",
            message=Message.assistant("ok"),
            usage=Usage(input_tokens=40, output_tokens=10, reported_cost_usd=reported),
        ),
        cost_usd=cost,
    )
    return est, res


# --- money conversion ------------------------------------------------------


def test_usd_to_nusd_rounds_up() -> None:
    assert usd_to_nusd(0) == 0
    assert usd_to_nusd(1.0) == 1_000_000_000
    assert usd_to_nusd(1e-12) == 1  # sub-nano costs are never rounded to zero
    assert nusd_to_usd(usd_to_nusd(0.25)) >= 0.25


@pytest.mark.parametrize("bad", [-0.01, float("inf"), float("nan")])
def test_usd_to_nusd_rejects_bad_values(bad: float) -> None:
    with pytest.raises(LedgerError):
        usd_to_nusd(bad)


# --- recording ---------------------------------------------------------------


def test_record_persists_all_fields(ledger: CostLedger, clock: FixedClock) -> None:
    est, res = make_pair(0.001, estimate=0.004, reported=0.0009)
    entry = ledger.record(est, res)
    assert isinstance(entry, LedgerEntry)
    assert entry.id == 1
    assert entry.ts == clock.now
    assert entry.project_id == "p1"
    assert entry.task_id == "t1"
    assert entry.tier is Tier.SMALL
    assert entry.model == "m-small"
    assert entry.response_model == "vendor/m-small-2026"
    assert (entry.input_tokens, entry.output_tokens) == (40, 10)
    assert entry.cost_nusd == usd_to_nusd(0.001)
    assert entry.estimate_nusd == usd_to_nusd(0.004)
    assert entry.reported_cost_nusd == usd_to_nusd(0.0009)
    assert entry.cost_usd == pytest.approx(0.001)
    assert ledger.get(1) == entry


def test_record_without_context_or_reported_cost(ledger: CostLedger) -> None:
    entry = ledger.record(*make_pair(0.5, project=None, task=None))
    assert entry.project_id is None and entry.task_id is None
    assert entry.reported_cost_nusd is None


def test_record_rejects_tier_mismatch(ledger: CostLedger) -> None:
    est, _ = make_pair(0.1)
    _, res = make_pair(0.1, tier=Tier.LARGE)
    with pytest.raises(LedgerError):
        ledger.record(est, res)
    assert ledger.entries() == []


def test_get_missing_raises(ledger: CostLedger) -> None:
    with pytest.raises(KeyError):
        ledger.get(42)


def test_naive_clock_is_rejected() -> None:
    with CostLedger(":memory:", clock=lambda: datetime(2026, 1, 1)) as led:
        with pytest.raises(LedgerError):
            led.record(*make_pair(0.1))


def test_non_utc_clock_is_normalised_to_utc_day() -> None:
    tehran = timezone(timedelta(hours=3, minutes=30))
    # 01:00 in Tehran on Nov 10 is still Nov 9 in UTC.
    clk = FixedClock(datetime(2026, 11, 10, 1, 0, tzinfo=tehran))
    with CostLedger(":memory:", clock=clk) as led:
        entry = led.record(*make_pair(0.1))
        assert entry.ts.tzinfo is not None and entry.ts.utcoffset() == timedelta(0)
        assert led.daily_cost(date(2026, 11, 9)) == pytest.approx(0.1)
        assert led.daily_cost(date(2026, 11, 10)) == 0


# --- aggregation -------------------------------------------------------------


def test_totals_by_project_and_day(ledger: CostLedger, clock: FixedClock) -> None:
    ledger.record(*make_pair(0.10, project="a"))
    ledger.record(*make_pair(0.20, project="b"))
    clock.now += timedelta(days=1)
    ledger.record(*make_pair(0.30, project="a"))
    ledger.record(*make_pair(0.05, project=None))

    assert ledger.total_cost() == pytest.approx(0.65)
    assert ledger.project_cost("a") == pytest.approx(0.40)
    assert ledger.project_cost("b") == pytest.approx(0.20)
    assert ledger.project_cost("missing") == 0
    assert ledger.daily_cost(date(2026, 11, 9)) == pytest.approx(0.30)
    assert ledger.daily_cost() == pytest.approx(0.35)  # "today" per clock = Nov 10
    assert ledger.spent_nusd(project_id="a", day=date(2026, 11, 10)) == usd_to_nusd(0.30)
    assert [e.project_id for e in ledger.entries(project_id="a")] == ["a", "a"]
    assert len(ledger.entries(day=date(2026, 11, 10))) == 2
    assert [e.id for e in ledger.entries()] == [1, 2, 3, 4]


def test_sum_never_undercounts_many_small_calls(ledger: CostLedger) -> None:
    for _ in range(1000):
        ledger.record(*make_pair(0.0001))
    assert ledger.total_cost() >= 0.1


def test_empty_ledger_is_zero(ledger: CostLedger) -> None:
    assert ledger.total_cost() == 0
    assert ledger.daily_cost() == 0
    assert ledger.entries() == []


# --- durability and schema ---------------------------------------------------


def test_persists_across_reopen(tmp_path: Path, clock: FixedClock) -> None:
    path = tmp_path / "nested" / "ledger.sqlite3"
    with CostLedger(path, clock=clock) as led:
        led.record(*make_pair(0.25))
    assert path.exists()
    with CostLedger(path, clock=clock) as led:
        assert led.total_cost() == pytest.approx(0.25)
        led.record(*make_pair(0.25))
        assert led.total_cost() == pytest.approx(0.50)


def test_for_workspace_location(tmp_path: Path) -> None:
    with CostLedger.for_workspace(tmp_path / "ws") as led:
        led.record(*make_pair(0.01))
    assert (tmp_path / "ws" / "ledger.sqlite3").exists()


def test_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "l.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(LedgerError, match="schema version"):
        CostLedger(path)


def test_rejects_foreign_database(tmp_path: Path) -> None:
    path = tmp_path / "other.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE stuff (x)")
    conn.commit()
    conn.close()
    with pytest.raises(LedgerError, match="not an Automa ledger"):
        CostLedger(path)


def test_database_rejects_negative_cost_even_bypassing_api(ledger: CostLedger) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        ledger._conn.execute(  # verify the DB-level constraint
            "INSERT INTO calls (ts, day, tier, model, response_model, input_tokens,"
            " output_tokens, cost_nusd, estimate_nusd) VALUES"
            " ('t', 'd', 'small', 'm', 'm', 0, 0, -1, 0)"
        )


def test_write_failure_propagates(tmp_path: Path, clock: FixedClock) -> None:
    led = CostLedger(tmp_path / "l.sqlite3", clock=clock)
    led.close()
    with pytest.raises(sqlite3.ProgrammingError):
        led.record(*make_pair(0.1))


# --- integration with the gateway -------------------------------------------


class LedgerOnlyGuard:
    """Minimal guard for this test: allows everything, records into the ledger."""

    def __init__(self, ledger: CostLedger):
        self.ledger = ledger

    def authorize(self, estimate: CallEstimate) -> None:
        return None

    def record(self, estimate: CallEstimate, result: LLMResult) -> None:
        self.ledger.record(estimate, result)

    def release(self, estimate: CallEstimate) -> None:
        return None


def test_gateway_calls_are_recorded(ledger: CostLedger) -> None:
    config = AutomaConfig.model_validate(
        {
            "tiers": {"small": "m-small"},
            "prices": {"m-small": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}},
        }
    )
    provider = FakeProvider(["hello", "world"])
    gw = LLMGateway(config, provider, guard=LedgerOnlyGuard(ledger))
    ctx = CallContext(project_id="proj", task_id="T1")
    r1 = gw.complete([Message.user("hi")], tier=Tier.SMALL, max_tokens=10, context=ctx)
    r2 = gw.complete([Message.user("again")], tier=Tier.SMALL, max_tokens=10, context=ctx)
    entries = ledger.entries(project_id="proj")
    assert len(entries) == 2
    assert entries[0].cost_nusd == usd_to_nusd(r1.cost_usd)
    assert entries[1].cost_nusd == usd_to_nusd(r2.cost_usd)
    assert all(e.task_id == "T1" and e.model == "m-small" for e in entries)
    assert ledger.project_cost("proj") >= r1.cost_usd + r2.cost_usd
