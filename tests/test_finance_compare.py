"""Tests for estimate vs actual spending (automa.finance.compare, D67).

Pure: events and ledger entries are built by hand, plus a round trip through a
real JSONL event log in tmp_path. No provider, no network, no model calls.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import automa.finance.compare as cmp_mod
from automa.budget import LedgerEntry
from automa.config import Tier
from automa.events import Event, JsonlEventLog
from automa.finance import (
    BudgetFit,
    ComparisonOutcome,
    EstimateComparison,
    EstimateRecord,
    SpendGroup,
    compare_estimate,
)

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
_ids = iter(range(1, 10_000))
Out = ComparisonOutcome


def entry(
    cost: int = 100,
    *,
    project: str | None = "p1",
    task: str | None = "t1",
    ts: datetime = T0 + timedelta(minutes=1),
    reserved: int = 1_000,
) -> LedgerEntry:
    return LedgerEntry(
        id=next(_ids),
        ts=ts,
        project_id=project,
        task_id=task,
        tier=Tier.MEDIUM,
        model="m-model",
        response_model="m-model",
        input_tokens=10,
        output_tokens=5,
        cost_nusd=cost,
        estimate_nusd=reserved,
    )


def data(**over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "plan_version": 1,
        "tasks": ["t1", "t2"],
        "excluded": [],
        "router": False,
        "expected_nusd": 500,
        "expected_complete": True,
        "unknown": [],
        "worst_nusd": 2_000,
        "project_remaining_nusd": 10_000,
        "assessment": "worst_fits",
    }
    d.update(over)
    return d


def est(ts: datetime = T0, project: str | None = "p1", **over: Any) -> Event:
    return Event(ts=ts, type="finance.estimate", project_id=project, data=data(**over))


def test_no_events_is_no_estimate() -> None:
    c = compare_estimate("p1", [], [entry()])
    assert c.outcome is Out.NO_ESTIMATE
    assert c.estimate is None and c.estimates == 0
    assert c.tasks == SpendGroup() and c.other == SpendGroup()
    assert c.expected_delta_nusd is None


def test_other_event_types_are_ignored() -> None:
    other = Event(ts=T0, type="task.passed", project_id="p1", data={"tasks": 1})
    assert compare_estimate("p1", [other], []).outcome is Out.NO_ESTIMATE


def test_estimate_without_spending() -> None:
    c = compare_estimate("p1", [est()], [])
    assert c.outcome is Out.NO_SPENDING
    assert c.estimate is not None and c.estimate.tasks == ("t1", "t2")
    assert c.estimates == 1
    assert c.expected_delta_nusd == -500


def test_within_expected() -> None:
    c = compare_estimate("p1", [est()], [entry(200, task="t1"), entry(250, task="t2")])
    assert c.outcome is Out.WITHIN_EXPECTED
    assert c.tasks.cost_nusd == 450 and c.tasks.calls == 2
    assert c.known.cost_nusd == 450
    assert c.expected_delta_nusd == -50


def test_exactly_expected_is_within() -> None:
    c = compare_estimate("p1", [est()], [entry(500)])
    assert c.outcome is Out.WITHIN_EXPECTED and c.expected_delta_nusd == 0


def test_above_expected_within_worst() -> None:
    c = compare_estimate("p1", [est()], [entry(900), entry(600, task="t2")])
    assert c.outcome is Out.ABOVE_EXPECTED
    assert c.expected_delta_nusd == 1_000


def test_exactly_worst_is_not_above_worst() -> None:
    c = compare_estimate("p1", [est()], [entry(2_000)])
    assert c.outcome is Out.ABOVE_EXPECTED


def test_above_worst() -> None:
    c = compare_estimate("p1", [est()], [entry(1_500), entry(501, task="t2")])
    assert c.outcome is Out.ABOVE_WORST


def test_unknown_tasks_compared_only_with_worst() -> None:
    ev = est(expected_nusd=100, expected_complete=False, unknown=["t2"])
    c = compare_estimate("p1", [ev], [entry(80, task="t1"), entry(1_000, task="t2")])
    assert c.outcome is Out.WITHIN_EXPECTED  # t1 (known) 80 <= 100
    assert c.tasks.cost_nusd == 1_080 and c.known.cost_nusd == 80
    assert c.expected_delta_nusd == -20


def test_only_unknown_tasks_spent() -> None:
    ev = est(expected_nusd=100, expected_complete=False, unknown=["t2"])
    c = compare_estimate("p1", [ev], [entry(1_000, task="t2")])
    assert c.outcome is Out.EXPECTED_UNKNOWN
    c2 = compare_estimate("p1", [ev], [entry(2_001, task="t2")])
    assert c2.outcome is Out.ABOVE_WORST


def test_all_unknown() -> None:
    ev = est(expected_nusd=0, expected_complete=False, unknown=["t1", "t2"])
    assert compare_estimate("p1", [ev], [entry(10)]).outcome is Out.EXPECTED_UNKNOWN


def test_spending_before_estimate_is_ignored() -> None:
    old = entry(5_000, ts=T0 - timedelta(seconds=1))
    c = compare_estimate("p1", [est()], [old, entry(100)])
    assert c.tasks.cost_nusd == 100 and c.outcome is Out.WITHIN_EXPECTED


def test_spending_at_estimate_timestamp_counts() -> None:
    c = compare_estimate("p1", [est()], [entry(100, ts=T0)])
    assert c.tasks.calls == 1


def test_naive_and_other_zone_ledger_timestamps() -> None:
    tz = timezone(timedelta(hours=3, minutes=30))
    before = entry(7, ts=(T0 - timedelta(minutes=1)).astimezone(tz))
    after = entry(9, ts=(T0 + timedelta(minutes=1)).replace(tzinfo=None))
    c = compare_estimate("p1", [est()], [before, after])
    assert c.tasks.cost_nusd == 9


def test_other_projects_ignored() -> None:
    c = compare_estimate(
        "p1",
        [est(), est(ts=T0 + timedelta(hours=1), project="p2")],
        [entry(100), entry(9_999, project="p2"), entry(9_999, project=None)],
    )
    assert c.estimate is not None and c.estimate.ts == T0  # p2's later event ignored
    assert c.estimates == 1
    assert c.total.cost_nusd == 100


def test_other_calls_bucket() -> None:
    c = compare_estimate(
        "p1",
        [est()],
        [entry(100, task="t1"), entry(30, task=None), entry(40, task="t9")],
    )
    assert c.tasks.cost_nusd == 100
    assert c.other.cost_nusd == 70 and c.other.calls == 2
    assert c.total.cost_nusd == 170 and c.total.calls == 3
    assert c.outcome is Out.WITHIN_EXPECTED  # other calls do not count against the estimate


def test_only_other_calls_is_no_spending() -> None:
    c = compare_estimate("p1", [est()], [entry(30, task=None)])
    assert c.outcome is Out.NO_SPENDING and c.other.cost_nusd == 30


def test_latest_estimate_is_used_regardless_of_order() -> None:
    later = est(ts=T0 + timedelta(hours=1), expected_nusd=50, worst_nusd=60, tasks=["t2"])
    entries = [entry(500, task="t1"), entry(55, task="t2", ts=T0 + timedelta(hours=2))]
    for events in ([est(), later], [later, est()]):
        c = compare_estimate("p1", events, entries)
        assert c.estimate is not None and c.estimate.expected_nusd == 50
        assert c.estimates == 2
        assert c.tasks.cost_nusd == 55 and c.other.cost_nusd == 0
        assert c.outcome is Out.ABOVE_EXPECTED


def test_tie_takes_later_in_input_order() -> None:
    a, b = est(expected_nusd=10), est(expected_nusd=20)
    c = compare_estimate("p1", [a, b], [])
    assert c.estimate is not None and c.estimate.expected_nusd == 20


def test_event_without_project_id_is_accepted() -> None:
    c = compare_estimate("p1", [est(project=None)], [entry(100)])
    assert c.outcome is Out.WITHIN_EXPECTED


@pytest.mark.parametrize(
    "over",
    [
        {"expected_nusd": -1},
        {"expected_nusd": "500"},
        {"expected_nusd": 1.5},
        {"expected_nusd": True},
        {"worst_nusd": 100},  # expected > worst
        {"tasks": ["t1", "t1"]},
        {"tasks": "t1"},
        {"tasks": [1]},
        {"unknown": ["t9"], "expected_complete": False},
        {"unknown": ["t1"], "expected_complete": True},
        {"expected_complete": False},
        {"assessment": "maybe"},
        {"assessment": 3},
        {"plan_version": -1},
    ],
)
def test_invalid_latest_estimate(over: dict[str, Any]) -> None:
    c = compare_estimate("p1", [est(), est(ts=T0 + timedelta(hours=1), **over)], [entry()])
    assert c.outcome is Out.INVALID_ESTIMATE
    assert c.estimate is None and c.invalid_reason and c.estimates == 2
    assert c.tasks == SpendGroup()  # an older estimate is never used instead


def test_missing_field_is_invalid() -> None:
    d = data()
    del d["worst_nusd"]
    ev = Event(ts=T0, type="finance.estimate", project_id="p1", data=d)
    assert compare_estimate("p1", [ev], []).outcome is Out.INVALID_ESTIMATE


def test_extra_fields_are_ignored() -> None:
    c = compare_estimate("p1", [est(future_field=1)], [])
    assert c.outcome is Out.NO_SPENDING


def test_record_from_event_rejects_other_types() -> None:
    with pytest.raises(ValueError):
        EstimateRecord.from_event(Event(ts=T0, type="task.passed", data=data()))


def test_record_fields() -> None:
    r = EstimateRecord.from_event(est(router=True, excluded=["t0"]))
    assert r.assessment is BudgetFit.WORST_FITS
    assert r.router is True and r.excluded == ("t0",)
    assert r.ts == T0


@pytest.mark.parametrize("pid", ["", None, 3])
def test_invalid_project_id(pid: Any) -> None:
    with pytest.raises(ValueError):
        compare_estimate(pid, [], [])


def test_invalid_inputs() -> None:
    with pytest.raises(TypeError):
        compare_estimate("p1", [data()], [])  # type: ignore[list-item]
    for events in ([], [est()], [est(expected_nusd=-1)]):
        with pytest.raises(TypeError):
            compare_estimate("p1", events, ["x"])  # type: ignore[list-item]


def test_inconsistent_comparison_rejected() -> None:
    with pytest.raises(ValidationError):
        EstimateComparison(project_id="p1", outcome=Out.WITHIN_EXPECTED)
    with pytest.raises(ValidationError):
        EstimateComparison(project_id="p1", outcome=Out.INVALID_ESTIMATE, estimates=1)
    with pytest.raises(ValidationError):
        EstimateComparison(project_id="p1", outcome=Out.NO_ESTIMATE, estimates=1)
    rec = EstimateRecord.from_event(est())
    with pytest.raises(ValidationError):
        EstimateComparison(
            project_id="p1",
            outcome=Out.WITHIN_EXPECTED,
            estimate=rec,
            estimates=1,
            tasks=SpendGroup(calls=1, cost_nusd=1),
            known=SpendGroup(calls=1, cost_nusd=2),
        )


def test_jsonl_round_trip(tmp_path: Path) -> None:
    log = JsonlEventLog(tmp_path / "events.jsonl")
    log.emit(est(expected_nusd=100, expected_complete=False, unknown=["t2"]))
    c = compare_estimate("p1", log.read(), [entry(90)])
    assert c.outcome is Out.WITHIN_EXPECTED
    assert c.estimate is not None and c.estimate.unknown == ("t2",)
    again = EstimateComparison.model_validate_json(c.model_dump_json())
    assert again == c


def test_inputs_not_mutated() -> None:
    ev = est()
    before = dict(ev.data)
    entries = [entry(100)]
    compare_estimate("p1", [ev], entries)
    assert ev.data == before and len(entries) == 1


def test_accepts_generators() -> None:
    c = compare_estimate("p1", (e for e in [est()]), (e for e in [entry(100)]))
    assert c.outcome is Out.WITHIN_EXPECTED


def test_module_is_pure() -> None:
    """No model, provider, gateway, ledger access, project store or file I/Out."""
    src = Path(cmp_mod.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any(
        m.startswith(("automa.llm", "automa.project", "automa.budget.")) for m in imported
    )
    for name in (
        "open(",
        "write_text",
        "sqlite3",
        "CostLedger",
        ".authorize(",
        ".record(",
        ".emit(",
    ):
        assert name not in src
