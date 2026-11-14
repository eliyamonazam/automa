"""Tests for the actual-spending report (automa.finance.report, D65).

Pure: ledger entries are built by hand, plus one round trip through a real
SQLite ledger in tmp_path. No provider, no network, no model calls.
"""

from __future__ import annotations

import ast
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import automa.finance.report as rep_mod
from automa.budget import CostLedger, LedgerEntry
from automa.config import Tier
from automa.finance import SpendGroup, SpendingReport, spending_report
from test_ledger import FixedClock, make_pair

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
_ids = iter(range(1, 10_000))


def entry(
    cost: int = 100,
    *,
    project: str | None = "p1",
    task: str | None = "t1",
    model: str = "m-model",
    ts: datetime = T0,
    inp: int = 10,
    out: int = 5,
    reserved: int = 1_000,
) -> LedgerEntry:
    return LedgerEntry(
        id=next(_ids),
        ts=ts,
        project_id=project,
        task_id=task,
        tier=Tier.MEDIUM,
        model=model,
        response_model=model,
        input_tokens=inp,
        output_tokens=out,
        cost_nusd=cost,
        estimate_nusd=reserved,
    )


def keys(groups: tuple[SpendGroup, ...]) -> list[str | None]:
    return [g.key for g in groups]


def test_empty() -> None:
    r = spending_report([])
    assert r.total == SpendGroup()
    assert r.first_ts is None and r.last_ts is None
    assert r.by_project == r.by_day == r.by_model == r.by_role == ()
    assert r.no_project.calls == r.no_task.calls == r.unknown_role.calls == 0


def test_totals_sum_everything() -> None:
    es = [entry(100, inp=1, out=2, reserved=10), entry(250, inp=3, out=4, reserved=20)]
    r = spending_report(es)
    assert r.total.calls == 2
    assert r.total.cost_nusd == 350
    assert (r.total.input_tokens, r.total.output_tokens) == (4, 6)
    assert r.total.reserved_nusd == 30
    assert r.total.cost_usd == pytest.approx(350e-9)


def test_by_project_sorted_by_cost_then_key() -> None:
    es = [
        entry(10, project="b"),
        entry(30, project="a"),
        entry(10, project="c"),
        entry(5, project="a"),
    ]
    r = spending_report(es)
    assert keys(r.by_project) == ["a", "b", "c"]
    assert [g.cost_nusd for g in r.by_project] == [35, 10, 10]
    assert r.by_project[0].calls == 2


def test_calls_without_project() -> None:
    r = spending_report([entry(7, project=None, task=None), entry(3)])
    assert r.no_project.cost_nusd == 7 and r.no_project.calls == 1
    assert keys(r.by_project) == ["p1"]


def test_by_day_chronological_and_utc() -> None:
    tehran = timezone(timedelta(hours=3, minutes=30))
    es = [
        entry(1, ts=datetime(2026, 11, 12, 1, 0, tzinfo=tehran)),  # 2026-11-11 21:30 UTC
        entry(50, ts=datetime(2026, 11, 10, 9, 0, tzinfo=UTC)),
        entry(2, ts=datetime(2026, 11, 11, 23, 59, tzinfo=UTC)),
    ]
    r = spending_report(es)
    assert keys(r.by_day) == ["2026-11-10", "2026-11-11"]
    assert [g.cost_nusd for g in r.by_day] == [50, 3]


def test_first_and_last_ts() -> None:
    es = [entry(ts=T0 + timedelta(hours=2)), entry(ts=T0), entry(ts=T0 + timedelta(hours=1))]
    r = spending_report(es)
    assert r.first_ts == T0 and r.last_ts == T0 + timedelta(hours=2)


def test_by_model() -> None:
    es = [entry(5, model="s-model"), entry(40, model="l-model"), entry(6, model="s-model")]
    r = spending_report(es)
    assert keys(r.by_model) == ["l-model", "s-model"]
    assert r.by_model[1].calls == 2


def test_by_role_from_the_map() -> None:
    roles = {("p1", "t1"): "writer", ("p1", "t2"): "developer", ("p2", "t1"): "developer"}
    es = [
        entry(10, task="t1"),
        entry(20, task="t2"),
        entry(30, project="p2", task="t1"),
        entry(4, task="t1"),
    ]
    r = spending_report(es, roles=roles)
    assert keys(r.by_role) == ["developer", "writer"]
    assert [g.cost_nusd for g in r.by_role] == [50, 14]
    assert r.unknown_role.calls == r.no_task.calls == 0


def test_role_is_per_project() -> None:
    """The same task id in another project does not borrow its role."""
    r = spending_report([entry(9, project="p2", task="t1")], roles={("p1", "t1"): "writer"})
    assert r.by_role == ()
    assert r.unknown_role.cost_nusd == 9


def test_calls_without_task_go_to_no_task() -> None:
    r = spending_report([entry(8, task=None), entry(2)], roles={("p1", "t1"): "writer"})
    assert r.no_task.cost_nusd == 8 and r.no_task.calls == 1
    assert keys(r.by_role) == ["writer"]


def test_unknown_task_and_task_without_project() -> None:
    r = spending_report(
        [entry(3, task="gone"), entry(4, project=None, task="t1")],
        roles={("p1", "t1"): "writer"},
    )
    assert r.unknown_role.cost_nusd == 7 and r.unknown_role.calls == 2
    assert r.by_role == ()


def test_no_role_map_means_every_task_is_unknown() -> None:
    r = spending_report([entry(3), entry(1, task=None)])
    assert r.unknown_role.cost_nusd == 3
    assert r.no_task.cost_nusd == 1


def test_project_filter() -> None:
    es = [entry(10, project="p1"), entry(20, project="p2"), entry(5, project=None, task=None)]
    r = spending_report(es, project_id="p2")
    assert r.project_id == "p2"
    assert r.total.cost_nusd == 20
    assert keys(r.by_project) == ["p2"]
    assert r.no_project.calls == 0


def test_day_filters_are_inclusive() -> None:
    es = [entry(1, ts=T0 + timedelta(days=d)) for d in range(5)]  # 10th..14th
    r = spending_report(es, since=date(2026, 11, 11), until=date(2026, 11, 13))
    assert keys(r.by_day) == ["2026-11-11", "2026-11-12", "2026-11-13"]
    assert r.total.calls == 3
    assert (r.since, r.until) == (date(2026, 11, 11), date(2026, 11, 13))
    assert spending_report(es, since=date(2026, 11, 14)).total.calls == 1
    assert spending_report(es, until=date(2026, 11, 10)).total.calls == 1


def test_since_after_until_rejected() -> None:
    with pytest.raises(ValueError, match="since"):
        spending_report([], since=date(2026, 11, 2), until=date(2026, 11, 1))


@pytest.mark.parametrize(
    "bad",
    [{"t1": "writer"}, {("p1",): "writer"}, {("p1", 1): "writer"}, {("p1", "t1"): ""}],
)
def test_bad_role_map_rejected(bad: Any) -> None:
    with pytest.raises(TypeError):
        spending_report([entry()], roles=bad)


def test_non_entries_rejected() -> None:
    with pytest.raises(TypeError):
        spending_report([{"cost_nusd": 1}])  # type: ignore[list-item]


def test_every_grouping_adds_up() -> None:
    roles = {("p1", "t1"): "writer"}
    es = [
        entry(11, model="a", ts=T0),
        entry(13, project="p2", task="x", model="b", ts=T0 + timedelta(days=1)),
        entry(17, project=None, task=None, model="a", ts=T0 + timedelta(days=2)),
        entry(19, task=None, model="c"),
    ]
    r = spending_report(es, roles=roles)
    total = r.total.cost_nusd
    assert total == 60
    assert sum(g.cost_nusd for g in r.by_project) + r.no_project.cost_nusd == total
    assert sum(g.cost_nusd for g in r.by_day) == total
    assert sum(g.cost_nusd for g in r.by_model) == total
    parts = sum(g.cost_nusd for g in r.by_role) + r.no_task.cost_nusd + r.unknown_role.cost_nusd
    assert parts == total
    calls = sum(g.calls for g in r.by_model)
    assert calls == r.total.calls == 4


def test_report_that_does_not_add_up_is_invalid() -> None:
    with pytest.raises(ValidationError, match="add up"):
        SpendingReport(total=SpendGroup(calls=1, cost_nusd=5))
    with pytest.raises(ValidationError):
        SpendGroup(cost_nusd=-1)


def test_frozen_and_json_round_trip() -> None:
    r = spending_report([entry(5)], roles={("p1", "t1"): "writer"})
    with pytest.raises(ValidationError):
        r.total = SpendGroup()  # type: ignore[misc]
    again = SpendingReport.model_validate_json(r.model_dump_json())
    assert again == r


def test_does_not_mutate_inputs() -> None:
    es = [entry(5), entry(6)]
    roles = {("p1", "t1"): "writer"}
    before = (list(es), dict(roles))
    spending_report(es, roles=roles)
    assert (es, roles) == before


def test_round_trip_through_real_ledger(tmp_path: Path) -> None:
    clock = FixedClock(datetime(2026, 11, 9, 12, 0, tzinfo=UTC))
    path = tmp_path / "ledger.sqlite3"
    with CostLedger(path, clock=clock) as led:
        led.record(*make_pair(0.01, project="p1"))
        led.record(*make_pair(0.02, project="p2"))
        expected = led.spent_nusd()
    with CostLedger(path, clock=clock, read_only=True) as ro:
        r = spending_report(ro.entries())
        assert r.total.cost_nusd == expected
        assert r.total.cost_nusd == ro.spent_nusd()
        assert keys(r.by_day) == ["2026-11-09"]
        assert keys(r.by_project) == ["p2", "p1"]
        assert ro.spent_nusd(project_id="p1") == r.by_project[1].cost_nusd


def test_module_is_pure() -> None:
    """No model, provider, gateway, ledger access, project store or file I/O."""
    src = Path(rep_mod.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any(
        m.startswith(("automa.llm", "automa.project", "automa.budget.")) for m in imported
    )
    for name in ("open(", "write_text", "sqlite3", "CostLedger", ".authorize(", ".record("):
        assert name not in src
