"""Real task costs from the ledger widen the cost range (D175). No model calls."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import test_finance_cost_range as fr
from automa.budget.ledger import CostLedger, LedgerEntry
from automa.config import Tier
from automa.finance import RangeBasis, TaskProgress, estimate_plan
from automa.finance.estimate import task_totals_by_role
from automa.finance.history import passed_task_roles, task_totals, workspace_task_totals
from automa.plan import Plan, TaskStatus
from automa.project import ProjectStore
from test_task_brake import est, res

M = fr.M
T0 = datetime(2026, 11, 22, tzinfo=UTC)


def entry(i: int, project: str | None, task: str | None, cost: int) -> LedgerEntry:
    return LedgerEntry(
        id=i,
        ts=T0,
        project_id=project,
        task_id=task,
        tier=Tier.SMALL,
        model="m",
        response_model="m",
        input_tokens=1,
        output_tokens=1,
        cost_nusd=cost,
        estimate_nusd=cost,
    )


# -- task_totals_by_role --------------------------------------------------------------


def test_totals_sum_every_call_of_a_task_by_role() -> None:
    entries = [
        entry(1, "p1", "a", 10),
        entry(2, "p1", "a", 5),  # a second call (e.g. the verifier) of the same task
        entry(3, "p1", "b", 7),
        entry(4, "p2", "a", 3),
        entry(5, "p1", None, 100),  # planning: no task
        entry(6, None, None, 100),
        entry(7, "p1", "c", 9),  # not a passed task
        entry(8, "p3", "z", 0),  # free: left out
    ]
    roles = {("p1", "a"): "writer", ("p1", "b"): "developer", ("p2", "a"): "writer"}
    roles[("p3", "z")] = "writer"
    assert task_totals_by_role(entries, roles) == {"writer": [15, 3], "developer": [7]}


# -- the range ------------------------------------------------------------------------


def test_real_totals_raise_the_high_end_of_fresh_tasks() -> None:
    base = fr.only(estimate_plan(fr.plan(fr.task("a")), fr.cfg(), fr.ROLES, []))
    big = [base.range_high_nusd * 3] * 3
    t = fr.only(
        estimate_plan(fr.plan(fr.task("a")), fr.cfg(), fr.ROLES, [], task_totals={"writer": big})
    )
    assert t.range_high_nusd == base.range_high_nusd * 3
    assert t.range_low_nusd == base.range_low_nusd  # the low end is unchanged
    assert t.range_basis is RangeBasis.MIXED


def test_real_totals_never_narrow_the_range() -> None:
    base = fr.only(estimate_plan(fr.plan(fr.task("a")), fr.cfg(), fr.ROLES, []))
    t = fr.only(
        estimate_plan(
            fr.plan(fr.task("a")), fr.cfg(), fr.ROLES, [], task_totals={"writer": [1] * 5}
        )
    )
    assert t == base


def test_thin_or_other_role_totals_are_ignored() -> None:
    base = fr.only(estimate_plan(fr.plan(fr.task("a")), fr.cfg(), fr.ROLES, []))
    for totals in ({"writer": [10**12, 10**12]}, {"developer": [10**12] * 5}):
        t = fr.only(
            estimate_plan(fr.plan(fr.task("a")), fr.cfg(), fr.ROLES, [], task_totals=totals)
        )
        assert t == base


def test_real_totals_stay_below_the_worst_case() -> None:
    t = fr.only(
        estimate_plan(
            fr.plan(fr.task("a")), fr.cfg(), fr.ROLES, [], task_totals={"writer": [10**15] * 3}
        )
    )
    assert t.range_high_nusd == t.worst_nusd


def test_resumed_tasks_keep_their_own_range() -> None:
    prog = {"a": TaskProgress(attempts=1, tier=Tier.MEDIUM, unjudged=True)}
    base = fr.only(estimate_plan(fr.plan(fr.task("a")), fr.cfg(), fr.ROLES, [], progress=prog))
    t = fr.only(
        estimate_plan(
            fr.plan(fr.task("a")),
            fr.cfg(),
            fr.ROLES,
            [],
            progress=prog,
            task_totals={"writer": [10**12] * 3},
        )
    )
    assert t == base


def test_history_basis_stays_history() -> None:
    costs = {("writer", "m-model"): [10 * M, 20 * M, 30 * M, 40 * M]}
    t = fr.only(
        estimate_plan(
            fr.plan(fr.task("a")),
            fr.cfg(),
            fr.ROLES,
            [],
            costs=costs,
            task_totals={"writer": [10**10] * 3},
        )
    )
    assert t.range_basis is RangeBasis.HISTORY and t.range_high_nusd == 10**10


# -- reading the workspace ------------------------------------------------------------


def _passed_project(store: ProjectStore, pid: str, monkeypatch: pytest.MonkeyPatch) -> None:
    store.create("g", project_id=pid)
    store.save_plan(pid, Plan(goal="g", tasks=[fr.task("a"), fr.task("b", role="developer")]))
    # A real passed state needs stored results and a pass verdict; only the status
    # matters here: "a" passed, "b" failed.
    states = {
        "a": SimpleNamespace(status=TaskStatus.PASSED),
        "b": SimpleNamespace(status=TaskStatus.FAILED),
    }
    monkeypatch.setattr(store, "task_states", lambda _pid: states)


def test_passed_roles_and_totals_from_store_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    _passed_project(store, "p1", monkeypatch)
    assert passed_task_roles(store) == {("p1", "a"): "writer"}
    with CostLedger(tmp_path / "ledger.sqlite3", clock=lambda: T0) as led:
        e = est(0.01, project="p1", task="a")
        led.record(e, res(0.01))
        led.record(est(0.02, project="p1", task="b"), res(0.02))
        assert task_totals(store, led) == {"writer": [10_000_000]}
    assert workspace_task_totals(store, tmp_path / "ledger.sqlite3") == {"writer": [10_000_000]}


def test_missing_or_broken_inputs_give_nothing(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    assert passed_task_roles(store) == {}
    assert workspace_task_totals(store, tmp_path / "missing.sqlite3") == {}
    bad = tmp_path / "bad.sqlite3"
    bad.write_bytes(b"not a database")
    assert workspace_task_totals(store, bad) == {}
    store.create("g", project_id="p1")  # no plan: skipped
    assert passed_task_roles(store) == {}
