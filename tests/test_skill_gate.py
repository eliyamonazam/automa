"""Tests for the benchmark gate of automatically created skills (D88)."""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from itertools import count
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from automa.benchmarks.results import BenchmarkRunRecord
from automa.plan.models import CheckResult
from automa.skill_gate import (
    GateReason,
    SkillGateDecision,
    SkillGateError,
    SkillGatePolicy,
)
from automa.skill_gate import evaluate_skill_gate as _evaluate

_ids = count()
BASE_FP = "a" * 64
CAND_FP = "b" * 64


def _stamp(records: Any, fp: str) -> Any:
    """Give unfingerprinted records of one side the side's skill set (D89)."""
    if not isinstance(records, list):
        return records
    return [
        r.model_copy(update={"skills_fingerprint": fp})
        if isinstance(r, BenchmarkRunRecord) and r.skills_fingerprint is None
        else r
        for r in records
    ]


def evaluate_skill_gate(baseline: Any, candidate: Any, **kw: Any) -> SkillGateDecision:
    """The gate with each side stamped as one distinct skill set."""
    return _evaluate(_stamp(baseline, BASE_FP), _stamp(candidate, CAND_FP), **kw)


def rec(bid: str = "cli-todo", ok: int = 2, n: int = 2, **over: Any) -> BenchmarkRunRecord:
    """A consistent record with ``ok`` of ``n`` checks passing."""
    t0 = datetime(2026, 11, 10, tzinfo=UTC)
    checks = [CheckResult(criterion=i, passed=i < ok) for i in range(n)]
    passed = ok == n
    base: dict[str, Any] = {
        "benchmark_id": bid,
        "title": "t",
        "kind": "cli",
        "project_id": f"bench-{next(_ids)}-{bid}",
        "automa_version": "0.1.0",
        "started_at": t0,
        "finished_at": t0 + timedelta(seconds=5),
        "stop": "completed" if passed else "tasks_failed",
        "project_status": "done" if passed else "failed",
        "planned": True,
        "passed": passed,
        "checks": checks,
        "score": ok / n,
        "budget_usd": 0.5,
        "cost_nusd": 1000,
        "model_calls": 2,
        "attempts": 1,
        "tasks_passed": ["build"] if passed else [],
        "tasks_failed": [] if passed else ["build"],
    }
    base.update(over)
    return BenchmarkRunRecord.model_validate(base)


def runs(oks: list[int], bid: str = "cli-todo", **over: Any) -> list[BenchmarkRunRecord]:
    return [rec(bid, ok, **over) for ok in oks]


# --- acceptance -------------------------------------------------------------------


def test_strictly_better_pass_rate_is_accepted() -> None:
    d = evaluate_skill_gate(runs([2, 1, 1]), runs([2, 2, 1]))
    assert d.accepted
    assert d.reasons == []
    (c,) = d.comparisons
    assert c.improved and not c.regressed and c.enough_runs
    assert c.baseline.passes == 1 and c.candidate.passes == 2


def test_better_score_without_more_passes_is_accepted() -> None:
    d = evaluate_skill_gate(runs([0, 0, 1]), runs([1, 1, 1]))
    assert d.accepted
    assert d.comparisons[0].baseline.pass_rate == d.comparisons[0].candidate.pass_rate == 0


def test_equal_results_are_rejected_as_no_improvement() -> None:
    d = evaluate_skill_gate(runs([2, 1, 0]), runs([0, 1, 2]))
    assert not d.accepted
    assert d.reasons == [GateReason.NO_IMPROVEMENT]


def test_regression_on_one_benchmark_rejects_even_with_gain_elsewhere() -> None:
    base = runs([1, 1, 1], "a") + runs([2, 2, 2], "b")
    cand = runs([2, 2, 2], "a") + runs([2, 2, 1], "b")
    d = evaluate_skill_gate(base, cand)
    assert not d.accepted
    assert d.reasons == [GateReason.REGRESSION]
    by = {c.benchmark_id: c for c in d.comparisons}
    assert by["a"].improved and by["b"].regressed


def test_more_passes_but_lower_score_is_a_regression() -> None:
    base = [rec(ok=4, n=4), rec(ok=3, n=4), rec(ok=3, n=4)]  # rate 1/3, score 10/12
    cand = [rec(ok=4, n=4), rec(ok=4, n=4), rec(ok=0, n=4)]  # rate 2/3, score 8/12
    d = evaluate_skill_gate(base, cand)
    c = d.comparisons[0]
    assert c.improved and c.regressed
    assert GateReason.REGRESSION in d.reasons and not d.accepted


def test_too_few_runs_rejects_even_when_better() -> None:
    d = evaluate_skill_gate(runs([0, 0]), runs([2, 2, 2]))
    assert d.reasons == [GateReason.TOO_FEW_RUNS]
    assert not d.comparisons[0].enough_runs
    d = evaluate_skill_gate(runs([0, 0, 0]), runs([2, 2]))
    assert d.reasons == [GateReason.TOO_FEW_RUNS]


def test_min_runs_policy() -> None:
    d = evaluate_skill_gate(runs([0]), runs([2]), policy=SkillGatePolicy(min_runs=1))
    assert d.accepted
    d = evaluate_skill_gate(runs([0] * 3), runs([2] * 3), policy=SkillGatePolicy(min_runs=4))
    assert d.reasons == [GateReason.TOO_FEW_RUNS]


def test_several_reasons_are_all_reported_in_fixed_order() -> None:
    base = runs([2, 2], "a")
    cand = runs([1, 1], "a", cost_nusd=10_000)
    d = evaluate_skill_gate(base, cand, policy=SkillGatePolicy(max_cost_ratio=2))
    assert d.reasons == [
        GateReason.TOO_FEW_RUNS,
        GateReason.REGRESSION,
        GateReason.NO_IMPROVEMENT,
        GateReason.COST,
    ]


def test_exact_ties_are_not_float_accidents() -> None:
    # 1/3 + 1/3 + 1/3 vs 1 + 0 + 0 with 3 checks: exactly equal mean score.
    base = [rec(ok=1, n=3) for _ in range(3)]
    cand = [rec(ok=3, n=3), rec(ok=0, n=3), rec(ok=0, n=3)]
    d = evaluate_skill_gate(base, cand)
    c = d.comparisons[0]
    assert c.baseline.mean_score == c.candidate.mean_score
    assert c.improved  # one pass vs none
    assert d.accepted


# --- cost --------------------------------------------------------------------------


def test_cost_ratio_disabled_by_default() -> None:
    d = evaluate_skill_gate(runs([1, 1, 1]), runs([2, 2, 2], cost_nusd=10**9))
    assert d.accepted


def test_cost_ratio_boundary_is_inclusive() -> None:
    pol = SkillGatePolicy(max_cost_ratio=1.5)
    ok = evaluate_skill_gate(runs([1] * 3), runs([2] * 3, cost_nusd=1500), policy=pol)
    assert ok.accepted
    over = evaluate_skill_gate(runs([1] * 3), runs([2] * 3, cost_nusd=1501), policy=pol)
    assert over.reasons == [GateReason.COST]
    assert over.comparisons[0].over_cost


def test_cost_ratio_with_zero_baseline_cost() -> None:
    pol = SkillGatePolicy(max_cost_ratio=1)
    free = evaluate_skill_gate(runs([1] * 3, cost_nusd=0), runs([2] * 3, cost_nusd=0), policy=pol)
    assert free.accepted
    paid = evaluate_skill_gate(runs([1] * 3, cost_nusd=0), runs([2] * 3, cost_nusd=1), policy=pol)
    assert paid.reasons == [GateReason.COST]


# --- unfair comparisons are errors -------------------------------------------------


@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_empty_side_is_an_error(side: str) -> None:
    args = {"baseline": runs([1] * 3), "candidate": runs([1] * 3)}
    args[side] = []
    with pytest.raises(SkillGateError, match=side):
        evaluate_skill_gate(args["baseline"], args["candidate"])


@pytest.mark.parametrize("bad", ["abc", b"x", None, 3, {"k": 1}, frozenset()])
def test_non_sequence_input_is_an_error(bad: Any) -> None:
    with pytest.raises(SkillGateError):
        evaluate_skill_gate(bad, runs([1] * 3))


def test_foreign_items_are_an_error() -> None:
    with pytest.raises(SkillGateError, match="not a benchmark run"):
        evaluate_skill_gate([*runs([1] * 3), {"benchmark_id": "x"}], runs([1] * 3))  # type: ignore[list-item]


def test_too_many_records_is_an_error() -> None:
    many = runs([1] * 3)
    with pytest.raises(SkillGateError, match="more than"):
        evaluate_skill_gate(many * 400, many)


def test_duplicate_project_within_or_across_sides() -> None:
    r = rec()
    with pytest.raises(SkillGateError, match="more than once"):
        evaluate_skill_gate([r, r, rec()], runs([2] * 3))
    with pytest.raises(SkillGateError, match="more than once"):
        evaluate_skill_gate([r, rec(), rec()], [r, rec(), rec()])


def test_different_versions_are_an_error() -> None:
    with pytest.raises(SkillGateError, match="versions"):
        evaluate_skill_gate(runs([1] * 3), runs([2] * 3, automa_version="0.2.0"))
    with pytest.raises(SkillGateError, match="versions"):
        evaluate_skill_gate([*runs([1] * 2), rec(automa_version="9")], runs([2] * 3))


def test_benchmark_on_one_side_only_is_an_error() -> None:
    with pytest.raises(SkillGateError, match="both sides: b"):
        evaluate_skill_gate(runs([1] * 3, "a") + runs([1] * 3, "b"), runs([2] * 3, "a"))
    with pytest.raises(SkillGateError, match="both sides: b"):
        evaluate_skill_gate(runs([1] * 3, "a"), runs([2] * 3, "a") + runs([2] * 3, "b"))


@pytest.mark.parametrize(
    "over",
    [{"budget_usd": 0.25}, {"kind": "web"}],
)
def test_changed_benchmark_definition_is_an_error(over: dict[str, Any]) -> None:
    with pytest.raises(SkillGateError, match="changed"):
        evaluate_skill_gate(runs([1] * 3), [*runs([2] * 2), rec(**over)])


def test_changed_number_of_checks_is_an_error() -> None:
    with pytest.raises(SkillGateError, match="changed"):
        evaluate_skill_gate(runs([1] * 3), [*runs([2] * 2), rec(ok=3, n=3)])


def test_required_benchmarks_must_all_be_covered() -> None:
    base, cand = runs([1] * 3, "a"), runs([2] * 3, "a")
    assert evaluate_skill_gate(base, cand, benchmarks=["a"]).accepted
    with pytest.raises(SkillGateError, match="no runs: b"):
        evaluate_skill_gate(base, cand, benchmarks=["a", "b"])


def test_records_outside_required_set_are_an_error() -> None:
    base = runs([1] * 3, "a") + runs([2] * 3, "b")
    cand = runs([2] * 3, "a") + runs([0] * 3, "b")
    with pytest.raises(SkillGateError, match="outside the set: b"):
        evaluate_skill_gate(base, cand, benchmarks=["a"])


@pytest.mark.parametrize("bad", [[], ["a", "a"], "a"])
def test_invalid_required_list(bad: Any) -> None:
    with pytest.raises(SkillGateError):
        evaluate_skill_gate(runs([1] * 3, "a"), runs([2] * 3, "a"), benchmarks=bad)


def test_policy_type_is_checked() -> None:
    with pytest.raises(SkillGateError, match="policy"):
        evaluate_skill_gate(runs([1] * 3), runs([2] * 3), policy={"min_runs": 1})  # type: ignore[arg-type]


# --- policy and decision models ----------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"min_runs": 0},
        {"min_runs": 51},
        {"min_runs": True},
        {"min_runs": 2.0},
        {"min_runs": "3"},
        {"max_cost_ratio": 0.5},
        {"max_cost_ratio": 101},
        {"max_cost_ratio": float("inf")},
        {"max_cost_ratio": float("nan")},
        {"other": 1},
    ],
)
def test_policy_rejects_invalid_values(kw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SkillGatePolicy(**kw)


def test_policy_defaults() -> None:
    p = SkillGatePolicy()
    assert p.min_runs == 3 and p.max_cost_ratio is None


def test_decision_is_frozen_and_deterministic() -> None:
    base = runs([1] * 3, "b") + runs([1] * 3, "a")
    cand = runs([2] * 3, "a") + runs([2] * 3, "b")
    d1 = evaluate_skill_gate(base, cand)
    d2 = evaluate_skill_gate(list(reversed(base)), list(reversed(cand)))
    assert [c.benchmark_id for c in d1.comparisons] == ["a", "b"]
    assert d1.model_dump() == d2.model_dump()
    with pytest.raises(ValidationError):
        d1.accepted = False  # type: ignore[misc]
    assert isinstance(d1, SkillGateDecision)


def test_event_data_contains_numbers_and_ids_only() -> None:
    base = runs([1] * 3, detail="secret detail text")
    cand = runs([2] * 3, detail="secret detail text")
    data = evaluate_skill_gate(base, cand).event_data()
    assert data["accepted"] is True
    assert data["reasons"] == []
    assert data["min_runs"] == 3 and data["max_cost_ratio"] is None
    assert data["automa_version"] == "0.1.0"
    (b,) = data["benchmarks"]
    assert b["benchmark_id"] == "cli-todo"
    assert b["baseline"] == {
        "runs": 3,
        "passes": 0,
        "pass_rate": 0.0,
        "mean_score": 0.5,
        "cost_nusd": 3000,
    }
    assert b["candidate"]["pass_rate"] == 1.0
    assert "secret" not in repr(data)
    assert "bench-" not in repr(data)  # no project ids


def test_inputs_are_not_mutated() -> None:
    base, cand = runs([1] * 3), runs([2] * 3)
    b0, c0 = list(base), list(cand)
    evaluate_skill_gate(base, cand)
    assert base == b0 and cand == c0


# --- structure ---------------------------------------------------------------------


def test_module_has_no_model_network_or_write_paths() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "automa" / "skill_gate.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = (
        "automa.llm",
        "automa.budget",
        "automa.agents",
        "automa.tools",
        "automa.skills",
        "httpx",
        "socket",
        "subprocess",
        "os",
        "shutil",
        "pathlib",
    )
    for mod in imported:
        assert not any(mod == f or mod.startswith(f + ".") for f in forbidden), mod
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "open" not in names
