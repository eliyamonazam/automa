"""Tests for the pre-run cost estimate (automa.finance, D62).

Pure: plans, configs and statistics are built by hand. No provider, no network,
no model calls, no files.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import automa.finance.estimate as est_mod
from automa.agents.verifier import VerifierSettings
from automa.agents.worker import WorkerSettings
from automa.budget import NANO_PER_USD, cap_to_nusd
from automa.config import AutomaConfig, ConfigError, Tier
from automa.finance import (
    BudgetFit,
    EstimateBasis,
    EstimateSettings,
    PlanEstimate,
    estimate_plan,
)
from automa.orchestrator import OrchestratorSettings
from automa.plan.models import AcceptanceCriterion, Plan, Task
from automa.quality import QualityStats
from automa.roles.registry import RoleRegistry
from automa.router import Router

PRICES = {
    "s-model": {"input_per_mtok": 0.1, "output_per_mtok": 0.4},
    "m-model": {"input_per_mtok": 1.0, "output_per_mtok": 4.0},
    "l-model": {"input_per_mtok": 5.0, "output_per_mtok": 20.0},
}
TIERS = {"small": "s-model", "medium": "m-model", "large": "l-model"}
ROLES = RoleRegistry.builtin()  # developer: medium, 40 steps; writer: medium, 30
CAP = cap_to_nusd(0.25)  # default per_call_usd


def cfg(tiers: dict[str, str] | None = None, **extra: Any) -> AutomaConfig:
    tiers = TIERS if tiers is None else tiers
    data: dict[str, Any] = {"tiers": tiers, "prices": {m: PRICES[m] for m in tiers.values()}}
    data.update(extra)
    return AutomaConfig.model_validate(data)


def task(tid: str, role: str = "writer", tier: Tier | None = None, deps: tuple[str, ...] = ()):
    return Task(
        id=tid,
        title=f"task {tid}",
        role=role,
        tier=tier,
        depends_on=list(deps),
        acceptance_criteria=[AcceptanceCriterion(description="done")],
    )


def plan(*tasks: Task) -> Plan:
    return Plan(goal="a goal", tasks=list(tasks))


def st(model: str, attempts: int, passed: int, cost_nusd: int = 0, role: str = "writer"):
    return QualityStats(
        role=role,
        model=model,
        tiers=[Tier.MEDIUM],
        attempts=attempts,
        passed=passed,
        cost_nusd=cost_nusd,
    )


def one(p: PlanEstimate):
    assert len(p.tasks) == 1
    return p.tasks[0]


# --- settings --------------------------------------------------------------------------


def test_settings_defaults_follow_agent_defaults() -> None:
    s = EstimateSettings()
    o = OrchestratorSettings()
    assert s.max_attempts == o.max_attempts
    assert s.escalate_after == o.escalate_after
    assert s.worker_max_steps == WorkerSettings().max_steps
    assert s.verifier_calls == VerifierSettings().max_attempts
    assert s.min_samples == 3


def test_settings_from_settings() -> None:
    s = EstimateSettings.from_settings(
        OrchestratorSettings(max_attempts=6, escalate_after=3),
        WorkerSettings(max_steps=10),
        VerifierSettings(max_attempts=1),
        min_samples=7,
    )
    assert (s.max_attempts, s.escalate_after, s.worker_max_steps) == (6, 3, 10)
    assert (s.verifier_calls, s.min_samples) == (1, 7)


@pytest.mark.parametrize(
    "kw",
    [
        {"max_attempts": 0},
        {"escalate_after": 0},
        {"worker_max_steps": 0},
        {"worker_max_steps": 201},
        {"verifier_calls": -1},
        {"min_samples": 0},
        {"unknown": 1},
    ],
)
def test_settings_rejects_bad_values(kw: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        EstimateSettings(**kw)


# --- worst case ------------------------------------------------------------------------


def test_worst_case_from_enforced_limits() -> None:
    e = one(estimate_plan(plan(task("a")), cfg(), ROLES, []))
    # writer: 30 steps (< worker 200); 4 attempts × (30 + 2 verifier calls)
    assert e.max_calls == 4 * (30 + 2)
    assert e.worst_nusd == e.max_calls * CAP


def test_worst_case_uses_role_steps_and_worker_cap() -> None:
    dev = one(estimate_plan(plan(task("a", "developer")), cfg(), ROLES, []))
    assert dev.max_calls == 4 * (40 + 2)
    s = EstimateSettings(worker_max_steps=5, verifier_calls=0, max_attempts=2)
    low = one(estimate_plan(plan(task("a", "developer")), cfg(), ROLES, [], settings=s))
    assert low.max_calls == 2 * 5


def test_worst_case_follows_per_call_cap() -> None:
    c = cfg(limits={"per_call_usd": 0.01, "daily_usd": 1, "project_usd": 1})
    e = one(estimate_plan(plan(task("a")), c, ROLES, []))
    assert e.worst_nusd == e.max_calls * cap_to_nusd(0.01)
    assert estimate_plan(plan(task("a")), c, ROLES, []).per_call_cap_nusd == cap_to_nusd(0.01)


def test_worst_case_ignores_history() -> None:
    a = one(estimate_plan(plan(task("a")), cfg(), ROLES, []))
    b = one(estimate_plan(plan(task("a")), cfg(), ROLES, [st("m-model", 10, 10, 10)]))
    assert a.worst_nusd == b.worst_nusd


def test_plan_totals_sum_tasks() -> None:
    p = plan(task("a"), task("b", "developer", deps=("a",)))
    e = estimate_plan(p, cfg(), ROLES, [])
    assert e.worst_nusd == sum(t.worst_nusd for t in e.tasks)
    assert [t.task_id for t in e.tasks] == ["a", "b"]


# --- expected --------------------------------------------------------------------------


def test_no_history_is_unknown() -> None:
    e = estimate_plan(plan(task("a")), cfg(), ROLES, [])
    t = one(e)
    assert t.basis is EstimateBasis.NO_DATA
    assert t.expected_nusd is None and t.expected_attempts is None
    assert t.samples == 0 and t.pass_rate is None
    assert e.unknown == ["a"] and not e.complete
    assert e.expected_nusd == 0


def test_too_few_samples_is_unknown() -> None:
    t = one(estimate_plan(plan(task("a")), cfg(), ROLES, [st("m-model", 2, 2, 200)]))
    assert t.basis is EstimateBasis.NO_DATA
    assert t.samples == 2 and t.pass_rate == 1.0
    s = EstimateSettings(min_samples=2)
    t2 = one(estimate_plan(plan(task("a")), cfg(), ROLES, [st("m-model", 2, 2, 200)], settings=s))
    assert t2.basis is EstimateBasis.HISTORY and t2.expected_nusd == 100


def test_always_passing_costs_one_attempt() -> None:
    e = estimate_plan(plan(task("a")), cfg(), ROLES, [st("m-model", 4, 4, 4_000)])
    t = one(e)
    assert t.basis is EstimateBasis.HISTORY
    assert t.expected_attempts == 1.0
    assert t.expected_nusd == 1_000
    assert e.complete and e.expected_nusd == 1_000


def test_partial_pass_rate_geometric_within_one_tier() -> None:
    # Only the medium tier is configured: no escalation; p = 0.5, 4 attempts max.
    c = cfg({"medium": "m-model"})
    t = one(estimate_plan(plan(task("a")), c, ROLES, [st("m-model", 4, 2, 4_000)]))
    exp_attempts = 1 + 0.5 + 0.25 + 0.125
    assert t.expected_attempts == pytest.approx(exp_attempts)
    assert t.expected_nusd == math.ceil(1_000 * exp_attempts)


def test_never_passing_costs_all_attempts() -> None:
    c = cfg({"medium": "m-model"})
    t = one(estimate_plan(plan(task("a")), c, ROLES, [st("m-model", 5, 0, 5_000)]))
    assert t.expected_attempts == 4.0
    assert t.expected_nusd == 4_000


def test_escalation_needs_history_of_higher_tier() -> None:
    # medium p=0.5 → after 2 rejections the large tier is reached with prob 0.25.
    stats = [st("m-model", 4, 2, 4_000)]
    t = one(estimate_plan(plan(task("a")), cfg(), ROLES, stats))
    assert t.basis is EstimateBasis.NO_DATA and t.expected_nusd is None
    stats.append(st("l-model", 3, 3, 30_000))
    t = one(estimate_plan(plan(task("a")), cfg(), ROLES, stats))
    assert t.basis is EstimateBasis.HISTORY
    # attempts: m (1), m (0.5), l (0.25), l passes always -> never 4th
    assert t.expected_attempts == pytest.approx(1.75)
    assert t.expected_nusd == math.ceil(1_000 + 500 + 0.25 * 10_000)


def test_unreachable_tier_needs_no_history() -> None:
    # medium always passes: the large tier is never reached, so no data needed for it.
    t = one(estimate_plan(plan(task("a")), cfg(), ROLES, [st("m-model", 3, 3, 300)]))
    assert t.basis is EstimateBasis.HISTORY and t.expected_nusd == 100


def test_escalate_after_setting_respected() -> None:
    stats = [st("m-model", 4, 2, 4_000)]
    s = EstimateSettings(escalate_after=4)  # never escalates within 4 attempts
    t = one(estimate_plan(plan(task("a")), cfg(), ROLES, stats, settings=s))
    assert t.basis is EstimateBasis.HISTORY
    assert t.expected_attempts == pytest.approx(1.875)


def test_top_tier_does_not_escalate() -> None:
    c = cfg()
    t = one(estimate_plan(plan(task("a", tier=Tier.LARGE)), c, ROLES, [st("l-model", 4, 0, 400)]))
    assert t.expected_attempts == 4.0 and t.expected_nusd == 400


def test_history_of_other_role_or_model_ignored() -> None:
    stats = [st("m-model", 9, 9, 900, role="developer"), st("s-model", 9, 9, 9)]
    t = one(estimate_plan(plan(task("a")), cfg(), ROLES, stats))
    assert t.basis is EstimateBasis.NO_DATA and t.samples == 0


def test_expected_rounds_up() -> None:
    c = cfg({"medium": "m-model"})
    t = one(estimate_plan(plan(task("a")), c, ROLES, [st("m-model", 3, 3, 10)]))
    assert t.expected_nusd == 4  # 10/3 rounded up, never under-counted


def test_zero_cost_history() -> None:
    t = one(estimate_plan(plan(task("a")), cfg(), ROLES, [st("m-model", 3, 3, 0)]))
    assert t.expected_nusd == 0 and t.basis is EstimateBasis.HISTORY


def test_task_tier_overrides_role_tier() -> None:
    t = one(estimate_plan(plan(task("a", tier=Tier.SMALL)), cfg(), ROLES, []))
    assert (t.requested, t.tier, t.model) == (Tier.SMALL, Tier.SMALL, "s-model")
    d = one(estimate_plan(plan(task("a")), cfg(), ROLES, []))
    assert (d.requested, d.tier, d.model) == (Tier.MEDIUM, Tier.MEDIUM, "m-model")


def test_plan_expected_sums_known_parts_only() -> None:
    p = plan(task("a"), task("b", "developer"))
    e = estimate_plan(p, cfg(), ROLES, [st("m-model", 3, 3, 300)])
    assert e.unknown == ["b"]
    assert e.expected_nusd == 100


# --- router ----------------------------------------------------------------------------


def test_router_changes_starting_tier() -> None:
    c = cfg(router={"min_attempts": 3, "min_pass_rate": 0.8})
    stats = [st("s-model", 3, 3, 30)]
    r = Router(c, stats)
    t = one(estimate_plan(plan(task("a")), c, ROLES, stats, router=r))
    assert t.requested is Tier.MEDIUM and t.tier is Tier.SMALL and t.routed
    assert t.model == "s-model" and t.expected_nusd == 10
    t0 = one(estimate_plan(plan(task("a")), c, ROLES, stats))
    assert t0.tier is Tier.MEDIUM and not t0.routed


def test_router_from_other_config_rejected() -> None:
    r = Router(cfg({"medium": "m-model"}), [])
    with pytest.raises(ValueError, match="different configuration"):
        estimate_plan(plan(task("a")), cfg(), ROLES, [], router=r)


def test_router_from_equal_config_accepted() -> None:
    r = Router(cfg(), [])
    t = one(estimate_plan(plan(task("a")), cfg(), ROLES, [], router=r))
    assert t.tier is Tier.MEDIUM


# --- skip, roles, errors ---------------------------------------------------------------


def test_skip_leaves_out_passed_tasks() -> None:
    p = plan(task("a"), task("b", deps=("a",)))
    e = estimate_plan(p, cfg(), ROLES, [], skip={"a"})
    assert [t.task_id for t in e.tasks] == ["b"] and e.skipped == ["a"]


def test_skip_unknown_id_rejected() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        estimate_plan(plan(task("a")), cfg(), ROLES, [], skip={"zzz"})


def test_unknown_role_costs_nothing() -> None:
    e = estimate_plan(plan(task("a", "nobody")), cfg(), ROLES, [])
    t = one(e)
    assert t.basis is EstimateBasis.UNKNOWN_ROLE
    assert t.worst_nusd == 0 and t.expected_nusd == 0 and t.max_calls == 0
    assert e.complete


def test_unconfigured_tier_is_config_error() -> None:
    with pytest.raises(ConfigError, match="large"):
        estimate_plan(plan(task("a", tier=Tier.LARGE)), cfg({"medium": "m-model"}), ROLES, [])


def test_duplicate_stats_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        estimate_plan(plan(task("a")), cfg(), ROLES, [st("m-model", 3, 3), st("m-model", 4, 4)])


# --- assess ----------------------------------------------------------------------------


def _est(expected: int, worst: int, unknown: bool) -> PlanEstimate:
    p = plan(task("a"))
    stats = [] if unknown else [st("m-model", 3, 3, 3 * expected)]
    e = estimate_plan(p, cfg(), ROLES, stats)
    return e.model_copy(
        update={"worst_nusd": worst, "expected_nusd": expected if not unknown else 0}
    )


def test_assess_worst_fits() -> None:
    e = _est(100, 1_000, unknown=True)
    assert e.assess(1_000) is BudgetFit.WORST_FITS


def test_assess_expected_fits() -> None:
    e = _est(100, 1_000, unknown=False)
    assert e.assess(999) is BudgetFit.EXPECTED_FITS
    assert e.assess(100) is BudgetFit.EXPECTED_FITS


def test_assess_expected_exceeds() -> None:
    e = _est(100, 1_000, unknown=False)
    assert e.assess(99) is BudgetFit.EXPECTED_EXCEEDS
    assert e.assess(0) is BudgetFit.EXPECTED_EXCEEDS


def test_assess_unknown() -> None:
    e = _est(0, 1_000, unknown=True)
    assert e.assess(10) is BudgetFit.UNKNOWN


def test_assess_negative_rejected() -> None:
    with pytest.raises(ValueError):
        _est(0, 1, unknown=True).assess(-1)


def test_realistic_worst_case_dwarfs_default_project_cap() -> None:
    # The worst case is a bound, not a forecast: the BudgetGuard caps real spending.
    e = estimate_plan(plan(task("a")), cfg(), ROLES, [])
    assert e.worst_nusd > 5 * NANO_PER_USD
    assert e.assess(5 * NANO_PER_USD) is BudgetFit.UNKNOWN


# --- purity ----------------------------------------------------------------------------


def test_module_is_pure() -> None:
    """No model, provider, gateway, ledger writes or file I/O in the estimate module."""
    src = Path(est_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    forbidden = {"automa.llm", "automa.llm.gateway", "automa.budget.ledger", "automa.project"}
    assert not imported & forbidden
    assert not any(m.startswith("automa.llm") for m in imported)
    for name in ("open(", "write_text", "sqlite3", ".authorize(", ".record("):
        assert name not in src
