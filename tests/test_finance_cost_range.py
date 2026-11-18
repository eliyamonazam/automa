"""Tests for the realistic cost range of the pre-run estimate (automa.finance, D119).

Pure: plans, configs and per-attempt costs are built by hand. No provider, no
network, no model calls, no files.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from automa.budget import cap_to_nusd
from automa.config import AutomaConfig, Tier
from automa.finance import (
    DEFAULT_TOKEN_PROFILES,
    EstimateSettings,
    RangeBasis,
    RoleTokenProfile,
    TaskProgress,
    TokenAllowance,
    attempt_costs,
    estimate_plan,
    percentile_nusd,
    profile_for,
    tokens_nusd,
)
from automa.finance.estimate import DEFAULT_PROFILE, RANGE_HIGH_ATTEMPTS, TaskEstimate
from automa.plan.models import AcceptanceCriterion, Plan, ResultStatus, Task, VerdictOutcome
from automa.quality import QualityRecord
from automa.roles.registry import RoleRegistry

PRICES = {
    "s-model": {"input_per_mtok": 0.1, "output_per_mtok": 0.4},
    "m-model": {"input_per_mtok": 1.0, "output_per_mtok": 4.0},
    "l-model": {"input_per_mtok": 5.0, "output_per_mtok": 20.0},
}
TIERS = {"small": "s-model", "medium": "m-model", "large": "l-model"}
ROLES = RoleRegistry.builtin()
M = 1_000_000

# writer profile at m-model (1000 nUSD per input token, 4000 per output token):
W_LOW = 6_000 * 1_000 + 1_500 * 4_000  # 12.0M
W_HIGH = 25_000 * 1_000 + 5_000 * 4_000  # 45.0M
W_VERIFY = 4_000 * 1_000 + 600 * 4_000  # 6.4M
# the same at l-model (5000 / 20000 nUSD per token)
L_HIGH = 25_000 * 5_000 + 5_000 * 20_000  # 225M
L_VERIFY = 4_000 * 5_000 + 600 * 20_000  # 32M


def cfg(**extra: Any) -> AutomaConfig:
    data: dict[str, Any] = {"tiers": TIERS, "prices": PRICES}
    data.update(extra)
    return AutomaConfig.model_validate(data)


def task(tid: str, role: str = "writer", tier: Tier | None = None) -> Task:
    return Task(
        id=tid,
        title=f"task {tid}",
        role=role,
        tier=tier,
        acceptance_criteria=[AcceptanceCriterion(description="done")],
    )


def plan(*tasks: Task) -> Plan:
    return Plan(goal="a goal", tasks=list(tasks))


def only(p: Any) -> TaskEstimate:
    assert len(p.tasks) == 1
    return p.tasks[0]


def rec(model: str, cost: int, attempt: int = 1, role: str = "writer") -> QualityRecord:
    return QualityRecord(
        project_id="p1",
        task_id="a",
        attempt=attempt,
        role=role,
        tier=Tier.MEDIUM,
        model=model,
        result_status=ResultStatus.COMPLETED,
        outcome=VerdictOutcome.PASS,
        model_evaluated=False,
        cost_nusd=cost,
        judged_at=datetime(2026, 11, 10, tzinfo=UTC),
    )


# --- helpers ----------------------------------------------------------------------------


def test_tokens_nusd_exact_and_rounded_up() -> None:
    price = cfg().price_for("m-model")
    assert tokens_nusd(price, TokenAllowance(input_tokens=0, output_tokens=0)) == 0
    assert tokens_nusd(price, TokenAllowance(input_tokens=1, output_tokens=1)) == 5_000
    tiny = AutomaConfig.model_validate(
        {
            "tiers": {"small": "x"},
            "prices": {"x": {"input_per_mtok": 0.0000001, "output_per_mtok": 0}},
        }
    ).price_for("x")
    assert tokens_nusd(tiny, TokenAllowance(input_tokens=1, output_tokens=0)) == 1


def test_percentile_nearest_rank() -> None:
    assert percentile_nusd([5], 25) == 5
    assert percentile_nusd([5], 90) == 5
    assert percentile_nusd([40, 10, 30, 20], 25) == 10
    assert percentile_nusd([40, 10, 30, 20], 90) == 40
    assert percentile_nusd(list(range(1, 101)), 90) == 90
    assert percentile_nusd(list(range(1, 101)), 25) == 25
    assert percentile_nusd([1, 2, 3], 100) == 3


@pytest.mark.parametrize("pct", [0, 101, -5])
def test_percentile_rejects_bad_pct(pct: int) -> None:
    with pytest.raises(ValueError):
        percentile_nusd([1, 2], pct)


def test_percentile_rejects_empty_and_negative() -> None:
    with pytest.raises(ValueError):
        percentile_nusd([], 25)
    with pytest.raises(ValueError):
        percentile_nusd([1, -1], 25)


def test_attempt_costs_groups_by_role_and_model() -> None:
    out = attempt_costs(
        [rec("m-model", 1), rec("m-model", 2, attempt=2), rec("l-model", 3, attempt=3)]
        + [rec("m-model", 4, attempt=4, role="developer")]
    )
    assert out == {
        ("writer", "m-model"): [1, 2],
        ("writer", "l-model"): [3],
        ("developer", "m-model"): [4],
    }
    assert attempt_costs([]) == {}


def test_profiles_are_ordered_and_cover_builtin_roles() -> None:
    for name in ROLES.names():
        assert name in DEFAULT_TOKEN_PROFILES
    for role in ("writer", "developer", "architect", "designer", "researcher"):
        prof = profile_for(role)
        assert prof.low.input_tokens <= prof.high.input_tokens
        assert prof.low.output_tokens <= prof.high.output_tokens
    # research reads web pages: more input than writing
    assert profile_for("researcher").high.input_tokens > profile_for("writer").high.input_tokens
    assert profile_for("my-custom-role") is DEFAULT_PROFILE


def test_profile_rejects_low_above_high() -> None:
    with pytest.raises(ValidationError):
        RoleTokenProfile(
            low=TokenAllowance(input_tokens=10, output_tokens=1),
            high=TokenAllowance(input_tokens=5, output_tokens=1),
            verify=TokenAllowance(input_tokens=0, output_tokens=0),
        )
    with pytest.raises(ValidationError):
        TokenAllowance(input_tokens=-1, output_tokens=0)


def test_task_estimate_rejects_inconsistent_range() -> None:
    base: dict[str, Any] = {
        "task_id": "a",
        "role": "writer",
        "requested": Tier.MEDIUM,
        "tier": Tier.MEDIUM,
        "model": "m-model",
        "routed": False,
        "basis": "no_data",
        "samples": 0,
        "max_calls": 1,
        "worst_nusd": 100,
    }
    TaskEstimate(**base, range_low_nusd=10, range_high_nusd=100)
    with pytest.raises(ValidationError):
        TaskEstimate(**base, range_low_nusd=50, range_high_nusd=10)
    with pytest.raises(ValidationError):
        TaskEstimate(**base, range_low_nusd=0, range_high_nusd=101)


# --- defaults ---------------------------------------------------------------------------


def test_fresh_task_without_history_uses_default_profile() -> None:
    t = only(estimate_plan(plan(task("a")), cfg(), ROLES, []))
    assert t.range_basis is RangeBasis.DEFAULTS
    assert t.range_low_nusd == W_LOW + W_VERIFY
    assert t.range_high_nusd == RANGE_HIGH_ATTEMPTS * (W_HIGH + W_VERIFY)
    assert 0 < t.range_low_nusd < t.range_high_nusd <= t.worst_nusd
    # the expected cost stays unknown, the range does not
    assert t.expected_nusd is None


def test_range_is_far_below_worst_case_with_realistic_prices() -> None:
    t = only(estimate_plan(plan(task("a")), cfg(), ROLES, []))
    assert t.range_high_nusd * 10 < t.worst_nusd


def test_range_uses_tier_model_price() -> None:
    small = only(estimate_plan(plan(task("a", tier=Tier.SMALL)), cfg(), ROLES, []))
    medium = only(estimate_plan(plan(task("a")), cfg(), ROLES, []))
    assert small.range_high_nusd < medium.range_high_nusd
    assert small.range_low_nusd < medium.range_low_nusd


def test_researcher_range_is_larger_than_writer() -> None:
    p = estimate_plan(plan(task("a"), task("b", role="researcher")), cfg(), ROLES, [])
    w, r = p.tasks
    assert r.range_high_nusd > w.range_high_nusd
    assert r.range_low_nusd > w.range_low_nusd


def test_high_end_follows_escalation_schedule() -> None:
    s = EstimateSettings(escalate_after=1)
    t = only(estimate_plan(plan(task("a")), cfg(), ROLES, [], settings=s))
    # first attempt on medium, the second (after one rejection) on large
    assert t.range_high_nusd == (W_HIGH + W_VERIFY) + (L_HIGH + L_VERIFY)
    assert t.range_low_nusd == W_LOW + W_VERIFY


def test_single_attempt_allowed() -> None:
    s = EstimateSettings(max_attempts=1)
    t = only(estimate_plan(plan(task("a")), cfg(), ROLES, [], settings=s))
    assert t.range_high_nusd == W_HIGH + W_VERIFY


def test_range_clamped_to_worst_case() -> None:
    c = cfg(limits={"project_usd": 1.0, "daily_usd": 1.0, "per_call_usd": 0.0001})
    t = only(estimate_plan(plan(task("a")), c, ROLES, []))
    assert t.worst_nusd == t.max_calls * cap_to_nusd(0.0001)
    assert t.range_high_nusd == t.worst_nusd
    assert t.range_low_nusd <= t.range_high_nusd


def test_free_model_gives_zero_range() -> None:
    c = AutomaConfig.model_validate(
        {
            "tiers": {"medium": "free"},
            "prices": {"free": {"input_per_mtok": 0, "output_per_mtok": 0}},
        }
    )
    t = only(estimate_plan(plan(task("a")), c, ROLES, []))
    assert (t.range_low_nusd, t.range_high_nusd) == (0, 0)


def test_unknown_role_has_zero_range() -> None:
    t = only(estimate_plan(plan(task("a", role="nobody")), cfg(), ROLES, []))
    assert t.range_basis is RangeBasis.UNKNOWN_ROLE
    assert (t.range_low_nusd, t.range_high_nusd) == (0, 0)


# --- history ----------------------------------------------------------------------------


def test_history_percentiles_used_with_enough_samples() -> None:
    costs = {("writer", "m-model"): [10 * M, 20 * M, 30 * M, 40 * M]}
    t = only(estimate_plan(plan(task("a")), cfg(), ROLES, [], costs=costs))
    assert t.range_basis is RangeBasis.HISTORY
    assert t.range_low_nusd == 10 * M + W_VERIFY
    assert t.range_high_nusd == 2 * (40 * M + W_VERIFY)


def test_thin_history_falls_back_to_defaults() -> None:
    costs = {("writer", "m-model"): [10 * M, 20 * M]}  # below min_samples (3)
    t = only(estimate_plan(plan(task("a")), cfg(), ROLES, [], costs=costs))
    assert t.range_basis is RangeBasis.DEFAULTS
    assert t.range_low_nusd == W_LOW + W_VERIFY


def test_min_samples_setting_is_respected() -> None:
    costs = {("writer", "m-model"): [10 * M, 20 * M]}
    s = EstimateSettings(min_samples=2)
    t = only(estimate_plan(plan(task("a")), cfg(), ROLES, [], costs=costs, settings=s))
    assert t.range_basis is RangeBasis.HISTORY


def test_history_of_other_role_or_model_is_ignored() -> None:
    costs = {
        ("developer", "m-model"): [1, 2, 3],
        ("writer", "s-model"): [1, 2, 3],
    }
    t = only(estimate_plan(plan(task("a")), cfg(), ROLES, [], costs=costs))
    assert t.range_basis is RangeBasis.DEFAULTS


def test_mixed_history_and_defaults_over_escalation() -> None:
    costs = {("writer", "m-model"): [10 * M, 20 * M, 30 * M, 40 * M]}
    s = EstimateSettings(escalate_after=1)
    t = only(estimate_plan(plan(task("a")), cfg(), ROLES, [], costs=costs, settings=s))
    assert t.range_basis is RangeBasis.MIXED
    assert t.range_high_nusd == (40 * M + W_VERIFY) + (L_HIGH + L_VERIFY)


def test_zero_cost_history_still_counts_verifier() -> None:
    costs = {("writer", "m-model"): [0, 0, 0]}
    t = only(estimate_plan(plan(task("a")), cfg(), ROLES, [], costs=costs))
    assert t.range_low_nusd == W_VERIFY
    assert t.range_high_nusd == 2 * W_VERIFY


# --- resume -----------------------------------------------------------------------------


def test_unjudged_attempt_low_is_its_verification_only() -> None:
    prog = {"a": TaskProgress(attempts=1, tier=Tier.MEDIUM, unjudged=True)}
    t = only(estimate_plan(plan(task("a")), cfg(), ROLES, [], progress=prog))
    assert t.range_low_nusd == W_VERIFY
    assert t.range_high_nusd == W_VERIFY + (W_HIGH + W_VERIFY)


def test_no_attempts_left_gives_zero_range() -> None:
    prog = {"a": TaskProgress(attempts=4, tier=Tier.MEDIUM, at_tier=1)}
    t = only(estimate_plan(plan(task("a")), cfg(), ROLES, [], progress=prog))
    assert t.worst_nusd == 0
    assert (t.range_low_nusd, t.range_high_nusd) == (0, 0)


def test_last_attempt_left_high_is_one_attempt() -> None:
    prog = {"a": TaskProgress(attempts=3, tier=Tier.MEDIUM, at_tier=1)}
    t = only(estimate_plan(plan(task("a")), cfg(), ROLES, [], progress=prog))
    # at_tier 1 + one more rejection escalates, but there is just one attempt left
    assert t.range_high_nusd == W_HIGH + W_VERIFY


# --- images -----------------------------------------------------------------------------


def img_cfg() -> AutomaConfig:
    return AutomaConfig.model_validate(
        {
            "tiers": TIERS,
            "prices": PRICES,
            "models": {
                "img": {
                    "family": "openai",
                    "capabilities": ["image_out"],
                    "input_per_mtok": 0,
                    "output_per_mtok": 0,
                    "per_image_usd": 0.04,
                }
            },
            "images": {"model": "img", "max_per_task": 3},
        }
    )


def test_image_role_adds_images_to_range() -> None:
    c = img_cfg()
    t = only(estimate_plan(plan(task("a", role="designer")), c, ROLES, []))
    image = 40 * M
    prof = profile_for("designer")
    price = c.price_for("m-model")
    low_tokens = tokens_nusd(price, prof.low) + tokens_nusd(price, prof.verify)
    high_tokens = tokens_nusd(price, prof.high) + tokens_nusd(price, prof.verify)
    assert t.range_low_nusd == low_tokens + image  # one image on the first attempt
    assert t.range_high_nusd == 2 * (high_tokens + 3 * image)
    assert t.range_high_nusd <= t.worst_nusd


# --- plan totals ------------------------------------------------------------------------


def test_plan_range_is_sum_of_tasks() -> None:
    p = estimate_plan(plan(task("a"), task("b", role="developer")), cfg(), ROLES, [])
    assert p.range_low_nusd == sum(t.range_low_nusd for t in p.tasks)
    assert p.range_high_nusd == sum(t.range_high_nusd for t in p.tasks)
    assert p.range_low_nusd < p.range_high_nusd <= p.worst_nusd
    assert p.range_basis is RangeBasis.DEFAULTS


def test_plan_range_basis_mixed_and_history() -> None:
    costs = {("writer", "m-model"): [1, 2, 3]}
    p = estimate_plan(plan(task("a"), task("b", role="developer")), cfg(), ROLES, [], costs=costs)
    assert p.range_basis is RangeBasis.MIXED
    p = estimate_plan(plan(task("a"), task("x", role="nobody")), cfg(), ROLES, [], costs=costs)
    assert p.range_basis is RangeBasis.HISTORY


def test_skipped_tasks_not_in_range() -> None:
    p = estimate_plan(plan(task("a"), task("b")), cfg(), ROLES, [], skip={"a"})
    assert p.range_low_nusd == W_LOW + W_VERIFY


def test_estimate_without_costs_argument_unchanged_otherwise() -> None:
    p1 = estimate_plan(plan(task("a")), cfg(), ROLES, [])
    p2 = estimate_plan(plan(task("a")), cfg(), ROLES, [], costs={})
    assert p1 == p2
