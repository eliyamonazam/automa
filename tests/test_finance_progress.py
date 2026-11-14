"""Tests for resumed tasks in the cost estimate (TaskProgress, D63).

Pure: plans, configs and statistics are built by hand. No provider, no model calls.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from automa.config import ConfigError, Tier
from automa.finance import EstimateBasis, EstimateSettings, TaskProgress, estimate_plan
from automa.router import Router
from test_finance_estimate import CAP, ROLES, cfg, one, plan, st, task

STEPS = 30 + 2  # writer max_steps + verifier calls per attempt


def test_progress_validation() -> None:
    assert TaskProgress().fresh
    assert not TaskProgress(attempts=1).fresh
    assert not TaskProgress(tier=Tier.LARGE).fresh
    with pytest.raises(ValidationError):
        TaskProgress(unjudged=True)  # no attempt
    with pytest.raises(ValidationError):
        TaskProgress(attempts=1, unjudged=True)  # no tier
    with pytest.raises(ValidationError):
        TaskProgress(attempts=1, tier=Tier.MEDIUM, at_tier=2)
    with pytest.raises(ValidationError):
        TaskProgress(attempts=-1)
    with pytest.raises(ValidationError):
        TaskProgress(attempts=1, bogus=1)  # type: ignore[call-arg]


def test_fresh_progress_equals_no_progress() -> None:
    c = cfg()
    stats = [st("m-model", 10, 5, 1000)]
    p = plan(task("a"))
    base = estimate_plan(p, c, ROLES, stats)
    with_progress = estimate_plan(p, c, ROLES, stats, progress={"a": TaskProgress()})
    assert base == with_progress


def test_worst_case_counts_remaining_attempts_only() -> None:
    c = cfg()
    p = plan(task("a"))
    e = one(
        estimate_plan(p, c, ROLES, [], progress={"a": TaskProgress(attempts=3, tier=Tier.LARGE)})
    )
    assert e.max_calls == 1 * STEPS
    assert e.worst_nusd == STEPS * CAP
    assert e.done_attempts == 3
    assert not e.reverify


def test_no_attempts_left_costs_nothing() -> None:
    c = cfg()
    stats = [st("l-model", 10, 0, 5000)]
    p = plan(task("a"))
    e = one(
        estimate_plan(p, c, ROLES, stats, progress={"a": TaskProgress(attempts=4, tier=Tier.LARGE)})
    )
    assert e.max_calls == 0
    assert e.worst_nusd == 0
    assert e.expected_nusd == 0
    assert e.expected_attempts == 0


def test_more_attempts_than_max_is_zero_not_negative() -> None:
    c = cfg()
    s = EstimateSettings(max_attempts=2)
    p = plan(task("a"))
    prog = {"a": TaskProgress(attempts=3, tier=Tier.LARGE)}
    e = one(estimate_plan(p, c, ROLES, [], settings=s, progress=prog))
    assert e.max_calls == 0


def test_unjudged_attempt_adds_only_verifier_calls() -> None:
    c = cfg()
    p = plan(task("a"))
    prog = {"a": TaskProgress(attempts=4, tier=Tier.LARGE, at_tier=1, unjudged=True)}
    e = one(estimate_plan(p, c, ROLES, [], progress=prog))
    assert e.max_calls == 2  # verifier only
    assert e.reverify


def test_unjudged_attempt_with_remaining_attempts() -> None:
    c = cfg()
    p = plan(task("a"))
    prog = {"a": TaskProgress(attempts=1, tier=Tier.MEDIUM, unjudged=True)}
    e = one(estimate_plan(p, c, ROLES, [], progress=prog))
    assert e.max_calls == 3 * STEPS + 2


def test_resumed_task_starts_at_saved_tier() -> None:
    c = cfg()
    stats = [st("l-model", 10, 10, 9000), st("m-model", 10, 0, 1000)]
    p = plan(task("a"))
    e = one(
        estimate_plan(p, c, ROLES, stats, progress={"a": TaskProgress(attempts=2, tier=Tier.LARGE)})
    )
    assert e.tier is Tier.LARGE
    assert e.model == "l-model"
    assert not e.routed
    assert e.basis is EstimateBasis.HISTORY
    assert e.expected_attempts == 1
    assert e.expected_nusd == 900


def test_resumed_task_continues_escalation_count() -> None:
    # One rejection already at medium: the next rejection escalates to large.
    c = cfg()
    stats = [st("m-model", 10, 5, 1000), st("l-model", 10, 10, 2000)]
    p = plan(task("a"))
    prog = {"a": TaskProgress(attempts=1, tier=Tier.MEDIUM, at_tier=1)}
    e = one(estimate_plan(p, c, ROLES, stats, progress=prog))
    # attempt 2 at medium (cost 100, pass .5), attempt 3 at large (reach .5, cost 200)
    assert e.expected_attempts == pytest.approx(1.5)
    assert e.expected_nusd == math.ceil(100 + 0.5 * 200)


def test_escalation_needs_history_of_the_higher_tier() -> None:
    c = cfg()
    stats = [st("m-model", 10, 5, 1000)]
    p = plan(task("a"))
    prog = {"a": TaskProgress(attempts=1, tier=Tier.MEDIUM, at_tier=1)}
    e = one(estimate_plan(p, c, ROLES, stats, progress=prog))
    assert e.expected_nusd is None
    assert e.basis is EstimateBasis.NO_DATA
    # without the previous rejection, medium is tried twice: still needs large
    # only if both fail (reach .25 > 0), so still unknown
    e2 = one(
        estimate_plan(
            p, c, ROLES, stats, progress={"a": TaskProgress(attempts=1, tier=Tier.MEDIUM)}
        )
    )
    assert e2.expected_nusd is None


def test_unjudged_attempt_costs_no_worker_call_but_may_pass() -> None:
    c = cfg()
    stats = [st("m-model", 10, 5, 1000), st("l-model", 10, 10, 2000)]
    p = plan(task("a"))
    prog = {"a": TaskProgress(attempts=1, tier=Tier.MEDIUM, unjudged=True)}
    e = one(estimate_plan(p, c, ROLES, stats, progress=prog))
    # unjudged (free, pass .5) -> attempt 2 medium (reach .5, 100) -> escalate after 2
    # rejections -> attempt 3 large (reach .25, 200)
    assert e.expected_attempts == pytest.approx(0.75)
    assert e.expected_nusd == math.ceil(0.5 * 100 + 0.25 * 200)


def test_unjudged_attempt_without_history_is_unknown() -> None:
    c = cfg()
    p = plan(task("a"))
    prog = {"a": TaskProgress(attempts=4, tier=Tier.MEDIUM, unjudged=True)}
    e = one(estimate_plan(p, c, ROLES, [], progress=prog))
    assert e.expected_nusd is None


def test_router_never_reroutes_a_resumed_task() -> None:
    c = cfg()
    stats = [st("s-model", 50, 50, 100), st("m-model", 50, 25, 1000)]
    router = Router(c, stats)
    p = plan(task("a"), task("b"))
    prog = {"a": TaskProgress(attempts=1, tier=Tier.MEDIUM)}
    res = estimate_plan(p, c, ROLES, stats, router=router, progress=prog)
    a, b = res.tasks
    assert a.tier is Tier.MEDIUM and not a.routed
    assert b.tier is Tier.SMALL and b.routed


def test_attempts_without_saved_tier_use_requested_and_no_router() -> None:
    c = cfg()
    stats = [st("s-model", 50, 50, 100), st("m-model", 50, 25, 1000)]
    router = Router(c, stats)
    p = plan(task("a"))
    e = one(
        estimate_plan(p, c, ROLES, stats, router=router, progress={"a": TaskProgress(attempts=1)})
    )
    assert e.tier is Tier.MEDIUM and not e.routed


def test_saved_tier_not_configured_is_config_error() -> None:
    c = cfg({"small": "s-model", "medium": "m-model"})
    p = plan(task("a"))
    with pytest.raises(ConfigError):
        estimate_plan(p, c, ROLES, [], progress={"a": TaskProgress(attempts=1, tier=Tier.LARGE)})


def test_unknown_progress_id_rejected() -> None:
    c = cfg()
    with pytest.raises(ValueError, match="progress"):
        estimate_plan(plan(task("a")), c, ROLES, [], progress={"zz": TaskProgress()})


def test_totals_include_progress() -> None:
    c = cfg()
    p = plan(task("a"), task("b"))
    res = estimate_plan(p, c, ROLES, [], progress={"a": TaskProgress(attempts=3, tier=Tier.MEDIUM)})
    assert res.worst_nusd == (1 * STEPS + 4 * STEPS) * CAP
