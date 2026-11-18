"""Pre-run cost estimate of a plan (Financial Controller, core; D62).

For every task of a plan that still has to run, two numbers:

* **worst case** — a hard upper bound that follows from the limits the code
  enforces: at most ``max_attempts`` attempts, each at most
  ``min(role.max_steps, worker_max_steps)`` worker calls plus ``verifier_calls``
  verifier calls, and no call may cost more than ``per_call_usd`` (the
  ``BudgetGuard`` rejects it before it is sent). Always known;
* **expected** — from the quality log history: the mean worker cost per attempt
  and the pass rate of the model behind each tier the task would go through
  (start tier, then the orchestrator's escalation schedule). Unknown
  (``None``) as soon as an attempt that can be reached with non-zero probability
  runs on a tier without at least ``min_samples`` judged attempts of that role.
  It counts worker cost only (verdicts do not store their cost; D56).

The estimate is pure and advisory: no model call, no write, no authorization.
Whatever it says, every call still goes through the gateway and the
``BudgetGuard``, which alone bounds spending.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from automa.agents.verifier import VerifierSettings
from automa.agents.worker import WorkerSettings
from automa.budget.guard import cap_to_nusd
from automa.config import AutomaConfig, ConfigError, Tier
from automa.orchestrator import OrchestratorSettings, next_tier
from automa.plan.models import MAX_TASK_ATTEMPTS, Plan
from automa.quality import QualityStats
from automa.roles.registry import RoleRegistry
from automa.router import Router

_WORKER_DEFAULTS = WorkerSettings()
_VERIFIER_DEFAULTS = VerifierSettings()
_ORCH_DEFAULTS = OrchestratorSettings()


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class EstimateSettings(_Frozen):
    """Bounds the estimate assumes; defaults match the agents' and orchestrator's."""

    max_attempts: int = Field(default=_ORCH_DEFAULTS.max_attempts, ge=1, le=MAX_TASK_ATTEMPTS)
    escalate_after: int = Field(default=_ORCH_DEFAULTS.escalate_after, ge=1, le=MAX_TASK_ATTEMPTS)
    worker_max_steps: int = Field(default=_WORKER_DEFAULTS.max_steps, ge=1, le=200)
    verifier_calls: int = Field(default=_VERIFIER_DEFAULTS.max_attempts, ge=0, le=5)
    # Judged attempts of (role, model) needed before history is trusted.
    min_samples: int = Field(default=3, ge=1, le=100_000)

    @classmethod
    def from_settings(
        cls,
        orchestrator: OrchestratorSettings,
        worker: WorkerSettings | None = None,
        verifier: VerifierSettings | None = None,
        *,
        min_samples: int = 3,
    ) -> EstimateSettings:
        w = worker if worker is not None else _WORKER_DEFAULTS
        v = verifier if verifier is not None else _VERIFIER_DEFAULTS
        return cls(
            max_attempts=orchestrator.max_attempts,
            escalate_after=orchestrator.escalate_after,
            worker_max_steps=w.max_steps,
            verifier_calls=v.max_attempts,
            min_samples=min_samples,
        )


class TaskProgress(_Frozen):
    """What a task has already done (a resumed run continues from it, D63).

    ``attempts`` stored results (the last one unjudged if ``unjudged``), ``tier`` the
    saved tier of the next attempt (``TaskState.tier``) and ``at_tier`` the
    consecutive rejections at ``tier`` before it (``rejections_at_tier``).
    """

    attempts: int = Field(default=0, ge=0, le=MAX_TASK_ATTEMPTS)
    tier: Tier | None = None
    at_tier: int = Field(default=0, ge=0, le=MAX_TASK_ATTEMPTS)
    unjudged: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> TaskProgress:
        if self.unjudged and (self.attempts < 1 or self.tier is None):
            raise ValueError("an unjudged attempt needs attempts >= 1 and its tier")
        if self.at_tier > self.attempts:
            raise ValueError("at_tier must not exceed attempts")
        return self

    @property
    def fresh(self) -> bool:
        """Nothing done yet: the router may still pick the starting tier."""
        return self.attempts == 0 and self.tier is None


class EstimateBasis(StrEnum):
    """Where a task's expected cost comes from."""

    HISTORY = "history"  # quality log statistics cover every reachable attempt
    NO_DATA = "no_data"  # some reachable attempt has no (or too little) history
    UNKNOWN_ROLE = "unknown_role"  # the orchestrator fails it without any model call


class TaskEstimate(_Frozen):
    """Estimate for one task that still has to run."""

    task_id: str
    role: str
    requested: Tier | None  # None only for an unknown role
    tier: Tier | None  # starting tier (after the router, if any)
    model: str | None
    routed: bool  # the router changed the starting tier
    basis: EstimateBasis
    samples: int = Field(ge=0)  # judged attempts behind the starting tier
    pass_rate: float | None = Field(default=None, ge=0, le=1)  # of the starting tier
    expected_attempts: float | None = Field(default=None, ge=0)
    expected_nusd: int | None = Field(default=None, ge=0)
    max_calls: int = Field(ge=0)
    worst_nusd: int = Field(ge=0)
    done_attempts: int = Field(default=0, ge=0)  # attempts already stored (resume)
    reverify: bool = False  # the last stored attempt still has to be judged


class BudgetFit(StrEnum):
    """How an estimate compares with the money that is left."""

    WORST_FITS = "worst_fits"  # even the worst case fits
    EXPECTED_FITS = "expected_fits"  # the complete expected cost fits, the worst case not
    UNKNOWN = "unknown"  # the known part fits, but the expected cost is incomplete
    EXPECTED_EXCEEDS = "expected_exceeds"  # the (known part of the) expected cost exceeds


class PlanEstimate(_Frozen):
    """Estimate of all tasks of a plan that still have to run."""

    tasks: list[TaskEstimate]
    skipped: list[str]  # task ids excluded (already passed)
    per_call_cap_nusd: int = Field(gt=0)
    expected_nusd: int = Field(ge=0)  # sum of the known expected costs
    worst_nusd: int = Field(ge=0)

    @property
    def unknown(self) -> list[str]:
        """Tasks whose expected cost is unknown."""
        return [t.task_id for t in self.tasks if t.expected_nusd is None]

    @property
    def complete(self) -> bool:
        return not self.unknown

    def assess(self, remaining_nusd: int) -> BudgetFit:
        """Compare with ``remaining_nusd`` (e.g. project cap minus spent so far)."""
        if remaining_nusd < 0:
            raise ValueError("remaining_nusd must not be negative")
        if self.worst_nusd <= remaining_nusd:
            return BudgetFit.WORST_FITS
        if self.expected_nusd > remaining_nusd:
            return BudgetFit.EXPECTED_EXCEEDS
        return BudgetFit.EXPECTED_FITS if self.complete else BudgetFit.UNKNOWN


_FRESH = TaskProgress()


def _tier_schedule(
    start: Tier, available: set[Tier], s: EstimateSettings, *, length: int, at_tier: int = 0
) -> list[Tier]:
    """The tier of every attempt if all of them are rejected (the orchestrator's rule)."""
    out: list[Tier] = []
    tier = start
    for _ in range(length):
        out.append(tier)
        at_tier += 1
        if at_tier >= s.escalate_after:
            higher = next_tier(tier, available)
            if higher is not None:
                tier, at_tier = higher, 0
    return out


def _expected(
    schedule: list[Tier],
    role: str,
    config: AutomaConfig,
    stats: dict[tuple[str, str], QualityStats],
    min_samples: int,
    *,
    first_done: bool = False,
) -> tuple[float, int] | None:
    """(expected attempts, expected worker cost) or ``None`` if history is missing.

    With ``first_done`` the first entry is an attempt whose work is already stored
    (only its verdict is pending): it costs no worker call but may still pass.
    """
    reach = 1.0  # probability that this attempt happens
    attempts = 0.0
    cost = 0.0
    for i, tier in enumerate(schedule):
        if reach <= 0.0:
            break
        st = stats.get((role, config.model_for_role(role, tier)))
        if st is None or st.attempts < min_samples:
            return None
        if not (first_done and i == 0):
            attempts += reach
            cost += reach * (st.cost_nusd / st.attempts)
        reach *= 1.0 - st.pass_rate
    return attempts, math.ceil(cost - 1e-6) if cost > 0 else 0


def estimate_plan(
    plan: Plan,
    config: AutomaConfig,
    roles: RoleRegistry,
    stats: Iterable[QualityStats],
    *,
    settings: EstimateSettings | None = None,
    router: Router | None = None,
    skip: Collection[str] = (),
    progress: Mapping[str, TaskProgress] | None = None,
) -> PlanEstimate:
    """Estimate the cost of running ``plan`` (tasks in ``skip`` are left out).

    ``progress`` describes tasks that already ran (resume, D63): they continue with
    the attempts that are left, from their saved tier and escalation count, are
    never routed again, and an unjudged last attempt only costs its verification.

    Raises ``ConfigError`` if a task's requested tier has no configured model
    (the orchestrator would stop before any call), ``ValueError`` for duplicate
    statistics, unknown ids in ``skip``/``progress`` or a router built from
    another config.
    """
    s = settings if settings is not None else EstimateSettings()
    if router is not None and router.config is not config and router.config != config:
        raise ValueError("router was built from a different configuration")
    by_key: dict[tuple[str, str], QualityStats] = {}
    for s_ in stats:
        key = (s_.role, s_.model)
        if key in by_key:
            raise ValueError(f"duplicate statistics for role {s_.role!r} model {s_.model!r}")
        by_key[key] = s_
    known_ids = {t.id for t in plan.tasks}
    bad = sorted(set(skip) - known_ids)
    if bad:
        raise ValueError(f"unknown task id(s) in skip: {', '.join(bad)}")
    done = dict(progress) if progress is not None else {}
    bad = sorted(set(done) - known_ids)
    if bad:
        raise ValueError(f"unknown task id(s) in progress: {', '.join(bad)}")

    cap = cap_to_nusd(config.limits.per_call_usd)
    available = set(config.tiers)
    out: list[TaskEstimate] = []
    for task in plan.topological_order():
        if task.id in skip:
            continue
        if task.role not in roles:
            out.append(
                TaskEstimate(
                    task_id=task.id,
                    role=task.role,
                    requested=None,
                    tier=None,
                    model=None,
                    routed=False,
                    basis=EstimateBasis.UNKNOWN_ROLE,
                    samples=0,
                    expected_attempts=0.0,
                    expected_nusd=0,
                    max_calls=0,
                    worst_nusd=0,
                )
            )
            continue
        role = roles.get(task.role)
        requested = task.tier or role.tier
        if requested not in available:
            raise ConfigError(f"no model configured for tier '{requested.value}'")
        prog = done.get(task.id, _FRESH)
        if prog.tier is not None and prog.tier not in available:
            raise ConfigError(f"no model configured for tier '{prog.tier.value}'")
        if not prog.fresh:
            start = prog.tier if prog.tier is not None else requested
        elif router is not None:
            # Role model lists are routed on their own models' evidence (D98).
            start = router.route(task.role, requested).tier
        else:
            start = requested
        model = config.model_for_role(task.role, start)
        st = by_key.get((task.role, model))
        left = max(0, s.max_attempts - prog.attempts)
        per_attempt = min(role.max_steps, s.worker_max_steps) + s.verifier_calls
        calls = left * per_attempt + (s.verifier_calls if prog.unjudged else 0)
        schedule = _tier_schedule(
            start,
            available,
            s,
            length=left + (1 if prog.unjudged else 0),
            at_tier=prog.at_tier,
        )
        exp = _expected(
            schedule, task.role, config, by_key, s.min_samples, first_done=prog.unjudged
        )
        out.append(
            TaskEstimate(
                task_id=task.id,
                role=task.role,
                requested=requested,
                tier=start,
                model=model,
                routed=start is not requested and prog.fresh,
                basis=EstimateBasis.HISTORY if exp is not None else EstimateBasis.NO_DATA,
                samples=st.attempts if st is not None else 0,
                pass_rate=st.pass_rate if st is not None else None,
                expected_attempts=exp[0] if exp is not None else None,
                expected_nusd=exp[1] if exp is not None else None,
                max_calls=calls,
                worst_nusd=calls * cap,
                done_attempts=prog.attempts,
                reverify=prog.unjudged,
            )
        )
    skipped = [t.id for t in plan.tasks if t.id in skip]
    return PlanEstimate(
        tasks=out,
        skipped=skipped,
        per_call_cap_nusd=cap,
        expected_nusd=sum(t.expected_nusd or 0 for t in out),
        worst_nusd=sum(t.worst_nusd for t in out),
    )
