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

Image generation (D101): a task whose role lists ``generate_image`` may also make
up to ``[images] max_per_task`` images per attempt at the ``[images]`` model's
``per_image_usd``. The worst case adds every one of them for every attempt left;
the expected cost (when known) conservatively adds the full image allowance for
every expected attempt, since the quality log does not hold image costs.

Cost range (D119): next to the worst case and the history-based expected cost,
every task gets a *realistic range* ``range_low_nusd``..``range_high_nusd`` that
is never "unknown". Each attempt in the range is priced at the catalog price of
the model it would run on, from

* the user's own history when the quality log holds at least ``min_samples``
  judged attempts of (role, model): the 25th (low) and 90th (high) percentile of
  the worker cost per attempt, plus the role's verifier token allowance; else
* the role's default token profile (``DEFAULT_TOKEN_PROFILES``: input/output
  tokens per attempt, low and high, taken from the first real runs on 1xAi).

The low end assumes the first attempt passes, the high end ``RANGE_HIGH_ATTEMPTS``
attempts on the orchestrator's escalation schedule. Both are clamped to the
worst case, which stays the only hard bound ("Guard cap").

The estimate is pure and advisory: no model call, no write, no authorization.
Whatever it says, every call still goes through the gateway and the
``BudgetGuard``, which alone bounds spending.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from automa.agents.verifier import VerifierSettings
from automa.agents.worker import WorkerSettings
from automa.budget.guard import NANO_PER_USD, cap_to_nusd
from automa.config import AutomaConfig, ConfigError, ModelPrice, Tier
from automa.orchestrator import OrchestratorSettings, next_tier, uses_images
from automa.plan.models import MAX_TASK_ATTEMPTS, Plan
from automa.quality import QualityRecord, QualityStats
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


class TokenAllowance(_Frozen):
    """Tokens of one attempt (all its calls together)."""

    input_tokens: int = Field(ge=0, le=10_000_000)
    output_tokens: int = Field(ge=0, le=10_000_000)


class RoleTokenProfile(_Frozen):
    """Default tokens per attempt of a role when its history is too thin (D119).

    ``low``/``high`` cover the worker's calls of one attempt; ``verify`` the
    verifier's calls of one attempt (added in both ends, also on history).
    """

    low: TokenAllowance
    high: TokenAllowance
    verify: TokenAllowance

    @model_validator(mode="after")
    def _ordered(self) -> RoleTokenProfile:
        if (
            self.low.input_tokens > self.high.input_tokens
            or self.low.output_tokens > self.high.output_tokens
        ):
            raise ValueError("low token allowance must not exceed high")
        return self


def _profile(
    li: int, lo: int, hi: int, ho: int, vi: int = 4_000, vo: int = 600
) -> RoleTokenProfile:
    return RoleTokenProfile(
        low=TokenAllowance(input_tokens=li, output_tokens=lo),
        high=TokenAllowance(input_tokens=hi, output_tokens=ho),
        verify=TokenAllowance(input_tokens=vi, output_tokens=vo),
    )


# Recalibrated on the owner's real runs (D173). The first profiles (D119: ~10-30k
# input, 2-5k output per attempt) came from tiny tasks and fell 7-10x short on real
# products: a C++ project with compile and test steps was estimated at $0.04-0.14
# and cost $0.93; one web-page task made 29 calls for $0.62. Every call of an
# attempt resends the whole conversation, so input grows with the number of steps;
# a real product attempt is 10-30 steps. The verifier's model-judged criteria read
# the produced files, so it uses far more than a short prompt.
DEFAULT_PROFILE = _profile(25_000, 4_000, 200_000, 20_000, 12_000, 1_200)
DEFAULT_TOKEN_PROFILES: dict[str, RoleTokenProfile] = {
    "writer": _profile(20_000, 4_000, 150_000, 16_000, 12_000, 1_200),
    "developer": _profile(25_000, 4_000, 200_000, 20_000, 12_000, 1_200),
    "architect": _profile(20_000, 4_000, 120_000, 12_000, 12_000, 1_200),
    "designer": _profile(20_000, 4_000, 150_000, 16_000, 12_000, 1_200),
    "researcher": _profile(40_000, 4_000, 250_000, 15_000, 12_000, 1_200),
}
RANGE_HIGH_ATTEMPTS = 2  # attempts the high end of the range assumes
RANGE_LOW_PERCENTILE = 25
RANGE_HIGH_PERCENTILE = 90


def profile_for(role: str) -> RoleTokenProfile:
    """The default token profile of ``role`` (``DEFAULT_PROFILE`` for custom roles)."""
    return DEFAULT_TOKEN_PROFILES.get(role, DEFAULT_PROFILE)


def tokens_nusd(price: ModelPrice, tokens: TokenAllowance) -> int:
    """Cost of ``tokens`` at ``price`` in nano-USD, rounded up (like the ledger)."""
    # USD per Mtok * 1e9 nUSD/USD / 1e6 tok/Mtok = per_mtok * 1000 nUSD per token.
    cost = (
        tokens.input_tokens * price.input_per_mtok + tokens.output_tokens * price.output_per_mtok
    ) * (NANO_PER_USD / 1_000_000)
    return math.ceil(cost - 1e-6) if cost > 0 else 0


def percentile_nusd(costs: Sequence[int], pct: int) -> int:
    """Nearest-rank percentile ``pct`` (1..100) of non-empty integer ``costs``."""
    if not costs:
        raise ValueError("no costs")
    if not 1 <= pct <= 100:
        raise ValueError("pct must be within 1..100")
    if any(c < 0 for c in costs):
        raise ValueError("costs must not be negative")
    ordered = sorted(costs)
    rank = math.ceil(pct * len(ordered) / 100)
    return ordered[max(rank, 1) - 1]


def attempt_costs(records: Iterable[QualityRecord]) -> dict[tuple[str, str], list[int]]:
    """Worker cost of every judged attempt, grouped by (role, model)."""
    out: dict[tuple[str, str], list[int]] = {}
    for rec in records:
        out.setdefault((rec.role, rec.model), []).append(rec.cost_nusd)
    return out


class RangeBasis(StrEnum):
    """Where a task's cost range comes from."""

    HISTORY = "history"  # every attempt in the range is priced from the user's history
    MIXED = "mixed"  # some attempts from history, some from default token profiles
    DEFAULTS = "defaults"  # default token profiles and catalog prices only
    UNKNOWN_ROLE = "unknown_role"  # no model call at all


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
    max_images: int = Field(default=0, ge=0)  # images all attempts left may generate
    image_worst_nusd: int = Field(default=0, ge=0)  # their cost (included in worst_nusd)
    done_attempts: int = Field(default=0, ge=0)  # attempts already stored (resume)
    reverify: bool = False  # the last stored attempt still has to be judged
    # Realistic range (D119); never above the worst case.
    range_low_nusd: int = Field(default=0, ge=0)
    range_high_nusd: int = Field(default=0, ge=0)
    range_basis: RangeBasis = RangeBasis.DEFAULTS

    @model_validator(mode="after")
    def _range_ordered(self) -> TaskEstimate:
        if self.range_low_nusd > self.range_high_nusd:
            raise ValueError("range low must not exceed range high")
        if self.range_high_nusd > self.worst_nusd:
            raise ValueError("range high must not exceed the worst case")
        return self


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
    def range_low_nusd(self) -> int:
        """Low end of the realistic range of the whole plan (D119)."""
        return sum(t.range_low_nusd for t in self.tasks)

    @property
    def range_high_nusd(self) -> int:
        """High end of the realistic range of the whole plan (D119)."""
        return sum(t.range_high_nusd for t in self.tasks)

    @property
    def range_basis(self) -> RangeBasis:
        """``history`` only if every task's range is from history (no-call tasks aside)."""
        kinds = {t.range_basis for t in self.tasks} - {RangeBasis.UNKNOWN_ROLE}
        if not kinds:
            return RangeBasis.HISTORY if self.tasks else RangeBasis.DEFAULTS
        if kinds == {RangeBasis.HISTORY}:
            return RangeBasis.HISTORY
        if kinds == {RangeBasis.DEFAULTS}:
            return RangeBasis.DEFAULTS
        return RangeBasis.MIXED

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


def _range(
    schedule: list[Tier],
    role: str,
    config: AutomaConfig,
    costs: Mapping[tuple[str, str], Sequence[int]],
    min_samples: int,
    *,
    left: int,
    unjudged: bool,
    images_per_attempt: int,
    image_nusd: int,
) -> tuple[int, int, RangeBasis]:
    """(low, high, basis) of the realistic range of one task (D119).

    ``schedule`` holds the tier of the unjudged stored attempt first (if any), then
    of every attempt left. Low: the stored attempt (if any) passes, else the first
    new attempt passes. High: ``RANGE_HIGH_ATTEMPTS`` attempts in all.
    """
    prof = profile_for(role)
    used_history = False
    used_defaults = False

    def verify_nusd(tier: Tier) -> int:
        return tokens_nusd(config.price_for(config.model_for_role(role, tier)), prof.verify)

    def attempt_nusd(tier: Tier, *, high: bool) -> int:
        nonlocal used_history, used_defaults
        model = config.model_for_role(role, tier)
        hist = costs.get((role, model))
        if hist is not None and len(hist) >= min_samples:
            used_history = True
            pct = RANGE_HIGH_PERCENTILE if high else RANGE_LOW_PERCENTILE
            worker = percentile_nusd(hist, pct)
        else:
            used_defaults = True
            worker = tokens_nusd(config.price_for(model), prof.high if high else prof.low)
        return worker + verify_nusd(tier)

    new_tiers = schedule[1:] if unjudged else schedule
    new_tiers = new_tiers[:left]
    low = 0
    high = 0
    if unjudged:
        low += verify_nusd(schedule[0])
        high += verify_nusd(schedule[0])
    else:
        if new_tiers:
            low += attempt_nusd(new_tiers[0], high=False) + min(images_per_attempt, 1) * image_nusd
    n_high = max(0, RANGE_HIGH_ATTEMPTS - (1 if unjudged else 0))
    for tier in new_tiers[:n_high]:
        high += attempt_nusd(tier, high=True) + images_per_attempt * image_nusd
    if used_history and used_defaults:
        basis = RangeBasis.MIXED
    elif used_history:
        basis = RangeBasis.HISTORY
    else:
        basis = RangeBasis.DEFAULTS
    return low, high, basis


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
    costs: Mapping[tuple[str, str], Sequence[int]] | None = None,
) -> PlanEstimate:
    """Estimate the cost of running ``plan`` (tasks in ``skip`` are left out).

    ``progress`` describes tasks that already ran (resume, D63): they continue with
    the attempts that are left, from their saved tier and escalation count, are
    never routed again, and an unjudged last attempt only costs its verification.

    ``costs`` maps (role, model) to the worker cost of every judged attempt
    (``attempt_costs(log.records())``); it feeds the history end of the realistic
    range (D119). Without it the range uses the default token profiles only.

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

    hist_costs: Mapping[tuple[str, str], Sequence[int]] = costs if costs is not None else {}
    cap = cap_to_nusd(config.limits.per_call_usd)
    image_price = config.image_price()
    # Rounded up, like the ledger: an image is never estimated below its price.
    image_nusd = math.ceil(image_price * NANO_PER_USD) if image_price is not None else 0
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
                    range_basis=RangeBasis.UNKNOWN_ROLE,
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
        images = 0
        if uses_images(role):
            if image_price is None:
                # The orchestrator stops such a task before any call (D101).
                raise ConfigError(
                    f"role '{task.role}' needs images: set [images] model in automa.toml"
                )
            images = left * config.images.max_per_task
            if exp is not None:
                extra = math.ceil(exp[0] * config.images.max_per_task * image_nusd)
                exp = (exp[0], exp[1] + extra)
        worst = calls * cap + images * image_nusd
        r_low, r_high, r_basis = _range(
            schedule,
            task.role,
            config,
            hist_costs,
            s.min_samples,
            left=left,
            unjudged=prog.unjudged,
            images_per_attempt=config.images.max_per_task if images else 0,
            image_nusd=image_nusd,
        )
        r_high = min(r_high, worst)
        r_low = min(r_low, r_high)
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
                worst_nusd=worst,
                max_images=images,
                image_worst_nusd=images * image_nusd,
                done_attempts=prog.attempts,
                reverify=prog.unjudged,
                range_low_nusd=r_low,
                range_high_nusd=r_high,
                range_basis=r_basis,
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
