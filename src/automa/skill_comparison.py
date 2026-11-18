"""Comparative benchmark run for a candidate skill (D105).

``run_skill_comparison`` measures one candidate ``SKILL.md`` against the current
skill set and hands the records to ``evaluate_skill_gate`` (D88/D89):

1. ``load_candidate_skill`` reads the candidate from a *staging* directory that
   is separate from ``<workspace>/skills/`` and the built-in skills, and that
   holds exactly one ``<name>/SKILL.md`` and nothing else (fails closed). The
   candidate is never installed: it exists only in the candidate side's
   in-memory ``SkillRegistry``.
2. ``plan_skill_comparison`` builds the candidate side with
   ``with_candidate_skill`` (text only, same policy), lays out every run
   (``runs`` baseline + ``runs`` candidate per benchmark, interleaved), and
   checks the worst-case total cost *before anything is spent*: the sum of each
   run's tightened project cap (``benchmark_limits``) must fit both the caller's
   ``max_total_usd`` and what is left of today's daily cap in the ledger.
3. Each run is a normal ``run_benchmark`` (its own project, its own
   ``BudgetGuard`` with the benchmark's tightened limits). Before every run the
   comparison checks again that the run's cap still fits the remaining total and
   daily budget; a run stopped by the infrastructure (provider, tool layer,
   configuration) or by the daily cap aborts the comparison, because such a
   failure on one side would make the other side look better. An aborted
   comparison has no gate decision.

The quality log is never written: candidate runs must not teach the router
anything about a skill that is not installed. Nothing here writes to the
skills directory, the roles or the roadmap.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from automa.bench import benchmark_limits, run_benchmark
from automa.benchmarks import Benchmark
from automa.benchmarks.results import BenchmarkRunRecord, BenchmarkStop
from automa.budget.ledger import CostLedger, usd_to_nusd
from automa.config import AutomaConfig, ConfigError
from automa.llm.images import ImageProvider
from automa.llm.provider import Provider
from automa.project import ProjectStore
from automa.roles import RoleRegistry
from automa.skill_gate import SkillGateDecision, SkillGatePolicy, evaluate_skill_gate
from automa.skill_variants import SkillVariantError, skillset_fingerprint, with_candidate_skill
from automa.skills import Skill, SkillError, SkillRegistry
from automa.skills.loader import BUILTIN_SKILLS_DIR, workspace_skills_dir
from automa.skills.registry import SKILL_FILE

MAX_RUNS_PER_SIDE = 20
MAX_COMPARISON_RUNS = 200  # baseline + candidate, all benchmarks
PROJECT_PREFIX = "skc"
_MAX_ID = 64

# Stops caused by the environment rather than by the skill: one of these on one
# side would bias the comparison, so the whole comparison is aborted.
INFRA_STOPS = frozenset({BenchmarkStop.PROVIDER, BenchmarkStop.TOOL_LAYER, BenchmarkStop.CONFIG})

Clock = Callable[[], datetime]


class SkillComparisonError(ValueError):
    """The comparison cannot be set up (nothing has been run or spent)."""


class ComparisonSide(StrEnum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"


class PlannedRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str
    side: ComparisonSide
    index: int = Field(ge=1)
    project_id: str
    max_cost_nusd: int = Field(gt=0)


class SkillComparisonPlan(BaseModel):
    """Everything decided before the first run (shown to the user before starting)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: str
    role: str
    runs_per_side: int
    benchmarks: tuple[str, ...]
    runs: tuple[PlannedRun, ...]
    max_total_nusd: int  # worst case of all runs together
    limit_nusd: int  # the caller's cap on the whole comparison
    baseline_fingerprint: str
    candidate_fingerprint: str

    @property
    def max_total_usd(self) -> float:
        return self.max_total_nusd / 1_000_000_000


class SkillComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: SkillComparisonPlan
    baseline: tuple[BenchmarkRunRecord, ...]
    candidate: tuple[BenchmarkRunRecord, ...]
    spent_nusd: int = Field(ge=0)
    aborted: str | None = None  # why the comparison stopped early (then no decision)
    decision: SkillGateDecision | None = None

    @property
    def accepted(self) -> bool:
        return self.decision is not None and self.decision.accepted


# --- candidate staging ------------------------------------------------------------


def _inside(a: Path, b: Path) -> bool:
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def load_candidate_skill(staging: Path, *, workspace: Path) -> Skill:
    """Read the one candidate skill from ``staging`` (fails closed).

    ``staging`` must be a real directory, overlap neither ``<workspace>/skills/``
    nor the built-in skills, and contain exactly one entry: a directory
    ``<name>/`` holding only ``SKILL.md``. Hidden entries count as entries.
    """
    staging = Path(staging)
    if staging.is_symlink() or not staging.is_dir():
        raise SkillComparisonError(f"candidate staging directory not found: {staging}")
    real = staging.resolve()
    for other, what in (
        (workspace_skills_dir(workspace), "the workspace skills directory"),
        (BUILTIN_SKILLS_DIR, "the built-in skills directory"),
    ):
        if _inside(real, Path(other).resolve()):
            raise SkillComparisonError(
                f"candidate staging directory must be separate from {what}: {staging}"
            )
    entries = sorted(os.scandir(real), key=lambda e: e.name)
    if len(entries) != 1:
        raise SkillComparisonError(
            f"candidate staging directory must hold exactly one <name>/{SKILL_FILE} "
            f"(found {len(entries)} entries)"
        )
    entry = entries[0]
    if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
        raise SkillComparisonError(f"{entry.name}: the candidate must be a directory")
    inner = sorted(e.name for e in os.scandir(real / entry.name))
    if inner != [SKILL_FILE]:
        raise SkillComparisonError(
            f"{entry.name}: the candidate directory must hold only {SKILL_FILE} (found {inner})"
        )
    try:
        registry = SkillRegistry.from_dir(real)
    except SkillError as e:
        raise SkillComparisonError(f"invalid candidate skill: {e}") from None
    if len(registry) != 1:  # pragma: no cover - guaranteed by the checks above
        raise SkillComparisonError("candidate staging directory must hold exactly one skill")
    return next(iter(registry))


# --- planning -----------------------------------------------------------------------


def comparison_project_id(
    stamp: datetime, bench_index: int, side: ComparisonSide, run: int, benchmark_id: str
) -> str:
    """``skc-<UTC stamp>-<bench#><b|c><run#>-<benchmark id>``, unique within a comparison."""
    ts = stamp.astimezone(UTC).strftime("%Y%m%d-%H%M%S")
    letter = "b" if side is ComparisonSide.BASELINE else "c"
    return f"{PROJECT_PREFIX}-{ts}-{bench_index:02d}{letter}{run:02d}-{benchmark_id}"[
        :_MAX_ID
    ].rstrip("-")


def _daily_left_nusd(config: AutomaConfig, ledger: CostLedger) -> int:
    daily = usd_to_nusd(config.limits.daily_usd)
    return daily - ledger.spent_nusd(day=ledger.today())


def plan_skill_comparison(
    benchmarks: Sequence[Benchmark],
    *,
    candidate: Skill,
    role: str,
    runs: int,
    max_total_usd: float,
    config: AutomaConfig,
    store: ProjectStore,
    ledger: CostLedger,
    roles: RoleRegistry,
    skills: SkillRegistry,
    policy: SkillGatePolicy | None = None,
    images_available: bool = False,
    now: datetime | None = None,
) -> tuple[SkillComparisonPlan, RoleRegistry, SkillRegistry]:
    """Check everything and lay out the runs. Raises ``SkillComparisonError``; spends nothing.

    Returns the plan and the candidate side's roles and skills.
    """
    policy = policy or SkillGatePolicy()
    if isinstance(runs, bool) or not isinstance(runs, int):
        raise SkillComparisonError("runs must be an integer")
    if not policy.min_runs <= runs <= MAX_RUNS_PER_SIDE:
        raise SkillComparisonError(
            f"runs must be between {policy.min_runs} (the gate's min_runs) "
            f"and {MAX_RUNS_PER_SIDE}, got {runs}"
        )
    if isinstance(max_total_usd, bool) or not isinstance(max_total_usd, int | float):
        raise SkillComparisonError("max_total_usd must be a number")
    if not math.isfinite(max_total_usd) or max_total_usd <= 0:
        raise SkillComparisonError("max_total_usd must be a positive, finite number")
    benches = list(benchmarks)
    if not benches or not all(isinstance(b, Benchmark) for b in benches):
        raise SkillComparisonError("at least one benchmark is needed")
    ids = [b.id for b in benches]
    if len(set(ids)) != len(ids):
        raise SkillComparisonError("a benchmark is listed more than once")
    if 2 * runs * len(benches) > MAX_COMPARISON_RUNS:
        raise SkillComparisonError(
            f"too many runs ({2 * runs * len(benches)}; at most {MAX_COMPARISON_RUNS})"
        )
    needing = [b.id for b in benches if b.needs_images]
    if needing and not images_available:
        raise SkillComparisonError(
            f"benchmarks {needing} need image generation: set [images] model in automa.toml"
        )
    try:
        cand_roles, cand_skills = with_candidate_skill(roles, skills, candidate, role=role)
    except SkillVariantError as e:
        raise SkillComparisonError(str(e)) from None
    base_fp = skillset_fingerprint(roles, skills)
    cand_fp = skillset_fingerprint(cand_roles, cand_skills)
    if base_fp == cand_fp:  # pragma: no cover - with_candidate_skill adds a skill name
        raise SkillComparisonError("the candidate side does not differ from the baseline")

    stamp = now or datetime.now(UTC)
    planned: list[PlannedRun] = []
    for bi, bench in enumerate(benches, start=1):
        cap = usd_to_nusd(benchmark_limits(config.limits, bench).project_usd)
        for i in range(1, runs + 1):
            for side in (ComparisonSide.BASELINE, ComparisonSide.CANDIDATE):
                pid = comparison_project_id(stamp, bi, side, i, bench.id)
                planned.append(
                    PlannedRun(
                        benchmark_id=bench.id,
                        side=side,
                        index=i,
                        project_id=pid,
                        max_cost_nusd=cap,
                    )
                )
    pids = [p.project_id for p in planned]
    if len(set(pids)) != len(pids):  # pragma: no cover - ids carry bench#, side, run#
        raise SkillComparisonError("project ids are not unique")
    taken = [p for p in pids if store.exists(p)]
    if taken:
        raise SkillComparisonError(f"project {taken[0]!r} already exists; try again")

    total = sum(p.max_cost_nusd for p in planned)
    limit = usd_to_nusd(max_total_usd)
    if total > limit:
        raise SkillComparisonError(
            f"worst-case cost ${total / 1e9:.4f} exceeds the comparison cap ${max_total_usd:.4f}"
        )
    left = _daily_left_nusd(config, ledger)
    if total > left:
        raise SkillComparisonError(
            f"worst-case cost ${total / 1e9:.4f} exceeds what is left of today's "
            f"daily cap (${max(left, 0) / 1e9:.4f})"
        )
    plan = SkillComparisonPlan(
        candidate=candidate.name,
        role=role,
        runs_per_side=runs,
        benchmarks=tuple(ids),
        runs=tuple(planned),
        max_total_nusd=total,
        limit_nusd=limit,
        baseline_fingerprint=base_fp,
        candidate_fingerprint=cand_fp,
    )
    return plan, cand_roles, cand_skills


# --- running ------------------------------------------------------------------------


def run_skill_comparison(
    benchmarks: Sequence[Benchmark],
    *,
    candidate: Skill,
    role: str,
    runs: int,
    max_total_usd: float,
    config: AutomaConfig,
    provider: Provider,
    store: ProjectStore,
    ledger: CostLedger,
    roles: RoleRegistry,
    skills: SkillRegistry,
    policy: SkillGatePolicy | None = None,
    image_provider: ImageProvider | None = None,
    clock: Clock | None = None,
    on_record: Callable[[ComparisonSide, BenchmarkRunRecord], None] | None = None,
) -> SkillComparisonResult:
    """Run every planned benchmark run and apply the gate.

    ``on_record`` is called after each run (e.g. to write the result file).
    ``SkillComparisonError`` / ``ConfigError`` before the first run mean nothing
    was spent. ``KeyboardInterrupt`` propagates (records already passed to
    ``on_record`` stay; there is no decision).
    """
    if image_provider is not None and not config.images.enabled:
        raise ConfigError("an image provider needs an [images] model in the configuration")
    policy = policy or SkillGatePolicy()
    clock = clock or (lambda: datetime.now(UTC))
    plan, cand_roles, cand_skills = plan_skill_comparison(
        benchmarks,
        candidate=candidate,
        role=role,
        runs=runs,
        max_total_usd=max_total_usd,
        config=config,
        store=store,
        ledger=ledger,
        roles=roles,
        skills=skills,
        policy=policy,
        images_available=image_provider is not None,
        now=clock(),
    )
    by_id = {b.id: b for b in benchmarks}
    side_setup = {
        ComparisonSide.BASELINE: (roles, skills),
        ComparisonSide.CANDIDATE: (cand_roles, cand_skills),
    }
    records: dict[ComparisonSide, list[BenchmarkRunRecord]] = {s: [] for s in ComparisonSide}
    spent = 0
    aborted: str | None = None
    for planned in plan.runs:
        if spent + planned.max_cost_nusd > plan.limit_nusd:
            aborted = "comparison cap reached"
            break
        if planned.max_cost_nusd > _daily_left_nusd(config, ledger):
            aborted = "daily cap reached"
            break
        side_roles, side_skills = side_setup[planned.side]
        record = run_benchmark(
            by_id[planned.benchmark_id],
            config=config,
            provider=provider,
            store=store,
            ledger=ledger,
            roles=side_roles,
            skills=side_skills,
            quality=None,
            project_id=planned.project_id,
            clock=clock,
            image_provider=image_provider,
        )
        expected_fp = (
            plan.baseline_fingerprint
            if planned.side is ComparisonSide.BASELINE
            else plan.candidate_fingerprint
        )
        if record.skills_fingerprint != expected_fp:  # pragma: no cover - defence in depth
            raise SkillComparisonError("a run used an unexpected skill set")
        records[planned.side].append(record)
        spent += record.cost_nusd
        if on_record is not None:
            on_record(planned.side, record)
        if record.stop in INFRA_STOPS:
            aborted = f"run {record.project_id} stopped by {record.stop.value}: {record.detail}"[
                :500
            ]
            break
        if record.stop is BenchmarkStop.BUDGET and _daily_left_nusd(config, ledger) <= 0:
            aborted = "daily cap reached"
            break

    decision = None
    if aborted is None:
        decision = evaluate_skill_gate(
            records[ComparisonSide.BASELINE],
            records[ComparisonSide.CANDIDATE],
            benchmarks=list(plan.benchmarks),
            policy=policy,
        )
    return SkillComparisonResult(
        plan=plan,
        baseline=tuple(records[ComparisonSide.BASELINE]),
        candidate=tuple(records[ComparisonSide.CANDIDATE]),
        spent_nusd=spent,
        aborted=aborted,
        decision=decision,
    )
