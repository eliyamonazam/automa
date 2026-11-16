"""Benchmark gate for automatically created skills (pure, no model).

A candidate skill may only enter the skill set when benchmark runs *with* it are
measurably better than runs *without* it (docs/ARCHITECTURE.md, "Skill Engine").
``evaluate_skill_gate`` compares two sets of ``BenchmarkRunRecord`` — baseline
(without the candidate) and candidate (with it) — and returns a
``SkillGateDecision``. The rules are deterministic and use exact arithmetic
(``fractions.Fraction``), so a tie is a tie and never a rounding accident:

* both sets must cover exactly the same benchmarks, each with at least
  ``min_runs`` runs on each side (otherwise ``too_few_runs``);
* no benchmark may get worse in pass rate or in mean score (``regression``);
* at least one benchmark must be strictly better in pass rate or mean score
  (otherwise ``no_improvement``);
* optionally, the candidate's mean cost per run may not exceed
  ``max_cost_ratio`` × the baseline's, per benchmark (``cost``).

Records that cannot be compared fairly (different Automa versions, a benchmark
whose budget, kind or number of checks changed, a project id used twice, a
benchmark missing from one side) raise ``SkillGateError`` — the gate never
guesses. Nothing here calls a model, writes a file or installs a skill; see D88.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum
from fractions import Fraction
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from automa.benchmarks.results import BenchmarkRunRecord

DEFAULT_MIN_RUNS = 3
MAX_MIN_RUNS = 50
MAX_GATE_RECORDS = 1000  # per side


class SkillGateError(ValueError):
    """The two record sets cannot be compared fairly."""


class GateReason(StrEnum):
    """Why a candidate was rejected (empty list = accepted)."""

    TOO_FEW_RUNS = "too_few_runs"
    REGRESSION = "regression"
    NO_IMPROVEMENT = "no_improvement"
    COST = "cost"


class SkillGatePolicy(BaseModel):
    """Thresholds of the gate. Stricter values are always allowed."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    min_runs: int = Field(default=DEFAULT_MIN_RUNS, ge=1, le=MAX_MIN_RUNS)
    max_cost_ratio: float | None = Field(default=None, ge=1, le=100)

    @field_validator("min_runs", mode="before")
    @classmethod
    def _strict_int(cls, v: Any) -> Any:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("min_runs must be an integer")
        return v


class SideStats(BaseModel):
    """Exact statistics of one side (baseline or candidate) for one benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runs: int = Field(ge=0)
    passes: int = Field(ge=0)
    # Mean score as an exact fraction (numerator / denominator).
    score_num: int = Field(ge=0)
    score_den: int = Field(ge=1)
    cost_nusd: int = Field(ge=0)

    @property
    def pass_rate(self) -> Fraction:
        return Fraction(self.passes, self.runs) if self.runs else Fraction(0)

    @property
    def mean_score(self) -> Fraction:
        return Fraction(self.score_num, self.score_den)

    @property
    def mean_cost(self) -> Fraction:
        return Fraction(self.cost_nusd, self.runs) if self.runs else Fraction(0)

    def summary(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "passes": self.passes,
            "pass_rate": float(self.pass_rate),
            "mean_score": float(self.mean_score),
            "cost_nusd": self.cost_nusd,
        }


class BenchmarkComparison(BaseModel):
    """Baseline vs candidate on one benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str
    baseline: SideStats
    candidate: SideStats
    enough_runs: bool
    regressed: bool
    improved: bool
    over_cost: bool


class SkillGateDecision(BaseModel):
    """The gate's verdict. ``accepted`` is true exactly when ``reasons`` is empty."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted: bool
    reasons: list[GateReason]
    policy: SkillGatePolicy
    automa_version: str
    comparisons: list[BenchmarkComparison]

    def event_data(self) -> dict[str, Any]:
        """Payload for a ``skill.gate`` event (numbers and ids only)."""
        return {
            "accepted": self.accepted,
            "reasons": [r.value for r in self.reasons],
            "min_runs": self.policy.min_runs,
            "max_cost_ratio": self.policy.max_cost_ratio,
            "automa_version": self.automa_version,
            "benchmarks": [
                {
                    "benchmark_id": c.benchmark_id,
                    "baseline": c.baseline.summary(),
                    "candidate": c.candidate.summary(),
                    "regressed": c.regressed,
                    "improved": c.improved,
                    "over_cost": c.over_cost,
                }
                for c in self.comparisons
            ],
        }


def _check_records(records: Sequence[BenchmarkRunRecord], side: str) -> None:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise SkillGateError(f"{side} records must be a sequence")
    if not records:
        raise SkillGateError(f"{side} has no benchmark runs")
    if len(records) > MAX_GATE_RECORDS:
        raise SkillGateError(f"{side} has more than {MAX_GATE_RECORDS} runs")
    for r in records:
        if not isinstance(r, BenchmarkRunRecord):
            raise SkillGateError(f"{side} contains something that is not a benchmark run")


def _group(records: Iterable[BenchmarkRunRecord]) -> dict[str, list[BenchmarkRunRecord]]:
    out: dict[str, list[BenchmarkRunRecord]] = {}
    for r in records:
        out.setdefault(r.benchmark_id, []).append(r)
    return out


def _stats(records: Sequence[BenchmarkRunRecord]) -> SideStats:
    total = sum(
        (Fraction(sum(c.passed for c in r.checks), len(r.checks)) for r in records), Fraction(0)
    )
    mean = total / len(records)
    return SideStats(
        runs=len(records),
        passes=sum(1 for r in records if r.passed),
        score_num=mean.numerator,
        score_den=mean.denominator,
        cost_nusd=sum(r.cost_nusd for r in records),
    )


def evaluate_skill_gate(
    baseline: Sequence[BenchmarkRunRecord],
    candidate: Sequence[BenchmarkRunRecord],
    *,
    benchmarks: Sequence[str] | None = None,
    policy: SkillGatePolicy | None = None,
) -> SkillGateDecision:
    """Decide whether ``candidate`` runs are better than ``baseline`` runs.

    ``benchmarks``, when given, is the exact set of benchmark ids that must be
    covered (e.g. the whole suite); a record for any other benchmark is an error,
    so a caller cannot drop an unfavourable benchmark by accident.
    """
    policy = policy if policy is not None else SkillGatePolicy()
    if not isinstance(policy, SkillGatePolicy):
        raise SkillGateError("policy must be a SkillGatePolicy")
    _check_records(baseline, "baseline")
    _check_records(candidate, "candidate")

    seen: set[str] = set()
    for r in [*baseline, *candidate]:
        if r.project_id in seen:
            raise SkillGateError(f"project {r.project_id} appears more than once")
        seen.add(r.project_id)

    versions = {r.automa_version for r in [*baseline, *candidate]}
    if len(versions) != 1:
        raise SkillGateError(
            "runs come from different Automa versions: " + ", ".join(sorted(versions))
        )
    (version,) = versions

    base_groups, cand_groups = _group(baseline), _group(candidate)
    if set(base_groups) != set(cand_groups):
        only = sorted(set(base_groups) ^ set(cand_groups))
        raise SkillGateError("benchmarks not run on both sides: " + ", ".join(only))
    if benchmarks is not None:
        if isinstance(benchmarks, (str, bytes)):
            raise SkillGateError("benchmarks must be a sequence of ids")
        required = list(benchmarks)
        if not required or len(set(required)) != len(required):
            raise SkillGateError("benchmarks must be a non-empty list without duplicates")
        missing = sorted(set(required) - set(base_groups))
        if missing:
            raise SkillGateError("required benchmarks have no runs: " + ", ".join(missing))
        extra = sorted(set(base_groups) - set(required))
        if extra:
            raise SkillGateError("runs for benchmarks outside the set: " + ", ".join(extra))

    ratio = Fraction(str(policy.max_cost_ratio)) if policy.max_cost_ratio is not None else None
    comparisons: list[BenchmarkComparison] = []
    for bid in sorted(base_groups):
        recs = [*base_groups[bid], *cand_groups[bid]]
        shapes = {(r.kind, r.budget_usd, len(r.checks)) for r in recs}
        if len(shapes) != 1:
            raise SkillGateError(
                f"benchmark {bid} changed between runs (kind, budget or number of checks)"
            )
        b, c = _stats(base_groups[bid]), _stats(cand_groups[bid])
        enough = b.runs >= policy.min_runs and c.runs >= policy.min_runs
        regressed = c.pass_rate < b.pass_rate or c.mean_score < b.mean_score
        improved = c.pass_rate > b.pass_rate or c.mean_score > b.mean_score
        over_cost = ratio is not None and c.mean_cost > ratio * b.mean_cost
        comparisons.append(
            BenchmarkComparison(
                benchmark_id=bid,
                baseline=b,
                candidate=c,
                enough_runs=enough,
                regressed=regressed,
                improved=improved,
                over_cost=over_cost,
            )
        )

    reasons: list[GateReason] = []
    if not all(x.enough_runs for x in comparisons):
        reasons.append(GateReason.TOO_FEW_RUNS)
    if any(x.regressed for x in comparisons):
        reasons.append(GateReason.REGRESSION)
    if not any(x.improved for x in comparisons):
        reasons.append(GateReason.NO_IMPROVEMENT)
    if any(x.over_cost for x in comparisons):
        reasons.append(GateReason.COST)
    return SkillGateDecision(
        accepted=not reasons,
        reasons=reasons,
        policy=policy,
        automa_version=version,
        comparisons=comparisons,
    )
