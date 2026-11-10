"""Data models of the benchmark suite: ``Benchmark`` and ``BenchmarkOutcome``.

A benchmark is a sample goal with a *deterministic* definition of success. Its
success criteria reuse the plan's ``AcceptanceCriterion`` checks, but every
criterion must have a check: whether a benchmark passed is decided by running
things (files, commands) through the Tool Layer, never by a model (see D4, D26).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from automa.plan.models import (
    MAX_ANSWER_CHARS,
    MAX_LIST_ITEMS,
    AcceptanceCriterion,
    CheckResult,
    LongText,
    Title,
)
from automa.project.store import PROJECT_ID_PATTERN

BENCHMARK_SCHEMA_VERSION = 1
# Hard upper bound for a single benchmark's project budget. A benchmark file is
# data in the repository; it must not be able to ask for an unbounded budget.
MAX_BENCHMARK_BUDGET_USD = 5.0
MAX_BENCHMARK_ATTEMPTS = 10


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class BenchmarkKind(StrEnum):
    CLI = "cli"
    WEB = "web"
    RESEARCH = "research"


class Benchmark(_Strict):
    """Contents of one ``benchmarks/<id>.toml`` file."""

    schema_version: int = BENCHMARK_SCHEMA_VERSION
    id: Annotated[str, Field(pattern=PROJECT_ID_PATTERN)]
    title: Title
    kind: BenchmarkKind
    goal: LongText
    # Benchmarks run unattended: this text answers every clarifying question.
    answer: Annotated[
        str, StringConstraints(strip_whitespace=True, max_length=MAX_ANSWER_CHARS)
    ] = ""
    budget_usd: float = Field(gt=0, le=MAX_BENCHMARK_BUDGET_USD)
    max_attempts: int = Field(default=4, ge=1, le=MAX_BENCHMARK_ATTEMPTS)
    success_criteria: list[AcceptanceCriterion] = Field(min_length=1, max_length=MAX_LIST_ITEMS)

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != BENCHMARK_SCHEMA_VERSION:
            raise ValueError(f"unsupported benchmark schema_version {v}")
        return v

    @field_validator("success_criteria")
    @classmethod
    def _all_deterministic(cls, v: list[AcceptanceCriterion]) -> list[AcceptanceCriterion]:
        missing = [i for i, c in enumerate(v) if c.check is None]
        if missing:
            raise ValueError(
                f"every success criterion needs a deterministic check (missing: {missing})"
            )
        return v


class BenchmarkOutcome(_Strict):
    """Result of evaluating a benchmark's success criteria on a project's files.

    Always consistent: one check result per criterion, in order, and ``passed``
    is true exactly when every check passed.
    """

    benchmark_id: Annotated[str, Field(pattern=PROJECT_ID_PATTERN)]
    passed: bool
    checks: list[CheckResult] = Field(min_length=1, max_length=MAX_LIST_ITEMS)

    @model_validator(mode="after")
    def _consistent(self) -> BenchmarkOutcome:
        if [c.criterion for c in self.checks] != list(range(len(self.checks))):
            raise ValueError("checks must cover criteria 0..n-1 in order")
        if self.passed != all(c.passed for c in self.checks):
            raise ValueError("passed must equal 'every check passed'")
        return self

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def score(self) -> float:
        """Fraction of success criteria met (0.0 – 1.0)."""
        return self.passed_count / len(self.checks)
