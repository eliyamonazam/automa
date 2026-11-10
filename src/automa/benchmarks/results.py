"""Benchmark run records: ``BenchmarkRunRecord`` and write-once result files.

A record says what one benchmark run measured: whether the deterministic success
criteria passed (``passed``/``score``/``checks``, from ``BenchmarkOutcome``), how
the run ended (``stop``), and what it cost (exact nano-USD from the ledger).
Records are written once to ``<results_dir>/<benchmark_id>/<project_id>.json``
and never overwritten (see D27). Nothing here calls a model.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from automa.benchmarks.models import (
    MAX_BENCHMARK_BUDGET_USD,
    BenchmarkKind,
    BenchmarkOutcome,
)
from automa.plan.models import MAX_LIST_ITEMS, CheckResult, Text, Title
from automa.project.store import PROJECT_ID_PATTERN, ProjectStatus

RESULT_SCHEMA_VERSION = 1
RESULT_SUFFIX = ".json"


class BenchmarkResultError(Exception):
    """A benchmark result could not be written (or already exists)."""


class BenchmarkStop(StrEnum):
    """How a benchmark run ended (the Orchestrator's stop reasons + planning)."""

    COMPLETED = "completed"
    TASKS_FAILED = "tasks_failed"
    BUDGET = "budget"
    APPROVAL = "approval"
    PROVIDER = "provider"
    VERIFICATION = "verification"
    TOOL_LAYER = "tool_layer"
    CONFIG = "config"
    PLANNING = "planning"  # the Executive gave no valid plan


_Id = Annotated[str, Field(pattern=PROJECT_ID_PATTERN)]


class BenchmarkRunRecord(BaseModel):
    """Contents of one result file. Always internally consistent."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: int = RESULT_SCHEMA_VERSION
    benchmark_id: _Id
    title: Title
    kind: BenchmarkKind
    project_id: _Id
    automa_version: Annotated[str, Field(min_length=1, max_length=64)]
    started_at: AwareDatetime
    finished_at: AwareDatetime
    stop: BenchmarkStop
    detail: Text = ""
    project_status: ProjectStatus
    planned: bool
    passed: bool
    checks: list[CheckResult] = Field(min_length=1, max_length=MAX_LIST_ITEMS)
    score: float = Field(ge=0, le=1)
    budget_usd: float = Field(gt=0, le=MAX_BENCHMARK_BUDGET_USD)
    cost_nusd: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    attempts: int = Field(ge=0)
    tasks_passed: list[str] = Field(default_factory=list)
    tasks_failed: list[str] = Field(default_factory=list)
    tasks_blocked: list[str] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != RESULT_SCHEMA_VERSION:
            raise ValueError(f"unsupported benchmark result schema_version {v}")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> BenchmarkRunRecord:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be before started_at")
        # Same invariants as BenchmarkOutcome: success is the checks, nothing else.
        outcome = BenchmarkOutcome(
            benchmark_id=self.benchmark_id, passed=self.passed, checks=self.checks
        )
        if abs(self.score - outcome.score) > 1e-9:
            raise ValueError("score must equal the fraction of passed checks")
        if self.stop is BenchmarkStop.PLANNING and self.planned:
            raise ValueError("a planning stop means there is no plan")
        if not self.planned and (self.attempts or self.tasks_passed or self.tasks_failed):
            raise ValueError("no task can have run without a plan")
        if self.stop is BenchmarkStop.COMPLETED and (
            not self.planned or self.tasks_failed or self.tasks_blocked
        ):
            raise ValueError("a completed run has a plan and no failed/blocked task")
        return self

    @property
    def cost_usd(self) -> float:
        return self.cost_nusd / 1_000_000_000

    @classmethod
    def build(
        cls,
        *,
        outcome: BenchmarkOutcome,
        title: str,
        kind: BenchmarkKind,
        project_id: str,
        automa_version: str,
        started_at: datetime,
        finished_at: datetime,
        stop: BenchmarkStop,
        detail: str,
        project_status: ProjectStatus,
        planned: bool,
        budget_usd: float,
        cost_nusd: int,
        model_calls: int,
        attempts: int,
        tasks_passed: list[str],
        tasks_failed: list[str],
        tasks_blocked: list[str],
    ) -> BenchmarkRunRecord:
        return cls(
            benchmark_id=outcome.benchmark_id,
            title=title,
            kind=kind,
            project_id=project_id,
            automa_version=automa_version,
            started_at=started_at,
            finished_at=finished_at,
            stop=stop,
            detail=detail[:1000],
            project_status=project_status,
            planned=planned,
            passed=outcome.passed,
            checks=outcome.checks,
            score=outcome.score,
            budget_usd=budget_usd,
            cost_nusd=cost_nusd,
            model_calls=model_calls,
            attempts=attempts,
            tasks_passed=tasks_passed,
            tasks_failed=tasks_failed,
            tasks_blocked=tasks_blocked,
        )


def _real_dir(path: Path, what: str) -> None:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        try:
            path.mkdir()
        except FileExistsError:
            pass
        except OSError as e:
            raise BenchmarkResultError(f"cannot create {what} ({e.strerror})") from None
        st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        raise BenchmarkResultError(f"{what} is not a real directory: {path}")


def prepare_results_dir(results_dir: Path, benchmark_id: str) -> Path:
    """Create (if needed) and check ``<results_dir>/<benchmark_id>/``.

    Called before a run starts, so a run never spends money it cannot record.
    The results directory's parent must exist; symlinks and non-directories are refused.
    """
    results_dir = Path(results_dir)
    if not results_dir.parent.is_dir():
        raise BenchmarkResultError(f"parent of results directory does not exist: {results_dir}")
    _real_dir(results_dir, "results directory")
    sub = results_dir / benchmark_id
    _real_dir(sub, "benchmark results directory")
    return sub


def write_result(results_dir: Path, record: BenchmarkRunRecord) -> Path:
    """Write ``record`` once; refuse symlinked directories and existing files."""
    sub = prepare_results_dir(results_dir, record.benchmark_id)
    path = sub / f"{record.project_id}{RESULT_SUFFIX}"
    text = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=sub)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise BenchmarkResultError(f"result already exists: {path}") from None
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    return path


def load_result(path: Path) -> BenchmarkRunRecord:
    """Read a result file (regular, non-symlinked); raises ``BenchmarkResultError``."""
    try:
        st = os.lstat(path)
    except OSError as e:
        raise BenchmarkResultError(f"cannot read result {path.name} ({e.strerror})") from None
    if not stat.S_ISREG(st.st_mode):
        raise BenchmarkResultError(f"result {path.name} is not a regular file")
    try:
        return BenchmarkRunRecord.model_validate_json(path.read_bytes())
    except (ValidationError, ValueError) as e:
        raise BenchmarkResultError(f"invalid result {path.name}: {e}") from None
