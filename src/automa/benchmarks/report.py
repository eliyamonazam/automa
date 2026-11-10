"""Failure analysis over recorded benchmark runs: ``scan_results`` and ``build_report``.

Read-only: it lists ``<results_dir>/<benchmark_id>/<project_id>.json`` files written
by ``write_result`` and aggregates them per benchmark (pass rate, score, cost, how
runs ended, which success criteria fail most). It never creates, changes or deletes
a file and never calls a model (see D28). Files that cannot be trusted (invalid,
symlinked, misplaced) are not silently skipped: they are listed as problems.
"""

from __future__ import annotations

import os
import stat
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from automa.benchmarks.loader import BenchmarkSuite
from automa.benchmarks.results import (
    RESULT_SUFFIX,
    BenchmarkResultError,
    BenchmarkRunRecord,
    BenchmarkStop,
    load_result,
)
from automa.project.store import _PROJECT_ID_RE

# Upper bound on files read by one scan, so a huge directory cannot exhaust memory.
MAX_RESULT_FILES = 10_000


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class FailureCause(StrEnum):
    """Why a run that did not pass failed, coarsest useful category first."""

    # The project finished with every task accepted by its own Verifier, but the
    # benchmark's deterministic criteria still failed: plan or Verifier too lenient.
    CRITERIA_MISSED = "criteria_missed"
    TASKS_FAILED = "tasks_failed"  # some task failed for good after all attempts
    PLANNING = "planning"  # no valid plan
    BUDGET = "budget"
    PROVIDER = "provider"
    OTHER_STOP = "other_stop"  # approval / verification / tool layer / config


def failure_cause(record: BenchmarkRunRecord) -> FailureCause | None:
    """``None`` for a passed run; otherwise the cause category of the failure."""
    if record.passed:
        return None
    match record.stop:
        case BenchmarkStop.COMPLETED:
            return FailureCause.CRITERIA_MISSED
        case BenchmarkStop.TASKS_FAILED:
            return FailureCause.TASKS_FAILED
        case BenchmarkStop.PLANNING:
            return FailureCause.PLANNING
        case BenchmarkStop.BUDGET:
            return FailureCause.BUDGET
        case BenchmarkStop.PROVIDER:
            return FailureCause.PROVIDER
        case _:
            return FailureCause.OTHER_STOP


class ResultProblem(_Frozen):
    """A file or directory under the results directory that was not used."""

    path: str
    reason: str


class ResultScan(_Frozen):
    records: list[BenchmarkRunRecord] = Field(default_factory=list)
    problems: list[ResultProblem] = Field(default_factory=list)


class CriterionFailure(_Frozen):
    criterion: int = Field(ge=0)
    description: str  # from the suite; "(unknown criterion)" if not found
    failed_runs: int = Field(ge=1)
    # Failed in runs that otherwise completed: the most informative signal.
    failed_in_completed_runs: int = Field(ge=0)


class BenchmarkSummary(_Frozen):
    benchmark_id: str
    title: str
    in_suite: bool
    runs: int = Field(ge=1)
    passed_runs: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    mean_score: float = Field(ge=0, le=1)
    total_cost_nusd: int = Field(ge=0)
    mean_cost_nusd: int = Field(ge=0)
    mean_attempts: float = Field(ge=0)
    stops: dict[str, int]
    failure_causes: dict[str, int]
    criteria_failures: list[CriterionFailure]
    versions: list[str]
    first_run_at: AwareDatetime
    last_run_at: AwareDatetime
    last_project_id: str
    last_passed: bool


class BenchmarkReport(_Frozen):
    benchmarks: list[BenchmarkSummary] = Field(default_factory=list)
    problems: list[ResultProblem] = Field(default_factory=list)
    total_runs: int = Field(ge=0)
    total_passed: int = Field(ge=0)
    total_cost_nusd: int = Field(ge=0)


def _is_real_dir(path: Path) -> bool:
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


def scan_results(results_dir: Path, *, benchmark_id: str | None = None) -> ResultScan:
    """Load every result file under ``results_dir`` (optionally one benchmark only).

    A missing results directory is an empty scan. A results directory that is a
    symlink or not a directory raises ``BenchmarkResultError``. Everything else that
    cannot be used becomes a ``ResultProblem``. Temporary files of an interrupted
    ``write_result`` (``.<name>.tmp``) are ignored.
    """
    results_dir = Path(results_dir)
    try:
        st = os.lstat(results_dir)
    except FileNotFoundError:
        return ResultScan()
    except OSError as e:
        raise BenchmarkResultError(f"cannot read results directory ({e.strerror})") from None
    if not stat.S_ISDIR(st.st_mode):
        raise BenchmarkResultError(f"results directory is not a real directory: {results_dir}")

    records: list[BenchmarkRunRecord] = []
    problems: list[ResultProblem] = []
    read = 0
    for sub in sorted(results_dir.iterdir()):
        name = sub.name
        if name.startswith("."):
            continue
        if benchmark_id is not None and name != benchmark_id:
            continue
        rel = name
        if not _is_real_dir(sub):
            if sub.is_symlink() or sub.is_dir():
                problems.append(ResultProblem(path=rel, reason="symlinked directory"))
            elif name != "README.md" and name != ".gitkeep":
                problems.append(ResultProblem(path=rel, reason="not a benchmark directory"))
            continue
        if not _PROJECT_ID_RE.fullmatch(name):
            problems.append(ResultProblem(path=rel, reason="invalid benchmark id"))
            continue
        for f in sorted(sub.iterdir()):
            fname = f.name
            if fname.startswith("."):
                continue
            frel = f"{name}/{fname}"
            if not fname.endswith(RESULT_SUFFIX):
                problems.append(ResultProblem(path=frel, reason="not a result file"))
                continue
            if read >= MAX_RESULT_FILES:
                problems.append(
                    ResultProblem(path=frel, reason=f"over {MAX_RESULT_FILES} files; not read")
                )
                continue
            read += 1
            try:
                record = load_result(f)
            except BenchmarkResultError as e:
                problems.append(ResultProblem(path=frel, reason=str(e)[:500]))
                continue
            if record.benchmark_id != name:
                problems.append(
                    ResultProblem(path=frel, reason=f"records benchmark {record.benchmark_id!r}")
                )
                continue
            if fname != f"{record.project_id}{RESULT_SUFFIX}":
                problems.append(
                    ResultProblem(path=frel, reason=f"records project {record.project_id!r}")
                )
                continue
            records.append(record)
    records.sort(key=lambda r: (r.benchmark_id, r.started_at, r.project_id))
    return ResultScan(records=records, problems=problems)


def _criterion_description(suite: BenchmarkSuite | None, benchmark_id: str, index: int) -> str:
    if suite is not None and benchmark_id in suite.ids():
        criteria = suite.get(benchmark_id).success_criteria
        if index < len(criteria):
            return criteria[index].description
    return "(unknown criterion)"


def _summarize(
    benchmark_id: str, runs: list[BenchmarkRunRecord], suite: BenchmarkSuite | None
) -> BenchmarkSummary:
    n = len(runs)
    passed = sum(1 for r in runs if r.passed)
    total_cost = sum(r.cost_nusd for r in runs)
    failed_runs: Counter[int] = Counter()
    failed_completed: Counter[int] = Counter()
    for r in runs:
        for c in r.checks:
            if not c.passed:
                failed_runs[c.criterion] += 1
                if r.stop is BenchmarkStop.COMPLETED:
                    failed_completed[c.criterion] += 1
    criteria = [
        CriterionFailure(
            criterion=idx,
            description=_criterion_description(suite, benchmark_id, idx),
            failed_runs=count,
            failed_in_completed_runs=failed_completed[idx],
        )
        for idx, count in failed_runs.items()
    ]
    criteria.sort(key=lambda c: (-c.failed_in_completed_runs, -c.failed_runs, c.criterion))
    causes = Counter(c.value for r in runs if (c := failure_cause(r)) is not None)
    last = max(runs, key=lambda r: (r.finished_at, r.project_id))
    in_suite = suite is not None and benchmark_id in suite.ids()
    title = suite.get(benchmark_id).title if in_suite and suite is not None else last.title
    return BenchmarkSummary(
        benchmark_id=benchmark_id,
        title=title,
        in_suite=in_suite,
        runs=n,
        passed_runs=passed,
        pass_rate=passed / n,
        mean_score=sum(r.score for r in runs) / n,
        total_cost_nusd=total_cost,
        mean_cost_nusd=total_cost // n,
        mean_attempts=sum(r.attempts for r in runs) / n,
        stops=dict(sorted(Counter(r.stop.value for r in runs).items())),
        failure_causes=dict(sorted(causes.items())),
        criteria_failures=criteria,
        versions=sorted({r.automa_version for r in runs}),
        first_run_at=min(r.started_at for r in runs),
        last_run_at=last.finished_at,
        last_project_id=last.project_id,
        last_passed=last.passed,
    )


def build_report(
    records: Iterable[BenchmarkRunRecord],
    *,
    suite: BenchmarkSuite | None = None,
    problems: Iterable[ResultProblem] = (),
) -> BenchmarkReport:
    """Aggregate run records per benchmark (sorted by id). Pure; touches no file."""
    by_id: dict[str, list[BenchmarkRunRecord]] = {}
    for r in records:
        by_id.setdefault(r.benchmark_id, []).append(r)
    summaries = [_summarize(bid, runs, suite) for bid, runs in sorted(by_id.items())]
    return BenchmarkReport(
        benchmarks=summaries,
        problems=list(problems),
        total_runs=sum(s.runs for s in summaries),
        total_passed=sum(s.passed_runs for s in summaries),
        total_cost_nusd=sum(s.total_cost_nusd for s in summaries),
    )
