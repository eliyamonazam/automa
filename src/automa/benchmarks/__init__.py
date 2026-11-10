"""Benchmark suite: sample goals with deterministic success criteria (see D26)."""

from automa.benchmarks.evaluate import evaluate_benchmark
from automa.benchmarks.loader import (
    BENCHMARK_SUFFIX,
    MAX_BENCHMARK_FILE_BYTES,
    BenchmarkError,
    BenchmarkSuite,
    load_benchmark,
    parse_benchmark,
)
from automa.benchmarks.models import (
    BENCHMARK_SCHEMA_VERSION,
    MAX_BENCHMARK_ATTEMPTS,
    MAX_BENCHMARK_BUDGET_USD,
    Benchmark,
    BenchmarkKind,
    BenchmarkOutcome,
)
from automa.benchmarks.report import (
    BenchmarkReport,
    BenchmarkSummary,
    CriterionFailure,
    FailureCause,
    ResultProblem,
    ResultScan,
    build_report,
    failure_cause,
    scan_results,
)
from automa.benchmarks.results import (
    RESULT_SCHEMA_VERSION,
    BenchmarkResultError,
    BenchmarkRunRecord,
    BenchmarkStop,
    load_result,
    prepare_results_dir,
    write_result,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BENCHMARK_SUFFIX",
    "MAX_BENCHMARK_ATTEMPTS",
    "MAX_BENCHMARK_BUDGET_USD",
    "MAX_BENCHMARK_FILE_BYTES",
    "RESULT_SCHEMA_VERSION",
    "Benchmark",
    "BenchmarkReport",
    "BenchmarkError",
    "BenchmarkKind",
    "BenchmarkOutcome",
    "BenchmarkResultError",
    "BenchmarkRunRecord",
    "BenchmarkStop",
    "BenchmarkSuite",
    "BenchmarkSummary",
    "CriterionFailure",
    "FailureCause",
    "ResultProblem",
    "ResultScan",
    "build_report",
    "evaluate_benchmark",
    "failure_cause",
    "load_benchmark",
    "load_result",
    "parse_benchmark",
    "prepare_results_dir",
    "scan_results",
    "write_result",
]
