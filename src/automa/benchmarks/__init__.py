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

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BENCHMARK_SUFFIX",
    "MAX_BENCHMARK_ATTEMPTS",
    "MAX_BENCHMARK_BUDGET_USD",
    "MAX_BENCHMARK_FILE_BYTES",
    "Benchmark",
    "BenchmarkError",
    "BenchmarkKind",
    "BenchmarkOutcome",
    "BenchmarkSuite",
    "evaluate_benchmark",
    "load_benchmark",
    "parse_benchmark",
]
