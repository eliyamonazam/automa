"""Judging a benchmark: run its success criteria on a project's ``files/``.

The checks go through the same path as the Verifier's deterministic checks: a
``ToolExecutor`` with the fixed ``verifier_policy()`` (read files, list
directories, run check commands; never write), so a benchmark check gets the
same confinement, clean environment, timeouts and ``tool.*`` events. No model
is called here.
"""

from __future__ import annotations

from pathlib import Path

from automa.agents.verifier import run_criteria_checks, verifier_policy
from automa.benchmarks.models import Benchmark, BenchmarkOutcome
from automa.events import EventSink
from automa.tools import DockerSandbox, default_tools
from automa.tools.executor import ToolExecutor
from automa.tools.workspace import Workspace


def evaluate_benchmark(
    benchmark: Benchmark,
    files_dir: Path,
    events: EventSink,
    *,
    project_id: str | None = None,
    sandbox: DockerSandbox | None = None,
) -> BenchmarkOutcome:
    """Run every success criterion of ``benchmark`` against ``files_dir``.

    ``files_dir`` is a project's ``files/`` directory (must be a real directory).
    ``project_id`` labels the emitted events (defaults to the benchmark id).
    ``sandbox`` runs the check commands in Docker (the same ``[sandbox]`` as the run).
    """
    executor = ToolExecutor(
        default_tools(sandbox),
        verifier_policy(),
        Workspace(files_dir),
        events,
        project_id or benchmark.id,
    )
    checks = run_criteria_checks(benchmark.success_criteria, executor)
    return BenchmarkOutcome(
        benchmark_id=benchmark.id,
        passed=all(c.passed for c in checks),
        checks=checks,
    )
