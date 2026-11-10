"""Failure analysis over recorded benchmark runs: ``scan_results``/``build_report`` and
``automa bench report``. No model is involved anywhere: records are built directly.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.benchmarks import (
    BenchmarkResultError,
    BenchmarkRunRecord,
    BenchmarkStop,
    BenchmarkSuite,
    FailureCause,
    build_report,
    failure_cause,
    scan_results,
    write_result,
)
from automa.benchmarks.report import MAX_RESULT_FILES
from automa.plan.models import CheckResult

REPO = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO / "benchmarks"
BID = "cli-word-count"
T0 = datetime(2026, 11, 10, tzinfo=UTC)


def rec(
    n: int,
    *,
    bid: str = BID,
    stop: str = "completed",
    failed: tuple[int, ...] = (),
    criteria: int = 6,
    cost: int = 1_000_000,
    attempts: int = 1,
    version: str = "0.1.0",
) -> BenchmarkRunRecord:
    checks = [CheckResult(criterion=i, passed=i not in failed) for i in range(criteria)]
    passed = not failed
    planned = stop != "planning"
    status = {"completed": "done", "tasks_failed": "failed"}.get(stop, "paused")
    data: dict[str, Any] = {
        "benchmark_id": bid,
        "title": "Title of " + bid,
        "kind": "cli",
        "project_id": f"bench-20261110t{n:06d}z-{bid}",
        "automa_version": version,
        "started_at": T0 + timedelta(minutes=n),
        "finished_at": T0 + timedelta(minutes=n, seconds=30),
        "stop": stop,
        "project_status": status,
        "planned": planned,
        "passed": passed,
        "checks": checks,
        "score": sum(c.passed for c in checks) / len(checks),
        "budget_usd": 0.5,
        "cost_nusd": cost,
        "model_calls": 3,
        "attempts": attempts if planned else 0,
        "tasks_passed": ["build"] if stop == "completed" else [],
        "tasks_failed": ["build"] if stop == "tasks_failed" else [],
    }
    return BenchmarkRunRecord.model_validate(data)


@pytest.fixture
def suite_dir(tmp_path: Path) -> Path:
    d = tmp_path / "suite"
    d.mkdir()
    for f in BENCH_DIR.glob("*.toml"):
        shutil.copy(f, d / f.name)
    return d


def tree(root: Path) -> list[tuple[str, int, int]]:
    out = []
    for dirpath, dirs, files in os.walk(root):
        for name in sorted(dirs + files):
            p = Path(dirpath) / name
            st = os.lstat(p)
            out.append((str(p.relative_to(root)), st.st_size, st.st_mtime_ns))
    return sorted(out)


def report_cli(*args: str) -> Any:
    return CliRunner().invoke(cli.app, ["bench", "report", *args])


# --- failure_cause --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stop", "want"),
    [
        ("completed", FailureCause.CRITERIA_MISSED),
        ("tasks_failed", FailureCause.TASKS_FAILED),
        ("planning", FailureCause.PLANNING),
        ("budget", FailureCause.BUDGET),
        ("provider", FailureCause.PROVIDER),
        ("approval", FailureCause.OTHER_STOP),
        ("verification", FailureCause.OTHER_STOP),
        ("tool_layer", FailureCause.OTHER_STOP),
        ("config", FailureCause.OTHER_STOP),
    ],
)
def test_failure_cause_of_each_stop(stop: str, want: FailureCause) -> None:
    assert failure_cause(rec(1, stop=stop, failed=(0,))) is want


def test_every_stop_has_a_cause() -> None:
    for stop in BenchmarkStop:
        assert isinstance(failure_cause(rec(1, stop=stop.value, failed=(0,))), FailureCause)


def test_passed_run_has_no_cause() -> None:
    assert failure_cause(rec(1)) is None


# --- build_report ----------------------------------------------------------------------


def test_empty_report() -> None:
    r = build_report([])
    assert r.benchmarks == [] and r.total_runs == 0 and r.total_cost_nusd == 0


def test_aggregates_per_benchmark(suite_dir: Path) -> None:
    suite = BenchmarkSuite.from_dir(suite_dir)
    records = [
        rec(1, cost=100, attempts=1),
        rec(2, failed=(3,), cost=200, attempts=2),  # completed but criterion 3 missed
        rec(3, stop="budget", failed=(1, 3, 4), cost=300, attempts=1, version="0.2.0"),
        rec(4, stop="tasks_failed", failed=(1,), cost=400, attempts=4),
        rec(5, bid="research-brief", criteria=5, cost=50),
    ]
    r = build_report(records, suite=suite)
    assert [s.benchmark_id for s in r.benchmarks] == ["cli-word-count", "research-brief"]
    assert r.total_runs == 5 and r.total_passed == 2 and r.total_cost_nusd == 1050

    s = r.benchmarks[0]
    assert s.in_suite and s.title == suite.get(BID).title
    assert (s.runs, s.passed_runs) == (4, 1)
    assert s.pass_rate == pytest.approx(0.25)
    assert s.mean_score == pytest.approx((1 + 5 / 6 + 3 / 6 + 5 / 6) / 4)
    assert s.total_cost_nusd == 1000 and s.mean_cost_nusd == 250
    assert s.mean_attempts == pytest.approx(2.0)
    assert s.stops == {"budget": 1, "completed": 2, "tasks_failed": 1}
    assert s.failure_causes == {"budget": 1, "criteria_missed": 1, "tasks_failed": 1}
    assert s.versions == ["0.1.0", "0.2.0"]
    assert s.first_run_at == T0 + timedelta(minutes=1)
    assert s.last_run_at == T0 + timedelta(minutes=4, seconds=30)
    assert s.last_project_id.endswith("000004z-cli-word-count") and not s.last_passed

    # criterion 3 failed in a completed run: listed first, even though 1 fails as often
    got = [(c.criterion, c.failed_runs, c.failed_in_completed_runs) for c in s.criteria_failures]
    assert got == [(3, 2, 1), (1, 2, 0), (4, 1, 0)]
    descs = suite.get(BID).success_criteria
    assert s.criteria_failures[0].description == descs[3].description


def test_benchmark_not_in_suite_keeps_recorded_title() -> None:
    r = build_report([rec(1, bid="gone-bench", failed=(7,), criteria=8)], suite=None)
    s = r.benchmarks[0]
    assert not s.in_suite and s.title == "Title of gone-bench"
    assert s.criteria_failures[0].description == "(unknown criterion)"


def test_criterion_index_beyond_suite_is_unknown(suite_dir: Path) -> None:
    suite = BenchmarkSuite.from_dir(suite_dir)
    r = build_report([rec(1, failed=(9,), criteria=10)], suite=suite)
    assert r.benchmarks[0].criteria_failures[0].description == "(unknown criterion)"


def test_report_does_not_change_input(suite_dir: Path) -> None:
    records = [rec(2), rec(1, failed=(0,))]
    before = [r.model_dump() for r in records]
    build_report(records, suite=BenchmarkSuite.from_dir(suite_dir))
    assert [r.model_dump() for r in records] == before


# --- scan_results ------------------------------------------------------------------------


def test_scan_missing_dir_is_empty_and_creates_nothing(tmp_path: Path) -> None:
    scan = scan_results(tmp_path / "nope")
    assert scan.records == [] and scan.problems == []
    assert not (tmp_path / "nope").exists()


def test_scan_reads_written_results_in_order(tmp_path: Path) -> None:
    res = tmp_path / "results"
    for r in (rec(3), rec(1, failed=(2,)), rec(2, bid="research-brief", criteria=5)):
        write_result(res, r)
    (res / ".gitkeep").write_text("", encoding="utf-8")
    (res / BID / ".x.json.tmp").write_text("partial", encoding="utf-8")  # interrupted write
    scan = scan_results(res)
    assert scan.problems == []
    assert [(r.benchmark_id, r.started_at.minute) for r in scan.records] == [
        (BID, 1),
        (BID, 3),
        ("research-brief", 2),
    ]


def test_scan_filter_by_benchmark(tmp_path: Path) -> None:
    res = tmp_path / "results"
    write_result(res, rec(1))
    write_result(res, rec(2, bid="research-brief", criteria=5))
    scan = scan_results(res, benchmark_id="research-brief")
    assert [r.benchmark_id for r in scan.records] == ["research-brief"]


def test_scan_lists_untrusted_files_as_problems(tmp_path: Path) -> None:
    res = tmp_path / "results"
    write_result(res, rec(1))
    sub = res / BID
    (sub / "broken.json").write_text("{not json", encoding="utf-8")
    (sub / "notes.txt").write_text("x", encoding="utf-8")
    # a valid record placed under the wrong benchmark, and under the wrong file name
    other = res / "research-brief"
    other.mkdir()
    shutil.copy(sub / f"{rec(1).project_id}.json", other / "a.json")
    shutil.copy(sub / f"{rec(1).project_id}.json", sub / "renamed.json")
    # a symlinked result file and a symlinked benchmark directory
    os.symlink(sub / f"{rec(1).project_id}.json", sub / "link.json")
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, res / "linked-bench")
    (res / "stray.json").write_text("{}", encoding="utf-8")
    (res / "Bad_Id").mkdir()

    scan = scan_results(res)
    assert len(scan.records) == 1
    reasons = {p.path: p.reason for p in scan.problems}
    assert set(reasons) == {
        f"{BID}/broken.json",
        f"{BID}/notes.txt",
        f"{BID}/renamed.json",
        f"{BID}/link.json",
        "research-brief/a.json",
        "linked-bench",
        "stray.json",
        "Bad_Id",
    }
    assert "invalid result" in reasons[f"{BID}/broken.json"]
    assert "not a regular file" in reasons[f"{BID}/link.json"]
    assert "records benchmark" in reasons["research-brief/a.json"]
    assert "records project" in reasons[f"{BID}/renamed.json"]
    assert reasons["linked-bench"] == "symlinked directory"


def test_scan_refuses_symlinked_or_file_results_dir(tmp_path: Path) -> None:
    real = tmp_path / "real"
    write_result(real, rec(1))
    os.symlink(real, tmp_path / "link")
    with pytest.raises(BenchmarkResultError, match="not a real directory"):
        scan_results(tmp_path / "link")
    (tmp_path / "file").write_text("x", encoding="utf-8")
    with pytest.raises(BenchmarkResultError, match="not a real directory"):
        scan_results(tmp_path / "file")


def test_scan_is_read_only(tmp_path: Path) -> None:
    res = tmp_path / "results"
    write_result(res, rec(1))
    (res / BID / "broken.json").write_text("{", encoding="utf-8")
    before = tree(tmp_path)
    scan_results(res)
    assert tree(tmp_path) == before


def test_scan_caps_files_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from automa.benchmarks import report as report_mod

    monkeypatch.setattr(report_mod, "MAX_RESULT_FILES", 2)
    res = tmp_path / "results"
    for n in range(1, 4):
        write_result(res, rec(n))
    scan = scan_results(res)
    assert len(scan.records) == 2
    assert len(scan.problems) == 1 and "not read" in scan.problems[0].reason
    assert MAX_RESULT_FILES >= 1000


# --- automa bench report ------------------------------------------------------------------


@pytest.fixture
def no_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The report needs no config, no key and must never build a provider."""
    for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
        monkeypatch.delenv(var, raising=False)

    def boom(config: Any) -> Any:
        raise AssertionError("bench report must not build a provider")

    monkeypatch.setattr(cli, "make_provider", boom)
    monkeypatch.chdir(tmp_path)


def test_cli_report_no_results(suite_dir: Path, no_model: None) -> None:
    r = report_cli("--suite", str(suite_dir))
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "no benchmark results recorded yet" in r.output
    assert not (suite_dir / "results").exists()


def test_cli_report_text(suite_dir: Path, no_model: None) -> None:
    res = suite_dir / "results"
    write_result(res, rec(1))
    write_result(res, rec(2, failed=(3,), cost=3_000_000_000))
    write_result(res, rec(3, stop="budget", failed=(3, 4), attempts=3))
    before = tree(suite_dir)
    r = report_cli("--suite", str(suite_dir))
    assert r.exit_code == cli.EXIT_OK, r.output
    out = r.output
    title = BenchmarkSuite.from_dir(suite_dir).get(BID).title
    assert f"{BID} — {title}" in out
    assert "runs: 3, passed: 1 (33%)" in out
    assert "mean attempts 1.7" in out
    assert "cost: total $3.0020, mean $1.0007 per run" in out
    assert "stops: budget 1, completed 2" in out
    assert "failure causes: budget 1, criteria_missed 1" in out
    assert "criterion 3 failed in 2/3 run(s) (1 completed)" in out
    assert out.index("criterion 3 failed") < out.index("criterion 4 failed")
    assert "last: bench-20261110t000003z-cli-word-count FAILED" in out
    assert "total: 3 run(s), 1 passed, cost $3.0020" in out
    assert tree(suite_dir) == before


def test_cli_report_json(suite_dir: Path, no_model: None) -> None:
    res = suite_dir / "results"
    write_result(res, rec(1, failed=(0,)))
    r = report_cli("--suite", str(suite_dir), "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    data = json.loads(r.output)
    assert data["total_runs"] == 1 and data["total_passed"] == 0
    s = data["benchmarks"][0]
    assert s["failure_causes"] == {"criteria_missed": 1}
    assert s["criteria_failures"][0]["criterion"] == 0
    assert data["problems"] == []


def test_cli_report_filter_and_results_option(
    suite_dir: Path, tmp_path: Path, no_model: None
) -> None:
    res = tmp_path / "elsewhere"
    write_result(res, rec(1))
    write_result(res, rec(2, bid="research-brief", criteria=5))
    r = report_cli("--suite", str(suite_dir), "--results", str(res), "--benchmark", BID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "research-brief" not in r.output and "total: 1 run(s)" in r.output


def test_cli_report_problems_exit_1_but_still_report(suite_dir: Path, no_model: None) -> None:
    res = suite_dir / "results"
    write_result(res, rec(1))
    (res / BID / "broken.json").write_text("{", encoding="utf-8")
    r = report_cli("--suite", str(suite_dir))
    assert r.exit_code == cli.EXIT_USER
    assert f"problem: {BID}/broken.json" in r.output
    assert "total: 1 run(s)" in r.output


def test_cli_report_bad_suite_fails_closed_without_results_option(
    tmp_path: Path, no_model: None
) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "x.toml").write_text("not = [valid", encoding="utf-8")
    r = report_cli("--suite", str(bad))
    assert r.exit_code == cli.EXIT_USER and "benchmark suite" in r.output


def test_cli_report_without_suite_when_results_given(tmp_path: Path, no_model: None) -> None:
    res = tmp_path / "res"
    write_result(res, rec(1, failed=(2,)))
    r = report_cli("--suite", str(tmp_path / "missing"), "--results", str(res))
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "[not in suite]" in r.output and "(unknown criterion)" in r.output


def test_cli_report_symlinked_results_dir(tmp_path: Path, no_model: None) -> None:
    real = tmp_path / "real"
    write_result(real, rec(1))
    os.symlink(real, tmp_path / "link")
    r = report_cli("--suite", str(tmp_path / "missing"), "--results", str(tmp_path / "link"))
    assert r.exit_code == cli.EXIT_USER and "not a real directory" in r.output


def test_cli_has_no_shadowed_top_level_functions() -> None:
    """A second ``def`` with the same name silently replaced ``automa run``'s output once."""
    import ast

    tree = ast.parse((REPO / "src" / "automa" / "cli.py").read_text(encoding="utf-8"))
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert len(names) == len(set(names)), sorted(n for n in names if names.count(n) > 1)
