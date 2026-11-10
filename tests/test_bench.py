"""Benchmark runner: ``automa bench list`` / ``automa bench run <id>`` and result records.

Only the provider factory is replaced by a scripted ``FakeProvider``; the rest is the
real stack (config, LLMGateway + BudgetGuard + CostLedger, Executive, Orchestrator,
Worker, Verifier, Tool Layer, ProjectStore, benchmark evaluation). No network.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from automa import bench, cli
from automa.benchmarks import (
    BenchmarkKind,
    BenchmarkOutcome,
    BenchmarkResultError,
    BenchmarkRunRecord,
    BenchmarkStop,
    BenchmarkSuite,
    load_result,
    prepare_results_dir,
    write_result,
)
from automa.budget import CostLedger
from automa.config import AutomaConfig, BudgetLimits
from automa.llm import (
    ChatRequest,
    ChatResponse,
    FakeProvider,
    FinishReason,
    Message,
    ProviderError,
    ToolCall,
    Usage,
)
from automa.llm import Role as MsgRole
from automa.orchestrator import StopReason
from automa.plan.models import CheckResult
from automa.project import ProjectStatus, ProjectStore
from automa.project.store import _PROJECT_ID_RE

REPO = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO / "benchmarks"
BID = "cli-word-count"

TOML = """
workspace = "{ws}"
[tiers]
small = "fake-small"
medium = "fake-medium"
large = "fake-large"
[prices."fake-small"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[prices."fake-medium"]
input_per_mtok = 2.0
output_per_mtok = 4.0
[prices."fake-large"]
input_per_mtok = 4.0
output_per_mtok = 8.0
[limits]
project_usd = {project}
daily_usd = 5.0
per_call_usd = {per_call}
"""

QUESTION = "Which Python version should the tool target?"

WC_PY = """\
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Count lines, words and characters.")
    parser.add_argument("path")
    args = parser.parse_args()
    try:
        with open(args.path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"wc.py: {args.path}: no such file", file=sys.stderr)
        return 1
    print(text.count("\\n"), len(text.split()), len(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
WC_BYTES_PY = WC_PY.replace("len(text))", "len(text.encode('utf-8')))")
README = "# wc.py\n\nRun `python3 wc.py <file>`.\n"

PLAN: dict[str, Any] = {
    "goal": "word count tool",
    "tasks": [
        {
            "id": "build",
            "title": "Implement wc.py and its README",
            "role": "developer",
            "acceptance_criteria": [
                {
                    "description": "wc.py exists",
                    "check": {"type": "file_exists", "path": "wc.py"},
                },
                {
                    "description": "README.md exists",
                    "check": {"type": "file_exists", "path": "README.md"},
                },
            ],
        }
    ],
}


def calls(*items: tuple[str, dict[str, Any]]) -> ChatResponse:
    tcs = tuple(
        ToolCall(id=f"c{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(items)
    )
    return ChatResponse(
        model="fake",
        message=Message(role=MsgRole.ASSISTANT, content=None, tool_calls=tcs),
        usage=Usage(input_tokens=200, output_tokens=40),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def build_attempt(code: str = WC_PY) -> list[Any]:
    return [
        calls(
            ("write_file", {"path": "wc.py", "content": code}),
            ("write_file", {"path": "README.md", "content": README}),
        ),
        "wrote wc.py and README.md",
    ]


def text_of(request: ChatRequest) -> str:
    return "\n".join(m.content or "" for m in request.messages)


class Env:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.tmp = tmp_path
        self.ws = tmp_path / "ws"
        self.config = tmp_path / "automa.toml"
        self.suite = tmp_path / "suite"
        self.suite.mkdir()
        shutil.copy(BENCH_DIR / f"{BID}.toml", self.suite / f"{BID}.toml")
        self.results = self.suite / "results"
        self.write_config()
        self.provider = FakeProvider()
        monkeypatch.chdir(tmp_path)
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)

        def factory(config: AutomaConfig) -> FakeProvider:
            return self.provider

        monkeypatch.setattr(cli, "make_provider", factory)

    def write_config(self, *, project: float = 5.0) -> None:
        self.config.write_text(
            TOML.format(ws=self.ws.as_posix(), project=project, per_call=min(0.25, project)),
            encoding="utf-8",
        )

    def set_budget(self, budget: float) -> None:
        path = self.suite / f"{BID}.toml"
        text = path.read_text(encoding="utf-8").replace(
            "budget_usd = 0.5", f"budget_usd = {budget}"
        )
        path.write_text(text, encoding="utf-8")

    def bench(self, *args: str, suite: bool = True) -> Any:
        extra = ["--suite", str(self.suite)] if suite else []
        return CliRunner().invoke(
            cli.app, ["bench", "run", *args, *extra, "--config", str(self.config)]
        )

    @property
    def store(self) -> ProjectStore:
        return ProjectStore.for_workspace(self.ws)

    def projects(self) -> list[str]:
        root = self.ws / "projects"
        return sorted(p.name for p in root.iterdir()) if root.is_dir() else []

    def results_files(self) -> list[Path]:
        d = self.results / BID
        return sorted(d.glob("*.json")) if d.is_dir() else []

    def record(self) -> BenchmarkRunRecord:
        files = self.results_files()
        assert len(files) == 1, files
        return load_result(files[0])

    def spent(self, pid: str) -> int:
        with CostLedger(self.ws / cli.LEDGER_FILENAME) as ledger:
            return ledger.spent_nusd(project_id=pid)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


# --- limits and ids -------------------------------------------------------------


@pytest.mark.parametrize(
    ("project", "daily", "per_call", "budget", "want_project", "want_per_call"),
    [
        (5.0, 2.0, 0.25, 0.5, 0.5, 0.25),  # benchmark tighter
        (0.3, 2.0, 0.25, 0.5, 0.3, 0.25),  # user tighter: user's cap wins
        (5.0, 2.0, 0.25, 0.1, 0.1, 0.1),  # per-call lowered to the project cap
        (1.0, 1.0, 1.0, 5.0, 1.0, 1.0),
    ],
)
def test_benchmark_limits_only_tighten(
    project: float,
    daily: float,
    per_call: float,
    budget: float,
    want_project: float,
    want_per_call: float,
) -> None:
    user = BudgetLimits(project_usd=project, daily_usd=daily, per_call_usd=per_call)
    b = BenchmarkSuite.from_dir(BENCH_DIR).get(BID).model_copy(update={"budget_usd": budget})
    got = bench.benchmark_limits(user, b)
    assert got.project_usd == want_project
    assert got.per_call_usd == want_per_call
    assert got.daily_usd == daily
    assert got.project_usd <= user.project_usd
    assert got.per_call_usd <= user.per_call_usd
    assert got.daily_usd <= user.daily_usd


def test_project_id_is_valid_and_bounded() -> None:
    now = datetime(2026, 11, 10, 14, 30, 5, tzinfo=UTC)
    assert bench.benchmark_project_id(BID, now) == f"bench-20261110-143005-{BID}"
    long_id = "a" * 40 + "-" + "b" * 23
    for bid in (BID, "x", long_id, "a" * 64):
        pid = bench.benchmark_project_id(bid, now)
        assert len(pid) <= 64 and _PROJECT_ID_RE.match(pid), pid
    # timestamps are UTC whatever the clock's zone
    tehran = now.astimezone(timezone(timedelta(hours=3, minutes=30)))
    assert bench.benchmark_project_id(BID, tehran) == bench.benchmark_project_id(BID, now)


def test_every_stop_reason_has_a_benchmark_stop() -> None:
    assert set(bench.STOP_FOR_REASON) == set(StopReason)
    assert {s.value for s in bench.STOP_FOR_REASON.values()} == {r.value for r in StopReason}
    assert BenchmarkStop.PLANNING not in bench.STOP_FOR_REASON.values()


# --- bench list -----------------------------------------------------------------


def test_bench_list_shows_the_real_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(cli.app, ["bench", "list", "--suite", str(BENCH_DIR)])
    assert r.exit_code == 0, r.output
    for bid in ("cli-word-count", "research-brief", "static-landing-page"):
        assert bid in r.output
    assert "$0.50" in r.output


def test_bench_list_bad_suite_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "broken.toml").write_text("id = [", encoding="utf-8")
    r = CliRunner().invoke(cli.app, ["bench", "list", "--suite", str(tmp_path)])
    assert r.exit_code == cli.EXIT_USER
    assert "benchmark suite" in r.output
    r = CliRunner().invoke(cli.app, ["bench", "list", "--suite", str(tmp_path / "nope")])
    assert r.exit_code == cli.EXIT_USER


# --- bench run: success -----------------------------------------------------------


def test_bench_run_passes_and_records(env: Env) -> None:
    env.provider.push(json.dumps({"questions": [QUESTION]}), json.dumps(PLAN), *build_attempt())
    r = env.bench(BID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "result: PASSED (score 1.00, 6/6 criteria)" in r.output
    assert env.provider.remaining == 0

    rec = env.record()
    assert rec.passed and rec.score == 1.0 and len(rec.checks) == 6
    assert rec.stop is BenchmarkStop.COMPLETED
    assert rec.project_status is ProjectStatus.DONE
    assert rec.planned and rec.attempts == 1
    assert rec.tasks_passed == ["build"] and not rec.tasks_failed and not rec.tasks_blocked
    assert rec.budget_usd == 0.5  # the benchmark's budget (config cap is 5)
    assert rec.kind is BenchmarkKind.CLI and rec.benchmark_id == BID
    assert rec.model_calls == len(env.provider.requests) == 4
    assert rec.cost_nusd == env.spent(rec.project_id) > 0
    assert rec.started_at <= rec.finished_at
    assert f"recorded: {env.results_files()[0]}" in r.output

    # a fresh project with the benchmark's goal; the fixed answer reached the planner
    assert env.projects() == [rec.project_id]
    assert rec.project_id.startswith("bench-") and rec.project_id.endswith(BID)
    project = env.store.load(rec.project_id)
    suite = BenchmarkSuite.from_dir(env.suite)
    assert project.goal == suite.get(BID).goal
    clar = env.store.load_clarifications(rec.project_id)
    assert [(c.question, c.answer) for c in clar] == [(QUESTION, suite.get(BID).answer)]
    assert suite.get(BID).answer in text_of(env.provider.requests[1])

    types = [e.type for e in env.store.events(rec.project_id).read()]
    assert types[1] == "benchmark.started" and types[-1] == "benchmark.finished"
    assert any(t.startswith("tool.") for t in types)

    # the result file is plain JSON that round-trips
    data = json.loads(env.results_files()[0].read_text(encoding="utf-8"))
    assert data["passed"] is True and data["stop"] == "completed"


def test_bench_run_without_questions_plans_directly(env: Env) -> None:
    env.provider.push(json.dumps({"questions": []}), json.dumps(PLAN), *build_attempt())
    r = env.bench(BID)
    assert r.exit_code == cli.EXIT_OK, r.output
    rec = env.record()
    assert env.store.load_clarifications(rec.project_id) == []


def test_two_runs_make_two_projects_and_two_records(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    times = iter(datetime(2026, 11, 10, 10, 0, s, tzinfo=UTC) for s in range(0, 60))
    monkeypatch.setattr(bench, "_utc_now", lambda: next(times))
    for _ in range(2):
        env.provider.push(json.dumps({"questions": []}), json.dumps(PLAN), *build_attempt())
        assert env.bench(BID).exit_code == cli.EXIT_OK
    assert len(env.projects()) == 2
    assert len(env.results_files()) == 2


# --- bench run: judged by the benchmark, not the plan ---------------------------------


def test_plan_passes_but_benchmark_criteria_fail(env: Env) -> None:
    # the plan's own criteria (files exist) pass; the benchmark's Unicode check fails
    env.provider.push(json.dumps({"questions": []}), json.dumps(PLAN), *build_attempt(WC_BYTES_PY))
    r = env.bench(BID)
    assert r.exit_code == cli.EXIT_BENCH_FAILED, r.output
    assert "result: FAILED" in r.output and "[FAIL] 3." in r.output
    rec = env.record()
    assert rec.stop is BenchmarkStop.COMPLETED and rec.project_status is ProjectStatus.DONE
    assert not rec.passed
    assert [c.criterion for c in rec.checks if not c.passed] == [3]
    assert rec.score == pytest.approx(5 / 6)


# --- bench run: stops are recorded ------------------------------------------------------


def test_benchmark_budget_caps_the_project(env: Env) -> None:
    env.set_budget(0.000001)  # below even the per-call reservation of the first call
    env.provider.push(json.dumps({"questions": []}), json.dumps(PLAN), *build_attempt())
    r = env.bench(BID)
    assert r.exit_code == cli.EXIT_BENCH_FAILED, r.output
    assert env.provider.requests == []  # refused before any provider call
    rec = env.record()
    assert rec.stop is BenchmarkStop.BUDGET and rec.budget_usd == 0.000001
    assert rec.project_status is ProjectStatus.PAUSED
    assert not rec.planned and not rec.passed and rec.score == 0.0
    assert rec.cost_nusd == 0 and rec.model_calls == 0


def test_user_cap_wins_when_tighter(env: Env) -> None:
    cap = 0.3  # below the benchmark's 0.5
    env.write_config(project=cap)
    env.provider.push(json.dumps({"questions": []}), json.dumps(PLAN), *build_attempt())
    r = env.bench(BID)
    assert r.exit_code == cli.EXIT_OK, r.output
    rec = env.record()
    assert rec.budget_usd == cap
    assert "of $0.30" in r.output


def test_budget_stops_at_every_cap_are_recorded_and_never_overspend(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sweep caps: each run stays within its cap and records how it ended.

    The guard reserves a worst-case cost per call, so where exactly a cap stops the
    run is an implementation detail; the sweep must hit both a budget stop (with the
    first call already spent at some caps) and a complete run. A mid-run pause is
    covered by the provider-error test below and the Orchestrator's own budget tests.
    """
    ticks = iter(range(10_000))
    t0 = datetime(2026, 11, 10, tzinfo=UTC)
    monkeypatch.setattr(bench, "_utc_now", lambda: t0 + timedelta(seconds=next(ticks)))
    seen: set[tuple[BenchmarkStop, bool]] = set()
    for i, cap in enumerate((0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.16, 0.5)):
        env.set_budget(cap)
        prev = FakeProvider()
        env.provider = prev
        prev.push(json.dumps({"questions": []}), json.dumps(PLAN), *build_attempt())
        r = env.bench(BID)
        assert r.exit_code in (cli.EXIT_OK, cli.EXIT_BENCH_FAILED), r.output
        records = env.results_files()
        assert len(records) == i + 1
        rec = load_result(records[-1])
        assert rec.budget_usd == cap
        assert rec.cost_nusd <= round(cap * 1e9)  # never past the benchmark's cap
        assert rec.cost_nusd == env.spent(rec.project_id)
        assert rec.model_calls == len(prev.requests)
        if rec.stop is BenchmarkStop.BUDGET:
            assert rec.project_status is ProjectStatus.PAUSED and not rec.passed
        else:
            assert rec.stop is BenchmarkStop.COMPLETED and rec.passed
        seen.add((rec.stop, rec.planned))
        # restore the original budget line for the next replacement
        path = env.suite / f"{BID}.toml"
        path.write_text(
            path.read_text(encoding="utf-8").replace(f"budget_usd = {cap}", "budget_usd = 0.5"),
            encoding="utf-8",
        )
    assert (BenchmarkStop.BUDGET, False) in seen
    assert (BenchmarkStop.COMPLETED, True) in seen


def test_planning_failure_is_recorded(env: Env) -> None:
    env.provider.push(json.dumps({"questions": []}), "not json", "{}", "[1]")
    r = env.bench(BID)
    assert r.exit_code == cli.EXIT_BENCH_FAILED, r.output
    assert "stop: planning" in r.output
    rec = env.record()
    assert rec.stop is BenchmarkStop.PLANNING and not rec.planned
    assert rec.project_status is ProjectStatus.FAILED
    assert rec.attempts == 0 and not rec.passed
    assert rec.model_calls == 4


def test_provider_error_during_clarification_is_recorded(env: Env) -> None:
    env.provider.push(ProviderError("upstream down"))
    r = env.bench(BID)
    assert r.exit_code == cli.EXIT_BENCH_FAILED, r.output
    rec = env.record()
    assert rec.stop is BenchmarkStop.PROVIDER and "upstream down" in rec.detail
    assert not rec.planned


def test_provider_error_during_run_pauses_and_is_recorded(env: Env) -> None:
    env.provider.push(json.dumps({"questions": []}), json.dumps(PLAN), ProviderError("down"))
    r = env.bench(BID)
    assert r.exit_code == cli.EXIT_BENCH_FAILED, r.output
    rec = env.record()
    assert rec.stop is BenchmarkStop.PROVIDER and rec.planned
    assert rec.project_status is ProjectStatus.PAUSED
    # the paused project can be resumed with the ordinary run command
    env.provider.push(*build_attempt())
    r = CliRunner().invoke(cli.app, ["run", rec.project_id, "--config", str(env.config)])
    assert r.exit_code == cli.EXIT_OK, r.output


def test_ctrl_c_records_nothing(env: Env) -> None:
    env.provider.push(json.dumps({"questions": []}), json.dumps(PLAN), KeyboardInterrupt())
    r = env.bench(BID)
    assert r.exit_code == cli.EXIT_INTERRUPTED, r.output
    assert "no result recorded" in r.output
    assert env.results_files() == []
    (pid,) = env.projects()
    assert env.store.load(pid).status is ProjectStatus.PAUSED


# --- bench run: refused before spending anything ----------------------------------------


def test_unknown_benchmark(env: Env) -> None:
    r = env.bench("nope")
    assert r.exit_code == cli.EXIT_USER
    assert "unknown benchmark" in r.output and BID in r.output
    assert env.projects() == [] and env.provider.requests == []


def test_missing_api_key_with_real_factory(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    from automa.llm.factory import provider_from_config

    monkeypatch.setattr(cli, "make_provider", provider_from_config)
    r = env.bench(BID)
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert env.projects() == []
    assert not (env.ws / cli.LEDGER_FILENAME).exists()


def test_unusable_results_dir_is_refused_before_the_run(env: Env) -> None:
    r = env.bench(BID, "--results", str(env.tmp / "missing" / "results"))
    assert r.exit_code == cli.EXIT_USER, r.output
    target = env.tmp / "elsewhere"
    target.mkdir()
    link = env.tmp / "linked"
    os.symlink(target, link)
    r = env.bench(BID, "--results", str(link))
    assert r.exit_code == cli.EXIT_USER, r.output
    assert env.projects() == [] and env.provider.requests == []


def test_results_option_redirects(env: Env) -> None:
    out = env.tmp / "out"
    env.provider.push(json.dumps({"questions": []}), json.dumps(PLAN), *build_attempt())
    r = env.bench(BID, "--results", str(out))
    assert r.exit_code == cli.EXIT_OK, r.output
    assert len(list((out / BID).glob("*.json"))) == 1
    assert env.results_files() == []


# --- result records -------------------------------------------------------------------


def _record(**over: Any) -> BenchmarkRunRecord:
    t0 = datetime(2026, 11, 10, tzinfo=UTC)
    base: dict[str, Any] = {
        "benchmark_id": BID,
        "title": "t",
        "kind": "cli",
        "project_id": f"bench-x-{BID}",
        "automa_version": "0.1.0",
        "started_at": t0,
        "finished_at": t0 + timedelta(seconds=5),
        "stop": "completed",
        "project_status": "done",
        "planned": True,
        "passed": False,
        "checks": [
            CheckResult(criterion=0, passed=True),
            CheckResult(criterion=1, passed=False),
        ],
        "score": 0.5,
        "budget_usd": 0.5,
        "cost_nusd": 10,
        "model_calls": 2,
        "attempts": 1,
        "tasks_passed": ["build"],
    }
    base.update(over)
    return BenchmarkRunRecord.model_validate(base)


def test_record_is_consistent() -> None:
    assert _record().score == 0.5
    with pytest.raises(ValidationError):
        _record(score=1.0)
    with pytest.raises(ValidationError):
        _record(passed=True)
    with pytest.raises(ValidationError):
        _record(stop="planning")  # planning stop with a plan
    with pytest.raises(ValidationError):
        _record(planned=False, stop="budget")  # tasks ran without a plan
    with pytest.raises(ValidationError):
        _record(tasks_failed=["build"])  # completed with a failed task
    with pytest.raises(ValidationError):
        _record(finished_at=datetime(2026, 11, 9, tzinfo=UTC))
    with pytest.raises(ValidationError):
        _record(budget_usd=6.0)
    with pytest.raises(ValidationError):
        _record(cost_nusd=-1)
    with pytest.raises(ValidationError):
        _record(started_at=datetime(2026, 11, 10))  # naive
    with pytest.raises(ValidationError):
        _record(schema_version=2)
    with pytest.raises(ValidationError):
        _record(extra=1)
    with pytest.raises(ValidationError):
        _record(checks=[CheckResult(criterion=1, passed=False)], score=0.0)


def test_build_matches_outcome() -> None:
    outcome = BenchmarkOutcome(
        benchmark_id=BID,
        passed=True,
        checks=[CheckResult(criterion=0, passed=True)],
    )
    t0 = datetime(2026, 11, 10, tzinfo=UTC)
    rec = BenchmarkRunRecord.build(
        outcome=outcome,
        title="t",
        kind=BenchmarkKind.CLI,
        project_id="p",
        automa_version="0",
        started_at=t0,
        finished_at=t0,
        stop=BenchmarkStop.COMPLETED,
        detail="x" * 5000,
        project_status=ProjectStatus.DONE,
        planned=True,
        budget_usd=1.0,
        cost_nusd=0,
        model_calls=0,
        attempts=1,
        tasks_passed=["a"],
        tasks_failed=[],
        tasks_blocked=[],
    )
    assert rec.passed and rec.score == 1.0 and len(rec.detail) == 1000


def test_write_result_is_write_once_and_round_trips(tmp_path: Path) -> None:
    rec = _record()
    path = write_result(tmp_path / "results", rec)
    assert path == tmp_path / "results" / BID / f"{rec.project_id}.json"
    assert load_result(path) == rec
    with pytest.raises(BenchmarkResultError, match="already exists"):
        write_result(tmp_path / "results", rec)
    assert load_result(path) == rec
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no temp files left


def test_write_result_refuses_symlinks_and_files(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    os.symlink(real, tmp_path / "results")
    with pytest.raises(BenchmarkResultError, match="real directory"):
        write_result(tmp_path / "results", _record())
    other = tmp_path / "r2"
    other.mkdir()
    os.symlink(real, other / BID)
    with pytest.raises(BenchmarkResultError, match="real directory"):
        write_result(other, _record())
    (tmp_path / "afile").write_text("x", encoding="utf-8")
    with pytest.raises(BenchmarkResultError):
        prepare_results_dir(tmp_path / "afile", BID)
    assert list(real.iterdir()) == []


def test_load_result_rejects_bad_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(BenchmarkResultError, match="invalid"):
        load_result(bad)
    good = write_result(tmp_path / "results", _record())
    os.symlink(good, tmp_path / "link.json")
    with pytest.raises(BenchmarkResultError, match="regular"):
        load_result(tmp_path / "link.json")
    with pytest.raises(BenchmarkResultError):
        load_result(tmp_path / "missing.json")


# --- structure ------------------------------------------------------------------------


def test_bench_wraps_the_provider_in_one_guarded_gateway_with_benchmark_limits() -> None:
    import inspect

    tree = ast.parse(inspect.getsource(bench))
    uses = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == "provider" and isinstance(n.ctx, ast.Load)
    ]
    gateways = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "LLMGateway"
    ]
    assert len(gateways) == 1
    (gw,) = gateways
    assert any(isinstance(a, ast.Name) and a.id == "provider" for a in gw.args)
    assert any(k.arg == "guard" for k in gw.keywords)
    assert len(uses) == 1 and uses[0] in gw.args
    guards = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "BudgetGuard"
    ]
    assert len(guards) == 1
    first = guards[0].args[0]
    assert isinstance(first, ast.Name) and first.id == "limits"
    src = inspect.getsource(bench)
    assert "limits = benchmark_limits(config.limits, benchmark)" in src
    assert ".complete(" not in src
