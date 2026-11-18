"""Comparative benchmark run for a candidate skill: ``run_skill_comparison`` (D105).

Real stack (config, LLMGateway + BudgetGuard + CostLedger, Executive, Orchestrator,
Worker, Verifier, Tool Layer, ProjectStore, benchmark evaluation) with a scripted
``FakeProvider`` only. No network, no real model API.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from automa import skill_comparison as sc
from automa.benchmarks import Benchmark, BenchmarkStop, BenchmarkSuite
from automa.budget import CostLedger
from automa.budget.ledger import usd_to_nusd
from automa.config import AutomaConfig, ConfigError, Tier, load_config
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
from automa.llm.gateway import CallContext, CallEstimate
from automa.llm.types import LLMResult
from automa.project import ProjectStore
from automa.project.store import _PROJECT_ID_RE
from automa.roles import RoleRegistry
from automa.skill_comparison import (
    ComparisonSide,
    SkillComparisonError,
    comparison_project_id,
    load_candidate_skill,
    plan_skill_comparison,
    run_skill_comparison,
)
from automa.skill_gate import GateReason, SkillGatePolicy
from automa.skill_variants import skillset_fingerprint
from automa.skills import Skill, SkillRegistry, load_skills
from automa.skills.loader import BUILTIN_SKILLS_DIR
from automa.tools import default_tools

REPO = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO / "benchmarks"
BID = "cli-word-count"
CANDIDATE_BODY = "CANDIDATE-MARKER: count characters, never bytes."

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
project_usd = 5.0
daily_usd = {daily}
per_call_usd = 0.25
"""

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
                {"description": "wc.py exists", "check": {"type": "file_exists", "path": "wc.py"}},
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


def one_run(code: str = WC_PY) -> list[Any]:
    """The four provider replies of one benchmark run (clarify, plan, two worker turns)."""
    return [
        json.dumps({"questions": []}),
        json.dumps(PLAN),
        calls(
            ("write_file", {"path": "wc.py", "content": code}),
            ("write_file", {"path": "README.md", "content": README}),
        ),
        "wrote wc.py and README.md",
    ]


def text_of(request: ChatRequest) -> str:
    return "\n".join(m.content or "" for m in request.messages)


def spend(ledger: CostLedger, usd: float) -> None:
    """Record spending of another project today (as an earlier/concurrent run would)."""
    est = CallEstimate(
        tier=Tier.SMALL,
        model="fake-small",
        max_input_tokens=1,
        max_output_tokens=1,
        max_cost_usd=usd,
        context=CallContext(project_id="other", task_id=None),
    )
    res = LLMResult(
        tier=Tier.SMALL,
        response=ChatResponse(
            model="fake-small",
            message=Message.assistant("ok"),
            usage=Usage(input_tokens=1, output_tokens=1),
        ),
        cost_usd=usd,
    )
    ledger.record(est, res)


def candidate_skill(name: str = "unicode-counts", body: str = CANDIDATE_BODY) -> Skill:
    return Skill(name=name, description="Count characters, not bytes.", body=body)


def write_staging(root: Path, name: str = "unicode-counts", body: str = CANDIDATE_BODY) -> Path:
    staging = root / "staging"
    (staging / name).mkdir(parents=True)
    (staging / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Count characters, not bytes.\n---\n{body}\n",
        encoding="utf-8",
    )
    return staging


class Env:
    def __init__(self, tmp_path: Path, *, daily: float = 5.0):
        self.tmp = tmp_path
        self.ws = tmp_path / "ws"
        self.config_path = tmp_path / "automa.toml"
        self.config_path.write_text(
            TOML.format(ws=self.ws.as_posix(), daily=daily), encoding="utf-8"
        )
        self.config: AutomaConfig = load_config(self.config_path)
        self.provider = FakeProvider()
        self.store = ProjectStore.for_workspace(self.ws)
        self.ledger = CostLedger(self.ws / "ledger.sqlite3")
        self.skills = load_skills(self.ws)
        self.roles = RoleRegistry.builtin(tools=default_tools(), skills=self.skills)
        self.bench = BenchmarkSuite.from_dir(BENCH_DIR).get(BID)
        self._t = datetime(2026, 11, 18, 12, 0, 0, tzinfo=UTC)

    def clock(self) -> datetime:
        self._t += timedelta(seconds=1)
        return self._t

    def run(self, **over: Any) -> sc.SkillComparisonResult:
        kwargs: dict[str, Any] = {
            "candidate": candidate_skill(),
            "role": "developer",
            "runs": 1,
            "max_total_usd": 2.0,
            "config": self.config,
            "provider": self.provider,
            "store": self.store,
            "ledger": self.ledger,
            "roles": self.roles,
            "skills": self.skills,
            "policy": SkillGatePolicy(min_runs=1),
            "clock": self.clock,
        }
        kwargs.update(over)
        benchmarks = kwargs.pop("benchmarks", [self.bench])
        return run_skill_comparison(benchmarks, **kwargs)

    def plan(self, **over: Any) -> sc.SkillComparisonPlan:
        kwargs: dict[str, Any] = {
            "candidate": candidate_skill(),
            "role": "developer",
            "runs": 1,
            "max_total_usd": 2.0,
            "config": self.config,
            "store": self.store,
            "ledger": self.ledger,
            "roles": self.roles,
            "skills": self.skills,
            "policy": SkillGatePolicy(min_runs=1),
            "now": datetime(2026, 11, 18, 12, 0, 0, tzinfo=UTC),
        }
        kwargs.update(over)
        benchmarks = kwargs.pop("benchmarks", [self.bench])
        return plan_skill_comparison(benchmarks, **kwargs)[0]

    def projects(self) -> list[str]:
        return self.store.list_ids()


@pytest.fixture
def env(tmp_path: Path) -> Any:
    e = Env(tmp_path)
    yield e
    e.ledger.close()


def bench_with(b: Benchmark, **update: Any) -> Benchmark:
    return b.model_copy(update=update)


# --- candidate staging -------------------------------------------------------------


def test_load_candidate_from_staging(tmp_path: Path) -> None:
    staging = write_staging(tmp_path)
    got = load_candidate_skill(staging, workspace=tmp_path / "ws")
    assert got.name == "unicode-counts" and CANDIDATE_BODY in got.body


def test_staging_missing_or_symlink_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SkillComparisonError, match="not found"):
        load_candidate_skill(tmp_path / "nope", workspace=tmp_path / "ws")
    staging = write_staging(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(staging, target_is_directory=True)
    with pytest.raises(SkillComparisonError, match="not found"):
        load_candidate_skill(link, workspace=tmp_path / "ws")
    f = tmp_path / "file"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(SkillComparisonError, match="not found"):
        load_candidate_skill(f, workspace=tmp_path / "ws")


@pytest.mark.parametrize("where", ["skills", "skills/inner", "."])
def test_staging_overlapping_workspace_skills_is_rejected(tmp_path: Path, where: str) -> None:
    ws = tmp_path / "ws"
    target = ws / where
    write_staging(target)  # creates target/staging/...
    staging = target / "staging" if where != "." else ws
    if where == ".":
        # the workspace itself contains skills/: overlapping in the other direction
        (ws / "skills").mkdir()
    with pytest.raises(SkillComparisonError, match="separate from the workspace skills"):
        load_candidate_skill(staging, workspace=ws)


def test_staging_inside_builtin_skills_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SkillComparisonError, match="built-in skills"):
        load_candidate_skill(BUILTIN_SKILLS_DIR, workspace=tmp_path / "ws")


def test_staging_reached_through_a_symlinked_parent_is_still_checked(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    write_staging(ws / "skills")
    alias = tmp_path / "alias"
    alias.symlink_to(ws / "skills", target_is_directory=True)
    with pytest.raises(SkillComparisonError, match="separate"):
        load_candidate_skill(alias / "staging", workspace=ws)


def test_staging_must_hold_exactly_one_entry(tmp_path: Path) -> None:
    staging = write_staging(tmp_path)
    (staging / "README.md").write_text("notes", encoding="utf-8")
    with pytest.raises(SkillComparisonError, match="exactly one"):
        load_candidate_skill(staging, workspace=tmp_path / "ws")
    (staging / "README.md").unlink()
    (staging / ".hidden").write_text("x", encoding="utf-8")
    with pytest.raises(SkillComparisonError, match="exactly one"):
        load_candidate_skill(staging, workspace=tmp_path / "ws")
    (staging / ".hidden").unlink()
    write_staging(tmp_path / "two", name="other")
    os.rename(tmp_path / "two" / "staging" / "other", staging / "other")
    with pytest.raises(SkillComparisonError, match=r"found 2 entries"):
        load_candidate_skill(staging, workspace=tmp_path / "ws")


def test_empty_staging_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "staging").mkdir()
    with pytest.raises(SkillComparisonError, match="found 0 entries"):
        load_candidate_skill(tmp_path / "staging", workspace=tmp_path / "ws")


def test_candidate_entry_must_be_a_real_directory(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "SKILL.md").write_text("---\nname: x\ndescription: y\n---\nz\n", encoding="utf-8")
    with pytest.raises(SkillComparisonError, match="must be a directory"):
        load_candidate_skill(staging, workspace=tmp_path / "ws")
    (staging / "SKILL.md").unlink()
    real = write_staging(tmp_path / "elsewhere") / "unicode-counts"
    (staging / "unicode-counts").symlink_to(real, target_is_directory=True)
    with pytest.raises(SkillComparisonError, match="must be a directory"):
        load_candidate_skill(staging, workspace=tmp_path / "ws")


def test_candidate_dir_must_hold_only_skill_md(tmp_path: Path) -> None:
    staging = write_staging(tmp_path)
    (staging / "unicode-counts" / "script.sh").write_text("rm -rf /", encoding="utf-8")
    with pytest.raises(SkillComparisonError, match="only SKILL.md"):
        load_candidate_skill(staging, workspace=tmp_path / "ws")


def test_candidate_dir_without_skill_md(tmp_path: Path) -> None:
    (tmp_path / "staging" / "x").mkdir(parents=True)
    with pytest.raises(SkillComparisonError, match="only SKILL.md"):
        load_candidate_skill(tmp_path / "staging", workspace=tmp_path / "ws")


@pytest.mark.parametrize(
    "text",
    [
        "no frontmatter\n",
        "---\nname: other-name\ndescription: d\n---\nbody\n",  # name != directory
        "---\nname: unicode-counts\ndescription: d\nallowed-tools: run_command\n---\nbody\n",
        "---\nname: unicode-counts\ndescription: d\n---\n\n",  # empty body
    ],
)
def test_invalid_candidate_file_is_rejected(tmp_path: Path, text: str) -> None:
    staging = write_staging(tmp_path)
    (staging / "unicode-counts" / "SKILL.md").write_text(text, encoding="utf-8")
    with pytest.raises(SkillComparisonError, match="invalid candidate skill"):
        load_candidate_skill(staging, workspace=tmp_path / "ws")


def test_symlinked_skill_md_is_rejected(tmp_path: Path) -> None:
    staging = write_staging(tmp_path)
    target = tmp_path / "real.md"
    skill_md = staging / "unicode-counts" / "SKILL.md"
    target.write_text(skill_md.read_text(encoding="utf-8"), encoding="utf-8")
    skill_md.unlink()
    skill_md.symlink_to(target)
    with pytest.raises(SkillComparisonError, match="invalid candidate skill"):
        load_candidate_skill(staging, workspace=tmp_path / "ws")


def test_loading_writes_nothing(tmp_path: Path) -> None:
    staging = write_staging(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    load_candidate_skill(staging, workspace=tmp_path / "ws")
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert not (tmp_path / "ws").exists()


# --- planning: checks before anything is spent -----------------------------------------


def test_plan_lays_out_interleaved_runs_with_unique_ids(env: Env) -> None:
    plan = env.plan(runs=3, policy=SkillGatePolicy(min_runs=3), max_total_usd=3.0)
    sides = [(r.side, r.index) for r in plan.runs]
    assert sides == [
        (ComparisonSide.BASELINE, 1),
        (ComparisonSide.CANDIDATE, 1),
        (ComparisonSide.BASELINE, 2),
        (ComparisonSide.CANDIDATE, 2),
        (ComparisonSide.BASELINE, 3),
        (ComparisonSide.CANDIDATE, 3),
    ]
    pids = [r.project_id for r in plan.runs]
    assert len(set(pids)) == len(pids)
    assert all(_PROJECT_ID_RE.fullmatch(p) and len(p) <= 64 for p in pids)
    cap = usd_to_nusd(0.5)  # the benchmark's budget (user's project cap is 5)
    assert all(r.max_cost_nusd == cap for r in plan.runs)
    assert plan.max_total_nusd == 6 * cap
    assert plan.max_total_usd == pytest.approx(3.0)
    assert plan.limit_nusd == usd_to_nusd(3.0) == plan.max_total_nusd  # exactly at the cap


def test_plan_fingerprints_match_the_sides(env: Env) -> None:
    plan = env.plan()
    assert plan.baseline_fingerprint == skillset_fingerprint(env.roles, env.skills)
    assert plan.candidate_fingerprint != plan.baseline_fingerprint
    assert plan.candidate == "unicode-counts" and plan.role == "developer"
    assert plan.benchmarks == (BID,)


def test_project_ids_are_bounded_and_distinct() -> None:
    now = datetime(2026, 11, 18, 12, 0, 0, tzinfo=UTC)
    long_id = "a" * 64
    seen = set()
    for bi in (1, 2, 99):
        for side in ComparisonSide:
            for run in (1, 20):
                pid = comparison_project_id(now, bi, side, run, long_id)
                assert _PROJECT_ID_RE.fullmatch(pid) and len(pid) <= 64
                seen.add(pid)
    assert len(seen) == 3 * 2 * 2
    assert comparison_project_id(now, 1, ComparisonSide.CANDIDATE, 2, BID) == (
        f"skc-20261118-120000-01c02-{BID}"
    )


def test_total_worst_case_over_comparison_cap_is_refused(env: Env) -> None:
    # 2 runs x $0.50 = $1.00 > $0.99
    with pytest.raises(SkillComparisonError, match="exceeds the comparison cap"):
        env.run(max_total_usd=0.99)
    assert env.provider.requests == [] and env.projects() == []
    env.plan(max_total_usd=1.0)  # exactly at the cap is allowed


def test_total_worst_case_over_daily_left_is_refused(tmp_path: Path) -> None:
    e = Env(tmp_path, daily=0.9)
    try:
        with pytest.raises(SkillComparisonError, match="daily cap"):
            e.run()
        assert e.provider.requests == [] and e.projects() == []
    finally:
        e.ledger.close()


def test_daily_spending_already_in_the_ledger_counts(env: Env) -> None:
    # spend $4.20 today from an earlier run: $0.80 left < $1.00 worst case
    spend(env.ledger, 4.2)
    with pytest.raises(SkillComparisonError, match="daily cap"):
        env.run()
    assert env.provider.requests == []


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "1"])
def test_bad_comparison_cap(env: Env, value: Any) -> None:
    with pytest.raises(SkillComparisonError, match="max_total_usd"):
        env.plan(max_total_usd=value)


@pytest.mark.parametrize("runs", [0, 2, 21, True, 1.0])
def test_runs_out_of_range(env: Env, runs: Any) -> None:
    with pytest.raises(SkillComparisonError, match="runs must"):
        env.plan(runs=runs, policy=SkillGatePolicy(min_runs=3))


def test_runs_below_gate_min_runs_is_refused_before_spending(env: Env) -> None:
    # default gate: min_runs = 3; one run per side could never be accepted
    with pytest.raises(SkillComparisonError, match="min_runs"):
        env.run(policy=None)
    assert env.provider.requests == []


def test_no_benchmark_or_duplicate_benchmark(env: Env) -> None:
    with pytest.raises(SkillComparisonError, match="at least one"):
        env.plan(benchmarks=[])
    with pytest.raises(SkillComparisonError, match="more than once"):
        env.plan(benchmarks=[env.bench, env.bench])
    with pytest.raises(SkillComparisonError, match="at least one"):
        env.plan(benchmarks=["cli-word-count"])


def test_too_many_runs_in_total(env: Env) -> None:
    benches = [bench_with(env.bench, id=f"b{i}") for i in range(6)]
    with pytest.raises(SkillComparisonError, match="too many runs"):
        env.plan(benchmarks=benches, runs=20, max_total_usd=1000.0)


def test_unknown_role_or_clashing_name(env: Env) -> None:
    with pytest.raises(SkillComparisonError, match="unknown role"):
        env.plan(role="nobody")
    existing = next(iter(env.skills), None)
    if existing is not None:
        with pytest.raises(SkillComparisonError, match="existing skill"):
            env.plan(candidate=candidate_skill(name=existing.name))
    clash = SkillRegistry([candidate_skill()])
    roles = RoleRegistry.builtin(tools=default_tools(), skills=clash)
    with pytest.raises(SkillComparisonError, match="existing skill"):
        env.plan(skills=clash, roles=roles)


def test_image_benchmark_without_images_is_refused_before_spending(env: Env) -> None:
    img = bench_with(env.bench, id="needs-img", needs_images=True)
    with pytest.raises(SkillComparisonError, match="need image generation"):
        env.run(benchmarks=[env.bench, img], max_total_usd=10.0)
    assert env.provider.requests == [] and env.projects() == []


def test_image_provider_without_images_config_is_config_error(env: Env) -> None:
    class NoImages:
        def generate(self, request: Any) -> Any:  # pragma: no cover - never called
            raise AssertionError

    with pytest.raises(ConfigError, match=r"\[images\]"):
        env.run(image_provider=NoImages())
    assert env.provider.requests == [] and env.projects() == []


def test_existing_project_id_is_refused(env: Env) -> None:
    first = env.plan().runs[1].project_id
    env.store.create("taken", project_id=first)
    with pytest.raises(SkillComparisonError, match="already exists"):
        env.plan()


def test_planning_spends_and_writes_nothing(env: Env) -> None:
    env.plan()
    assert env.projects() == []
    assert env.ledger.entries() == []
    assert env.provider.requests == []


# --- running ------------------------------------------------------------------------------


def test_candidate_better_is_accepted(env: Env) -> None:
    # baseline writes the byte-counting version (fails a criterion); candidate is correct
    env.provider.push(*one_run(WC_BYTES_PY), *one_run(WC_PY))
    seen: list[tuple[ComparisonSide, str]] = []
    result = env.run(on_record=lambda side, rec: seen.append((side, rec.project_id)))
    assert env.provider.remaining == 0
    assert result.aborted is None
    assert result.decision is not None and result.accepted
    assert result.decision.reasons == []
    assert [r.passed for r in result.baseline] == [False]
    assert [r.passed for r in result.candidate] == [True]
    assert seen == [
        (ComparisonSide.BASELINE, result.baseline[0].project_id),
        (ComparisonSide.CANDIDATE, result.candidate[0].project_id),
    ]
    assert [r.project_id for r in result.plan.runs] == [p for _, p in seen]
    assert result.spent_nusd == sum(r.cost_nusd for r in (*result.baseline, *result.candidate))
    assert result.spent_nusd == env.ledger.spent_nusd() > 0
    assert sorted(env.projects()) == sorted(p for _, p in seen)


def test_candidate_text_reaches_only_candidate_workers(env: Env) -> None:
    env.provider.push(*one_run(), *one_run())
    result = env.run()
    reqs = env.provider.requests
    assert len(reqs) == 8
    baseline_texts = [text_of(r) for r in reqs[:4]]
    candidate_texts = [text_of(r) for r in reqs[4:]]
    assert not any(CANDIDATE_BODY in t for t in baseline_texts)
    # the worker turns (3rd and 4th call of a run) carry the candidate skill's text
    assert any(CANDIDATE_BODY in t for t in candidate_texts[2:])
    # executive calls never get worker skills
    assert not any(CANDIDATE_BODY in t for t in candidate_texts[:2])
    assert result.baseline[0].skills_fingerprint == result.plan.baseline_fingerprint
    assert result.candidate[0].skills_fingerprint == result.plan.candidate_fingerprint


def test_tie_is_rejected_no_improvement(env: Env) -> None:
    env.provider.push(*one_run(), *one_run())
    result = env.run()
    assert result.decision is not None and not result.accepted
    assert result.decision.reasons == [GateReason.NO_IMPROVEMENT]


def test_regression_is_rejected(env: Env) -> None:
    env.provider.push(*one_run(WC_PY), *one_run(WC_BYTES_PY))
    result = env.run()
    assert result.decision is not None and not result.accepted
    assert GateReason.REGRESSION in result.decision.reasons


def test_three_runs_per_side_with_default_gate(env: Env) -> None:
    for _ in range(3):
        env.provider.push(*one_run(WC_BYTES_PY), *one_run(WC_PY))
    result = env.run(runs=3, policy=None, max_total_usd=3.0)
    assert env.provider.remaining == 0
    assert len(result.baseline) == len(result.candidate) == 3
    assert result.accepted
    assert len(env.projects()) == 6


def test_two_benchmarks_cover_both_in_the_decision(env: Env) -> None:
    other = bench_with(env.bench, id="cli-word-count-2")
    env.provider.push(*one_run(WC_BYTES_PY), *one_run(WC_PY), *one_run(), *one_run())
    result = env.run(benchmarks=[env.bench, other], max_total_usd=2.0)
    assert result.decision is not None and result.accepted
    assert {c.benchmark_id for c in result.decision.comparisons} == {BID, "cli-word-count-2"}


def test_each_run_has_the_benchmark_tightened_cap(env: Env) -> None:
    env.provider.push(*one_run(), *one_run())
    result = env.run()
    assert all(r.budget_usd == 0.5 for r in (*result.baseline, *result.candidate))


def test_provider_failure_aborts_without_decision(env: Env) -> None:
    env.provider.push(*one_run(WC_PY), ProviderError("gateway down"))
    result = env.run()
    assert result.decision is None and not result.accepted
    assert result.aborted is not None and "provider" in result.aborted
    assert len(result.baseline) == 1 and len(result.candidate) == 1
    assert result.candidate[0].stop is BenchmarkStop.PROVIDER


def test_infrastructure_failure_on_baseline_aborts_before_candidate(env: Env) -> None:
    # a failing baseline would make any candidate look better: stop, no decision
    env.provider.push(ProviderError("gateway down"), *one_run())
    result = env.run()
    assert result.decision is None and result.aborted is not None
    assert len(result.baseline) == 1 and result.candidate == ()
    assert env.provider.remaining == 4  # the candidate run never started


def test_comparison_cap_is_rechecked_before_each_run(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pretend the first run cost more than authorised (a gateway ignoring max_tokens):
    # the next run no longer fits the comparison cap and is not started.
    env.provider.push(*one_run(), *one_run())
    real = sc.run_benchmark

    def costly(*a: Any, **kw: Any) -> Any:
        rec = real(*a, **kw)
        return rec.model_copy(update={"cost_nusd": usd_to_nusd(0.6)})

    monkeypatch.setattr(sc, "run_benchmark", costly)
    result = env.run(max_total_usd=1.0)
    assert result.aborted == "comparison cap reached"
    assert result.decision is None
    assert len(result.baseline) == 1 and result.candidate == ()


def test_daily_cap_is_rechecked_before_each_run(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    # Another process spends most of today's cap after planning.
    env.provider.push(*one_run(), *one_run())
    real = sc.run_benchmark

    def then_spend(*a: Any, **kw: Any) -> Any:
        rec = real(*a, **kw)
        spend(env.ledger, 4.7)
        return rec

    monkeypatch.setattr(sc, "run_benchmark", then_spend)
    result = env.run()
    assert result.aborted == "daily cap reached" and result.decision is None
    assert len(result.baseline) == 1 and result.candidate == ()


def test_budget_stop_of_a_run_is_a_measurement(env: Env) -> None:
    # A tiny benchmark budget stops both sides with "budget": recorded, decided (tie).
    tiny = bench_with(env.bench, budget_usd=0.000001)
    result = env.run(benchmarks=[tiny])
    assert env.provider.requests == []
    assert result.aborted is None
    assert [r.stop for r in (*result.baseline, *result.candidate)] == [BenchmarkStop.BUDGET] * 2
    assert result.decision is not None
    assert result.decision.reasons == [GateReason.NO_IMPROVEMENT]


def test_keyboard_interrupt_propagates(env: Env) -> None:
    env.provider.push(*one_run(), KeyboardInterrupt())
    seen: list[str] = []
    with pytest.raises(KeyboardInterrupt):
        env.run(on_record=lambda side, rec: seen.append(rec.project_id))
    assert len(seen) == 1


def test_quality_log_and_skills_dir_are_never_written(env: Env) -> None:
    env.provider.push(*one_run(WC_BYTES_PY), *one_run(WC_PY))
    env.run()
    assert not (env.ws / "skills").exists()
    assert not list(env.ws.glob("quality*"))


def test_input_registries_are_unchanged(env: Env) -> None:
    before = skillset_fingerprint(env.roles, env.skills)
    names = env.skills.names()
    env.provider.push(*one_run(), *one_run())
    env.run()
    assert skillset_fingerprint(env.roles, env.skills) == before
    assert env.skills.names() == names
    assert "unicode-counts" not in env.skills


def test_candidate_role_policy_is_unchanged(env: Env) -> None:
    _, cand_roles, _ = plan_skill_comparison(
        [env.bench],
        candidate=candidate_skill(),
        role="developer",
        runs=1,
        max_total_usd=2.0,
        config=env.config,
        store=env.store,
        ledger=env.ledger,
        roles=env.roles,
        skills=env.skills,
        policy=SkillGatePolicy(min_runs=1),
    )
    for r in env.roles:
        assert cand_roles.get(r.name).policy() == r.policy()
    assert "unicode-counts" in cand_roles.get("developer").skills


# --- structure -----------------------------------------------------------------------------


def test_module_never_calls_a_model_or_installs_anything() -> None:
    src = inspect.getsource(sc)
    tree = ast.parse(src)
    imported = {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module is not None
    }
    assert "automa.llm.gateway" not in imported
    assert "shutil" not in src and "subprocess" not in src
    for bad in (".complete(", "write_text", "write_bytes", "open(", "mkdir", "BudgetGuard("):
        assert bad not in src, bad
    # quality is never passed through
    assert "quality=None" in src


def test_cli_module_is_not_needed() -> None:
    assert "automa.cli" not in inspect.getsource(sc)
