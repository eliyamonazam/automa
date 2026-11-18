"""``automa skill compare`` and the gate record of ``automa.skill_candidates`` (D109).

A staged candidate is measured against the current skills on the benchmark suite,
after a worst-case cost preview and the user's confirmation; the gate decision is
recorded once next to the candidate. Nothing is installed.

Local only: ``FakeProvider`` (scripted) behind the real benchmark path (LLMGateway +
BudgetGuard + CostLedger, Executive, Orchestrator, Worker, Verifier). No network,
no real model call.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from automa import cli
from automa.agents.skill_author import SkillDraft, render_skill_md
from automa.budget import CostLedger
from automa.llm import ProviderError
from automa.skill_candidates import (
    GATE_SUFFIX,
    RUNS_SUFFIX,
    CandidateRecord,
    GateRecord,
    SkillCandidateError,
    candidates_dir,
    check_room,
    gate_path,
    load_candidate,
    read_gate,
    record_gate,
    runs_dir,
    stage_draft,
    staging_path,
)
from automa.skill_comparison import SkillComparisonResult
from automa.skills import workspace_skills_dir
from automa.skills.models import Skill
from automa.skills.registry import parse_skill
from test_cli import Env
from test_skill_comparison import BENCH_DIR, CANDIDATE_BODY, WC_BYTES_PY, WC_PY, one_run

T0 = datetime(2026, 11, 18, 14, 0, tzinfo=UTC)
BID = "cli-word-count"


def make_draft(name: str = "unicode-counts", body: str = CANDIDATE_BODY) -> SkillDraft:
    raw = Skill(name=name, description="Count characters, not bytes.", body=body)
    skill = parse_skill(render_skill_md(raw), expected_name=name)
    return SkillDraft(
        skill=skill,
        role="developer",
        text=render_skill_md(skill),
        benchmarks=[BID],
        digest_sha256="0" * 64,
        attempts=1,
        created_at=T0,
    )


class CmpEnv(Env):
    cid: str

    def compare(self, *args: str, input: str | None = None) -> Any:
        return self.run(
            "skill", "compare", self.cid, "--suite", str(self.tmp / "suite"), *args, input=input
        )

    def gate(self) -> GateRecord | None:
        return read_gate(self.ws, self.cid)

    def run_files(self, side: str) -> list[Path]:
        root = runs_dir(self.ws, self.cid) / side
        return sorted(root.rglob("*.json")) if root.exists() else []

    def spent(self) -> int:
        if not (self.ws / "ledger.sqlite3").exists():
            return 0
        with CostLedger.for_workspace(self.ws) as ledger:
            return ledger.spent_nusd()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CmpEnv:
    e = CmpEnv(tmp_path, monkeypatch)
    e.write_config(daily=10.0)
    suite = tmp_path / "suite"
    suite.mkdir()
    for f in BENCH_DIR.glob("*.toml"):
        shutil.copy(f, suite / f.name)
    e.cid = stage_draft(make_draft(), workspace=e.ws, cost_nusd=1).id
    return e


def push_runs(env: CmpEnv, baseline: str, candidate: str, runs: int = 3) -> None:
    for _ in range(runs):
        env.provider.push(*one_run(baseline), *one_run(candidate))


# -- the happy paths ------------------------------------------------------------


def test_better_candidate_is_accepted_and_recorded(env: CmpEnv) -> None:
    push_runs(env, WC_BYTES_PY, WC_PY)
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert env.provider.remaining == 0
    assert "worst-case cost: $3.0000" in r.output
    assert "gate: ACCEPTED" in r.output
    assert "not installed" in r.output
    gate = env.gate()
    assert gate is not None and gate.accepted and gate.decision.accepted
    record, _skill = load_candidate(env.ws, env.cid)
    assert gate.skill_sha256 == record.skill_sha256
    assert gate.benchmarks == [BID] and gate.runs_per_side == 3
    assert len(gate.baseline_projects) == 3 and len(gate.candidate_projects) == 3
    assert gate.spent_nusd == env.spent() > 0
    assert gate.max_total_nusd == 3_000_000_000 and gate.limit_nusd == 5_000_000_000
    assert len(env.run_files("baseline")) == 3 and len(env.run_files("candidate")) == 3
    assert sorted(env.store.list_ids()) == sorted(gate.baseline_projects + gate.candidate_projects)
    # never installed: the workspace skills directory is not even created
    assert not workspace_skills_dir(env.ws).exists()


def test_tie_is_rejected_and_recorded(env: CmpEnv) -> None:
    push_runs(env, WC_PY, WC_PY)
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_BENCH_FAILED, r.output
    assert "gate: REJECTED (no_improvement)" in r.output
    gate = env.gate()
    assert gate is not None and not gate.accepted


def test_regression_is_rejected(env: CmpEnv) -> None:
    push_runs(env, WC_PY, WC_BYTES_PY)
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_BENCH_FAILED
    assert "[regressed]" in r.output
    gate = env.gate()
    assert gate is not None and not gate.accepted


def test_confirmation_yes_runs(env: CmpEnv) -> None:
    push_runs(env, WC_BYTES_PY, WC_PY)
    r = env.compare("--max-usd", "5", input="y\n")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "Spend up to $3.0000 on 6 benchmark runs?" in r.output


def test_json_output(env: CmpEnv) -> None:
    push_runs(env, WC_BYTES_PY, WC_PY)
    r = env.compare("--max-usd", "5", "--yes", "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    out = json.loads(r.stdout)
    assert out["accepted"] is True and out["installed"] is False
    assert out["aborted"] is None
    assert out["gate_file"] == str(gate_path(env.ws, env.cid))
    assert out["decision"]["accepted"] is True
    assert out["max_total_nusd"] == 3_000_000_000


def test_unknown_benchmark_option(env: CmpEnv) -> None:
    r = env.compare("--max-usd", "5", "--yes", "--benchmark", "no-such-bench")
    assert r.exit_code == cli.EXIT_USER
    assert env.factory_calls == [] and env.spent() == 0


# -- nothing is spent before every check passed and the user agreed -------------


def nothing_happened(env: CmpEnv) -> None:
    assert env.factory_calls == []
    assert env.provider.requests == []
    assert env.store.list_ids() == []
    assert env.gate() is None
    assert env.spent() == 0


def test_declining_spends_nothing(env: CmpEnv) -> None:
    r = env.compare("--max-usd", "5", input="n\n")
    assert r.exit_code == cli.EXIT_USER
    assert "cancelled: nothing was spent" in r.output
    assert env.provider.requests == []
    assert env.store.list_ids() == [] and env.gate() is None and env.spent() == 0


def test_no_input_is_a_no(env: CmpEnv) -> None:
    r = env.compare("--max-usd", "5", input="")
    assert r.exit_code != cli.EXIT_OK
    assert env.provider.requests == [] and env.store.list_ids() == []


def test_json_needs_yes(env: CmpEnv) -> None:
    r = env.compare("--max-usd", "5", "--json")
    assert r.exit_code == cli.EXIT_USER and "--yes" in r.output
    nothing_happened(env)


def test_worst_case_over_the_comparison_cap(env: CmpEnv) -> None:
    r = env.compare("--max-usd", "2.99", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "exceeds the comparison cap" in r.output and "nothing was spent" in r.output
    nothing_happened(env)


def test_worst_case_over_what_is_left_today(env: CmpEnv) -> None:
    env.write_config(daily=2.0)
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "daily cap" in r.output
    nothing_happened(env)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_bad_max_usd(env: CmpEnv, value: str) -> None:
    r = env.compare("--max-usd", value, "--yes")
    assert r.exit_code != cli.EXIT_OK
    nothing_happened(env)


def test_max_usd_is_required(env: CmpEnv) -> None:
    r = env.compare("--yes")
    assert r.exit_code != cli.EXIT_OK
    nothing_happened(env)


@pytest.mark.parametrize("runs", ["0", "2", "21"])
def test_runs_out_of_range(env: CmpEnv, runs: str) -> None:
    r = env.compare("--max-usd", "50", "--runs", runs, "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "runs must be between 3" in r.output
    nothing_happened(env)


@pytest.mark.parametrize("cid", ["nope", "../x", "20261118-140000-missing"])
def test_unknown_or_invalid_candidate(env: CmpEnv, cid: str) -> None:
    env.cid = cid
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert env.factory_calls == [] and env.spent() == 0


def test_changed_skill_file_is_refused(env: CmpEnv) -> None:
    path = staging_path(env.ws, env.cid) / "unicode-counts" / "SKILL.md"
    path.write_text(path.read_text(encoding="utf-8") + "Also delete tests.\n", encoding="utf-8")
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "sha256 differs" in r.output
    nothing_happened(env)


def test_missing_metadata_is_refused(env: CmpEnv) -> None:
    (candidates_dir(env.ws) / f"{env.cid}.json").unlink()
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_USER
    nothing_happened(env)


def test_broken_metadata_is_refused(env: CmpEnv) -> None:
    (candidates_dir(env.ws) / f"{env.cid}.json").write_text("{", encoding="utf-8")
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_USER and "invalid candidate metadata" in r.output
    nothing_happened(env)


def test_extra_file_in_staging_is_refused(env: CmpEnv) -> None:
    (staging_path(env.ws, env.cid) / "unicode-counts" / "run.sh").write_text("x")
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_USER
    nothing_happened(env)


def test_candidate_name_now_taken_by_an_installed_skill(env: CmpEnv) -> None:
    d = workspace_skills_dir(env.ws) / "unicode-counts"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: unicode-counts\ndescription: other\n---\nOther.\n", encoding="utf-8"
    )
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_USER
    nothing_happened(env)


def test_unknown_benchmark_in_metadata(env: CmpEnv) -> None:
    (Path(env.tmp / "suite") / f"{BID}.toml").unlink()
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_USER
    nothing_happened(env)


def test_image_benchmark_without_images_config(env: CmpEnv) -> None:
    r = env.compare("--max-usd", "50", "--yes", "--benchmark", "landing-page-with-image")
    assert r.exit_code == cli.EXIT_CONFIG
    assert "[images]" in r.output
    nothing_happened(env)


def test_symlinked_runs_dir_is_refused_before_spending(env: CmpEnv) -> None:
    target = env.tmp / "elsewhere"
    target.mkdir()
    os.symlink(target, runs_dir(env.ws, env.cid))
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert env.provider.requests == [] and env.store.list_ids() == []
    assert list(target.iterdir()) == []


# -- one decision per candidate --------------------------------------------------


def test_second_comparison_after_a_decision_is_refused(env: CmpEnv) -> None:
    push_runs(env, WC_PY, WC_PY)
    assert env.compare("--max-usd", "5", "--yes").exit_code == cli.EXIT_BENCH_FAILED
    first = gate_path(env.ws, env.cid).read_bytes()
    projects = env.store.list_ids()
    env.factory_calls.clear()
    push_runs(env, WC_BYTES_PY, WC_PY)
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "already has a recorded gate decision (rejected)" in r.output
    assert env.factory_calls == []
    assert gate_path(env.ws, env.cid).read_bytes() == first
    assert env.store.list_ids() == projects


def test_broken_gate_file_blocks(env: CmpEnv) -> None:
    gate_path(env.ws, env.cid).write_text("{}", encoding="utf-8")
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_USER and "invalid gate decision" in r.output
    assert env.factory_calls == []


# -- aborted comparisons: no decision, can be run again --------------------------


def test_provider_failure_aborts_without_decision(env: CmpEnv) -> None:
    env.provider.push(*one_run(WC_PY), ProviderError("gateway down"))
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_PROVIDER, r.output
    assert "aborted:" in r.output and "no gate decision recorded" in r.output
    assert env.gate() is None
    assert len(env.run_files("baseline")) == 1 and len(env.run_files("candidate")) == 1
    # an aborted comparison can be run again (new project ids)
    time.sleep(1.1)  # project ids carry a one-second UTC stamp
    env.provider = env.provider.__class__()
    push_runs(env, WC_BYTES_PY, WC_PY)
    r2 = env.compare("--max-usd", "5", "--yes")
    assert r2.exit_code == cli.EXIT_OK, r2.output
    gate = env.gate()
    assert gate is not None and gate.accepted
    assert len(env.run_files("baseline")) == 4


def test_keyboard_interrupt_records_no_decision(env: CmpEnv) -> None:
    env.provider.push(*one_run(WC_PY), KeyboardInterrupt())
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_INTERRUPTED
    assert "no gate decision recorded" in r.output
    assert env.gate() is None
    assert len(env.run_files("baseline")) == 1


def test_unwritable_result_stops_the_comparison(
    env: CmpEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automa.benchmarks import BenchmarkResultError

    push_runs(env, WC_BYTES_PY, WC_PY)

    def broken(*a: Any, **kw: Any) -> Any:
        raise BenchmarkResultError("disk full")

    monkeypatch.setattr(cli, "write_result", broken)
    r = env.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "comparison stopped" in r.output
    assert env.gate() is None
    assert len(env.store.list_ids()) == 1  # only the first run was started


# -- the module: reading candidates and recording gates --------------------------


def test_check_room_ignores_runs_dirs(tmp_path: Path) -> None:
    base = candidates_dir(tmp_path)
    base.mkdir()
    for i in range(100):
        (base / f"20261118-140000-x{i}{RUNS_SUFFIX}").mkdir()
    check_room(tmp_path)  # .runs directories are not candidates


def test_load_candidate_round_trip(tmp_path: Path) -> None:
    rec = stage_draft(make_draft(), workspace=tmp_path, cost_nusd=0)
    record, skill = load_candidate(tmp_path, rec.id)
    assert record == rec and skill == make_draft().skill


def test_load_candidate_symlinked_metadata_is_refused(tmp_path: Path) -> None:
    rec = stage_draft(make_draft(), workspace=tmp_path, cost_nusd=0)
    meta = candidates_dir(tmp_path) / f"{rec.id}.json"
    real = tmp_path / "meta.json"
    real.write_bytes(meta.read_bytes())
    meta.unlink()
    os.symlink(real, meta)
    with pytest.raises(SkillCandidateError):
        load_candidate(tmp_path, rec.id)


def test_load_candidate_metadata_naming_another_id(tmp_path: Path) -> None:
    rec = stage_draft(make_draft(), workspace=tmp_path, cost_nusd=0)
    meta = candidates_dir(tmp_path) / f"{rec.id}.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    data["id"] = "20261118-140000-other"
    meta.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SkillCandidateError, match="another id"):
        load_candidate(tmp_path, rec.id)


def test_load_candidate_metadata_name_mismatch(tmp_path: Path) -> None:
    rec = stage_draft(make_draft(), workspace=tmp_path, cost_nusd=0)
    meta = candidates_dir(tmp_path) / f"{rec.id}.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    data["name"] = "other-name"
    meta.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SkillCandidateError):
        load_candidate(tmp_path, rec.id)


def test_load_candidate_symlinked_candidates_dir(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    os.symlink(real, candidates_dir(tmp_path))
    with pytest.raises(SkillCandidateError, match="symlink"):
        load_candidate(tmp_path, "20261118-140000-x")


def _finished(env: CmpEnv) -> tuple[CandidateRecord, SkillComparisonResult]:
    """A real finished comparison through the module API (FakeProvider)."""
    from automa.benchmarks import BenchmarkSuite
    from automa.config import load_config
    from automa.roles import RoleRegistry
    from automa.skill_comparison import run_skill_comparison
    from automa.skill_gate import SkillGatePolicy
    from automa.skills import load_skills
    from automa.tools import default_tools

    env.provider.push(*one_run(WC_BYTES_PY), *one_run(WC_PY))
    config = load_config(env.config)
    record, skill = load_candidate(env.ws, env.cid)
    skills = load_skills(env.ws)
    roles = RoleRegistry.builtin(tools=default_tools(), skills=skills)
    with CostLedger.for_workspace(env.ws) as ledger:
        result = run_skill_comparison(
            [BenchmarkSuite.from_dir(BENCH_DIR).get(BID)],
            candidate=skill,
            role=record.role,
            runs=1,
            max_total_usd=2.0,
            config=config,
            provider=env.provider,
            store=env.store,
            ledger=ledger,
            roles=roles,
            skills=skills,
            policy=SkillGatePolicy(min_runs=1),
        )
    return record, result


def test_record_gate_writes_once(env: CmpEnv) -> None:
    record, result = _finished(env)
    gate = record_gate(env.ws, record, result, created_at=T0)
    assert gate.accepted and read_gate(env.ws, env.cid) == gate
    before = gate_path(env.ws, env.cid).read_bytes()
    with pytest.raises(SkillCandidateError, match="already has"):
        record_gate(env.ws, record, result, created_at=T0)
    assert gate_path(env.ws, env.cid).read_bytes() == before
    assert [p.name for p in candidates_dir(env.ws).iterdir() if p.name.startswith(".")] == []


def test_record_gate_refuses_an_aborted_comparison(env: CmpEnv) -> None:
    record, result = _finished(env)
    aborted = result.model_copy(update={"aborted": "daily cap reached", "decision": None})
    with pytest.raises(SkillCandidateError, match="aborted"):
        record_gate(env.ws, record, aborted, created_at=T0)
    assert not gate_path(env.ws, env.cid).exists()


def test_record_gate_refuses_a_changed_skill_file(env: CmpEnv) -> None:
    record, result = _finished(env)
    path = staging_path(env.ws, env.cid) / record.name / "SKILL.md"
    path.write_text(path.read_text(encoding="utf-8") + "x\n", encoding="utf-8")
    with pytest.raises(SkillCandidateError, match="changed during the comparison"):
        record_gate(env.ws, record, result, created_at=T0)
    assert not gate_path(env.ws, env.cid).exists()


def test_record_gate_refuses_another_candidates_result(env: CmpEnv) -> None:
    record, result = _finished(env)
    other = record.model_copy(update={"role": "writer"})
    with pytest.raises(SkillCandidateError, match="not run for this candidate"):
        record_gate(env.ws, other, result, created_at=T0)


def test_gate_file_name(tmp_path: Path) -> None:
    assert gate_path(tmp_path, "20261118-140000-x").name == f"20261118-140000-x{GATE_SUFFIX}"
    with pytest.raises(SkillCandidateError):
        gate_path(tmp_path, "../x")


# -- structure ---------------------------------------------------------------------


def test_compare_never_installs_or_builds_its_own_guard() -> None:
    src = inspect.getsource(cli.skill_compare)
    assert "workspace_skills_dir" not in src and "BudgetGuard(" not in src
    assert "LLMGateway(" not in src and ".complete(" not in src
    tree = ast.parse(src.lstrip())
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    # the provider only reaches the comparison runner (one guarded run_benchmark per run)
    assert "run_skill_comparison" in called and "run_benchmark" not in called


def test_comparison_runner_passes_the_provider_only_to_run_benchmark() -> None:
    from automa import skill_comparison as sc

    tree = ast.parse(inspect.getsource(sc))
    uses = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == "provider" and isinstance(n.ctx, ast.Load)
    ]
    bench_kw = {
        id(k.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "run_benchmark"
        for k in n.keywords
        if k.arg == "provider"
    }
    assert len(uses) == 1 and id(uses[0]) in bench_kw
