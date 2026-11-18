"""``automa skill draft`` and ``automa.skill_candidates`` (D108): a candidate skill is
drafted from recorded benchmark failures and staged, never installed.

Local only: ``FakeProvider`` behind the real guarded ``LLMGateway`` + ``BudgetGuard``
+ ``CostLedger``. No network, no real model call.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from automa import cli
from automa.agents.skill_author import DEFAULT_BUDGET_ID, SkillDraft, render_skill_md
from automa.benchmarks import write_result
from automa.budget import CostLedger
from automa.llm import ChatResponse, FinishReason, Message, ProviderError, Usage
from automa.skill_candidates import (
    CANDIDATES_DIRNAME,
    MAX_CANDIDATES,
    CandidateRecord,
    SkillCandidateError,
    candidate_id,
    candidates_dir,
    check_room,
    stage_draft,
    staging_path,
)
from automa.skill_comparison import load_candidate_skill
from automa.skills import load_skills, workspace_skills_dir
from automa.skills.models import Skill
from automa.skills.registry import parse_skill
from test_bench_report import BENCH_DIR, rec
from test_cli import Env

T0 = datetime(2026, 11, 18, 14, 0, tzinfo=UTC)


def skill_md(name: str = "check-outputs", desc: str = "Check outputs before finishing") -> str:
    return f"---\nname: {name}\ndescription: {desc}\n---\nRun every check twice.\n"


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    e = Env(tmp_path, monkeypatch)
    suite = tmp_path / "suite"
    suite.mkdir()
    for f in BENCH_DIR.glob("*.toml"):
        shutil.copy(f, suite / f.name)
    results = suite / "results"
    write_result(results, rec(1, failed=(2,)))
    write_result(results, rec(2))
    write_result(results, rec(3, stop="tasks_failed", failed=(0, 1)))
    return e


def draft(env: Env, *args: str) -> Any:
    return env.run("skill", "draft", "--suite", str(env.tmp / "suite"), *args)


def staged(env: Env) -> list[str]:
    base = env.ws / CANDIDATES_DIRNAME
    return sorted(os.listdir(base)) if base.exists() else []


def tree_bytes(root: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    if not root.exists():
        return out
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            out[str(p.relative_to(root))] = p.read_bytes()
    return out


# -- happy path ----------------------------------------------------------------------


def test_drafts_and_stages_a_candidate(env: Env) -> None:
    env.provider.push(skill_md())
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_OK, r.output
    entries = staged(env)
    assert len(entries) == 2
    cid = next(e for e in entries if not e.endswith(".json"))
    assert f"{cid}.json" in entries
    assert cid.endswith("-check-outputs")
    staging = staging_path(env.ws, cid)
    skill = load_candidate_skill(staging, workspace=env.ws)
    assert skill.name == "check-outputs"
    assert (staging / "check-outputs" / "SKILL.md").read_text(encoding="utf-8") == (
        render_skill_md(skill)
    )
    assert f"candidate: {cid}" in r.output
    assert "skill: check-outputs for role developer" in r.output
    assert "learned from: cli-word-count" in r.output
    assert "cost: $" in r.output
    assert "not installed" in r.output
    assert len(env.provider.requests) == 1


def test_never_installs(env: Env) -> None:
    env.provider.push(skill_md())
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert not workspace_skills_dir(env.ws).exists()
    assert "check-outputs" not in load_skills(env.ws).names()
    r = env.run("skills", "--json")
    assert all(row["name"] != "check-outputs" for row in json.loads(r.output))


def test_metadata_records_cost_and_hash(env: Env) -> None:
    env.provider.push(skill_md())
    r = draft(env, "--role", "developer", "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    out = json.loads(r.output)
    assert out["installed"] is False
    assert out["role"] == "developer"
    assert out["benchmarks"] == ["cli-word-count"]
    assert out["attempts"] == 1
    assert out["cost_nusd"] > 0
    meta = env.ws / CANDIDATES_DIRNAME / f"{out['id']}.json"
    record = CandidateRecord.model_validate_json(meta.read_text(encoding="utf-8"))
    assert record.cost_nusd == out["cost_nusd"]
    text = Path(out["skill_md"]).read_text(encoding="utf-8")
    import hashlib

    assert record.skill_sha256 == hashlib.sha256(text.encode()).hexdigest()
    with CostLedger.for_workspace(env.ws) as ledger:
        assert ledger.spent_nusd(project_id=DEFAULT_BUDGET_ID) == record.cost_nusd


def test_prompt_contains_failures(env: Env) -> None:
    env.provider.push(skill_md())
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_OK, r.output
    prompt = env.provider.requests[0].messages[-1].content or ""
    assert "<analysis>" in prompt and "cli-word-count" in prompt
    assert '"name": "developer"' in prompt or "developer" in prompt


def test_invalid_reply_is_retried(env: Env) -> None:
    env.provider.push("no frontmatter here", skill_md())
    r = draft(env, "--role", "developer", "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert json.loads(r.output)["attempts"] == 2


def test_two_drafts_get_two_candidates(env: Env) -> None:
    env.provider.push(skill_md("first-skill"), skill_md("second-skill"))
    assert draft(env, "--role", "developer").exit_code == cli.EXIT_OK
    assert draft(env, "--role", "writer").exit_code == cli.EXIT_OK
    assert len([e for e in staged(env) if e.endswith(".json")]) == 2


# -- refusals before any model call --------------------------------------------------


def test_unknown_role(env: Env) -> None:
    r = draft(env, "--role", "nobody")
    assert r.exit_code == cli.EXIT_USER
    assert "unknown role" in r.output and "developer" in r.output
    assert env.provider.requests == [] and staged(env) == []


def test_unknown_benchmark(env: Env) -> None:
    r = draft(env, "--role", "developer", "--benchmark", "no-such-bench")
    assert r.exit_code == cli.EXIT_USER
    assert "nothing was staged" in r.output
    assert env.provider.requests == [] and staged(env) == []


def test_no_failures(env: Env) -> None:
    results = env.tmp / "suite" / "results"
    shutil.rmtree(results)
    write_result(results, rec(1))
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_USER
    assert env.provider.requests == [] and staged(env) == []


def test_no_results(env: Env) -> None:
    shutil.rmtree(env.tmp / "suite" / "results")
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_USER
    assert env.provider.requests == [] and staged(env) == []


def test_broken_result_file_refuses(env: Env) -> None:
    bad = env.tmp / "suite" / "results" / "cli-word-count" / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_USER
    assert "unusable result file" in r.output
    assert env.provider.requests == [] and staged(env) == []


def test_missing_suite(env: Env) -> None:
    r = env.run("skill", "draft", "--role", "developer", "--suite", str(env.tmp / "nope"))
    assert r.exit_code == cli.EXIT_USER
    assert env.provider.requests == []


def test_explicit_results_without_suite(env: Env) -> None:
    env.provider.push(skill_md())
    r = env.run(
        "skill",
        "draft",
        "--role",
        "developer",
        "--suite",
        str(env.tmp / "nope"),
        "--results",
        str(env.tmp / "suite" / "results"),
    )
    assert r.exit_code == cli.EXIT_OK, r.output


def test_missing_tier_is_config_error(env: Env) -> None:
    env.config.write_text(
        env.config.read_text(encoding="utf-8").replace('medium = "fake-medium"\n', ""),
        encoding="utf-8",
    )
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_CONFIG
    assert env.provider.requests == []


def test_symlinked_candidates_dir_refuses(env: Env) -> None:
    target = env.tmp / "elsewhere"
    target.mkdir()
    env.ws.mkdir(parents=True, exist_ok=True)
    (env.ws / CANDIDATES_DIRNAME).symlink_to(target)
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_USER
    assert "symlink" in r.output
    assert env.provider.requests == [] and os.listdir(target) == []


def test_full_store_refuses(env: Env) -> None:
    base = env.ws / CANDIDATES_DIRNAME
    for i in range(MAX_CANDIDATES):
        (base / f"20260101-0000{i:02d}-x{i}").mkdir(parents=True)
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_USER
    assert env.provider.requests == []


def test_no_key_uses_real_factory_and_fails(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    from automa.llm.factory import provider_from_config

    monkeypatch.setattr(cli, "make_provider", provider_from_config)
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_CONFIG
    assert "AUTOMA_API_KEY" in r.output
    assert staged(env) == []


# -- model / budget failures stage nothing -------------------------------------------


def test_budget_stop(env: Env) -> None:
    env.write_config(per_call=0.000001)
    env.provider.push(skill_md())
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_BUDGET
    assert "budget stop" in r.output and "nothing was staged" in r.output
    assert env.provider.requests == [] and staged(env) == []


def test_provider_error(env: Env) -> None:
    env.provider.push(ProviderError("gateway down"))
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_PROVIDER
    assert staged(env) == []


def test_no_valid_reply(env: Env) -> None:
    env.provider.push("bad", "bad", "bad")
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_PLANNING
    assert "nothing was staged" in r.output
    assert staged(env) == []
    assert len(env.provider.requests) == 3


def test_existing_skill_name_is_rejected(env: Env) -> None:
    ws_skill = workspace_skills_dir(env.ws) / "check-outputs"
    ws_skill.mkdir(parents=True)
    (ws_skill / "SKILL.md").write_text(skill_md(), encoding="utf-8")
    env.provider.push(skill_md(), skill_md(), skill_md())
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_PLANNING
    assert staged(env) == []


def test_permission_claims_are_rejected(env: Env) -> None:
    bad = "---\nname: x-skill\ndescription: d\nallowed-tools: run_command\n---\nbody\n"
    env.provider.push(bad, bad, bad)
    r = draft(env, "--role", "developer")
    assert r.exit_code == cli.EXIT_PLANNING
    assert staged(env) == []


def test_truncated_reply_counts_as_invalid(env: Env) -> None:
    cut = ChatResponse(
        model="fake-medium",
        message=Message.assistant(skill_md()),
        usage=Usage(input_tokens=10, output_tokens=10),
        finish_reason=FinishReason.LENGTH,
    )
    env.provider.push(cut, skill_md("other-skill"))
    r = draft(env, "--role", "developer", "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert json.loads(r.output)["name"] == "other-skill"


# -- stage_draft / check_room --------------------------------------------------------


def make_draft(name: str = "check-outputs", **kw: Any) -> SkillDraft:
    raw = Skill(name=name, description="Check outputs", body="Run every check twice.")
    skill = parse_skill(render_skill_md(raw), expected_name=name)
    data: dict[str, Any] = {
        "skill": skill,
        "role": "developer",
        "text": render_skill_md(skill),
        "benchmarks": ["cli-word-count"],
        "digest_sha256": "0" * 64,
        "attempts": 1,
        "created_at": T0,
    }
    data.update(kw)
    return SkillDraft.model_validate(data)


def test_stage_draft_layout(tmp_path: Path) -> None:
    record = stage_draft(make_draft(), workspace=tmp_path, cost_nusd=5)
    assert record.id == "20261118-140000-check-outputs" == candidate_id(T0, "check-outputs")
    base = candidates_dir(tmp_path)
    assert sorted(os.listdir(base)) == [record.id, f"{record.id}.json"]
    assert os.listdir(base / record.id) == ["check-outputs"]
    assert os.listdir(base / record.id / "check-outputs") == ["SKILL.md"]
    assert load_candidate_skill(base / record.id, workspace=tmp_path) == make_draft().skill


def test_stage_draft_never_overwrites(tmp_path: Path) -> None:
    stage_draft(make_draft(), workspace=tmp_path, cost_nusd=5)
    before = tree_bytes(tmp_path)
    with pytest.raises(SkillCandidateError, match="already exists"):
        stage_draft(make_draft(), workspace=tmp_path, cost_nusd=7)
    assert tree_bytes(tmp_path) == before
    assert not [e for e in os.listdir(candidates_dir(tmp_path)) if e.startswith(".")]


def test_stage_draft_existing_metadata_blocks(tmp_path: Path) -> None:
    base = candidates_dir(tmp_path)
    base.mkdir(parents=True)
    (base / "20261118-140000-check-outputs.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SkillCandidateError, match="already exists"):
        stage_draft(make_draft(), workspace=tmp_path, cost_nusd=5)
    assert sorted(os.listdir(base)) == ["20261118-140000-check-outputs.json"]


def test_stage_draft_text_mismatch_leaves_nothing(tmp_path: Path) -> None:
    other = Skill(name="check-outputs", description="Other", body="Different.\n")
    with pytest.raises(SkillCandidateError, match="differs"):
        stage_draft(make_draft(text=render_skill_md(other)), workspace=tmp_path, cost_nusd=0)
    assert os.listdir(candidates_dir(tmp_path)) == []


@pytest.mark.parametrize("cost", [-1, 1.5, True, "3"])
def test_stage_draft_bad_cost(tmp_path: Path, cost: Any) -> None:
    with pytest.raises(SkillCandidateError):
        stage_draft(make_draft(), workspace=tmp_path, cost_nusd=cost)
    assert not candidates_dir(tmp_path).exists()


def test_stage_draft_requires_draft(tmp_path: Path) -> None:
    with pytest.raises(SkillCandidateError):
        stage_draft("not a draft", workspace=tmp_path, cost_nusd=0)  # type: ignore[arg-type]


def test_check_room_file_instead_of_dir(tmp_path: Path) -> None:
    candidates_dir(tmp_path).write_text("x", encoding="utf-8")
    with pytest.raises(SkillCandidateError, match="not a directory"):
        check_room(tmp_path)


def test_check_room_ignores_metadata_and_hidden(tmp_path: Path) -> None:
    base = candidates_dir(tmp_path)
    base.mkdir()
    for i in range(MAX_CANDIDATES):
        (base / f"{i}.json").write_text("{}", encoding="utf-8")
        (base / f".tmp{i}").mkdir()
    check_room(tmp_path)


@pytest.mark.parametrize("cid", ["", "../x", "20261118-140000-A", "x" * 81, "a/b"])
def test_staging_path_rejects_bad_ids(tmp_path: Path, cid: str) -> None:
    with pytest.raises(SkillCandidateError):
        staging_path(tmp_path, cid)


def test_candidate_id_is_bounded() -> None:
    cid = candidate_id(T0, "a" * 64)
    assert len(cid) <= 80 and not cid.endswith("-")


def test_candidates_are_separate_from_installed_skills(tmp_path: Path) -> None:
    record = stage_draft(make_draft(), workspace=tmp_path, cost_nusd=0)
    staging = staging_path(tmp_path, record.id).resolve()
    skills = workspace_skills_dir(tmp_path).resolve()
    assert skills not in staging.parents and staging != skills


def test_draft_command_has_no_install_option() -> None:
    import inspect

    params = inspect.signature(cli.skill_draft).parameters
    assert not any("install" in p or "yes" in p for p in params)
