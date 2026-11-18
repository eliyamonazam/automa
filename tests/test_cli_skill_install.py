"""``automa skill install`` and ``automa.skill_candidates.install_candidate`` (D110).

An accepted, staged candidate is copied byte for byte to ``<workspace>/skills/<name>/``
after the user's confirmation, only while the measured text (sha256) and the measured
roles and skills (gate fingerprints) still hold; nothing existing is ever replaced and
automa.toml is never edited.

Local only: the accepted gate decision comes from a real ``automa skill compare`` run
with the scripted ``FakeProvider``. Installing itself makes no model call.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from automa import cli, skill_candidates
from automa.skill_candidates import (
    SkillCandidateError,
    candidates_dir,
    check_install,
    gate_path,
    install_candidate,
    stage_draft,
    staging_path,
)
from automa.skill_variants import skillset_fingerprint
from automa.skills import SkillRegistry, load_skills, workspace_skills_dir
from automa.skills.registry import SKILL_FILE
from test_cli_skill_compare import CmpEnv, make_draft, push_runs
from test_skill_comparison import BENCH_DIR, WC_BYTES_PY, WC_PY

NAME = "unicode-counts"


class InstEnv(CmpEnv):
    def install(self, *args: str, input: str | None = None) -> Any:
        return self.run("skill", "install", self.cid, *args, input=input)

    @property
    def target(self) -> Path:
        return workspace_skills_dir(self.ws) / NAME

    @property
    def staged_bytes(self) -> bytes:
        return (staging_path(self.ws, self.cid) / NAME / SKILL_FILE).read_bytes()

    def roles_and_skills(self) -> Any:
        return cli._roles_and_skills(cli._load(self.config))

    def rewrite_gate(self, **changes: Any) -> None:
        path = gate_path(self.ws, self.cid)
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in changes.items():
            if isinstance(value, dict):
                data[key].update(value)
            else:
                data[key] = value
        path.unlink()
        path.write_text(json.dumps(data), encoding="utf-8")

    def leftovers(self) -> list[str]:
        d = workspace_skills_dir(self.ws)
        return sorted(e for e in os.listdir(d) if e.startswith(".")) if d.exists() else []


def _make_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InstEnv:
    e = InstEnv(tmp_path, monkeypatch)
    e.write_config(daily=10.0)
    suite = tmp_path / "suite"
    suite.mkdir()
    for f in BENCH_DIR.glob("*.toml"):
        shutil.copy(f, suite / f.name)
    e.cid = stage_draft(make_draft(), workspace=e.ws, cost_nusd=1).id
    return e


@pytest.fixture
def staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InstEnv:
    """A staged candidate without a gate decision."""
    return _make_env(tmp_path, monkeypatch)


@pytest.fixture
def env(staged: InstEnv) -> InstEnv:
    """A staged candidate the gate accepted."""
    push_runs(staged, WC_BYTES_PY, WC_PY)
    r = staged.compare("--max-usd", "5", "--yes")
    assert r.exit_code == cli.EXIT_OK, r.output
    staged.factory_calls.clear()
    return staged


def reject(staged: InstEnv) -> None:
    push_runs(staged, WC_PY, WC_PY)
    assert staged.compare("--max-usd", "5", "--yes").exit_code == cli.EXIT_BENCH_FAILED
    staged.factory_calls.clear()


def not_installed(env: InstEnv) -> None:
    assert not env.target.exists() and not env.target.is_symlink()
    assert env.leftovers() == []
    assert env.factory_calls == []


# -- the happy path ---------------------------------------------------------------


def test_install_accepted_candidate(env: InstEnv) -> None:
    r = env.install("--yes")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert (env.target / SKILL_FILE).read_bytes() == env.staged_bytes
    assert sorted(os.listdir(env.target)) == [SKILL_FILE]
    assert f"installed: {env.target / SKILL_FILE}" in r.output
    assert "gate: ACCEPTED (cli-word-count 0/3 -> 3/3)" in r.output
    assert "[roles.developer]" in r.output and f'skills = ["{NAME}"]' in r.output
    assert env.factory_calls == [] and env.leftovers() == []
    skills = load_skills(env.ws)
    assert NAME in skills
    # installed but not used: no role lists it until automa.toml says so
    roles, _ = env.roles_and_skills()
    assert all(NAME not in role.skills for role in roles)


def test_staged_candidate_and_gate_are_kept(env: InstEnv) -> None:
    gate_before = gate_path(env.ws, env.cid).read_bytes()
    staged_before = env.staged_bytes
    assert env.install("--yes").exit_code == cli.EXIT_OK
    assert gate_path(env.ws, env.cid).read_bytes() == gate_before
    assert env.staged_bytes == staged_before


def test_printed_line_gives_the_measured_candidate_skill_set(env: InstEnv) -> None:
    gate = skill_candidates.read_gate(env.ws, env.cid)
    assert gate is not None
    assert env.install("--yes").exit_code == cli.EXIT_OK
    env.write_config(daily=10.0, extra=f'\n[roles.developer]\nskills = ["{NAME}"]\n')
    roles, skills = env.roles_and_skills()
    assert NAME in roles.get("developer").skills
    assert skillset_fingerprint(roles, skills) == gate.decision.candidate_fingerprint


def test_printed_line_keeps_existing_role_skills(env: InstEnv) -> None:
    config = cli._load(env.config)
    lines = cli._role_skills_line(config, "developer", NAME)
    assert lines == ["[roles.developer]", f'skills = ["{NAME}"]']
    other = workspace_skills_dir(env.ws) / "house-style"
    other.mkdir(parents=True)
    (other / SKILL_FILE).write_text(
        "---\nname: house-style\ndescription: Our style.\n---\nWrite tersely.\n",
        encoding="utf-8",
    )
    env.write_config(daily=10.0, extra='\n[roles.developer]\nskills = ["house-style"]\n')
    lines = cli._role_skills_line(cli._load(env.config), "developer", NAME)
    assert lines[1] == f'skills = ["house-style", "{NAME}"]'


def test_confirm_yes(env: InstEnv) -> None:
    r = env.install(input="y\n")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert f"Install skill {NAME}?" in r.output
    assert env.target.is_dir()


def test_confirm_no_installs_nothing(env: InstEnv) -> None:
    r = env.install(input="n\n")
    assert r.exit_code == cli.EXIT_USER
    assert "cancelled: nothing was installed" in r.output
    not_installed(env)


def test_no_input_is_a_no(env: InstEnv) -> None:
    r = env.install(input="")
    assert r.exit_code != cli.EXIT_OK
    not_installed(env)


def test_json_needs_yes(env: InstEnv) -> None:
    r = env.install("--json")
    assert r.exit_code == cli.EXIT_USER
    assert "--json needs --yes" in r.output
    not_installed(env)


def test_json_output(env: InstEnv) -> None:
    r = env.install("--yes", "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    out = json.loads(r.stdout)
    assert out["installed"] is True and out["role_in_use"] is False
    assert out["name"] == NAME and out["role"] == "developer"
    assert out["skill_md"] == str(env.target / SKILL_FILE)
    assert out["automa_toml"] == f'[roles.developer]\nskills = ["{NAME}"]'


def test_help_lists_install() -> None:
    from typer.testing import CliRunner

    r = CliRunner().invoke(cli.app, ["skill", "--help"])
    assert "install" in r.output


# -- refused: no accepted decision for exactly this text ---------------------------


def test_without_gate_decision_is_refused(staged: InstEnv) -> None:
    r = staged.install("--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "no recorded gate decision" in r.output
    assert "nothing was installed" in r.output
    not_installed(staged)


def test_rejected_candidate_is_refused(staged: InstEnv) -> None:
    reject(staged)
    r = staged.install("--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "rejected by the benchmark gate (no_improvement)" in r.output
    not_installed(staged)


def test_forged_accepted_flag_is_refused(staged: InstEnv) -> None:
    reject(staged)
    staged.rewrite_gate(accepted=True)
    r = staged.install("--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "rejected by the benchmark gate" in r.output
    not_installed(staged)


def test_forged_decision_with_reasons_is_refused(env: InstEnv) -> None:
    env.rewrite_gate(decision={"reasons": ["no_improvement"]})
    r = env.install("--yes")
    assert r.exit_code == cli.EXIT_USER
    not_installed(env)


def test_edited_skill_after_compare_is_refused(env: InstEnv) -> None:
    path = staging_path(env.ws, env.cid) / NAME / SKILL_FILE
    path.write_bytes(path.read_bytes() + b"\nAlso do something else.\n")
    r = env.install("--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "sha256 differs" in r.output
    not_installed(env)


def test_gate_for_another_text_is_refused(env: InstEnv) -> None:
    env.rewrite_gate(skill_sha256="a" * 64)
    r = env.install("--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "not the text the gate measured" in r.output
    not_installed(env)


def test_gate_for_another_role_is_refused(env: InstEnv) -> None:
    env.rewrite_gate(role="writer")
    r = env.install("--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "another skill or role" in r.output
    not_installed(env)


def test_broken_gate_file_is_refused(env: InstEnv) -> None:
    path = gate_path(env.ws, env.cid)
    path.unlink()
    path.write_text("{not json", encoding="utf-8")
    r = env.install("--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "invalid gate decision" in r.output
    not_installed(env)


def test_symlinked_gate_file_is_refused(env: InstEnv, tmp_path: Path) -> None:
    path = gate_path(env.ws, env.cid)
    real = tmp_path / "gate-copy.json"
    real.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(real)
    r = env.install("--yes")
    assert r.exit_code == cli.EXIT_USER
    not_installed(env)


def test_unknown_and_invalid_ids(env: InstEnv) -> None:
    for bad in ("20260101-000000-nope", "../escape", "UPPER"):
        r = env.run("skill", "install", bad, "--yes")
        assert r.exit_code == cli.EXIT_USER, bad
    not_installed(env)


# -- refused: the measured roles and skills changed ---------------------------------


def _add_workspace_skill(env: InstEnv, name: str) -> None:
    d = workspace_skills_dir(env.ws) / name
    d.mkdir(parents=True)
    (d / SKILL_FILE).write_text(
        f"---\nname: {name}\ndescription: Something else.\n---\nBe careful.\n", encoding="utf-8"
    )


def test_changed_role_skills_since_compare_is_refused(env: InstEnv) -> None:
    _add_workspace_skill(env, "house-style")
    env.write_config(daily=10.0, extra='\n[roles.writer]\nskills = ["house-style"]\n')
    r = env.install("--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "baseline fingerprint differs" in r.output
    assert not env.target.exists()


def test_unused_new_workspace_skill_does_not_block(env: InstEnv) -> None:
    _add_workspace_skill(env, "house-style")  # no role uses it: same fingerprint
    r = env.install("--yes")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert NAME in load_skills(env.ws)


def test_forged_candidate_fingerprint_is_refused(env: InstEnv) -> None:
    env.rewrite_gate(decision={"candidate_fingerprint": "b" * 64})
    r = env.install("--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "candidate fingerprint differs" in r.output
    not_installed(env)


# -- never replaces anything --------------------------------------------------------


def test_second_install_is_refused_and_keeps_the_first(env: InstEnv) -> None:
    assert env.install("--yes").exit_code == cli.EXIT_OK
    before = (env.target / SKILL_FILE).read_bytes()
    r = env.install("--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "never replaced" in r.output
    assert (env.target / SKILL_FILE).read_bytes() == before


def test_existing_folder_with_the_name_is_not_replaced(env: InstEnv) -> None:
    env.target.mkdir(parents=True)
    (env.target / "notes.txt").write_text("mine", encoding="utf-8")
    r = env.install("--yes")
    assert r.exit_code != cli.EXIT_OK
    assert (env.target / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert not (env.target / SKILL_FILE).exists()


def test_empty_folder_with_the_name_is_not_replaced(env: InstEnv) -> None:
    env.target.mkdir(parents=True)
    r = env.install("--yes")
    assert r.exit_code != cli.EXIT_OK
    assert env.target.is_dir() and os.listdir(env.target) == []


def test_symlink_with_the_name_is_refused(env: InstEnv, tmp_path: Path) -> None:
    workspace_skills_dir(env.ws).mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    env.target.symlink_to(elsewhere)
    r = env.install("--yes")
    assert r.exit_code != cli.EXIT_OK
    assert os.listdir(elsewhere) == []


def test_symlinked_skills_dir_is_refused(env: InstEnv, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    workspace_skills_dir(env.ws).symlink_to(elsewhere)
    r = env.install("--yes")
    assert r.exit_code == cli.EXIT_CONFIG  # the workspace skills do not load at all
    assert os.listdir(elsewhere) == []


def test_skills_path_that_is_a_file_is_refused(env: InstEnv) -> None:
    workspace_skills_dir(env.ws).write_text("x", encoding="utf-8")
    r = env.install("--yes")
    assert r.exit_code != cli.EXIT_OK
    assert workspace_skills_dir(env.ws).read_text(encoding="utf-8") == "x"


# -- the library function: rechecks and all-or-nothing ------------------------------


def _rs(env: InstEnv) -> tuple[Any, SkillRegistry]:
    roles, skills = env.roles_and_skills()
    return roles, skills


def test_check_install_writes_nothing(env: InstEnv) -> None:
    roles, skills = _rs(env)
    check = check_install(env.ws, env.cid, roles=roles, skills=skills)
    assert check.target == env.target and check.skill.name == NAME
    assert not workspace_skills_dir(env.ws).exists()


def test_check_install_rejects_wrong_types(env: InstEnv) -> None:
    roles, skills = _rs(env)
    with pytest.raises(TypeError):
        check_install(env.ws, env.cid, roles=skills, skills=skills)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        check_install(env.ws, env.cid, roles=roles, skills=roles)  # type: ignore[arg-type]


def test_install_rechecks_after_the_preview(env: InstEnv) -> None:
    roles, skills = _rs(env)
    check_install(env.ws, env.cid, roles=roles, skills=skills)
    path = staging_path(env.ws, env.cid) / NAME / SKILL_FILE
    path.write_bytes(path.read_bytes().replace(b"Count", b"Tally"))
    with pytest.raises(SkillCandidateError, match="sha256 differs"):
        install_candidate(env.ws, env.cid, roles=roles, skills=skills)
    not_installed(env)


def test_name_taken_between_check_and_install(
    env: InstEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    roles, skills = _rs(env)
    real_mkdtemp = skill_candidates.tempfile.mkdtemp

    def racing_mkdtemp(*args: Any, **kwargs: Any) -> str:
        tmp = real_mkdtemp(*args, **kwargs)
        env.target.mkdir()  # someone else takes the name meanwhile
        (env.target / "theirs.txt").write_text("theirs", encoding="utf-8")
        return tmp

    monkeypatch.setattr(skill_candidates.tempfile, "mkdtemp", racing_mkdtemp)
    with pytest.raises(SkillCandidateError, match="never replaced"):
        install_candidate(env.ws, env.cid, roles=roles, skills=skills)
    assert sorted(os.listdir(env.target)) == ["theirs.txt"]
    assert env.leftovers() == []


def test_failed_write_leaves_nothing(env: InstEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    roles, skills = _rs(env)

    def broken_rename(src: Any, dst: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(skill_candidates.os, "rename", broken_rename)
    with pytest.raises(OSError):
        install_candidate(env.ws, env.cid, roles=roles, skills=skills)
    monkeypatch.undo()
    not_installed(env)


def test_failed_load_back_removes_the_install(
    env: InstEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    roles, skills = _rs(env)
    monkeypatch.setattr(skill_candidates, "load_skills", lambda *a, **k: SkillRegistry())
    with pytest.raises(SkillCandidateError, match="did not load back"):
        install_candidate(env.ws, env.cid, roles=roles, skills=skills)
    not_installed(env)


def test_broken_load_back_removes_the_install(
    env: InstEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    roles, skills = _rs(env)

    def broken(*a: Any, **k: Any) -> SkillRegistry:
        from automa.skills import SkillError

        raise SkillError("boom")

    monkeypatch.setattr(skill_candidates, "load_skills", broken)
    with pytest.raises(SkillCandidateError, match="boom"):
        install_candidate(env.ws, env.cid, roles=roles, skills=skills)
    not_installed(env)


def test_install_never_touches_the_candidate_store(env: InstEnv) -> None:
    before = sorted(p.name for p in candidates_dir(env.ws).iterdir())
    roles, skills = _rs(env)
    install_candidate(env.ws, env.cid, roles=roles, skills=skills)
    assert sorted(p.name for p in candidates_dir(env.ws).iterdir()) == before


def test_installing_does_not_need_a_provider_or_api_key() -> None:
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(skill_candidates))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & {"make_provider", "LLMGateway", "BudgetGuard", "Provider"}
    src = inspect.getsource(cli.skill_install)
    assert "make_provider" not in src and "LLMGateway" not in src
