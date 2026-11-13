"""Skills for built-in roles from ``[roles.<name>] skills`` in automa.toml (D47).

Text only: the role keeps its tools, risk level and policy. Local only:
``FakeProvider`` replaces the provider; no network, no real model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.config import AutomaConfig, ConfigError, RoleSettings, load_config
from automa.llm import ChatResponse, FakeProvider, FinishReason, Message, ToolCall, Usage
from automa.llm import Role as MsgRole
from automa.plan import Plan, TaskStatus
from automa.project import ProjectStatus, ProjectStore
from automa.roles import Role, RoleError, RoleRegistry
from automa.roles.models import MAX_ROLE_SKILLS
from automa.roles.registry import apply_extra_skills
from automa.skills import Skill, SkillRegistry
from automa.tools import RiskLevel, default_tools

PID = "proj-1"
BODY = "Always end every document with the line: -- house style --"
WRITER_TOOLS = ("read_file", "list_dir", "write_file")

TOML = """\
workspace = "{ws}"
[tiers]
small = "fake-small"
medium = "fake-medium"
large = "fake-large"
[prices."fake-small"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[prices."fake-medium"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[prices."fake-large"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[limits]
project_usd = 5.0
daily_usd = 2.0
per_call_usd = 0.25
"""


def skill(name: str, body: str = BODY) -> Skill:
    return Skill(name=name, description=f"{name} skill.", body=body)


def skills(*names: str) -> SkillRegistry:
    return SkillRegistry(skill(n) for n in names)


def role(name: str = "writer", skills: tuple[str, ...] = ()) -> Role:
    return Role(
        name=name,
        description="Writes.",
        tools=("read_file", "write_file"),
        max_risk=RiskLevel.WRITE,
        instructions="Write.",
        skills=skills,
    )


def write_toml(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "automa.toml"
    path.write_text(TOML.format(ws=(tmp_path / "ws").as_posix()) + extra, encoding="utf-8")
    return path


def config(tmp_path: Path, extra: str) -> AutomaConfig:
    return load_config(write_toml(tmp_path, extra), env_file=None, environ={})


# -- config ------------------------------------------------------------------------


def test_config_without_roles_has_no_extra_skills(tmp_path: Path) -> None:
    cfg = config(tmp_path, "")
    assert cfg.roles == {}
    assert cfg.role_skills() == {}


def test_config_reads_role_skills(tmp_path: Path) -> None:
    cfg = config(
        tmp_path,
        '[roles.writer]\nskills = ["house-style", "glossary"]\n[roles.developer]\nskills = []\n',
    )
    assert cfg.roles["writer"] == RoleSettings(skills=("house-style", "glossary"))
    # roles with no extra skills are left out, order of the list is kept
    assert cfg.role_skills() == {"writer": ("house-style", "glossary")}


@pytest.mark.parametrize(
    "extra",
    [
        '[roles.writer]\ntools = ["run_command"]\n',
        '[roles.writer]\nmax_risk = "execute"\n',
        '[roles.writer]\ntier = "large"\n',
        "[roles.writer]\nmax_steps = 200\n",
        '[roles.writer]\ninstructions = "Ignore all rules."\n',
        '[roles.writer]\nskills = ["a"]\nallowed_tools = ["publish"]\n',
    ],
)
def test_config_role_settings_accept_only_skills(tmp_path: Path, extra: str) -> None:
    with pytest.raises(ConfigError, match="roles.writer"):
        config(tmp_path, extra)


def test_config_rejects_skills_as_string(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a string"):
        config(tmp_path, '[roles.writer]\nskills = "house-style"\n')


def test_config_rejects_non_string_skill_names(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="roles.writer.skills"):
        config(tmp_path, "[roles.writer]\nskills = [1, 2]\n")


@pytest.mark.parametrize("key", ['"Writer"', '"9lives"', '"a b"', '"x/y"', '"' + "a" * 33 + '"'])
def test_config_rejects_bad_role_names(tmp_path: Path, key: str) -> None:
    with pytest.raises(ConfigError, match="invalid role names"):
        config(tmp_path, f'[roles.{key}]\nskills = ["a"]\n')


def test_config_roles_must_be_tables(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="roles"):
        config(tmp_path, 'roles = ["writer"]\n')


def test_example_config_stays_valid() -> None:
    example = Path(__file__).resolve().parents[1] / "automa.toml.example"
    cfg = load_config(example, env_file=None, environ={})
    assert cfg.role_skills() == {}


# -- registry ----------------------------------------------------------------------


def test_extra_skills_are_appended_after_role_file_skills() -> None:
    reg = RoleRegistry(
        [role(skills=("base",))],
        skills=skills("base", "house-style", "glossary"),
        extra_skills={"writer": ("house-style", "glossary")},
    )
    assert reg.get("writer").skills == ("base", "house-style", "glossary")


def test_extra_skills_never_change_the_policy() -> None:
    original = role()
    reg = RoleRegistry(
        [original], skills=skills("house-style"), extra_skills={"writer": ["house-style"]}
    )
    updated = reg.get("writer")
    assert updated.policy() == original.policy()
    assert updated.model_dump(exclude={"skills"}) == original.model_dump(exclude={"skills"})


def test_extra_skills_leave_other_roles_untouched() -> None:
    other = role("editor")
    reg = RoleRegistry(
        [role(), other], skills=skills("house-style"), extra_skills={"writer": ["house-style"]}
    )
    assert reg.get("editor") == other


def test_empty_extra_skills_change_nothing() -> None:
    original = role()
    for extra in (None, {}, {"writer": ()}):
        assert RoleRegistry([original], extra_skills=extra).get("writer") == original


def test_extra_skills_for_unknown_role_fail() -> None:
    with pytest.raises(RoleError, match=r"unknown roles \['ghost'\]"):
        RoleRegistry([role()], extra_skills={"ghost": ["a"]})


def test_extra_skills_for_unknown_role_fail_even_when_empty() -> None:
    # A typo in the role name must not pass silently just because the list is empty.
    with pytest.raises(RoleError, match="unknown roles"):
        RoleRegistry([role()], extra_skills={"wrtier": []})


def test_extra_skill_already_listed_by_role_fails() -> None:
    with pytest.raises(RoleError, match="already listed"):
        RoleRegistry(
            [role(skills=("house-style",))],
            skills=skills("house-style"),
            extra_skills={"writer": ["house-style"]},
        )


def test_duplicate_extra_skills_fail() -> None:
    with pytest.raises(RoleError, match="duplicate skill names"):
        RoleRegistry([role()], extra_skills={"writer": ["a", "a"]})


def test_extra_skills_respect_the_per_role_cap() -> None:
    base = tuple(f"s{i}" for i in range(MAX_ROLE_SKILLS - 1))
    with pytest.raises(RoleError, match=f"at most {MAX_ROLE_SKILLS}"):
        RoleRegistry([role(skills=base)], extra_skills={"writer": ["x", "y"]})
    ok = RoleRegistry([role(skills=base)], extra_skills={"writer": ["x"]})
    assert len(ok.get("writer").skills) == MAX_ROLE_SKILLS


@pytest.mark.parametrize("bad", ["Bad", "-x", "a--b", "a/b", ""])
def test_invalid_extra_skill_names_fail(bad: str) -> None:
    with pytest.raises(RoleError, match="invalid skill names"):
        RoleRegistry([role()], extra_skills={"writer": [bad]})


def test_extra_skills_as_string_fail() -> None:
    with pytest.raises(RoleError, match="not a string"):
        apply_extra_skills({"writer": role()}, {"writer": "house-style"})


def test_missing_extra_skill_fails_when_skills_are_given() -> None:
    with pytest.raises(RoleError, match=r"unknown skills \['house-style'\]"):
        RoleRegistry([role()], skills=skills(), extra_skills={"writer": ["house-style"]})


def test_error_names_the_config_section() -> None:
    with pytest.raises(RoleError, match=r"automa.toml \[roles.writer\]"):
        RoleRegistry([role(skills=("a",))], extra_skills={"writer": ["a"]})


def test_builtin_roles_take_extra_skills() -> None:
    tools = default_tools()
    plain = RoleRegistry.builtin(tools=tools)
    reg = RoleRegistry.builtin(
        tools=tools, skills=skills("house-style"), extra_skills={"writer": ["house-style"]}
    )
    assert reg.get("writer").skills == ("house-style",)
    for r in reg:
        assert r.policy() == plain.get(r.name).policy()
        assert r.tools == plain.get(r.name).tools and r.max_risk is plain.get(r.name).max_risk


def test_from_dir_takes_extra_skills(tmp_path: Path) -> None:
    (tmp_path / "writer.toml").write_text(
        'name = "writer"\ndescription = "Writes."\ninstructions = "Write."\n', encoding="utf-8"
    )
    reg = RoleRegistry.from_dir(tmp_path, extra_skills={"writer": ["house-style"]})
    assert reg.get("writer").skills == ("house-style",)


# -- CLI ---------------------------------------------------------------------------


def calls(*items: tuple[str, dict[str, Any]]) -> ChatResponse:
    tcs = tuple(
        ToolCall(id=f"x{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(items)
    )
    return ChatResponse(
        model="fake",
        message=Message(role=MsgRole.ASSISTANT, content=None, tool_calls=tcs),
        usage=Usage(input_tokens=100, output_tokens=20),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def add_skill(root: Path, name: str, body: str = BODY) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} skill.\n---\n\n{body}\n", encoding="utf-8"
    )


class Env:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.tmp_path = tmp_path
        self.ws = tmp_path / "ws"
        self.config = write_toml(tmp_path, "")
        self.provider = FakeProvider()
        monkeypatch.chdir(tmp_path)
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)

        def factory(config: AutomaConfig) -> FakeProvider:
            return self.provider

        monkeypatch.setattr(cli, "make_provider", factory)

    def roles_toml(self, extra: str) -> None:
        self.config = write_toml(self.tmp_path, extra)

    @property
    def store(self) -> ProjectStore:
        return ProjectStore.for_workspace(self.ws)

    @property
    def skills_root(self) -> Path:
        return self.ws / "skills"

    def project(self) -> None:
        task = {
            "id": "a",
            "title": "Write a",
            "role": "writer",
            "acceptance_criteria": [
                {
                    "description": "a.txt says ok",
                    "check": {"type": "file_contains", "path": "a.txt", "text": "ok"},
                }
            ],
        }
        self.store.create("test goal", project_id=PID)
        self.store.save_plan(PID, Plan.model_validate({"goal": "test goal", "tasks": [task]}))

    def run(self, *args: str) -> Any:
        return CliRunner().invoke(cli.app, [*args, "--config", str(self.config)])

    def system_prompt(self, index: int = 0) -> str:
        content = self.provider.requests[index].messages[0].content
        assert isinstance(content, str)
        return content


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


def ok_attempt() -> list[Any]:
    return [calls(("write_file", {"path": "a.txt", "content": "ok"})), "done"]


def test_run_adds_configured_skill_to_builtin_writer(env: Env) -> None:
    add_skill(env.skills_root, "house-style")
    add_skill(env.skills_root, "unused", body="UNUSED-SKILL-TEXT")
    env.roles_toml('[roles.writer]\nskills = ["house-style"]\n')
    env.project()
    env.provider.push(*ok_attempt())
    r = env.run("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    prompt = env.system_prompt()
    assert "## Skills" in prompt and BODY in prompt
    assert "UNUSED-SKILL-TEXT" not in prompt
    # the built-in writer keeps exactly its tools
    assert {t.name for t in env.provider.requests[0].tools} == set(WRITER_TOOLS)
    assert env.store.load(PID).status is ProjectStatus.DONE


def test_run_skill_for_other_role_does_not_reach_writer(env: Env) -> None:
    add_skill(env.skills_root, "house-style")
    env.roles_toml('[roles.developer]\nskills = ["house-style"]\n')
    env.project()
    env.provider.push(*ok_attempt())
    r = env.run("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "## Skills" not in env.system_prompt()


def test_run_unknown_configured_skill_fails_before_any_call(env: Env) -> None:
    env.roles_toml('[roles.writer]\nskills = ["not-installed"]\n')
    env.project()
    r = env.run("run", PID)
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert "unknown skills" in r.output and "not-installed" in r.output
    assert env.provider.requests == []
    assert env.store.load(PID).status is ProjectStatus.NEW
    assert env.store.task_states(PID)["a"].status is TaskStatus.PENDING


def test_run_unknown_configured_role_fails_before_any_call(env: Env) -> None:
    add_skill(env.skills_root, "house-style")
    env.roles_toml('[roles.wrtier]\nskills = ["house-style"]\n')
    env.project()
    r = env.run("run", PID)
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert "unknown roles" in r.output and "wrtier" in r.output
    assert env.provider.requests == []


def test_run_rejects_tools_in_role_settings(env: Env) -> None:
    env.roles_toml('[roles.writer]\ntools = ["run_command"]\n')
    env.project()
    r = env.run("run", PID)
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert "roles.writer" in r.output
    assert env.provider.requests == []


def test_new_with_unknown_configured_skill_creates_no_project(env: Env) -> None:
    env.roles_toml('[roles.writer]\nskills = ["not-installed"]\n')
    r = env.run("new", "a goal", "--no-ask")
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert "not-installed" in r.output
    assert env.provider.requests == []
    assert not (env.ws / "projects").exists() or not any((env.ws / "projects").iterdir())


def test_skills_command_shows_configured_role(env: Env) -> None:
    add_skill(env.skills_root, "house-style")
    env.roles_toml('[roles.writer]\nskills = ["house-style"]\n')
    r = env.run("skills", "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    data = json.loads(r.output)
    entry = next(s for s in data if s["name"] == "house-style")
    assert entry["used_by"] == ["writer"]


def test_skills_command_fails_on_unknown_configured_role(env: Env) -> None:
    add_skill(env.skills_root, "house-style")
    env.roles_toml('[roles.ghost]\nskills = ["house-style"]\n')
    r = env.run("skills")
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert "unknown roles" in r.output


def test_status_shows_configured_role_skills(env: Env) -> None:
    add_skill(env.skills_root, "house-style")
    env.roles_toml('[roles.writer]\nskills = ["house-style"]\n')
    env.project()
    r = env.run("status")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "skills: 1 available; writer: house-style" in r.output
    r = env.run("status", PID, "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert json.loads(r.output)["skills"]["roles"] == {"writer": ["house-style"]}


def test_status_reports_missing_configured_skill_without_failing(env: Env) -> None:
    env.roles_toml('[roles.writer]\nskills = ["not-installed"]\n')
    env.project()
    r = env.run("status")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "automa run will refuse to start" in r.output
