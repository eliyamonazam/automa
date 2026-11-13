"""``automa skills`` and the skills line in ``automa status`` (D46).

Read-only commands: no API key, no provider, no model call, no files written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.config import AutomaConfig
from automa.project import ProjectStore
from automa.roles import Role, RoleRegistry
from automa.skills import SkillRegistry
from automa.tools import RiskLevel

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


def skill_md(name: str, description: str = "House writing style.", body: str = "Be brief.") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"


def add_skill(root: Path, name: str, text: str | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text if text is not None else skill_md(name), encoding="utf-8")
    return d


def tree(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        os.path.relpath(os.path.join(d, f), root)
        for d, dirs, files in os.walk(root)
        for f in [*dirs, *files]
    }


class Env:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.ws = tmp_path / "ws"
        self.config = tmp_path / "automa.toml"
        self.config.write_text(TOML.format(ws=self.ws.as_posix()), encoding="utf-8")
        self.monkeypatch = monkeypatch
        monkeypatch.chdir(tmp_path)
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)

        def no_provider(config: AutomaConfig) -> Any:
            raise AssertionError("read-only command must not build a provider")

        monkeypatch.setattr(cli, "make_provider", no_provider)

    @property
    def skills_root(self) -> Path:
        return self.ws / "skills"

    def roles_with_skills(self, **by_role: tuple[str, ...]) -> None:
        """Replace the built-in roles with roles listing the given skills."""

        class _Roles:
            @staticmethod
            def builtin(
                *,
                tools: Any,
                skills: SkillRegistry | None = None,
                extra_skills: Any = None,
            ) -> RoleRegistry:
                roles = [
                    Role(
                        name=name,
                        description="Writes.",
                        tools=("read_file", "write_file"),
                        max_risk=RiskLevel.WRITE,
                        instructions="Write.",
                        skills=names,
                    )
                    for name, names in by_role.items()
                ]
                return RoleRegistry(roles, tools=tools, skills=skills, extra_skills=extra_skills)

        self.monkeypatch.setattr(cli, "RoleRegistry", _Roles)

    def run(self, *args: str) -> Any:
        return CliRunner().invoke(cli.app, [*args, "--config", str(self.config)])


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


# -- automa skills -------------------------------------------------------------


def test_skills_empty_creates_nothing(env: Env) -> None:
    r = env.run("skills")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "no skills yet" in r.output and "SKILL.md" in r.output
    assert not env.ws.exists()


def test_skills_empty_json(env: Env) -> None:
    r = env.run("skills", "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert json.loads(r.output) == []


def test_skills_lists_workspace_skills_sorted(env: Env) -> None:
    add_skill(env.skills_root, "zeta", skill_md("zeta", "Last one."))
    add_skill(env.skills_root, "alpha", skill_md("alpha", "First one."))
    before = tree(env.ws)
    r = env.run("skills")
    assert r.exit_code == cli.EXIT_OK, r.output
    lines = r.output.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("alpha  workspace") and "roles: -" in lines[0]
    assert "First one." in lines[0]
    assert lines[1].startswith("zeta")
    assert tree(env.ws) == before  # read-only


def test_skills_json_fields(env: Env) -> None:
    add_skill(
        env.skills_root,
        "house-style",
        "---\nname: house-style\ndescription: Style.\nlicense: MIT\n---\nBe brief.\n",
    )
    r = env.run("skills", "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert json.loads(r.output) == [
        {
            "name": "house-style",
            "source": "workspace",
            "description": "Style.",
            "license": "MIT",
            "used_by": [],
            "body_chars": len("Be brief."),
        }
    ]


def test_skills_shows_roles_that_use_them(env: Env) -> None:
    add_skill(env.skills_root, "house-style")
    add_skill(env.skills_root, "tone")
    add_skill(env.skills_root, "unused")
    env.roles_with_skills(writer=("house-style", "tone"), editor=("tone",))
    r = env.run("skills", "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    used = {row["name"]: row["used_by"] for row in json.loads(r.output)}
    assert used == {"house-style": ["writer"], "tone": ["editor", "writer"], "unused": []}
    text = env.run("skills").output
    assert "roles: editor, writer" in text


def test_skills_marks_builtin_source(env: Env, tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    add_skill(builtin, "shipped")
    add_skill(env.skills_root, "mine")
    env.monkeypatch.setattr(cli, "builtin_skills", lambda: SkillRegistry.from_dir(builtin))
    rows = json.loads(env.run("skills", "--json").output)
    assert {r["name"]: r["source"] for r in rows} == {"mine": "workspace", "shipped": "builtin"}


def test_skills_name_clash_is_config_error(env: Env, tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    add_skill(builtin, "same")
    add_skill(env.skills_root, "same")
    env.monkeypatch.setattr(cli, "builtin_skills", lambda: SkillRegistry.from_dir(builtin))
    r = env.run("skills")
    assert r.exit_code == cli.EXIT_CONFIG
    assert "same name as built-in" in r.output


def test_skills_broken_skill_is_config_error(env: Env) -> None:
    add_skill(env.skills_root, "bad", "---\nname: other\ndescription: x\n---\nbody\n")
    r = env.run("skills")
    assert r.exit_code == cli.EXIT_CONFIG
    assert "workspace skills" in r.output


def test_skills_tools_field_is_config_error(env: Env) -> None:
    add_skill(env.skills_root, "evil", "---\nname: evil\ndescription: x\ntools: x\n---\nb\n")
    r = env.run("skills", "--json")
    assert r.exit_code == cli.EXIT_CONFIG
    assert "never grants tools" in r.output


def test_skills_role_with_missing_skill_is_config_error(env: Env) -> None:
    env.roles_with_skills(writer=("not-installed",))
    r = env.run("skills")
    assert r.exit_code == cli.EXIT_CONFIG
    assert "unknown skills" in r.output and "not-installed" in r.output


def test_skills_symlinked_dir_is_config_error(env: Env, tmp_path: Path) -> None:
    real = tmp_path / "elsewhere"
    add_skill(real, "x")
    env.ws.mkdir()
    env.skills_root.symlink_to(real, target_is_directory=True)
    r = env.run("skills")
    assert r.exit_code == cli.EXIT_CONFIG


def test_skills_escapes_control_characters_in_description(env: Env) -> None:
    add_skill(env.skills_root, "x", skill_md("x", "Nice\x1b[2Jtext"))
    r = env.run("skills")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "\x1b" not in r.output and "\\x1b" in r.output


def test_skills_persian_description(env: Env) -> None:
    add_skill(env.skills_root, "fa", skill_md("fa", "سبک نوشتن فارسی"))
    r = env.run("skills", "--json")
    assert json.loads(r.output)[0]["description"] == "سبک نوشتن فارسی"
    assert "سبک نوشتن فارسی" in env.run("skills").output


def test_skills_invalid_config(env: Env) -> None:
    env.config.write_text("workspace = [", encoding="utf-8")
    assert env.run("skills").exit_code == cli.EXIT_CONFIG


# -- automa status -------------------------------------------------------------


def make_project(env: Env) -> str:
    ProjectStore.for_workspace(env.ws).create("goal", project_id="p1")
    return "p1"


def test_status_one_shows_no_skills(env: Env) -> None:
    pid = make_project(env)
    r = env.run("status", pid)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "skills: none" in r.output


def test_status_one_shows_role_skills(env: Env) -> None:
    pid = make_project(env)
    add_skill(env.skills_root, "house-style")
    add_skill(env.skills_root, "tone")
    env.roles_with_skills(writer=("house-style", "tone"))
    r = env.run("status", pid)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "skills: 2 available; writer: house-style, tone" in r.output
    data = json.loads(env.run("status", pid, "--json").output)
    assert data["skills"] == {
        "available": ["house-style", "tone"],
        "roles": {"writer": ["house-style", "tone"]},
        "error": None,
    }


def test_status_unused_skills(env: Env) -> None:
    pid = make_project(env)
    add_skill(env.skills_root, "house-style")
    r = env.run("status", pid)
    assert "skills: 1 available; no role uses them" in r.output


def test_status_reports_broken_skills_without_failing(env: Env) -> None:
    pid = make_project(env)
    add_skill(env.skills_root, "bad", "---\nname: other\ndescription: x\n---\nbody\n")
    r = env.run("status", pid)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "skills: error:" in r.output and "automa run will refuse to start" in r.output
    data = json.loads(env.run("status", pid, "--json").output)
    assert data["skills"]["error"] and data["skills"]["available"] == []


def test_status_missing_role_skill_is_reported(env: Env) -> None:
    pid = make_project(env)
    env.roles_with_skills(writer=("gone",))
    r = env.run("status", pid)
    assert r.exit_code == cli.EXIT_OK
    assert "skills: error:" in r.output and "gone" in r.output


def test_status_all_shows_skills_line(env: Env) -> None:
    make_project(env)
    add_skill(env.skills_root, "house-style")
    r = env.run("status")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "skills: 1 available" in r.output


def test_status_all_json_unchanged_shape(env: Env) -> None:
    make_project(env)
    rows = json.loads(env.run("status", "--json").output)
    assert isinstance(rows, list) and "skills" not in rows[0]


def test_status_is_read_only_with_skills(env: Env) -> None:
    pid = make_project(env)
    add_skill(env.skills_root, "house-style")
    before = tree(env.ws)
    env.run("status", pid)
    env.run("status", pid, "--json")
    env.run("status")
    assert tree(env.ws) == before
