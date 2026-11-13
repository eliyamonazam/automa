"""Tests for connecting skills to roles (Role.skills, RoleRegistry and WorkerAgent).

Local only: FakeProvider behind a real LLMGateway + BudgetGuard. No real model calls.
Skills only add text to the worker's system prompt; they never change the policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from automa.agents import WorkerAgent
from automa.agents.worker import (
    MAX_SKILLS_PROMPT_CHARS,
    SKILLS_HEADER,
    WORKER_PROTOCOL,
    skills_prompt,
    system_prompt,
)
from automa.plan import ResultStatus
from automa.roles import MAX_ROLE_SKILLS, Role, RoleError, RoleRegistry, parse_role
from automa.roles.registry import BUILTIN_ROLES_DIR
from automa.skills import Skill, SkillRegistry
from automa.tools import default_tools
from test_worker import Rig, make_task


def role(**kw: Any) -> Role:
    d: dict[str, Any] = {
        "name": "developer",
        "description": "Writes code.",
        "tools": ["read_file", "write_file"],
        "max_risk": "write",
        "instructions": "You are a developer.",
    }
    d.update(kw)
    return Role.model_validate(d)


def skill(name: str, body: str = "Do it well.", description: str = "How to do X.") -> Skill:
    return Skill(name=name, description=description, body=body)


def registry(*names: str) -> SkillRegistry:
    return SkillRegistry(skill(n, body=f"Body of {n}.") for n in names)


ROLE_TOML = """\
name = "developer"
description = "Writes code."
tools = ["read_file", "write_file"]
max_risk = "write"
instructions = "You are a developer."
"""


# --- Role.skills field ---------------------------------------------------------


def test_skills_default_empty() -> None:
    assert role().skills == ()


def test_skills_from_toml() -> None:
    r = parse_role(ROLE_TOML + 'skills = ["pdf-report", "tests"]\n', expected_name="developer")
    assert r.skills == ("pdf-report", "tests")


@pytest.mark.parametrize(
    "bad",
    [
        "pdf-report",  # a string, not a list
        ["Bad"],
        ["-x"],
        ["x-"],
        ["a--b"],
        ["a b"],
        [""],
        [1],
        ["../etc"],
        ["a" * 65],
        ["dup", "dup"],
    ],
)
def test_bad_skill_lists_rejected(bad: Any) -> None:
    with pytest.raises(ValidationError):
        role(skills=bad)


def test_too_many_skills_rejected() -> None:
    names = [f"s{i}" for i in range(MAX_ROLE_SKILLS + 1)]
    with pytest.raises(ValidationError, match="at most"):
        role(skills=names)
    assert len(role(skills=names[:MAX_ROLE_SKILLS]).skills) == MAX_ROLE_SKILLS


def test_skills_do_not_change_policy() -> None:
    plain = role()
    with_skills = role(skills=["pdf-report", "tests"])
    assert with_skills.policy() == plain.policy()


def test_skills_is_immutable() -> None:
    r = role(skills=["a"])
    with pytest.raises(ValidationError):
        r.skills = ("b",)  # type: ignore[misc]


# --- RoleRegistry skill checks ---------------------------------------------------


def test_registry_without_skill_registry_does_not_check() -> None:
    reg = RoleRegistry([role(skills=["missing"])])
    assert reg.get("developer").skills == ("missing",)


def test_registry_rejects_unknown_skill() -> None:
    with pytest.raises(RoleError, match="unknown skills"):
        RoleRegistry([role(skills=["a", "missing"])], skills=registry("a"))


def test_registry_rejects_skill_with_empty_skill_registry() -> None:
    with pytest.raises(RoleError, match="unknown skills"):
        RoleRegistry([role(skills=["a"])], skills=SkillRegistry())


def test_registry_accepts_known_skills() -> None:
    reg = RoleRegistry([role(skills=["a", "b"])], skills=registry("a", "b", "c"))
    assert reg.get("developer").skills == ("a", "b")


def test_registry_rejects_wrong_skills_type() -> None:
    with pytest.raises(TypeError):
        RoleRegistry([role()], skills={"a": skill("a")})  # type: ignore[arg-type]


def test_from_dir_checks_skills(tmp_path: Path) -> None:
    (tmp_path / "developer.toml").write_text(ROLE_TOML + 'skills = ["a"]\n', encoding="utf-8")
    assert RoleRegistry.from_dir(tmp_path, skills=registry("a")).get("developer").skills == ("a",)
    with pytest.raises(RoleError, match="unknown skills"):
        RoleRegistry.from_dir(tmp_path, skills=registry("b"))


def test_builtin_roles_load_with_empty_skills() -> None:
    reg = RoleRegistry.builtin(tools=default_tools(), skills=SkillRegistry())
    assert len(reg) > 0
    assert all(r.skills == () for r in reg)
    assert BUILTIN_ROLES_DIR.is_dir()


# --- system prompt ---------------------------------------------------------------


def test_prompt_without_skills_is_unchanged() -> None:
    r = role()
    expected = r.instructions.strip() + "\n" + WORKER_PROTOCOL
    assert system_prompt(r) == expected
    assert system_prompt(r, registry("a")) == expected
    assert skills_prompt(r, None) == ""


def test_prompt_order_instructions_skills_protocol() -> None:
    r = role(skills=["b", "a"])
    text = system_prompt(r, registry("a", "b"))
    i_instr = text.index("You are a developer.")
    i_header = text.index(SKILLS_HEADER.strip())
    i_b = text.index("Body of b.")
    i_a = text.index("Body of a.")
    i_proto = text.index(WORKER_PROTOCOL.strip())
    # Role order decides precedence, not alphabetical order.
    assert i_instr < i_header < i_b < i_a < i_proto
    assert "### Skill `b`: How to do X." in text
    assert text.endswith(WORKER_PROTOCOL)


def test_prompt_states_skills_grant_nothing() -> None:
    assert "do not" in SKILLS_HEADER and "permission" in SKILLS_HEADER


def test_prompt_only_includes_listed_skills() -> None:
    text = system_prompt(role(skills=["a"]), registry("a", "b"))
    assert "Body of a." in text and "Body of b." not in text


def test_prompt_missing_registry_is_error() -> None:
    with pytest.raises(ValueError, match="none are loaded"):
        system_prompt(role(skills=["a"]), None)


def test_prompt_missing_skill_is_error() -> None:
    with pytest.raises(ValueError, match="unknown skills"):
        system_prompt(role(skills=["a", "zz"]), registry("a"))


def test_prompt_size_limit() -> None:
    body = "x" * 19_000
    reg = SkillRegistry(skill(f"s{i}", body=body) for i in range(4))
    within = role(skills=["s0", "s1", "s2"])
    assert len(skills_prompt(within, reg)) <= MAX_SKILLS_PROMPT_CHARS
    with pytest.raises(ValueError, match="limit"):
        skills_prompt(role(skills=["s0", "s1", "s2", "s3"]), reg)


# --- WorkerAgent -----------------------------------------------------------------


def test_worker_rejects_wrong_skills_type(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [])
    with pytest.raises(TypeError):
        WorkerAgent(rig.gateway, skills={"a": skill("a")})  # type: ignore[arg-type]


def test_worker_sends_skill_text_in_system_prompt(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["done"])
    base = rig.roles.get("developer")
    r = base.model_copy(update={"skills": ("pdf-report",)})
    reg = SkillRegistry([skill("pdf-report", body="Always add a title page.")])
    result = rig.run(role=r, executor=rig.executor(base), agent_kw={"skills": reg})
    assert result.status is ResultStatus.COMPLETED
    system = rig.provider.requests[0].messages[0]
    assert system.content == system_prompt(r, reg)
    assert "Always add a title page." in (system.content or "")
    started = rig.events.events[0]
    assert started.type == "worker.started" and started.data["skills"] == ["pdf-report"]


def test_worker_skills_do_not_widen_tools(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["done"])
    base = rig.roles.get("writer")
    r = base.model_copy(update={"skills": ("a",)})
    rig.run(
        role=r,
        task=make_task("writer"),
        executor=rig.executor(base),
        agent_kw={"skills": registry("a")},
    )
    sent = {t.name for t in rig.provider.requests[0].tools}
    assert sent <= set(base.tools)


def test_worker_missing_skill_fails_before_model_call(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["done"])
    base = rig.roles.get("developer")
    r = base.model_copy(update={"skills": ("missing",)})
    with pytest.raises(ValueError, match="unknown skills"):
        rig.run(role=r, executor=rig.executor(base), agent_kw={"skills": registry("a")})
    with pytest.raises(ValueError, match="none are loaded"):
        rig.run(role=r, executor=rig.executor(base))
    assert rig.provider.requests == []


def test_worker_without_skills_prompt_unchanged(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["done"])
    r = rig.roles.get("developer")
    rig.run(role=r, agent_kw={"skills": registry("a")})
    system = rig.provider.requests[0].messages[0]
    assert system.content == r.instructions.strip() + "\n" + WORKER_PROTOCOL
    assert rig.events.events[0].data["skills"] == []
