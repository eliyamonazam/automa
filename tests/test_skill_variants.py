"""Tests for skill-set fingerprints, candidate variants and the gate's skill-set checks (D89)."""

from __future__ import annotations

import ast
import re
from datetime import UTC, datetime, timedelta
from itertools import count
from pathlib import Path
from typing import Any

import pytest

from automa.benchmarks.results import BenchmarkRunRecord
from automa.plan.models import CheckResult
from automa.roles import RoleRegistry
from automa.roles.models import Role
from automa.skill_gate import SkillGateError, evaluate_skill_gate
from automa.skill_variants import (
    FINGERPRINT_PATTERN,
    SkillVariantError,
    skillset_fingerprint,
    with_candidate_skill,
)
from automa.skills import Skill, SkillRegistry
from automa.tools import default_tools


def skill(name: str = "tidy", body: str = "Keep functions short.", **over: Any) -> Skill:
    return Skill(name=name, description=f"{name} guidance", body=body, **over)


def role(name: str = "coder", skills: tuple[str, ...] = (), **over: Any) -> Role:
    data: dict[str, Any] = {
        "name": name,
        "description": f"{name} role",
        "tools": ["read_file", "write_file"],
        "max_risk": "write",
        "instructions": f"You are the {name}.",
        "skills": skills,
    }
    data.update(over)
    return Role.model_validate(data)


def registry(*roles: Role, skills: SkillRegistry | None = None) -> RoleRegistry:
    return RoleRegistry(list(roles), skills=skills)


# --- fingerprint -------------------------------------------------------------------


def test_fingerprint_is_hex_sha256_and_deterministic() -> None:
    skills = SkillRegistry([skill()])
    roles = registry(role(skills=("tidy",)), role("writer"), skills=skills)
    fp = skillset_fingerprint(roles, skills)
    assert re.fullmatch(FINGERPRINT_PATTERN, fp)
    again = registry(role("writer"), role(skills=("tidy",)), skills=SkillRegistry([skill()]))
    assert skillset_fingerprint(again, SkillRegistry([skill()])) == fp


def test_fingerprint_changes_with_skill_text_and_role_fields() -> None:
    skills = SkillRegistry([skill()])
    roles = registry(role(skills=("tidy",)), skills=skills)
    fp = skillset_fingerprint(roles, skills)
    for other in (
        SkillRegistry([skill(body="Keep functions tiny.")]),
        SkillRegistry([Skill(name="tidy", description="changed", body="Keep functions short.")]),
        SkillRegistry([skill(license="MIT")]),
    ):
        assert skillset_fingerprint(registry(role(skills=("tidy",)), skills=other), other) != fp
    changed_role = registry(role(skills=("tidy",), instructions="Other."), skills=skills)
    assert skillset_fingerprint(changed_role, skills) != fp
    more_tools = registry(
        role(skills=("tidy",), tools=["read_file", "write_file", "list_dir"]), skills=skills
    )
    assert skillset_fingerprint(more_tools, skills) != fp


def test_fingerprint_ignores_unused_skills_but_not_skill_order_on_a_role() -> None:
    one = SkillRegistry([skill()])
    two = SkillRegistry([skill(), skill("unused")])
    roles = registry(role(skills=("tidy",)), skills=two)
    assert skillset_fingerprint(roles, one) == skillset_fingerprint(roles, two)
    ab = SkillRegistry([skill("a"), skill("b")])
    first = registry(role(skills=("a", "b")), skills=ab)
    second = registry(role(skills=("b", "a")), skills=ab)
    assert skillset_fingerprint(first, ab) != skillset_fingerprint(second, ab)


def test_fingerprint_with_missing_skills_does_not_raise_and_differs() -> None:
    roles = registry(role(skills=("tidy",)))
    present = SkillRegistry([skill()])
    assert skillset_fingerprint(roles, None) == skillset_fingerprint(roles, SkillRegistry())
    assert skillset_fingerprint(roles, None) != skillset_fingerprint(roles, present)


def test_fingerprint_type_checks() -> None:
    with pytest.raises(TypeError):
        skillset_fingerprint([role()], None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        skillset_fingerprint(registry(role()), [skill()])  # type: ignore[arg-type]


def test_builtin_roles_have_a_fingerprint() -> None:
    skills = SkillRegistry()
    roles = RoleRegistry.builtin(tools=default_tools(), skills=skills)
    assert re.fullmatch(FINGERPRINT_PATTERN, skillset_fingerprint(roles, skills))


# --- candidate variant -------------------------------------------------------------


def test_candidate_is_added_to_one_role_only_as_text() -> None:
    base_skills = SkillRegistry([skill()])
    roles = registry(role(skills=("tidy",)), role("writer"), skills=base_skills)
    new_roles, new_skills = with_candidate_skill(
        roles, base_skills, skill("tests-first"), role="coder"
    )
    assert new_skills.names() == {"tidy", "tests-first"}
    assert new_roles.get("coder").skills == ("tidy", "tests-first")
    assert new_roles.get("writer") == roles.get("writer")
    for r in roles:
        assert new_roles.get(r.name).policy() == r.policy()
    # the inputs are unchanged and the two sides differ
    assert roles.get("coder").skills == ("tidy",) and base_skills.names() == {"tidy"}
    assert skillset_fingerprint(new_roles, new_skills) != skillset_fingerprint(roles, base_skills)


def test_candidate_on_builtin_roles() -> None:
    skills = SkillRegistry()
    roles = RoleRegistry.builtin(tools=default_tools(), skills=skills)
    target = sorted(roles.names())[0]
    new_roles, new_skills = with_candidate_skill(roles, skills, skill(), role=target)
    assert new_roles.get(target).skills[-1] == "tidy"
    assert new_roles.get(target).policy() == roles.get(target).policy()
    assert len(new_skills) == 1


def test_candidate_errors() -> None:
    base_skills = SkillRegistry([skill()])
    roles = registry(role(skills=("tidy",)), skills=base_skills)
    with pytest.raises(SkillVariantError, match="unknown role"):
        with_candidate_skill(roles, base_skills, skill("new"), role="nobody")
    with pytest.raises(SkillVariantError, match="existing skill"):
        with_candidate_skill(roles, base_skills, skill(body="different"), role="coder")
    full = tuple(f"s{i}" for i in range(8))
    full_skills = SkillRegistry([skill(n) for n in full])
    full_roles = registry(role(skills=full), skills=full_skills)
    with pytest.raises(SkillVariantError, match="cannot add"):
        with_candidate_skill(full_roles, full_skills, skill("extra"), role="coder")


def test_candidate_type_checks() -> None:
    skills = SkillRegistry()
    roles = registry(role())
    with pytest.raises(TypeError):
        with_candidate_skill([role()], skills, skill(), role="coder")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        with_candidate_skill(roles, None, skill(), role="coder")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        with_candidate_skill(roles, skills, {"name": "x"}, role="coder")  # type: ignore[arg-type]


def test_candidate_cannot_carry_permissions() -> None:
    # A skill is text only: the model rejects any extra field such as tools.
    with pytest.raises(ValueError):
        Skill.model_validate(
            {"name": "x", "description": "d", "body": "b", "allowed_tools": ["shell"]}
        )


# --- the gate requires one distinct skill set per side ------------------------------

_ids = count()


def rec(fp: str | None, ok: int = 2, **over: Any) -> BenchmarkRunRecord:
    t0 = datetime(2026, 11, 10, tzinfo=UTC)
    passed = ok == 2
    data: dict[str, Any] = {
        "benchmark_id": "cli-todo",
        "title": "t",
        "kind": "cli",
        "project_id": f"bench-{next(_ids)}-cli-todo",
        "automa_version": "0.1.0",
        "started_at": t0,
        "finished_at": t0 + timedelta(seconds=1),
        "stop": "completed" if passed else "tasks_failed",
        "project_status": "done" if passed else "failed",
        "planned": True,
        "passed": passed,
        "checks": [CheckResult(criterion=i, passed=i < ok) for i in range(2)],
        "score": ok / 2,
        "budget_usd": 0.5,
        "cost_nusd": 10,
        "model_calls": 1,
        "attempts": 1,
        "tasks_passed": ["build"] if passed else [],
        "tasks_failed": [] if passed else ["build"],
        "skills_fingerprint": fp,
    }
    data.update(over)
    return BenchmarkRunRecord.model_validate(data)


A, B, C = "a" * 64, "b" * 64, "c" * 64


def test_gate_accepts_two_distinct_skill_sets_and_reports_them() -> None:
    d = evaluate_skill_gate([rec(A, 1) for _ in range(3)], [rec(B) for _ in range(3)])
    assert d.accepted
    assert d.baseline_fingerprint == A and d.candidate_fingerprint == B
    data = d.event_data()
    assert data["baseline_fingerprint"] == A and data["candidate_fingerprint"] == B


def test_gate_refuses_same_skill_set_on_both_sides() -> None:
    with pytest.raises(SkillGateError, match="same skill set"):
        evaluate_skill_gate([rec(A, 1) for _ in range(3)], [rec(A) for _ in range(3)])


def test_gate_refuses_mixed_skill_sets_within_a_side() -> None:
    with pytest.raises(SkillGateError, match="baseline runs used 2 different"):
        evaluate_skill_gate([rec(A, 1), rec(A, 1), rec(C, 1)], [rec(B) for _ in range(3)])
    with pytest.raises(SkillGateError, match="candidate runs used 2 different"):
        evaluate_skill_gate([rec(A, 1) for _ in range(3)], [rec(B), rec(B), rec(C)])


def test_gate_refuses_runs_without_fingerprint() -> None:
    with pytest.raises(SkillGateError, match="baseline runs without a skill-set fingerprint"):
        evaluate_skill_gate([rec(A, 1), rec(None, 1), rec(A, 1)], [rec(B) for _ in range(3)])
    with pytest.raises(SkillGateError, match="candidate runs without"):
        evaluate_skill_gate([rec(A, 1) for _ in range(3)], [rec(None) for _ in range(3)])


def test_candidate_runs_cannot_be_relabelled_as_baseline() -> None:
    # Putting a candidate run among the baseline (to hide a regression) is caught.
    base = [rec(A, 2), rec(A, 2), rec(B, 0)]
    with pytest.raises(SkillGateError, match="different skill sets"):
        evaluate_skill_gate(base, [rec(B, 2) for _ in range(3)])


# --- structure ---------------------------------------------------------------------


def test_module_has_no_model_network_or_write_paths() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "automa" / "skill_variants.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    banned = {
        "automa.llm",
        "automa.budget",
        "automa.agents",
        "automa.tools",
        "os",
        "pathlib",
        "subprocess",
        "socket",
        "httpx",
        "shutil",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(alias.name == b or alias.name.startswith(b + ".") for b in banned)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not any(mod == b or mod.startswith(b + ".") for b in banned), mod
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open"
