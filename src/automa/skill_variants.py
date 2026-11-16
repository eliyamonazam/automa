"""Skill-set fingerprints and candidate variants for benchmark comparisons (D89).

Two building blocks of the comparative benchmark run for automatically created
skills (docs/ARCHITECTURE.md, "Skill Engine"):

* ``skillset_fingerprint(roles, skills)`` — a SHA-256 over everything that the
  roles and skills put into a worker's system prompt: every role (all fields,
  including its tools, risk level and instructions) and the full text of every
  skill some role uses. Unused skills do not change what a worker is told, so
  they do not change the fingerprint. ``run_benchmark`` stores it in each
  ``BenchmarkRunRecord`` so runs made with different skill sets are never mixed
  up, and ``evaluate_skill_gate`` refuses sides that are not one skill set each.
* ``with_candidate_skill(roles, skills, candidate, role=...)`` — the candidate
  side: the same roles with ``candidate`` appended to one role's skills and the
  same skills plus ``candidate``. Text only: the role's tools, risk level and
  policy stay exactly the same (checked twice), and a candidate may not reuse an
  existing skill name.

Pure: no model, no file, no network. Nothing is installed anywhere.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from automa.roles import RoleError, RoleRegistry
from automa.skills import Skill, SkillError, SkillRegistry

FINGERPRINT_VERSION = 1
FINGERPRINT_PATTERN = r"^[0-9a-f]{64}$"


class SkillVariantError(ValueError):
    """A candidate variant cannot be built (bad role, name clash, changed policy)."""


def _skill_data(skill: Skill) -> dict[str, Any]:
    return {
        "name": skill.name,
        "description": skill.description,
        "license": skill.license,
        "body": skill.body,
    }


def skillset_fingerprint(roles: RoleRegistry, skills: SkillRegistry | None) -> str:
    """Deterministic SHA-256 (hex) of the roles and the skills they use.

    A skill a role lists but ``skills`` lacks is recorded as missing (the worker
    would refuse to run that role anyway); it never raises for that.
    """
    if not isinstance(roles, RoleRegistry):
        raise TypeError("roles must be a RoleRegistry")
    if skills is not None and not isinstance(skills, SkillRegistry):
        raise TypeError("skills must be a SkillRegistry or None")
    used = sorted({name for role in roles for name in role.skills})
    skill_items: list[dict[str, Any]] = []
    for name in used:
        if skills is not None and name in skills:
            skill_items.append(_skill_data(skills.get(name)))
        else:
            skill_items.append({"name": name, "missing": True})
    payload = {
        "v": FINGERPRINT_VERSION,
        "roles": [role.model_dump(mode="json") for role in roles],  # sorted by name
        "skills": skill_items,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def with_candidate_skill(
    roles: RoleRegistry, skills: SkillRegistry, candidate: Skill, *, role: str
) -> tuple[RoleRegistry, SkillRegistry]:
    """The candidate side: ``candidate`` added to ``role`` (text only) and to the skills.

    Raises ``SkillVariantError`` for an unknown role, a candidate whose name is
    already a skill, a role already at its skill cap, or any change other than
    the one added skill name.
    """
    if not isinstance(roles, RoleRegistry):
        raise TypeError("roles must be a RoleRegistry")
    if not isinstance(skills, SkillRegistry):
        raise TypeError("skills must be a SkillRegistry")
    if not isinstance(candidate, Skill):
        raise TypeError("candidate must be a Skill")
    if role not in roles:
        raise SkillVariantError(f"unknown role {role!r} (known: {sorted(roles.names())})")
    if candidate.name in skills:
        raise SkillVariantError(
            f"candidate skill {candidate.name!r} has the name of an existing skill"
        )
    try:
        new_skills = SkillRegistry([*skills, candidate])
        new_roles = RoleRegistry(
            list(roles), skills=new_skills, extra_skills={role: [candidate.name]}
        )
    except (RoleError, SkillError) as e:
        raise SkillVariantError(f"cannot add candidate skill to role {role!r}: {e}") from None

    # Defence in depth: exactly one change — the candidate's name on one role.
    if new_roles.names() != roles.names():
        raise SkillVariantError("adding the candidate changed the set of roles")
    for old in roles:
        new = new_roles.get(old.name)
        if new.policy() != old.policy():
            raise SkillVariantError(f"adding the candidate changed the policy of {old.name!r}")
        expected = old.skills + (candidate.name,) if old.name == role else old.skills
        if new.model_dump(exclude={"skills"}) != old.model_dump(exclude={"skills"}) or (
            new.skills != expected
        ):
            raise SkillVariantError(f"adding the candidate changed role {old.name!r}")
    return new_roles, new_skills
