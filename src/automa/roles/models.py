"""Role definitions: data, not code.

A role is a TOML file ``<name>.toml`` with a system prompt (``instructions``),
the tools it may use, the highest risk level it may reach, a default model
tier and optional ``skills`` (names of ``SKILL.md`` skills whose text is added
to the worker's system prompt; skills never change the policy).
``Role.policy()`` turns it into an immutable ``PermissionPolicy``; the policy,
not the prompt, is what the ``ToolExecutor`` enforces.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from automa.config import Tier
from automa.plan.models import ROLE_PATTERN
from automa.skills.models import SKILL_NAME_PATTERN
from automa.tools.base import TOOL_NAME_RE, RiskLevel
from automa.tools.policy import PermissionPolicy

ROLE_SCHEMA_VERSION = 1
MAX_INSTRUCTIONS_CHARS = 20_000
MAX_ROLE_STEPS = 200
MAX_ROLE_SKILLS = 8
_SKILL_NAME_RE = re.compile(SKILL_NAME_PATTERN)


class Role(BaseModel):
    """One worker role. Strict: unknown fields, bad names or ``external`` are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: int = ROLE_SCHEMA_VERSION
    name: str = Field(pattern=ROLE_PATTERN)
    description: str = Field(min_length=1, max_length=500)
    tier: Tier = Tier.MEDIUM
    tools: tuple[str, ...] = ()
    max_risk: RiskLevel = RiskLevel.READ
    max_steps: int = Field(default=30, ge=1, le=MAX_ROLE_STEPS)
    instructions: str = Field(min_length=1, max_length=MAX_INSTRUCTIONS_CHARS)
    # Text only: skills add instructions to the prompt and never grant tools or risk.
    skills: tuple[str, ...] = ()

    @field_validator("schema_version")
    @classmethod
    def _schema(cls, v: int) -> int:
        if v != ROLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {v} (expected {ROLE_SCHEMA_VERSION})")
        return v

    @field_validator("description", "instructions")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("tools", mode="before")
    @classmethod
    def _tool_list(cls, v: Iterable[str]) -> tuple[str, ...]:
        if isinstance(v, str):
            raise ValueError("tools must be a list of names, not a string")
        names = tuple(v)
        bad = sorted({n for n in names if not isinstance(n, str) or not TOOL_NAME_RE.fullmatch(n)})
        if bad:
            raise ValueError(f"invalid tool names: {bad}")
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate tool names: {dupes}")
        return names

    @field_validator("skills", mode="before")
    @classmethod
    def _skill_list(cls, v: Iterable[str]) -> tuple[str, ...]:
        if isinstance(v, str):
            raise ValueError("skills must be a list of names, not a string")
        names = tuple(v)
        bad = sorted(
            {
                repr(n)
                for n in names
                if not isinstance(n, str) or not _SKILL_NAME_RE.fullmatch(n) or "--" in n
            }
        )
        if bad:
            raise ValueError(f"invalid skill names: {bad}")
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate skill names: {dupes}")
        if len(names) > MAX_ROLE_SKILLS:
            raise ValueError(f"at most {MAX_ROLE_SKILLS} skills per role")
        return names

    @field_validator("max_risk")
    @classmethod
    def _not_external(cls, v: RiskLevel) -> RiskLevel:
        if v is RiskLevel.EXTERNAL:
            raise ValueError("max_risk cannot be 'external': external tools need approval")
        return v

    def policy(self) -> PermissionPolicy:
        """The immutable permission policy this role runs under."""
        return PermissionPolicy(allowed_tools=frozenset(self.tools), max_risk=self.max_risk)
