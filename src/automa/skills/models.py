"""Skill definitions: instructions as data, never permissions.

A skill is a directory ``<name>/`` holding a ``SKILL.md`` file: a small
frontmatter block (``name``, ``description``, optional ``license``) and a
Markdown body of instructions. A skill can only add text to a prompt; it never
grants tools, risk levels or budget. Those come from the role's
``PermissionPolicy`` and the ``BudgetGuard`` only.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

SKILL_NAME_PATTERN = r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$"
MAX_SKILL_DESCRIPTION_CHARS = 1024
MAX_SKILL_BODY_CHARS = 20_000


class Skill(BaseModel):
    """One skill. Strict: unknown fields and blank text are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=SKILL_NAME_PATTERN)
    description: str = Field(min_length=1, max_length=MAX_SKILL_DESCRIPTION_CHARS)
    license: str | None = Field(default=None, min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=MAX_SKILL_BODY_CHARS)

    @field_validator("name")
    @classmethod
    def _no_double_hyphen(cls, v: str) -> str:
        if "--" in v:
            raise ValueError("name must not contain '--'")
        return v

    @field_validator("description", "body", "license")
    @classmethod
    def _not_blank(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("description", "license")
    @classmethod
    def _single_line(cls, v: str | None) -> str | None:
        if v is not None and any(c in v for c in "\r\n\x00"):
            raise ValueError("must be a single line")
        return v
