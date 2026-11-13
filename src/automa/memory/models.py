"""Data model of one memory entry (D48)."""

from __future__ import annotations

import unicodedata
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from automa.project import ProjectStoreError, validate_project_id

MEMORY_SCHEMA_VERSION = 1
MEMORY_ID_PATTERN = r"^mem-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$"
MAX_MEMORY_TEXT_CHARS = 2000
_ALLOWED_CONTROL = {"\n", "\t"}


class MemoryScope(StrEnum):
    USER = "user"  # preferences of the user, shared by every project
    PROJECT = "project"  # knowledge specific to one project
    LESSON = "lesson"  # general lessons learned, shared by every project


class MemorySource(StrEnum):
    USER = "user"  # written by the user
    REVIEW = "review"  # extracted by the post-project review step


def normalize_text(text: str) -> str:
    """Key used to detect duplicates in one layer: NFC, case-folded, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def clean_memory_text(v: str, *, max_chars: int = MAX_MEMORY_TEXT_CHARS) -> str:
    """NFC, CRLF to LF, stripped; no control characters except newline and tab.

    Raises ``ValueError`` for empty, too long or control-character text. Shared by
    ``MemoryEntry`` and the review step's proposals, so a proposal that passes here
    can always be stored as a memory entry.
    """
    v = unicodedata.normalize("NFC", v).replace("\r\n", "\n").strip()
    if not v:
        raise ValueError("memory text must not be empty")
    for ch in v:
        if ch in _ALLOWED_CONTROL:
            continue
        if unicodedata.category(ch) in {"Cc", "Cs"}:
            raise ValueError("memory text must not contain control characters")
    limit = min(max_chars, MAX_MEMORY_TEXT_CHARS)
    if len(v) > limit:
        raise ValueError(f"memory text longer than {limit} characters")
    return v


class MemoryEntry(BaseModel):
    """Contents of one ``memory/.../<id>.json`` file. Immutable; written once."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = MEMORY_SCHEMA_VERSION
    id: str = Field(pattern=MEMORY_ID_PATTERN)
    scope: MemoryScope
    text: str = Field(min_length=1, max_length=MAX_MEMORY_TEXT_CHARS)
    project_id: str | None = None
    source: MemorySource = MemorySource.USER
    created_at: datetime

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != MEMORY_SCHEMA_VERSION:
            raise ValueError(f"unsupported memory schema version {v}")
        return v

    @field_validator("text")
    @classmethod
    def _clean_text(cls, v: str) -> str:
        return clean_memory_text(v)

    @field_validator("project_id")
    @classmethod
    def _project_id(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            return validate_project_id(v)
        except ProjectStoreError as e:
            raise ValueError(str(e)) from None

    @field_validator("created_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return v.astimezone(UTC)

    @model_validator(mode="after")
    def _scope_fields(self) -> MemoryEntry:
        if self.scope is MemoryScope.PROJECT and self.project_id is None:
            raise ValueError("a project memory needs project_id")
        if self.scope is MemoryScope.USER:
            if self.project_id is not None:
                raise ValueError("a user memory cannot have project_id")
            if self.source is not MemorySource.USER:
                raise ValueError("user preferences can only be written by the user")
        return self
