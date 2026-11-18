"""Staged skill candidates: where ``automa skill draft`` puts a drafted ``SKILL.md`` (D108).

A candidate is never installed here. Layout, under ``<workspace>/skill-candidates/``::

    <id>/<name>/SKILL.md   the staging directory ``<id>/`` read by
                           ``automa.skill_comparison.load_candidate_skill``
    <id>.json              metadata (role, benchmarks, digest, attempts, cost, sha256)

``<id>`` is ``<UTC stamp>-<name>``. Staging is all-or-nothing: the candidate directory
is built under a hidden temporary name and moved into place with one ``os.rename``;
an existing id, a symlinked or non-directory ``skill-candidates`` and a full store are
errors (fails closed), checked by :func:`check_room` before any model call.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from automa.agents.skill_author import SkillDraft
from automa.skill_comparison import SkillComparisonError, load_candidate_skill
from automa.skills.registry import SKILL_FILE

CANDIDATES_DIRNAME = "skill-candidates"
MAX_CANDIDATES = 100
METADATA_SUFFIX = ".json"
CANDIDATE_ID_PATTERN = r"^[0-9]{8}-[0-9]{6}-[a-z0-9]+(?:-[a-z0-9]+)*$"
_ID_RE = re.compile(CANDIDATE_ID_PATTERN)
_MAX_ID = 80


class SkillCandidateError(Exception):
    """The candidate cannot be staged (nothing is left behind when this is raised)."""


class CandidateRecord(BaseModel):
    """Metadata of one staged candidate (``<id>.json``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=CANDIDATE_ID_PATTERN, max_length=_MAX_ID)
    name: str
    role: str
    benchmarks: list[str]
    digest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    skill_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempts: int = Field(ge=1)
    cost_nusd: int = Field(ge=0)
    created_at: datetime


def candidates_dir(workspace: str | Path) -> Path:
    """``<workspace>/skill-candidates``: staged, never-installed skill candidates."""
    return Path(workspace) / CANDIDATES_DIRNAME


def candidate_id(created_at: datetime, name: str) -> str:
    stamp = created_at.astimezone(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{name}"[:_MAX_ID].rstrip("-")


def staging_path(workspace: str | Path, cid: str) -> Path:
    """The staging directory of candidate ``cid`` (what ``load_candidate_skill`` reads)."""
    if not isinstance(cid, str) or not _ID_RE.fullmatch(cid) or len(cid) > _MAX_ID:
        raise SkillCandidateError(f"invalid candidate id {cid!r}")
    return candidates_dir(workspace) / cid


def _existing_dir(workspace: str | Path) -> Path | None:
    base = candidates_dir(workspace)
    if base.is_symlink():
        raise SkillCandidateError(f"{base} must not be a symlink")
    if not base.exists():
        return None
    if not base.is_dir():
        raise SkillCandidateError(f"{base} is not a directory")
    return base


def _count(base: Path) -> int:
    return sum(1 for e in os.scandir(base) if not e.name.startswith(".") and e.is_dir())


def check_room(workspace: str | Path) -> None:
    """Raise if a new candidate could not be staged (run before spending money)."""
    base = _existing_dir(workspace)
    if base is not None and _count(base) >= MAX_CANDIDATES:
        raise SkillCandidateError(
            f"{base} already holds {MAX_CANDIDATES} candidates; remove old ones first"
        )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stage_draft(draft: SkillDraft, *, workspace: str | Path, cost_nusd: int) -> CandidateRecord:
    """Write ``draft`` as a new candidate; never overwrites, never installs.

    The staged directory is read back with ``load_candidate_skill`` and must give
    exactly ``draft.skill``; otherwise everything written is removed again.
    """
    if not isinstance(draft, SkillDraft):
        raise SkillCandidateError("draft must be a SkillDraft")
    if isinstance(cost_nusd, bool) or not isinstance(cost_nusd, int) or cost_nusd < 0:
        raise SkillCandidateError("cost_nusd must be a non-negative integer")
    check_room(workspace)
    cid = candidate_id(draft.created_at, draft.skill.name)
    record = CandidateRecord(
        id=cid,
        name=draft.skill.name,
        role=draft.role,
        benchmarks=list(draft.benchmarks),
        digest_sha256=draft.digest_sha256,
        skill_sha256=_sha256(draft.text),
        attempts=draft.attempts,
        cost_nusd=cost_nusd,
        created_at=draft.created_at,
    )
    base = candidates_dir(workspace)
    base.mkdir(parents=True, exist_ok=True)
    final = staging_path(workspace, cid)
    meta = base / f"{cid}{METADATA_SUFFIX}"
    if final.exists() or final.is_symlink() or meta.exists() or meta.is_symlink():
        raise SkillCandidateError(f"candidate {cid} already exists")
    tmp = Path(tempfile.mkdtemp(prefix=f".{cid}.", dir=base))
    moved = False
    try:
        (tmp / draft.skill.name).mkdir()
        skill_file = tmp / draft.skill.name / SKILL_FILE
        with open(skill_file, "x", encoding="utf-8", newline="") as f:
            f.write(draft.text)
            f.flush()
            os.fsync(f.fileno())
        try:
            staged = load_candidate_skill(tmp, workspace=Path(workspace))
        except SkillComparisonError as e:
            raise SkillCandidateError(f"staged candidate does not load back: {e}") from None
        if staged != draft.skill:
            raise SkillCandidateError("staged candidate differs from the draft")
        text = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
        with open(tmp / ".meta", "x", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp / ".meta", meta)  # fails if it exists: never overwrites
        os.unlink(tmp / ".meta")
        try:
            os.rename(tmp, final)
        except OSError:
            os.unlink(meta)
            raise
        moved = True
    except FileExistsError:
        raise SkillCandidateError(f"candidate {cid} already exists") from None
    finally:
        if not moved:
            shutil.rmtree(tmp, ignore_errors=True)
    return record


__all__ = [
    "CANDIDATES_DIRNAME",
    "CANDIDATE_ID_PATTERN",
    "MAX_CANDIDATES",
    "CandidateRecord",
    "SkillCandidateError",
    "candidate_id",
    "candidates_dir",
    "check_room",
    "stage_draft",
    "staging_path",
]
