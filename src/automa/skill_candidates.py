"""Staged skill candidates: where ``automa skill draft`` puts a drafted ``SKILL.md`` (D108).

A candidate is never installed here. Layout, under ``<workspace>/skill-candidates/``::

    <id>/<name>/SKILL.md   the staging directory ``<id>/`` read by
                           ``automa.skill_comparison.load_candidate_skill``
    <id>.json              metadata (role, benchmarks, digest, attempts, cost, sha256)
    <id>.runs/<side>/      result files of ``automa skill compare`` (baseline/candidate)
    <id>.gate.json         the one recorded gate decision (written once, never replaced)

``install_candidate`` (D110) copies an accepted candidate's measured ``SKILL.md`` to
``<workspace>/skills/<name>/`` — the only way a candidate leaves staging.

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
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from automa.agents.skill_author import SkillDraft
from automa.roles import RoleRegistry
from automa.skill_comparison import (
    SkillComparisonError,
    SkillComparisonResult,
    load_candidate_skill,
)
from automa.skill_gate import SkillGateDecision
from automa.skill_variants import SkillVariantError, skillset_fingerprint, with_candidate_skill
from automa.skills import Skill, SkillError, SkillRegistry, workspace_skills_dir
from automa.skills.loader import BUILTIN_SKILLS_DIR, load_skills
from automa.skills.registry import SKILL_FILE, parse_skill

CANDIDATES_DIRNAME = "skill-candidates"
MAX_CANDIDATES = 100
METADATA_SUFFIX = ".json"
GATE_SUFFIX = ".gate.json"
RUNS_SUFFIX = ".runs"
_MAX_META_BYTES = 1 << 20
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
    # Only candidate directories count (not ``<id>.runs`` or hidden temporary ones).
    return sum(1 for e in os.scandir(base) if _ID_RE.fullmatch(e.name) and e.is_dir())


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


# --- reading a staged candidate and recording its gate decision (D109) -------------


def _read_small_file(path: Path, what: str) -> bytes:
    """Bytes of a regular, non-symlinked file of at most 1 MiB (fails closed)."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise SkillCandidateError(f"{what} not found: {path}") from None
    except OSError as e:
        raise SkillCandidateError(f"cannot read {what} {path} ({e.strerror})") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SkillCandidateError(f"{what} is not a regular file: {path}")
        if st.st_size > _MAX_META_BYTES:
            raise SkillCandidateError(f"{what} is too large: {path}")
        with os.fdopen(fd, "rb") as f:
            fd = -1
            data = f.read(_MAX_META_BYTES + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > _MAX_META_BYTES:
        raise SkillCandidateError(f"{what} is too large: {path}")
    return data


def skill_file_sha256(workspace: str | Path, record: CandidateRecord) -> str:
    """SHA-256 of the staged ``SKILL.md`` bytes as they are on disk now."""
    path = staging_path(workspace, record.id) / record.name / SKILL_FILE
    return hashlib.sha256(_read_small_file(path, "candidate skill file")).hexdigest()


def load_candidate(workspace: str | Path, cid: str) -> tuple[CandidateRecord, Skill]:
    """The metadata and the skill of staged candidate ``cid`` (fails closed).

    The metadata must be valid and name this id; the staged ``SKILL.md`` must still
    have the recorded ``skill_sha256`` (before and after it is parsed) and hold the
    recorded skill name.
    """
    staging = staging_path(workspace, cid)
    base = _existing_dir(workspace)
    if base is None:
        raise SkillCandidateError(f"unknown candidate {cid!r} (no {CANDIDATES_DIRNAME}/)")
    meta = base / f"{cid}{METADATA_SUFFIX}"
    if not (staging.exists() or staging.is_symlink()) and not (meta.exists() or meta.is_symlink()):
        raise SkillCandidateError(f"unknown candidate {cid!r}")
    try:
        record = CandidateRecord.model_validate_json(_read_small_file(meta, "candidate metadata"))
    except (ValidationError, ValueError):
        raise SkillCandidateError(f"invalid candidate metadata {meta.name}") from None
    if record.id != cid:
        raise SkillCandidateError(f"candidate metadata {meta.name} names another id")
    if skill_file_sha256(workspace, record) != record.skill_sha256:
        raise SkillCandidateError(
            f"candidate {cid}: {SKILL_FILE} changed since it was drafted (sha256 differs)"
        )
    try:
        skill = load_candidate_skill(staging, workspace=Path(workspace))
    except SkillComparisonError as e:
        raise SkillCandidateError(f"candidate {cid}: {e}") from None
    if skill.name != record.name:
        raise SkillCandidateError(f"candidate {cid}: skill name differs from its metadata")
    if skill_file_sha256(workspace, record) != record.skill_sha256:
        raise SkillCandidateError(f"candidate {cid}: {SKILL_FILE} changed while being read")
    return record, skill


class GateRecord(BaseModel):
    """The gate decision of one candidate (``<id>.gate.json``), written exactly once."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(pattern=CANDIDATE_ID_PATTERN, max_length=_MAX_ID)
    name: str
    role: str
    skill_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmarks: list[str]
    runs_per_side: int = Field(ge=1)
    accepted: bool
    decision: SkillGateDecision
    spent_nusd: int = Field(ge=0)
    max_total_nusd: int = Field(ge=0)
    limit_nusd: int = Field(ge=0)
    baseline_projects: list[str]
    candidate_projects: list[str]
    created_at: datetime


def gate_path(workspace: str | Path, cid: str) -> Path:
    staging_path(workspace, cid)  # validates the id
    return candidates_dir(workspace) / f"{cid}{GATE_SUFFIX}"


def runs_dir(workspace: str | Path, cid: str) -> Path:
    """``<id>.runs/``: the comparison's result files (``<side>/<benchmark>/…``)."""
    staging_path(workspace, cid)
    return candidates_dir(workspace) / f"{cid}{RUNS_SUFFIX}"


def read_gate(workspace: str | Path, cid: str) -> GateRecord | None:
    """The recorded decision of ``cid``; ``None`` if none; a broken file is an error."""
    path = gate_path(workspace, cid)
    if not (path.exists() or path.is_symlink()):
        return None
    try:
        gate = GateRecord.model_validate_json(_read_small_file(path, "gate decision"))
    except (ValidationError, ValueError):
        raise SkillCandidateError(f"invalid gate decision {path.name}") from None
    if gate.candidate_id != cid:
        raise SkillCandidateError(f"gate decision {path.name} names another candidate")
    return gate


def record_gate(
    workspace: str | Path,
    record: CandidateRecord,
    result: SkillComparisonResult,
    *,
    created_at: datetime,
) -> GateRecord:
    """Write the gate decision of a finished (not aborted) comparison; never overwrites."""
    if result.decision is None or result.aborted is not None:
        raise SkillCandidateError("an aborted comparison has no decision to record")
    if result.plan.candidate != record.name or result.plan.role != record.role:
        raise SkillCandidateError("the comparison was not run for this candidate")
    if skill_file_sha256(workspace, record) != record.skill_sha256:
        raise SkillCandidateError(
            f"candidate {record.id}: {SKILL_FILE} changed during the comparison; "
            "no decision recorded"
        )
    gate = GateRecord(
        candidate_id=record.id,
        name=record.name,
        role=record.role,
        skill_sha256=record.skill_sha256,
        benchmarks=list(result.plan.benchmarks),
        runs_per_side=result.plan.runs_per_side,
        accepted=result.decision.accepted,
        decision=result.decision,
        spent_nusd=result.spent_nusd,
        max_total_nusd=result.plan.max_total_nusd,
        limit_nusd=result.plan.limit_nusd,
        baseline_projects=[r.project_id for r in result.baseline],
        candidate_projects=[r.project_id for r in result.candidate],
        created_at=created_at,
    )
    path = gate_path(workspace, record.id)
    text = json.dumps(gate.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)  # fails if a decision exists: written once
        except FileExistsError:
            raise SkillCandidateError(
                f"candidate {record.id} already has a recorded gate decision"
            ) from None
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    return gate


# --- installing an accepted candidate (D110) ----------------------------------------


@dataclass(frozen=True)
class InstallCheck:
    """Everything ``install_candidate`` checked; ``target`` is the skill's new folder."""

    record: CandidateRecord
    gate: GateRecord
    skill: Skill
    target: Path


def _skills_dir(workspace: str | Path) -> Path:
    directory = workspace_skills_dir(workspace)
    if directory.is_symlink():
        raise SkillCandidateError(f"{directory} must not be a symlink")
    if directory.exists() and not directory.is_dir():
        raise SkillCandidateError(f"{directory} is not a directory")
    return directory


def check_install(
    workspace: str | Path, cid: str, *, roles: RoleRegistry, skills: SkillRegistry
) -> InstallCheck:
    """Raise unless candidate ``cid`` may be installed now (nothing is written).

    The candidate must load (metadata valid, ``SKILL.md`` unchanged since drafting),
    have a recorded gate decision that accepted exactly this text (sha256), and the
    current roles and skills must be the ones it was measured against (the gate's
    baseline fingerprint); its folder under ``<workspace>/skills/`` must not exist.
    """
    if not isinstance(roles, RoleRegistry):
        raise TypeError("roles must be a RoleRegistry")
    if not isinstance(skills, SkillRegistry):
        raise TypeError("skills must be a SkillRegistry")
    record, skill = load_candidate(workspace, cid)
    gate = read_gate(workspace, cid)
    if gate is None:
        raise SkillCandidateError(
            f"candidate {cid} has no recorded gate decision; run: automa skill compare {cid}"
        )
    if not gate.accepted or not gate.decision.accepted or gate.decision.reasons:
        reasons = ", ".join(r.value for r in gate.decision.reasons) or "not accepted"
        raise SkillCandidateError(
            f"candidate {cid} was rejected by the benchmark gate ({reasons}); "
            "only an accepted candidate can be installed"
        )
    if gate.name != record.name or gate.role != record.role:
        raise SkillCandidateError(f"gate decision of {cid} names another skill or role")
    if gate.skill_sha256 != record.skill_sha256:
        raise SkillCandidateError(
            f"candidate {cid}: {SKILL_FILE} is not the text the gate measured (sha256 differs)"
        )
    target = _skills_dir(workspace) / record.name
    if target.exists() or target.is_symlink():
        raise SkillCandidateError(f"{target} already exists; an installed skill is never replaced")
    if skillset_fingerprint(roles, skills) != gate.decision.baseline_fingerprint:
        raise SkillCandidateError(
            f"candidate {cid}: the roles or skills changed since the comparison (baseline "
            "fingerprint differs), so the gate's evidence no longer holds; draft and compare "
            "a new candidate"
        )
    try:
        cand_roles, cand_skills = with_candidate_skill(roles, skills, skill, role=record.role)
    except SkillVariantError as e:
        raise SkillCandidateError(f"candidate {cid}: {e}") from None
    if skillset_fingerprint(cand_roles, cand_skills) != gate.decision.candidate_fingerprint:
        raise SkillCandidateError(
            f"candidate {cid}: the skill set with this candidate is not the one the gate "
            "measured (candidate fingerprint differs)"
        )
    return InstallCheck(record=record, gate=gate, skill=skill, target=target)


def install_candidate(
    workspace: str | Path,
    cid: str,
    *,
    roles: RoleRegistry,
    skills: SkillRegistry,
    builtin_dir: Path = BUILTIN_SKILLS_DIR,
) -> InstallCheck:
    """Install accepted candidate ``cid`` as ``<workspace>/skills/<name>/SKILL.md``.

    Every check of :func:`check_install` is repeated here. The exact measured bytes
    are written under a hidden temporary folder (ignored by the skill loader), the
    target name is claimed with ``mkdir`` (fails if it exists) and the folder is
    moved into place with one ``os.rename``. The workspace skills are then loaded
    again and must hold exactly the measured skill; otherwise the new folder is
    removed. No role uses the skill until ``[roles.<role>] skills`` names it.
    """
    check = check_install(workspace, cid, roles=roles, skills=skills)
    record, gate, skill = check.record, check.gate, check.skill
    source = staging_path(workspace, cid) / record.name / SKILL_FILE
    data = _read_small_file(source, "candidate skill file")
    if hashlib.sha256(data).hexdigest() != gate.skill_sha256:
        raise SkillCandidateError(f"candidate {cid}: {SKILL_FILE} changed while being installed")
    try:
        again = parse_skill(data.decode("utf-8"), expected_name=record.name, source=str(source))
    except (SkillError, UnicodeDecodeError):
        raise SkillCandidateError(f"candidate {cid}: {SKILL_FILE} no longer parses") from None
    if again != skill:
        raise SkillCandidateError(f"candidate {cid}: {SKILL_FILE} changed while being installed")

    directory = _skills_dir(workspace)
    directory.mkdir(parents=True, exist_ok=True)
    directory = _skills_dir(workspace)  # still not a symlink after creating it
    target = directory / record.name
    tmp = Path(tempfile.mkdtemp(prefix=f".{record.name}.", dir=directory))
    claimed = moved = False
    try:
        with open(tmp / SKILL_FILE, "xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.mkdir(target)  # claims the name: fails if anything is there
        except FileExistsError:
            raise SkillCandidateError(
                f"{target} already exists; an installed skill is never replaced"
            ) from None
        claimed = True
        os.rename(tmp, target)  # replaces only the empty folder claimed above
        moved = True
    finally:
        if not moved:
            shutil.rmtree(tmp, ignore_errors=True)
            if claimed:
                try:
                    os.rmdir(target)
                except OSError:
                    pass
    try:
        installed = load_skills(workspace, builtin_dir=builtin_dir)
        ok = (
            record.name in installed
            and installed.get(record.name) == skill
            and hashlib.sha256((target / SKILL_FILE).read_bytes()).hexdigest() == gate.skill_sha256
        )
        problem = "" if ok else "the installed skill differs from the measured one"
    except (SkillError, OSError) as e:
        problem = str(e)
    if problem:
        shutil.rmtree(target, ignore_errors=True)
        raise SkillCandidateError(
            f"candidate {cid}: installed skill did not load back ({problem}); removed again"
        )
    return InstallCheck(record=record, gate=gate, skill=skill, target=target)


__all__ = [
    "CANDIDATES_DIRNAME",
    "CANDIDATE_ID_PATTERN",
    "GATE_SUFFIX",
    "MAX_CANDIDATES",
    "RUNS_SUFFIX",
    "CandidateRecord",
    "GateRecord",
    "InstallCheck",
    "SkillCandidateError",
    "candidate_id",
    "candidates_dir",
    "check_install",
    "check_room",
    "gate_path",
    "install_candidate",
    "load_candidate",
    "read_gate",
    "record_gate",
    "runs_dir",
    "skill_file_sha256",
    "stage_draft",
    "staging_path",
]
