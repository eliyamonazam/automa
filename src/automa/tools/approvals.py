"""Approval Gate store: requests for actions that need the user's approval.

When a tool call needs approval (any ``external`` tool), the ``ToolExecutor``
records an ``ApprovalRequest`` here and stops (``ApprovalRequired``); the project
pauses. The user approves or rejects the request (``decide``). On the next run, a
call to the **same tool with the same arguments in the same task** finds that
decision:

* approved → the call runs **once** (the grant is consumed before the tool runs);
* rejected → the call does not run; the model is told the user rejected it.

Layout (``<project>/approvals/``)::

    <id>.json   the request and its decision (atomic rewrite on decide)
    <id>.used   created exclusively when the decision is consumed (at most once)

Guarantees (fail closed):

* A request matches only on project, task, tool and a SHA-256 fingerprint of the
  canonical (validated) arguments; any difference needs a new approval.
* A decision can be made only on a ``pending`` request, and only once.
* A decision is consumed at most once, even across processes (``O_EXCL`` marker).
* Corrupt, unknown or symlinked files raise ``ApprovalError``; nothing is repaired.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from automa.tools.base import TOOL_NAME_RE, RiskLevel

Clock = Callable[[], datetime]

APPROVAL_SCHEMA_VERSION = 1
APPROVAL_ID_PATTERN = r"^apr-[0-9a-f]{12}$"
_APPROVAL_ID_RE = re.compile(APPROVAL_ID_PATTERN)
MAX_ARGUMENTS_JSON_CHARS = 2 * 1024 * 1024
MAX_NOTE_CHARS = 2_000
MAX_REASON_CHARS = 1_000
MAX_FILE_BYTES = MAX_ARGUMENTS_JSON_CHARS + 64 * 1024


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ApprovalError(Exception):
    """Invalid approval operation or unreadable approval file."""


class ApprovalNotFound(ApprovalError):
    pass


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


def canonical_arguments(arguments: Mapping[str, Any]) -> str:
    """Canonical JSON of the arguments (sorted keys, no whitespace)."""
    return json.dumps(dict(arguments), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(tool: str, arguments: Mapping[str, Any]) -> str:
    payload = json.dumps([tool, canonical_arguments(arguments)], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = APPROVAL_SCHEMA_VERSION
    id: str = Field(pattern=APPROVAL_ID_PATTERN)
    project_id: str = Field(min_length=1, max_length=64)
    task_id: str | None = Field(default=None, max_length=64)
    tool: str
    risk: RiskLevel
    arguments: dict[str, Any]
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str = Field(max_length=MAX_REASON_CHARS)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime
    decided_at: datetime | None = None
    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)

    @field_validator("schema_version")
    @classmethod
    def _schema(cls, v: int) -> int:
        if v != APPROVAL_SCHEMA_VERSION:
            raise ValueError(f"unsupported approval schema_version {v}")
        return v

    @field_validator("tool")
    @classmethod
    def _tool(cls, v: str) -> str:
        if not TOOL_NAME_RE.fullmatch(v):
            raise ValueError("invalid tool name")
        return v

    @field_validator("created_at", "decided_at")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> ApprovalRequest:
        if fingerprint(self.tool, self.arguments) != self.fingerprint:
            raise ValueError("fingerprint does not match tool and arguments")
        decided = self.status is not ApprovalStatus.PENDING
        if decided != (self.decided_at is not None):
            raise ValueError("decided_at must be set exactly when the request is decided")
        return self

    @property
    def decided(self) -> bool:
        return self.status is not ApprovalStatus.PENDING


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _write_exclusive(path: Path, text: str) -> bool:
    """Write a complete file at ``path`` only if it does not exist (``os.link``)."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        return True
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _dump(req: ApprovalRequest) -> str:
    return json.dumps(req.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)


class ApprovalStore:
    """Approval requests of one project, stored in ``directory``."""

    def __init__(self, directory: str | Path, project_id: str, *, clock: Clock = _utc_now):
        self._dir = Path(directory)
        self._project_id = project_id
        self._clock = clock

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def project_id(self) -> str:
        return self._project_id

    # -- paths -------------------------------------------------------------

    def _path(self, approval_id: str, suffix: str = ".json") -> Path:
        if not isinstance(approval_id, str) or not _APPROVAL_ID_RE.fullmatch(approval_id):
            raise ApprovalNotFound(f"invalid approval id: {approval_id!r}")
        return self._dir / f"{approval_id}{suffix}"

    def _ensure_dir(self) -> None:
        if self._dir.is_symlink():
            raise ApprovalError(f"{self._dir}: approvals directory must not be a symlink")
        self._dir.mkdir(parents=True, exist_ok=True)

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise ApprovalError("clock returned a naive datetime")
        return now.astimezone(UTC)

    # -- reading -----------------------------------------------------------

    def get(self, approval_id: str) -> ApprovalRequest:
        path = self._path(approval_id)
        if path.is_symlink():
            raise ApprovalError(f"{path}: approval file must not be a symlink")
        if not path.is_file():
            raise ApprovalNotFound(f"no such approval request: {approval_id!r}")
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ApprovalError(f"{path}: approval file is too large")
            req = ApprovalRequest.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError) as e:
            raise ApprovalError(f"{path}: unreadable approval file") from e
        if req.id != approval_id or req.project_id != self._project_id:
            raise ApprovalError(f"{path}: approval file does not belong here")
        return req

    def list(self, status: ApprovalStatus | None = None) -> list[ApprovalRequest]:
        """All requests (oldest first), optionally only those with ``status``."""
        if self._dir.is_symlink():
            raise ApprovalError(f"{self._dir}: approvals directory must not be a symlink")
        if not self._dir.is_dir():
            return []
        out = []
        for path in self._dir.iterdir():
            name = path.name
            if name.endswith(".json") and _APPROVAL_ID_RE.fullmatch(name[: -len(".json")]):
                out.append(self.get(name[: -len(".json")]))
        out.sort(key=lambda r: (r.created_at, r.id))
        return [r for r in out if status is None or r.status is status]

    def is_used(self, approval_id: str) -> bool:
        return os.path.lexists(self._path(approval_id, ".used"))

    # -- writing -----------------------------------------------------------

    def request(
        self,
        *,
        tool: str,
        risk: RiskLevel,
        arguments: Mapping[str, Any],
        reason: str,
        task_id: str | None = None,
    ) -> ApprovalRequest:
        """Record a pending request; an identical pending one is returned instead."""
        args = dict(arguments)
        if len(canonical_arguments(args)) > MAX_ARGUMENTS_JSON_CHARS:
            raise ApprovalError("arguments are too large to record for approval")
        fp = fingerprint(tool, args)
        for existing in self.list(ApprovalStatus.PENDING):
            if existing.fingerprint == fp and existing.task_id == task_id:
                return existing
        self._ensure_dir()
        reason = reason if len(reason) <= MAX_REASON_CHARS else reason[: MAX_REASON_CHARS - 6]
        for _ in range(5):
            req = ApprovalRequest(
                id=f"apr-{secrets.token_hex(6)}",
                project_id=self._project_id,
                task_id=task_id,
                tool=tool,
                risk=risk,
                arguments=args,
                fingerprint=fp,
                reason=reason,
                created_at=self._now(),
            )
            if _write_exclusive(self._path(req.id), _dump(req)):
                return req
        raise ApprovalError("could not allocate an approval id")

    def decide(
        self, approval_id: str, *, approve: bool, note: str | None = None
    ) -> ApprovalRequest:
        """Approve or reject a pending request (once)."""
        req = self.get(approval_id)
        if req.decided:
            raise ApprovalError(f"approval request {approval_id!r} is already {req.status.value}")
        if note is not None and len(note) > MAX_NOTE_CHARS:
            raise ApprovalError(f"note is longer than {MAX_NOTE_CHARS} characters")
        decided = req.model_copy(
            update={
                "status": ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED,
                "decided_at": self._now(),
                "note": note,
            }
        )
        decided = ApprovalRequest.model_validate(decided.model_dump())
        _write_atomic(self._path(approval_id), _dump(decided))
        return decided

    def find_decision(
        self, *, tool: str, arguments: Mapping[str, Any], task_id: str | None
    ) -> ApprovalRequest | None:
        """The oldest unconsumed decision for exactly this call, if any."""
        fp = fingerprint(tool, dict(arguments))
        for req in self.list():
            if (
                req.decided
                and req.fingerprint == fp
                and req.tool == tool
                and req.task_id == task_id
                and not self.is_used(req.id)
            ):
                return req
        return None

    def consume(self, approval_id: str) -> bool:
        """Mark a decision used. True only for the one caller that consumed it."""
        req = self.get(approval_id)
        if not req.decided:
            raise ApprovalError(f"approval request {approval_id!r} is not decided")
        try:
            fd = os.open(
                self._path(approval_id, ".used"),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError:
            return False
        try:
            os.write(fd, self._now().isoformat().encode("ascii"))
            os.fsync(fd)
        finally:
            os.close(fd)
        return True
