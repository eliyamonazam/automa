"""Questions Automa asks the user in the middle of a task (D159).

When a worker needs something only the user can provide (a file, data it cannot
reach, a credential, a decision), it calls the ``ask_user`` tool. That records a
``UserRequest`` here and the project pauses (``StopReason.USER_INPUT``) without
spending anything more. The user answers with text and/or files (CLI or web UI)
or says they cannot provide it; the next run starts the task again with every
answer in front of the worker.

Storage: one JSON file per request in ``<project>/requests/`` (atomic writes, no
symlinks). Uploaded answer files are written through ``Workspace`` into the
project's ``files/inputs/<request id>/`` so the worker reads them with its normal
tools. A request is answered or declined once; nothing here calls a model or
grants a tool, a path or a budget.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import unicodedata
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from automa.tools.workspace import Workspace, WorkspaceError

Clock = Callable[[], datetime]

REQUEST_SCHEMA_VERSION = 1
MAX_QUESTION_CHARS = 1_500
MAX_WHY_CHARS = 600
MAX_ANSWER_CHARS = 20_000
MAX_REASON_CHARS = 1_000
MAX_ANSWER_FILES = 10
MAX_ANSWER_FILE_BYTES = 20 * 1024 * 1024
MAX_REQUESTS_PER_TASK = 6
MAX_FILE_BYTES = 256 * 1024
INPUTS_DIR = "inputs"
_ID_RE = re.compile(r"^req-[0-9a-f]{12}$")
_UNSAFE_NAME = re.compile(r"[^\w.\- ]+", re.UNICODE)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class UserRequestError(Exception):
    """The request store or an answer was refused (message shown to the user)."""


class UserRequestNotFound(UserRequestError):
    pass


class RequestStatus(StrEnum):
    PENDING = "pending"
    ANSWERED = "answered"
    DECLINED = "declined"


RequestKind = Literal["text", "file"]


def question_key(question: str) -> str:
    """Normalized question text: the same question asked again finds its answer."""
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


class UserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = REQUEST_SCHEMA_VERSION
    id: str
    project_id: str
    task_id: str
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    why: str = Field(default="", max_length=MAX_WHY_CHARS)
    kind: RequestKind = "text"
    status: RequestStatus = RequestStatus.PENDING
    created_at: datetime
    answered_at: datetime | None = None
    answer: str = Field(default="", max_length=MAX_ANSWER_CHARS)
    files: tuple[str, ...] = Field(default=(), max_length=MAX_ANSWER_FILES)
    via: str | None = Field(default=None, max_length=20)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not _ID_RE.fullmatch(v):
            raise ValueError(f"invalid request id {v!r}")
        return v

    @field_validator("question")
    @classmethod
    def _question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v

    @property
    def key(self) -> str:
        return question_key(self.question)

    def as_text(self) -> str:
        """What the worker sees about this request (tool result and task message)."""
        if self.status is RequestStatus.PENDING:
            return f"Q: {self.question}\nA: (not answered yet)"
        if self.status is RequestStatus.DECLINED:
            reason = f" Reason: {self.answer}" if self.answer.strip() else ""
            return (
                f"Q: {self.question}\nA: The user cannot provide this.{reason} Do not ask "
                "again: continue without it, never invent it, and state clearly in your "
                "final summary what is missing because of it."
            )
        parts = [f"Q: {self.question}", f"A: {self.answer.strip() or '(no text)'}"]
        if self.files:
            parts.append(
                "Files the user gave (read them with your file tools): " + ", ".join(self.files)
            )
        return "\n".join(parts)


def _write_atomic(path: Path, text: str, *, exclusive: bool = False) -> bool:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if exclusive:
            try:
                os.link(tmp, path)
            except FileExistsError:
                return False
            return True
        os.replace(tmp, path)
        return True
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _dump(req: UserRequest) -> str:
    return json.dumps(req.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)


def safe_file_name(name: str) -> str:
    """A plain file name from an uploaded name (no directories, no odd characters)."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    base = unicodedata.normalize("NFKC", base).strip()
    base = _UNSAFE_NAME.sub("_", base).strip(" .")
    if not base or base in (".", ".."):
        base = "file"
    if len(base) > 120:
        stem, dot, ext = base.rpartition(".")
        base = (stem[:100] + dot + ext[:15]) if dot else base[:120]
    return base


class UserRequestStore:
    """The user requests of one project, stored in ``directory``."""

    def __init__(
        self,
        directory: str | Path,
        project_id: str,
        files_dir: str | Path,
        *,
        clock: Clock = _utc_now,
    ):
        self._dir = Path(directory)
        self._project_id = project_id
        self._files_dir = Path(files_dir)
        self._clock = clock

    @property
    def project_id(self) -> str:
        return self._project_id

    def _path(self, request_id: str) -> Path:
        if not isinstance(request_id, str) or not _ID_RE.fullmatch(request_id):
            raise UserRequestNotFound(f"invalid request id: {request_id!r}")
        return self._dir / f"{request_id}.json"

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise UserRequestError("clock returned a naive datetime")
        return now.astimezone(UTC)

    def _ensure_dir(self) -> None:
        if self._dir.is_symlink():
            raise UserRequestError(f"{self._dir}: requests directory must not be a symlink")
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- reading -----------------------------------------------------------

    def get(self, request_id: str) -> UserRequest:
        path = self._path(request_id)
        if path.is_symlink():
            raise UserRequestError(f"{path}: request file must not be a symlink")
        if not path.is_file():
            raise UserRequestNotFound(f"no such request: {request_id!r}")
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise UserRequestError(f"{path}: request file is too large")
            req = UserRequest.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError) as e:
            raise UserRequestError(f"{path}: unreadable request file") from e
        if req.id != request_id or req.project_id != self._project_id:
            raise UserRequestError(f"{path}: request file does not belong here")
        return req

    def list(self, status: RequestStatus | None = None) -> list[UserRequest]:
        """All requests (oldest first), optionally only those with ``status``."""
        if self._dir.is_symlink():
            raise UserRequestError(f"{self._dir}: requests directory must not be a symlink")
        if not self._dir.is_dir():
            return []
        out = []
        for path in self._dir.iterdir():
            name = path.name
            if name.endswith(".json") and _ID_RE.fullmatch(name[: -len(".json")]):
                out.append(self.get(name[: -len(".json")]))
        out.sort(key=lambda r: (r.created_at, r.id))
        return [r for r in out if status is None or r.status is status]

    def pending(self) -> list[UserRequest]:
        return self.list(RequestStatus.PENDING)

    def for_task(self, task_id: str) -> list[UserRequest]:
        return [r for r in self.list() if r.task_id == task_id]

    def find(self, task_id: str, question: str) -> UserRequest | None:
        """The latest request of ``task_id`` with the same (normalized) question."""
        key = question_key(question)
        found = [r for r in self.for_task(task_id) if r.key == key]
        return found[-1] if found else None

    # -- writing -----------------------------------------------------------

    def request(
        self, task_id: str, question: str, *, why: str = "", kind: RequestKind = "text"
    ) -> UserRequest:
        """Record a pending question (an identical pending one is returned instead)."""
        existing = self.find(task_id, question)
        if existing is not None and existing.status is RequestStatus.PENDING:
            return existing
        self._ensure_dir()
        for _ in range(5):
            req = UserRequest(
                id=f"req-{secrets.token_hex(6)}",
                project_id=self._project_id,
                task_id=task_id,
                question=question.strip(),
                why=why.strip(),
                kind=kind,
                created_at=self._now(),
            )
            if _write_atomic(self._path(req.id), _dump(req), exclusive=True):
                return req
        raise UserRequestError("could not allocate a request id")  # pragma: no cover

    def _decide(self, req: UserRequest, **update: object) -> UserRequest:
        if req.status is not RequestStatus.PENDING:
            raise UserRequestError(f"request {req.id} is already {req.status.value}")
        new = UserRequest.model_validate(req.model_dump() | update | {"answered_at": self._now()})
        current = self.get(req.id)  # re-read: refuse a second decision that raced us
        if current.status is not RequestStatus.PENDING:
            raise UserRequestError(f"request {req.id} is already {current.status.value}")
        _write_atomic(self._path(req.id), _dump(new))
        return new

    def answer(
        self,
        request_id: str,
        text: str = "",
        files: Sequence[tuple[str, bytes]] = (),
        *,
        via: str,
    ) -> UserRequest:
        """Answer a pending request with ``text`` and/or uploaded ``files``.

        Files are written into ``files/inputs/<request id>/`` (through ``Workspace``,
        never following a symlink); their workspace paths are recorded.
        """
        req = self.get(request_id)
        if req.status is not RequestStatus.PENDING:
            raise UserRequestError(f"request {req.id} is already {req.status.value}")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if len(text) > MAX_ANSWER_CHARS:
            raise UserRequestError(f"answer longer than {MAX_ANSWER_CHARS} characters")
        if not text.strip() and not files:
            raise UserRequestError("an answer needs text or at least one file")
        if len(files) > MAX_ANSWER_FILES:
            raise UserRequestError(f"at most {MAX_ANSWER_FILES} files per answer")
        paths: list[str] = []
        if files:
            self._files_dir.mkdir(parents=True, exist_ok=True)
            try:
                ws = Workspace(self._files_dir)
            except WorkspaceError as e:
                raise UserRequestError(str(e)) from None
            used: set[str] = set()
            for name, data in files:
                if len(data) > MAX_ANSWER_FILE_BYTES:
                    raise UserRequestError(
                        f"{safe_file_name(name)} is larger than {MAX_ANSWER_FILE_BYTES} bytes"
                    )
                base = safe_file_name(name)
                candidate, n = base, 1
                while candidate in used:
                    n += 1
                    stem, dot, ext = base.rpartition(".")
                    candidate = f"{stem}-{n}.{ext}" if dot else f"{base}-{n}"
                used.add(candidate)
                path = f"{INPUTS_DIR}/{req.id}/{candidate}"
                try:
                    ws.write_bytes(path, data, max_bytes=MAX_ANSWER_FILE_BYTES)
                except WorkspaceError as e:
                    raise UserRequestError(f"could not save {candidate}: {e}") from None
                paths.append(path)
        return self._decide(
            req, status=RequestStatus.ANSWERED, answer=text.strip(), files=tuple(paths), via=via
        )

    def decline(self, request_id: str, reason: str = "", *, via: str) -> UserRequest:
        """The user cannot (or will not) provide this; the task continues without it."""
        reason = reason.replace("\r\n", "\n").strip()
        if len(reason) > MAX_REASON_CHARS:
            raise UserRequestError(f"reason longer than {MAX_REASON_CHARS} characters")
        req = self.get(request_id)
        return self._decide(req, status=RequestStatus.DECLINED, answer=reason, via=via)
