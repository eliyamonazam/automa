"""Continue an attempt after the user answers, instead of starting it again (D166).

When a worker calls ``ask_user`` the attempt stops (``UserInputRequired``). Before
this, the next run started the attempt from scratch: the worker re-read and
re-wrote everything it had already done, which in a real run cost more than the
rest of the task. Now the worker attaches a ``SuspendedAttempt`` (the
conversation so far and the tool calls of the reply that were not run yet) to
the exception; the orchestrator saves it as ``<project>/suspended/<task>.json``,
and when the question is answered (or declined) the next run hands it back to
the worker, which runs the remaining calls (the ``ask_user`` call now returns the
answer) and goes on from there.

The file is data only (it never grants a tool, a path or a budget): the resumed
attempt uses the same executor, policy and BudgetGuard as any attempt. It is used
at most once (removed when read) and only for the same task and attempt number;
anything unreadable or mismatched is ignored and the attempt starts afresh.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from automa.config import Tier
from automa.llm.types import Message, ToolCall

SUSPENDED_DIR = "suspended"
SUSPEND_SCHEMA_VERSION = 1
MAX_SUSPENDED_BYTES = 4 * 1024 * 1024


class SuspendedAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SUSPEND_SCHEMA_VERSION
    task_id: str
    attempt: int = Field(ge=1)
    request_id: str
    messages: tuple[Message, ...] = Field(min_length=2)
    calls_left: tuple[ToolCall, ...] = Field(min_length=1)
    intact_left: tuple[bool, ...]
    steps: int = Field(ge=0)
    files: tuple[str, ...] = ()
    cost_nusd: int = Field(ge=0)
    tier: Tier
    model: str
    started_at: datetime


def _path(project_dir: Path, task_id: str) -> Path:
    if "/" in task_id or task_id in ("", ".", ".."):
        raise ValueError(f"invalid task id {task_id!r}")
    return project_dir / SUSPENDED_DIR / f"{task_id}.json"


def save_suspended(project_dir: Path, s: SuspendedAttempt) -> None:
    """Store ``s`` atomically (replacing an older one of the same task)."""
    directory = project_dir / SUSPENDED_DIR
    if directory.is_symlink():
        raise OSError(f"{directory} must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    data = s.model_dump_json()
    if len(data.encode("utf-8")) > MAX_SUSPENDED_BYTES:
        raise OSError("suspended attempt too large to keep")
    fd, tmp = tempfile.mkstemp(prefix=".susp.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, _path(project_dir, s.task_id))
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def take_suspended(project_dir: Path, task_id: str, attempt: int) -> SuspendedAttempt | None:
    """The saved attempt of ``task_id`` if it is for ``attempt``; removed either way."""
    path = _path(project_dir, task_id)
    if path.is_symlink() or not path.is_file():
        if path.is_symlink():
            path.unlink(missing_ok=True)
        return None
    try:
        if path.stat().st_size > MAX_SUSPENDED_BYTES:
            return None
        s = SuspendedAttempt.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return None
    finally:
        path.unlink(missing_ok=True)
    if s.task_id != task_id or s.attempt != attempt or len(s.intact_left) != len(s.calls_left):
        return None
    return s


def has_suspended(project_dir: Path, task_id: str) -> bool:
    path = _path(project_dir, task_id)
    return path.is_file() and not path.is_symlink()
