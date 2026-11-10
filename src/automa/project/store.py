"""``ProjectStore``: on-disk project state under ``<workspace>/projects/<project-id>/``.

Layout (see docs/ARCHITECTURE.md, "Project Store")::

    project.json   goal, status, timestamps, revision
    plan.json      tasks and dependencies
    clarifications.json  the Executive's questions and the user's answers
    tasks/<id>/    state.json, result-<n>.json, verdict-<n>.json (attempt history)
    approvals/     Approval Gate requests and decisions (see automa.approvals)
    files/         the project's real outputs
    events.jsonl   append-only event log

Guarantees:

* Project ids are validated before any path is built; resolved paths must stay
  inside the store root (no traversal, no symlinked project directories).
* ``project.json`` is written atomically (temp file + fsync + ``os.replace``), so a
  crash never leaves a half-written file.
* Saves use optimistic concurrency: ``save`` only succeeds if the on-disk
  ``revision`` equals the revision the caller loaded; otherwise ``ProjectConflict``.
* Corrupt or unknown files raise ``ProjectStoreError``; nothing is silently repaired.
* Attempt history (``result-<n>.json``/``verdict-<n>.json``) is write-once and
  sequential; a task state is only saved if it agrees with that history (a task is
  only ``passed`` with a stored passing verdict for its last attempt).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from automa.events import Event, JsonlEventLog
from automa.plan import (
    TASK_ID_PATTERN,
    Clarification,
    Clarifications,
    Plan,
    Task,
    TaskResult,
    TaskState,
    TaskStatus,
    Verdict,
)
from automa.tools.approvals import ApprovalStore

SCHEMA_VERSION = 1
PROJECTS_DIRNAME = "projects"
PROJECT_FILE = "project.json"
PLAN_FILE = "plan.json"
CLARIFICATIONS_FILE = "clarifications.json"
EVENTS_FILE = "events.jsonl"
TASKS_DIR = "tasks"
TASK_STATE_FILE = "state.json"
RESULT_FILE_FMT = "result-{attempt}.json"
VERDICT_FILE_FMT = "verdict-{attempt}.json"
_TASK_ID_RE = re.compile(TASK_ID_PATTERN)
FILES_DIR = "files"
APPROVALS_DIR = "approvals"

PROJECT_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$|^[a-z0-9]$"
_PROJECT_ID_RE = re.compile(PROJECT_ID_PATTERN)
MAX_GOAL_CHARS = 10_000

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ProjectStoreError(Exception):
    """The project store or a project on disk is invalid or unusable."""


class ProjectNotFound(ProjectStoreError):
    """No project with the given id exists."""


class ProjectExists(ProjectStoreError):
    """A project with the given id already exists."""


class ProjectConflict(ProjectStoreError):
    """The project changed on disk since it was loaded (stale revision)."""


class ProjectStatus(StrEnum):
    NEW = "new"
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"


def validate_project_id(project_id: str) -> str:
    """Return ``project_id`` if it is a safe directory name, else raise ``ProjectStoreError``.

    Allowed: 1-64 chars of ``a-z``, ``0-9`` and ``-``, not starting/ending with ``-``.
    """
    if not isinstance(project_id, str) or not _PROJECT_ID_RE.fullmatch(project_id):
        raise ProjectStoreError(f"invalid project id: {project_id!r}")
    return project_id


class Project(BaseModel):
    """Contents of ``project.json``. Immutable; use ``model_copy(update=...)`` to change."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: int = SCHEMA_VERSION
    id: str = Field(pattern=PROJECT_ID_PATTERN)
    goal: str = Field(min_length=1, max_length=MAX_GOAL_CHARS)
    status: ProjectStatus = ProjectStatus.NEW
    created_at: datetime
    updated_at: datetime
    revision: int = Field(default=0, ge=0)

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(f"unsupported project schema version {v} (expected {SCHEMA_VERSION})")
        return v

    @field_validator("goal")
    @classmethod
    def _goal_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("goal must not be blank")
        return v

    @field_validator("created_at", "updated_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return v.astimezone(UTC)


class ProjectEventLog(JsonlEventLog):
    """``events.jsonl`` of one project. Events carry this project's id.

    Events without ``project_id`` get it filled in; events for another project are
    rejected (raise), so one project's log can never record another's activity.
    Usable as the ``EventSink`` of ``BudgetGuard``.
    """

    def __init__(self, path: str | Path, project_id: str):
        super().__init__(path)
        self._project_id = validate_project_id(project_id)

    @property
    def project_id(self) -> str:
        return self._project_id

    def emit(self, event: Event) -> None:
        if event.project_id is None:
            event = event.model_copy(update={"project_id": self._project_id})
        elif event.project_id != self._project_id:
            raise ProjectStoreError(
                f"event for project {event.project_id!r} sent to log of {self._project_id!r}"
            )
        super().emit(event)


def _write_json_atomic(path: Path, data: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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


def _write_json_exclusive(path: Path, data: Any) -> None:
    """Write ``path`` once: complete temp file, then ``os.link`` (fails if it exists).

    Never overwrites, even across processes, and never leaves a partial file.
    """
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise ProjectConflict(f"{path.name} already exists") from None
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _read_json(path: Path, what: str) -> Any:
    """Read a regular (non-symlink) JSON file or raise ``ProjectStoreError``."""
    if path.is_symlink():
        raise ProjectStoreError(f"{path}: {what} file must not be a symlink")
    if not path.is_file():
        raise ProjectStoreError(f"{path}: {what} file is missing")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ProjectStoreError(f"{path}: unreadable {what} file") from e


class ProjectStore:
    """Creates, loads and saves projects under ``root`` (``<workspace>/projects``)."""

    def __init__(self, root: str | Path, *, clock: Clock = _utc_now):
        self._root = Path(root)
        self._clock = clock
        self._lock = threading.Lock()

    @classmethod
    def for_workspace(cls, workspace: str | Path, *, clock: Clock = _utc_now) -> ProjectStore:
        """Open the store at ``<workspace>/projects``."""
        return cls(Path(workspace) / PROJECTS_DIRNAME, clock=clock)

    @property
    def root(self) -> Path:
        return self._root

    # -- paths -------------------------------------------------------------

    def project_dir(self, project_id: str) -> Path:
        """Directory of ``project_id`` (validated; guaranteed to be inside the root)."""
        validate_project_id(project_id)
        path = self._root / project_id
        root = self._root.resolve()
        if path.is_symlink() or path.resolve().parent != root:
            raise ProjectStoreError(f"project directory escapes the store: {project_id!r}")
        return path

    def project_file(self, project_id: str) -> Path:
        return self.project_dir(project_id) / PROJECT_FILE

    def plan_file(self, project_id: str) -> Path:
        return self.project_dir(project_id) / PLAN_FILE

    def tasks_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / TASKS_DIR

    def files_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / FILES_DIR

    def approvals(self, project_id: str) -> ApprovalStore:
        """The project's Approval Gate store (the project must exist)."""
        self._require(project_id)
        return ApprovalStore(
            self.project_dir(project_id) / APPROVALS_DIR, project_id, clock=self._clock
        )

    def events(self, project_id: str) -> ProjectEventLog:
        """The project's ``events.jsonl`` (the project must exist)."""
        self._require(project_id)
        return ProjectEventLog(self.project_dir(project_id) / EVENTS_FILE, project_id)

    # -- queries -----------------------------------------------------------

    def exists(self, project_id: str) -> bool:
        return self.project_file(project_id).is_file()

    def list_ids(self) -> list[str]:
        """Ids of all projects (directories with a ``project.json``), sorted."""
        if not self._root.is_dir():
            return []
        ids = []
        for child in self._root.iterdir():
            if child.is_symlink() or not child.is_dir():
                continue
            if _PROJECT_ID_RE.fullmatch(child.name) and (child / PROJECT_FILE).is_file():
                ids.append(child.name)
        return sorted(ids)

    def list_projects(self) -> list[Project]:
        """All projects, oldest first (ties broken by id). Raises on any corrupt project."""
        projects = [self.load(pid) for pid in self.list_ids()]
        return sorted(projects, key=lambda p: (p.created_at, p.id))

    def load(self, project_id: str) -> Project:
        path = self._require(project_id)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ProjectStoreError(f"{path}: unreadable project file") from e
        try:
            project = Project.model_validate(raw)
        except ValidationError as e:
            raise ProjectStoreError(f"{path}: invalid project file") from e
        if project.id != project_id:
            raise ProjectStoreError(
                f"{path}: id {project.id!r} does not match directory {project_id!r}"
            )
        return project

    # -- mutations ---------------------------------------------------------

    def new_id(self) -> str:
        """A fresh id like ``p-20261109-143005-a1b2c3`` (UTC time + random suffix)."""
        now = self._now()
        return f"p-{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"

    def create(self, goal: str, *, project_id: str | None = None) -> Project:
        """Create a new project directory with ``project.json`` and a ``project.created`` event.

        Raises ``ProjectExists`` if the id is taken (never overwrites).
        """
        now = self._now()
        pid = validate_project_id(project_id) if project_id is not None else self.new_id()
        try:
            project = Project(id=pid, goal=goal, created_at=now, updated_at=now)
        except ValidationError as e:
            raise ProjectStoreError("invalid project") from e
        with self._lock:
            self._root.mkdir(parents=True, exist_ok=True)
            pdir = self.project_dir(pid)
            try:
                pdir.mkdir()  # exclusive: fails if another process created it first
            except FileExistsError:
                raise ProjectExists(f"project already exists: {pid!r}") from None
            (pdir / TASKS_DIR).mkdir()
            (pdir / FILES_DIR).mkdir()
            _write_json_atomic(pdir / PROJECT_FILE, project.model_dump(mode="json"))
        self.events(pid).emit(
            Event(ts=now, type="project.created", project_id=pid, data={"goal": goal})
        )
        return project

    def save(self, project: Project) -> Project:
        """Persist ``project``; returns the stored copy (``revision`` + 1, new ``updated_at``).

        Fails with ``ProjectConflict`` if the project changed on disk since ``project``
        was loaded, and with ``ProjectNotFound`` if it does not exist.
        """
        with self._lock:
            current = self.load(project.id)
            if project.revision != current.revision:
                raise ProjectConflict(
                    f"project {project.id!r} changed on disk "
                    f"(revision {current.revision}, yours {project.revision})"
                )
            if project.created_at != current.created_at:
                raise ProjectStoreError("created_at of a project cannot change")
            now = max(self._now(), current.updated_at)
            stored = project.model_copy(
                update={"revision": current.revision + 1, "updated_at": now}
            )
            try:  # model_copy skips validation; re-validate the copy before writing
                stored = Project.model_validate(stored.model_dump())
            except ValidationError as e:
                raise ProjectStoreError(f"invalid project {project.id!r}") from e
            _write_json_atomic(self.project_file(project.id), stored.model_dump(mode="json"))
        return stored

    def set_status(self, project_id: str, status: ProjectStatus) -> Project:
        """Load, change status, save, and log a ``project.status`` event."""
        current = self.load(project_id)
        stored = self.save(current.model_copy(update={"status": ProjectStatus(status)}))
        self.events(project_id).emit(
            Event(
                ts=stored.updated_at,
                type="project.status",
                project_id=project_id,
                data={"from": current.status.value, "to": stored.status.value},
            )
        )
        return stored

    # -- plan --------------------------------------------------------------

    def has_plan(self, project_id: str) -> bool:
        self._require(project_id)
        return self.plan_file(project_id).is_file()

    def load_plan(self, project_id: str) -> Plan:
        """Read ``plan.json``. Raises ``ProjectStoreError`` if missing, corrupt or invalid."""
        self._require(project_id)
        path = self.plan_file(project_id)
        if path.is_symlink() or not path.is_file():
            raise ProjectStoreError(f"project {project_id!r} has no plan")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ProjectStoreError(f"{path}: unreadable plan file") from e
        try:
            return Plan.model_validate(raw)
        except ValidationError as e:
            raise ProjectStoreError(f"{path}: invalid plan file") from e

    def save_plan(self, project_id: str, plan: Plan) -> Plan:
        """Atomically write ``plan.json`` and log a ``plan.saved`` event.

        The plan is re-validated before writing. A plan may only replace an older one:
        its ``version`` must be greater than the stored plan's (re-plans never go back,
        and an existing plan is never silently overwritten with the same version).
        """
        self._require(project_id)
        try:  # model_copy/construct may skip validation; never write an unvalidated plan
            plan = Plan.model_validate(plan.model_dump())
        except ValidationError as e:
            raise ProjectStoreError(f"invalid plan for project {project_id!r}") from e
        path = self.plan_file(project_id)
        with self._lock:
            if path.is_symlink():
                raise ProjectStoreError(f"{path}: plan file must not be a symlink")
            if path.exists():
                current = self.load_plan(project_id)
                if plan.version <= current.version:
                    raise ProjectConflict(
                        f"plan version {plan.version} does not replace stored version "
                        f"{current.version} of project {project_id!r}"
                    )
            _write_json_atomic(path, plan.model_dump(mode="json"))
        self.events(project_id).emit(
            Event(
                ts=self._now(),
                type="plan.saved",
                project_id=project_id,
                data={"version": plan.version, "tasks": len(plan.tasks)},
            )
        )
        return plan

    # -- clarifications ----------------------------------------------------

    def clarifications_file(self, project_id: str) -> Path:
        return self.project_dir(project_id) / CLARIFICATIONS_FILE

    def has_clarifications(self, project_id: str) -> bool:
        self._require(project_id)
        return self.clarifications_file(project_id).is_file()

    def save_clarifications(
        self, project_id: str, items: list[Clarification] | tuple[Clarification, ...]
    ) -> list[Clarification]:
        """Write ``clarifications.json`` once (atomic) and log ``clarifications.saved``.

        The record of what the user answered is never overwritten: a second save raises
        ``ProjectConflict``. Items are re-validated before writing.
        """
        self._require(project_id)
        try:
            record = Clarifications.model_validate(
                {"items": [Clarification.model_validate(i).model_dump() for i in items]}
            )
        except ValidationError as e:
            raise ProjectStoreError(f"invalid clarifications for project {project_id!r}") from e
        path = self.clarifications_file(project_id)
        with self._lock:
            if path.is_symlink():
                raise ProjectStoreError(f"{path}: clarifications file must not be a symlink")
            if path.exists():
                raise ProjectConflict(f"project {project_id!r} already has clarifications")
            _write_json_atomic(path, record.model_dump(mode="json"))
        self.events(project_id).emit(
            Event(
                ts=self._now(),
                type="clarifications.saved",
                project_id=project_id,
                data={"count": len(record.items)},
            )
        )
        return list(record.items)

    def load_clarifications(self, project_id: str) -> list[Clarification]:
        """Read ``clarifications.json``; an absent file means no clarifications (``[]``)."""
        self._require(project_id)
        path = self.clarifications_file(project_id)
        if path.is_symlink():
            raise ProjectStoreError(f"{path}: clarifications file must not be a symlink")
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ProjectStoreError(f"{path}: unreadable clarifications file") from e
        try:
            return list(Clarifications.model_validate(raw).items)
        except ValidationError as e:
            raise ProjectStoreError(f"{path}: invalid clarifications file") from e

    # -- tasks (attempt history and state) ---------------------------------

    def task_dir(self, project_id: str, task_id: str) -> Path:
        """``tasks/<task-id>/`` (validated; never a symlink, always inside ``tasks/``)."""
        if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id):
            raise ProjectStoreError(f"invalid task id: {task_id!r}")
        tasks = self.tasks_dir(project_id)
        if tasks.is_symlink():
            raise ProjectStoreError(f"{tasks}: tasks directory must not be a symlink")
        path = tasks / task_id
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ProjectStoreError(f"task directory escapes the store: {task_id!r}")
        if path.resolve().parent != tasks.resolve():
            raise ProjectStoreError(f"task directory escapes the store: {task_id!r}")
        return path

    def _plan_task(self, project_id: str, task_id: str) -> tuple[Plan, Task]:
        plan = self.load_plan(project_id)
        try:
            return plan, plan.task(task_id)
        except KeyError:
            raise ProjectStoreError(
                f"task {task_id!r} is not in the plan of project {project_id!r}"
            ) from None

    def _ensure_task_dir(self, project_id: str, task_id: str) -> Path:
        path = self.task_dir(project_id, task_id)
        tasks = self.tasks_dir(project_id)
        tasks.mkdir(exist_ok=True)
        path.mkdir(exist_ok=True)
        return self.task_dir(project_id, task_id)  # re-check after creating

    def attempt_count(self, project_id: str, task_id: str) -> int:
        """Number of stored ``TaskResult``s (attempts are numbered 1..n without gaps)."""
        self._require(project_id)
        path = self.task_dir(project_id, task_id)
        if not path.exists():
            return 0
        n = 0
        while (path / RESULT_FILE_FMT.format(attempt=n + 1)).exists() or (
            path / RESULT_FILE_FMT.format(attempt=n + 1)
        ).is_symlink():
            n += 1
        return n

    def save_task_result(self, project_id: str, result: TaskResult) -> TaskResult:
        """Store ``result`` as ``result-<attempt>.json`` (write-once) + ``task.result_saved``.

        The task must be in the plan and ``attempt`` must be exactly the next attempt
        number, so history has no gaps and is never rewritten (``ProjectConflict``).
        """
        self._require(project_id)
        try:
            result = TaskResult.model_validate(result.model_dump())
        except ValidationError as e:
            raise ProjectStoreError(f"invalid task result for project {project_id!r}") from e
        self._plan_task(project_id, result.task_id)
        with self._lock:
            path = self._ensure_task_dir(project_id, result.task_id)
            expected = self.attempt_count(project_id, result.task_id) + 1
            if result.attempt != expected:
                raise ProjectConflict(
                    f"task {result.task_id!r}: attempt {result.attempt} is not the next "
                    f"attempt ({expected})"
                )
            if expected > 1 and not self._verdict_path(path, expected - 1).exists():
                raise ProjectConflict(
                    f"task {result.task_id!r}: attempt {expected - 1} has no verdict yet"
                )
            _write_json_exclusive(
                path / RESULT_FILE_FMT.format(attempt=result.attempt),
                result.model_dump(mode="json"),
            )
        self.events(project_id).emit(
            Event(
                ts=self._now(),
                type="task.result_saved",
                project_id=project_id,
                data={
                    "task_id": result.task_id,
                    "attempt": result.attempt,
                    "status": result.status.value,
                },
            )
        )
        return result

    def load_task_result(self, project_id: str, task_id: str, attempt: int) -> TaskResult:
        self._require(project_id)
        path = self.task_dir(project_id, task_id) / RESULT_FILE_FMT.format(attempt=int(attempt))
        try:
            result = TaskResult.model_validate(_read_json(path, "task result"))
        except ValidationError as e:
            raise ProjectStoreError(f"{path}: invalid task result file") from e
        if result.task_id != task_id or result.attempt != attempt:
            raise ProjectStoreError(f"{path}: task id/attempt do not match the file name")
        return result

    @staticmethod
    def _verdict_path(task_dir: Path, attempt: int) -> Path:
        return task_dir / VERDICT_FILE_FMT.format(attempt=attempt)

    def save_verdict(self, project_id: str, verdict: Verdict) -> Verdict:
        """Store ``verdict`` as ``verdict-<attempt>.json`` (write-once) + ``task.verdict_saved``.

        The result of that attempt must already be stored and the verdict must match
        it and the planned task (``Verdict.matches``).
        """
        self._require(project_id)
        try:
            verdict = Verdict.model_validate(verdict.model_dump())
        except ValidationError as e:
            raise ProjectStoreError(f"invalid verdict for project {project_id!r}") from e
        _, task = self._plan_task(project_id, verdict.task_id)
        with self._lock:
            path = self.task_dir(project_id, verdict.task_id)
            if verdict.attempt > self.attempt_count(project_id, verdict.task_id):
                raise ProjectConflict(
                    f"task {verdict.task_id!r}: no stored result for attempt {verdict.attempt}"
                )
            result = self.load_task_result(project_id, verdict.task_id, verdict.attempt)
            if not verdict.matches(task, result):
                raise ProjectStoreError(
                    f"verdict does not match task {verdict.task_id!r} attempt {verdict.attempt}"
                )
            _write_json_exclusive(
                self._verdict_path(path, verdict.attempt), verdict.model_dump(mode="json")
            )
        self.events(project_id).emit(
            Event(
                ts=self._now(),
                type="task.verdict_saved",
                project_id=project_id,
                data={
                    "task_id": verdict.task_id,
                    "attempt": verdict.attempt,
                    "outcome": verdict.outcome.value,
                },
            )
        )
        return verdict

    def load_verdict(self, project_id: str, task_id: str, attempt: int) -> Verdict | None:
        """The verdict of ``attempt``, or ``None`` if it was not judged (yet)."""
        self._require(project_id)
        path = self._verdict_path(self.task_dir(project_id, task_id), int(attempt))
        if not path.exists() and not path.is_symlink():
            return None
        try:
            verdict = Verdict.model_validate(_read_json(path, "verdict"))
        except ValidationError as e:
            raise ProjectStoreError(f"{path}: invalid verdict file") from e
        if verdict.task_id != task_id or verdict.attempt != attempt:
            raise ProjectStoreError(f"{path}: task id/attempt do not match the file name")
        return verdict

    def task_history(
        self, project_id: str, task_id: str
    ) -> list[tuple[TaskResult, Verdict | None]]:
        """All attempts of a task, oldest first, each with its verdict (if any)."""
        return [
            (
                self.load_task_result(project_id, task_id, n),
                self.load_verdict(project_id, task_id, n),
            )
            for n in range(1, self.attempt_count(project_id, task_id) + 1)
        ]

    def load_task_state(self, project_id: str, task_id: str) -> TaskState | None:
        """``tasks/<id>/state.json``, or ``None`` if the task was never started."""
        self._require(project_id)
        path = self.task_dir(project_id, task_id) / TASK_STATE_FILE
        if not path.exists() and not path.is_symlink():
            return None
        try:
            state = TaskState.model_validate(_read_json(path, "task state"))
        except ValidationError as e:
            raise ProjectStoreError(f"{path}: invalid task state file") from e
        if state.task_id != task_id:
            raise ProjectStoreError(f"{path}: task id does not match the directory")
        return state

    def save_task_state(self, project_id: str, state: TaskState) -> TaskState:
        """Persist ``state`` (``revision`` + 1) and log ``task.status`` on a status change.

        Optimistic concurrency as in ``save``: a new state needs ``revision`` 0, an
        existing one the stored revision (``ProjectConflict``). The state must agree
        with the stored history: ``attempts`` equals the number of stored results, and
        ``passed`` requires a stored ``pass`` verdict for the last attempt.
        """
        self._require(project_id)
        try:
            state = TaskState.model_validate(state.model_dump())
        except ValidationError as e:
            raise ProjectStoreError(f"invalid task state for project {project_id!r}") from e
        plan, _ = self._plan_task(project_id, state.task_id)
        if state.plan_version != plan.version:
            raise ProjectStoreError(
                f"task state is for plan version {state.plan_version}, "
                f"stored plan is version {plan.version}"
            )
        with self._lock:
            tdir = self._ensure_task_dir(project_id, state.task_id)
            path = tdir / TASK_STATE_FILE
            current = self.load_task_state(project_id, state.task_id)
            stored_rev = current.revision if current is not None else 0
            if state.revision != stored_rev:
                raise ProjectConflict(
                    f"task {state.task_id!r} state changed on disk "
                    f"(revision {stored_rev}, yours {state.revision})"
                )
            attempts = self.attempt_count(project_id, state.task_id)
            if state.attempts != attempts:
                raise ProjectStoreError(
                    f"task {state.task_id!r}: state says {state.attempts} attempt(s), "
                    f"{attempts} stored"
                )
            if state.status is TaskStatus.PASSED:
                verdict = self.load_verdict(project_id, state.task_id, attempts)
                if verdict is None or not verdict.passed:
                    raise ProjectStoreError(
                        f"task {state.task_id!r} cannot be passed without a stored "
                        "passing verdict for its last attempt"
                    )
            now = self._now()
            if current is not None:
                now = max(now, current.updated_at)
            stored = TaskState.model_validate(
                state.model_copy(
                    update={"revision": stored_rev + 1, "updated_at": now}
                ).model_dump()
            )
            _write_json_atomic(path, stored.model_dump(mode="json"))
        old = current.status.value if current is not None else None
        if old != stored.status.value:
            self.events(project_id).emit(
                Event(
                    ts=stored.updated_at,
                    type="task.status",
                    project_id=project_id,
                    data={"task_id": stored.task_id, "from": old, "to": stored.status.value},
                )
            )
        return stored

    def task_states(self, project_id: str) -> dict[str, TaskState]:
        """States of the current plan's tasks (``pending`` defaults for unstarted ones)."""
        plan = self.load_plan(project_id)
        states: dict[str, TaskState] = {}
        for task in plan.tasks:
            state = self.load_task_state(project_id, task.id)
            if state is None or state.plan_version != plan.version:
                state = TaskState(
                    task_id=task.id,
                    plan_version=plan.version,
                    attempts=self.attempt_count(project_id, task.id),
                    updated_at=self._now(),
                    revision=state.revision if state is not None else 0,
                )
            states[task.id] = state
        return states

    def passed_tasks(self, project_id: str) -> set[str]:
        """Ids of the current plan's tasks whose state is ``passed``."""
        return {
            tid for tid, s in self.task_states(project_id).items() if s.status is TaskStatus.PASSED
        }

    # -- helpers -----------------------------------------------------------

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise ProjectStoreError("clock returned a naive datetime")
        return now.astimezone(UTC)

    def _require(self, project_id: str) -> Path:
        path = self.project_file(project_id)
        if not path.is_file():
            raise ProjectNotFound(f"no such project: {project_id!r}")
        return path
