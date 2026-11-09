"""``ProjectStore``: on-disk project state under ``<workspace>/projects/<project-id>/``.

Layout (see docs/ARCHITECTURE.md, "Project Store")::

    project.json   goal, status, timestamps, revision
    plan.json      tasks and dependencies (written by later milestones)
    tasks/         per-task inputs, outputs, verdicts, attempt history
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
from automa.plan import Plan

SCHEMA_VERSION = 1
PROJECTS_DIRNAME = "projects"
PROJECT_FILE = "project.json"
PLAN_FILE = "plan.json"
EVENTS_FILE = "events.jsonl"
TASKS_DIR = "tasks"
FILES_DIR = "files"

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
