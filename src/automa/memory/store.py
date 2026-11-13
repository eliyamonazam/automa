"""File storage of the three memory layers (D48).

Layout under ``<workspace>/memory/``::

    user/<id>.json                 user preferences
    lessons/<id>.json              general lessons
    projects/<project_id>/<id>.json  knowledge of one project

Every entry is its own write-once JSON file; forgetting deletes the file. Reading
fails closed: a symlinked directory or file, a non-regular or oversized file,
invalid JSON or an entry stored in the wrong place rejects the whole layer, so
a broken memory is never silently ignored or partly used.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from automa.memory.models import (
    MEMORY_ID_PATTERN,
    MemoryEntry,
    MemoryScope,
    MemorySource,
    normalize_text,
)
from automa.plan.models import _format_errors
from automa.project import ProjectStoreError, validate_project_id
from automa.project.store import _write_json_exclusive

MEMORY_DIRNAME = "memory"
MAX_MEMORY_FILE_BYTES = 16 * 1024
MAX_ENTRIES_PER_LAYER = 200
_MEMORY_ID_RE = re.compile(MEMORY_ID_PATTERN)
_LAYER_DIRS = {MemoryScope.USER: "user", MemoryScope.LESSON: "lessons"}
_PROJECTS_DIR = "projects"

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryStoreError(Exception):
    """Memory cannot be read or written."""


class MemoryNotFound(MemoryStoreError):
    pass


class MemoryFull(MemoryStoreError):
    """A layer already holds ``MAX_ENTRIES_PER_LAYER`` entries."""


def validate_memory_id(memory_id: str) -> str:
    if not isinstance(memory_id, str) or not _MEMORY_ID_RE.fullmatch(memory_id):
        raise MemoryStoreError(f"invalid memory id: {memory_id!r}")
    return memory_id


def _project_id(project_id: str) -> str:
    try:
        return validate_project_id(project_id)
    except ProjectStoreError as e:
        raise MemoryStoreError(str(e)) from None


def _read_entry_file(path: Path) -> bytes:
    # O_NONBLOCK: opening a FIFO must not hang; it is then rejected as non-regular.
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise MemoryNotFound(f"{path}: memory file is missing") from None
    except OSError as e:
        raise MemoryStoreError(f"{path}: cannot open memory file ({e.strerror})") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise MemoryStoreError(f"{path}: memory file is not a regular file")
        if st.st_size > MAX_MEMORY_FILE_BYTES:
            raise MemoryStoreError(f"{path}: memory file larger than {MAX_MEMORY_FILE_BYTES} bytes")
        with os.fdopen(fd, "rb", closefd=True) as f:
            fd = -1
            data = f.read(MAX_MEMORY_FILE_BYTES + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > MAX_MEMORY_FILE_BYTES:
        raise MemoryStoreError(f"{path}: memory file larger than {MAX_MEMORY_FILE_BYTES} bytes")
    return data


class MemoryContext(BaseModel):
    """The memory that applies to one project: all three layers, oldest first."""

    model_config = ConfigDict(frozen=True)

    user: tuple[MemoryEntry, ...] = ()
    project: tuple[MemoryEntry, ...] = ()
    lessons: tuple[MemoryEntry, ...] = ()

    def is_empty(self) -> bool:
        return not (self.user or self.project or self.lessons)


class MemoryStore:
    """Reads and writes ``<workspace>/memory/``. Never calls a model."""

    def __init__(self, root: str | Path, *, clock: Clock = _utc_now):
        self._root = Path(root)
        self._clock = clock
        self._lock = threading.Lock()

    @classmethod
    def for_workspace(cls, workspace: str | Path, *, clock: Clock = _utc_now) -> MemoryStore:
        return cls(Path(workspace) / MEMORY_DIRNAME, clock=clock)

    @property
    def root(self) -> Path:
        return self._root

    # -- paths ----------------------------------------------------------------

    def layer_dir(self, scope: MemoryScope, project_id: str | None = None) -> Path:
        scope = MemoryScope(scope)
        if scope is MemoryScope.PROJECT:
            if project_id is None:
                raise MemoryStoreError("project memory needs a project id")
            return self._root / _PROJECTS_DIR / _project_id(project_id)
        if project_id is not None:
            raise MemoryStoreError(f"{scope.value} memory is not tied to a project")
        return self._root / _LAYER_DIRS[scope]

    def _check_dirs(self, directory: Path) -> bool:
        """True if ``directory`` exists; raise if it or any parent up to the root is a symlink."""
        chain = [directory]
        parent = directory
        while parent != self._root and self._root in parent.parents:
            parent = parent.parent
            chain.append(parent)
        for d in reversed(chain):
            if d.is_symlink():
                raise MemoryStoreError(f"{d}: memory directory must not be a symlink")
            if d.exists() and not d.is_dir():
                raise MemoryStoreError(f"{d}: memory path is not a directory")
        return directory.is_dir()

    # -- reading --------------------------------------------------------------

    def _load(self, path: Path, scope: MemoryScope, project_id: str | None) -> MemoryEntry:
        data = _read_entry_file(path)
        try:
            raw = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise MemoryStoreError(f"{path}: unreadable memory file") from None
        try:
            entry = MemoryEntry.model_validate(raw)
        except ValidationError as e:
            raise MemoryStoreError(f"{path}: invalid memory file:\n{_format_errors(e)}") from None
        if entry.id != path.name.removesuffix(".json"):
            raise MemoryStoreError(f"{path}: id {entry.id!r} does not match the file name")
        if entry.scope is not scope:
            raise MemoryStoreError(
                f"{path}: {entry.scope.value} memory stored in the {scope.value} layer"
            )
        if scope is MemoryScope.PROJECT and entry.project_id != project_id:
            raise MemoryStoreError(
                f"{path}: memory of project {entry.project_id!r} in the wrong place"
            )
        return entry

    def entries(self, scope: MemoryScope, project_id: str | None = None) -> list[MemoryEntry]:
        """All entries of one layer, oldest first. Raises on any broken entry (fail closed)."""
        scope = MemoryScope(scope)
        directory = self.layer_dir(scope, project_id)
        if not self._check_dirs(directory):
            return []
        result = []
        for child in sorted(directory.iterdir(), key=lambda p: p.name):
            name = child.name
            if name.startswith("."):
                continue  # temp files of an in-progress write
            stem = name.removesuffix(".json")
            if not name.endswith(".json") or not _MEMORY_ID_RE.fullmatch(stem):
                raise MemoryStoreError(f"{child}: unexpected entry in the memory directory")
            result.append(self._load(child, scope, project_id))
        return sorted(result, key=lambda e: (e.created_at, e.id))

    def project_ids(self) -> list[str]:
        """Projects that have a project memory directory."""
        projects = self._root / _PROJECTS_DIR
        if not self._check_dirs(projects):
            return []
        ids = []
        for child in sorted(projects.iterdir(), key=lambda p: p.name):
            if child.name.startswith("."):
                continue
            if child.is_symlink() or not child.is_dir():
                raise MemoryStoreError(f"{child}: unexpected entry in the project memory directory")
            ids.append(_project_id(child.name))
        return ids

    def context(self, project_id: str | None = None) -> MemoryContext:
        """User preferences, lessons and (if given) the project's own memory."""
        return MemoryContext(
            user=tuple(self.entries(MemoryScope.USER)),
            project=tuple(self.entries(MemoryScope.PROJECT, project_id)) if project_id else (),
            lessons=tuple(self.entries(MemoryScope.LESSON)),
        )

    def find(self, memory_id: str) -> MemoryEntry:
        validate_memory_id(memory_id)
        layers: list[tuple[MemoryScope, str | None]] = [
            (MemoryScope.USER, None),
            (MemoryScope.LESSON, None),
        ]
        layers += [(MemoryScope.PROJECT, pid) for pid in self.project_ids()]
        for scope, pid in layers:
            directory = self.layer_dir(scope, pid)
            path = directory / f"{memory_id}.json"
            if not self._check_dirs(directory):
                continue
            if path.is_symlink() or path.exists():
                return self._load(path, scope, pid)
        raise MemoryNotFound(f"no such memory: {memory_id!r}")

    # -- writing --------------------------------------------------------------

    def add(
        self,
        scope: MemoryScope,
        text: str,
        *,
        project_id: str | None = None,
        source: MemorySource = MemorySource.USER,
    ) -> MemoryEntry:
        """Write one new entry. The same text twice in one layer is an error."""
        scope = MemoryScope(scope)
        if scope is MemoryScope.PROJECT and project_id is None:
            raise MemoryStoreError("project memory needs a project id")
        directory = self.layer_dir(scope, project_id if scope is MemoryScope.PROJECT else None)
        with self._lock:
            now = self._clock()
            fields = {
                "id": f"mem-{now.astimezone(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}",
                "scope": scope,
                "text": text,
                "project_id": project_id,
                "source": source,
                "created_at": now,
            }
            try:
                entry = MemoryEntry.model_validate(fields)
            except ValidationError as e:
                raise MemoryStoreError(f"invalid memory:\n{_format_errors(e)}") from None
            existing = self.entries(scope, project_id if scope is MemoryScope.PROJECT else None)
            if len(existing) >= MAX_ENTRIES_PER_LAYER:
                raise MemoryFull(
                    f"{scope.value} memory is full ({MAX_ENTRIES_PER_LAYER} entries); "
                    "forget some entries first"
                )
            key = normalize_text(entry.text)
            for other in existing:
                if normalize_text(other.text) == key:
                    raise MemoryStoreError(f"the same memory already exists: {other.id}")
            directory.mkdir(parents=True, exist_ok=True)
            self._check_dirs(directory)
            try:
                _write_json_exclusive(directory / f"{entry.id}.json", entry.model_dump(mode="json"))
            except ProjectStoreError as e:
                raise MemoryStoreError(str(e)) from None
            return entry

    def forget(self, memory_id: str) -> MemoryEntry:
        """Delete one entry and return what it was."""
        with self._lock:
            entry = self.find(memory_id)
            pid = entry.project_id if entry.scope is MemoryScope.PROJECT else None
            path = self.layer_dir(entry.scope, pid) / f"{entry.id}.json"
            try:
                path.unlink()
            except FileNotFoundError:
                raise MemoryNotFound(f"no such memory: {memory_id!r}") from None
            return entry
