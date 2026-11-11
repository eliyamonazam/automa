"""Event log: structured, append-only records of what happened (``events.jsonl``).

Components emit ``Event`` objects to an ``EventSink``. ``JsonlEventLog`` appends one
JSON object per line to a file (the per-project ``events.jsonl`` of the Project
Store, M2); ``MemoryEventSink`` keeps them in memory (tests, inspection).
Sinks must raise on failure; callers decide how to fail closed.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Event(BaseModel):
    """One log record. ``type`` is a dotted name, e.g. ``budget.warning``."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    ts: datetime
    type: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
    project_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ts")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("event timestamps must be timezone-aware")
        return v.astimezone(UTC)


class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...


class MemoryEventSink:
    """Keeps events in a list. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[Event] = []

    def emit(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def of_type(self, type_: str) -> list[Event]:
        return [e for e in self.events if e.type == type_]


# One lock per log file, shared by every ``JsonlEventLog`` on that path in this
# process, so lines from concurrent writers (parallel tasks, D35) never interleave.
_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _path_lock(path: Path) -> threading.Lock:
    key = str(path.absolute())
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = _PATH_LOCKS[key] = threading.Lock()
        return lock


class JsonlEventLog:
    """Appends events as JSON lines to ``path`` (created with its parent dirs)."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._lock = _path_lock(self._path)

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event: Event) -> None:
        line = event.model_dump_json() + "\n"
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()

    def read(self) -> list[Event]:
        """All events in file order. Raises on a corrupt line (never skips silently)."""
        if not self._path.exists():
            return []
        out: list[Event] = []
        with self._path.open(encoding="utf-8") as f:
            for n, raw in enumerate(f, start=1):
                if not raw.strip():
                    continue
                try:
                    out.append(Event.model_validate(json.loads(raw)))
                except (ValueError, TypeError) as e:
                    raise ValueError(f"{self._path}: invalid event on line {n}") from e
        return out
