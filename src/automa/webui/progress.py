"""Live progress of one project for the web UI (D127).

Read-only and pure apart from reading files that already exist: the project's
``events.jsonl`` (bounded, never followed through a symlink, never created) and
the Ledger entries the caller read. It gives the project page:

* how many tasks passed out of the plan,
* the cost of the latest run so far next to the cost range estimated at its start
  (the same ``compare_estimate``/``check_range`` as ``automa run``, D122),
* the latest events, newest first, with only short scalar fields (never
  ``detail``, error text or model output),
* whether the page should refresh itself (a simple ``<meta http-equiv="refresh">``,
  no script, D127).

Nothing here writes, calls a model, authorizes or stops anything.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from automa.budget import nusd_to_usd
from automa.budget.ledger import LedgerEntry
from automa.events import Event
from automa.finance.compare import check_range, compare_estimate
from automa.project.store import EVENTS_FILE, ProjectStore

MAX_EVENTS_BYTES = 16 * 1024 * 1024  # a larger event log is reported, not read
RECENT_EVENTS = 15
REFRESH_SECONDS = 5
SUMMARY_VALUE_CHARS = 60
# Only these short fields of an event are shown; everything else (``detail``,
# ``error``, arguments, …) may hold model or tool text and is left out.
SUMMARY_KEYS = (
    "task",
    "attempt",
    "tier",
    "model",
    "status",
    "from",
    "to",
    "reason",
    "approval",
    "tool",
    "plan_version",
)
PASSED = "passed"


class _View(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EventLine(_View):
    ts: datetime
    type: str
    summary: str = ""


class CostProgress(_View):
    """The latest run's cost so far against its estimated range."""

    estimated_at: datetime
    spent_usd: float  # calls of the estimated tasks since the estimate
    other_usd: float  # other calls of the project since the estimate
    low_usd: float
    high_usd: float
    worst_usd: float
    verdict: str  # below / within / above (final only once the run is over)


class ProgressView(_View):
    live: bool = False  # the page refreshes itself
    passed: int = 0
    total: int = 0
    cost: CostProgress | None = None
    events: tuple[EventLine, ...] = ()
    errors: tuple[str, ...] = ()


def read_events(store: ProjectStore, project_id: str) -> list[Event]:
    """All events of a project, read-only. ``ValueError`` for a symlink, a non-regular
    file, a file above :data:`MAX_EVENTS_BYTES` or a corrupt line (never skipped)."""
    path = store.project_dir(project_id) / EVENTS_FILE
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return []
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("events.jsonl is a symlink")
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("events.jsonl is not a regular file")
    if st.st_size > MAX_EVENTS_BYTES:
        raise ValueError(f"events.jsonl is larger than {MAX_EVENTS_BYTES} bytes")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as f:
        raw = f.read(MAX_EVENTS_BYTES + 1)
    if len(raw) > MAX_EVENTS_BYTES:
        raise ValueError(f"events.jsonl is larger than {MAX_EVENTS_BYTES} bytes")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("events.jsonl is not UTF-8") from e
    out: list[Event] = []
    for n, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(Event.model_validate(json.loads(line)))
        except (ValueError, TypeError) as e:
            raise ValueError(f"events.jsonl: invalid event on line {n}") from e
    return out


def _short(value: Any) -> str | None:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float | str):
        s = " ".join(str(value).split())
        if len(s) > SUMMARY_VALUE_CHARS:
            s = s[: SUMMARY_VALUE_CHARS - 1] + "…"
        return s
    return None  # lists, objects, None: left out


def event_line(event: Event) -> EventLine:
    """One event as a short line: only :data:`SUMMARY_KEYS` with scalar values."""
    parts = []
    for key in SUMMARY_KEYS:
        if key in event.data:
            value = _short(event.data[key])
            if value is not None:
                parts.append(f"{key}={value}")
    return EventLine(ts=event.ts, type=event.type, summary=" ".join(parts))


def recent_events(events: Sequence[Event], limit: int = RECENT_EVENTS) -> tuple[EventLine, ...]:
    """The latest ``limit`` events, newest first (stable for equal timestamps)."""
    if limit < 0:
        raise ValueError("limit must not be negative")
    indexed = sorted(enumerate(events), key=lambda p: (p[1].ts, p[0]), reverse=True)
    return tuple(event_line(e) for _, e in indexed[:limit])


def cost_progress(
    project_id: str, events: Sequence[Event], entries: Sequence[LedgerEntry]
) -> CostProgress | None:
    """The latest estimate's range next to what was spent since it, or ``None`` when
    there is no readable estimate with a range (D120)."""
    own = [e for e in entries if e.project_id == project_id]
    comparison = compare_estimate(project_id, list(events), own)
    check = check_range(comparison)
    if check is None or comparison.estimate is None:
        return None
    return CostProgress(
        estimated_at=comparison.estimate.ts,
        spent_usd=nusd_to_usd(check.actual_nusd),
        other_usd=nusd_to_usd(check.other_nusd),
        low_usd=nusd_to_usd(check.low_nusd),
        high_usd=nusd_to_usd(check.high_nusd),
        worst_usd=nusd_to_usd(check.worst_nusd),
        verdict=check.verdict.value,
    )


def build_progress(
    store: ProjectStore,
    project_id: str,
    task_states: Sequence[str | None],
    entries: Sequence[LedgerEntry],
    *,
    running: bool,
    pending_approvals: int = 0,
) -> ProgressView:
    """The progress part of a project page.

    ``task_states`` is the state of each plan task (``None`` = not started);
    ``running`` is whether the project or a background run is running. The page
    refreshes itself only while running and while no approval waits (a refresh
    must never wipe a half-filled decision form).
    """
    errors: list[str] = []
    events: list[Event] = []
    try:
        events = read_events(store, project_id)
    except (ValueError, OSError) as e:
        errors.append(f"events: {e}")
    cost = None
    if events:
        try:
            cost = cost_progress(project_id, events, entries)
        except (ValueError, TypeError) as e:
            errors.append(f"cost range: {e}")
    return ProgressView(
        live=running and pending_approvals == 0,
        passed=sum(1 for s in task_states if s == PASSED),
        total=len(task_states),
        cost=cost,
        events=recent_events(events),
        errors=tuple(errors),
    )
