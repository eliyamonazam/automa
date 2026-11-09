"""Tests for the event log primitives (automa.events)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from automa.events import Event, JsonlEventLog, MemoryEventSink

TS = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)


def test_event_normalizes_to_utc() -> None:
    tehran = timezone(timedelta(hours=3, minutes=30))
    e = Event(ts=datetime(2026, 11, 9, 15, 30, tzinfo=tehran), type="budget.warning")
    assert e.ts == TS and e.ts.tzinfo is UTC


def test_event_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError):
        Event(ts=datetime(2026, 11, 9, 12, 0), type="budget.warning")  # noqa: DTZ001


@pytest.mark.parametrize("bad", ["", "budget", "Budget.Warning", "budget.", ".x", "a b.c"])
def test_event_type_must_be_dotted_lowercase(bad: str) -> None:
    with pytest.raises(ValidationError):
        Event(ts=TS, type=bad)


def test_event_is_frozen_and_strict() -> None:
    e = Event(ts=TS, type="budget.stop")
    with pytest.raises(ValidationError):
        e.type = "x.y"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Event(ts=TS, type="a.b", unknown=1)  # type: ignore[call-arg]


def test_memory_sink_keeps_order_and_filters() -> None:
    sink = MemoryEventSink()
    sink.emit(Event(ts=TS, type="a.one"))
    sink.emit(Event(ts=TS, type="a.two"))
    sink.emit(Event(ts=TS, type="a.one", data={"n": 2}))
    assert [e.type for e in sink.events] == ["a.one", "a.two", "a.one"]
    assert [e.data for e in sink.of_type("a.one")] == [{}, {"n": 2}]


def test_jsonl_roundtrip_appends_and_creates_dirs(tmp_path: Path) -> None:
    path = tmp_path / "projects" / "p1" / "events.jsonl"
    log = JsonlEventLog(path)
    assert log.read() == []
    first = Event(ts=TS, type="budget.warning", project_id="p1", data={"threshold_pct": 50})
    second = Event(ts=TS, type="budget.stop", project_id="p1")
    log.emit(first)
    JsonlEventLog(path).emit(second)  # a second writer appends, never truncates
    assert path.read_text(encoding="utf-8").count("\n") == 2
    assert log.read() == [first, second]


def test_jsonl_corrupt_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = JsonlEventLog(path)
    log.emit(Event(ts=TS, type="a.b"))
    with path.open("a", encoding="utf-8") as f:
        f.write("{not json\n")
    with pytest.raises(ValueError, match="line 2"):
        log.read()
