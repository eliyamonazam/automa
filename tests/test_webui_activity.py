"""The live "what is it doing now" line (D171)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from automa.events import Event
from automa.webui.activity import Activity, current_activity
from automa.webui.i18n import MESSAGES

T0 = datetime(2026, 11, 22, 10, 0, tzinfo=UTC)


def ev(n: int, type_: str, **data: object) -> Event:
    return Event(ts=T0 + timedelta(seconds=n), type=type_, project_id="p", data=data)


def test_nothing_before_any_step() -> None:
    assert current_activity([]) is None
    assert current_activity([ev(0, "orchestrator.started")]) is None


def test_working_with_last_file_action() -> None:
    events = [
        ev(0, "orchestrator.started"),
        ev(1, "orchestrator.attempt", task="t1", attempt=1, tier="cheap"),
        ev(2, "worker.started", task="t1", attempt=1),
        ev(3, "tool.call", tool="read_file", risk="read", ok=True, path="a.md"),
        ev(4, "tool.call", tool="write_file", risk="write", ok=True, path="site/index.html"),
    ]
    a = current_activity(events, {"t1": "Build the page"})
    assert a == Activity(
        phase="working",
        task_title="Build the page",
        action_key="activity.wrote",
        action_arg="site/index.html",
        at=T0 + timedelta(seconds=4),
    )


def test_failed_tools_and_unknown_tools_are_not_shown() -> None:
    events = [
        ev(1, "worker.started", task="t1"),
        ev(2, "tool.call", tool="edit_file", ok=True, path="x.py"),
        ev(3, "tool.call", tool="write_file", ok=False, path="bad.py"),
        ev(4, "tool.call", tool="something_new", ok=True, path="secret"),
    ]
    a = current_activity(events)
    assert a is not None
    assert (a.action_key, a.action_arg) == ("activity.edited", "x.py")
    assert a.task_title == "t1"  # no title known: the id


def test_only_audit_fields_reach_the_line() -> None:
    events = [
        ev(1, "worker.started", task="t1"),
        ev(2, "tool.call", tool="run_command", ok=True, program="pytest", cwd="w", argv="x"),
        ev(3, "tool.call", tool="fetch_url", ok=True, host="example.com", url="https://s?k=1"),
    ]
    a = current_activity(events)
    assert a is not None
    assert (a.action_key, a.action_arg) == ("activity.fetched", "example.com")
    events.append(ev(4, "tool.call", tool="web_search", ok=True, query="private words"))
    b = current_activity(events)
    assert b is not None
    assert (b.action_key, b.action_arg) == ("activity.searched", "")


def test_checking_then_next_task_resets_action() -> None:
    events = [
        ev(1, "worker.started", task="t1"),
        ev(2, "tool.call", tool="write_file", ok=True, path="a"),
        ev(3, "worker.finished", task="t1"),
        ev(4, "verifier.checks", task="t1", attempt=1, passed=1, failed=0),
        # A tool call while checking (the verifier runs checks) is not the worker's.
        ev(5, "tool.call", tool="run_command", ok=True, program="pytest"),
    ]
    a = current_activity(events, {"t1": "One"})
    assert a is not None
    assert (a.phase, a.task_title, a.action_key) == ("checking", "One", "activity.wrote")
    events.append(ev(6, "orchestrator.attempt", task="t2", attempt=1))
    b = current_activity(events, {"t1": "One", "t2": "Two"})
    assert b is not None
    assert (b.phase, b.task_title, b.action_key) == ("working", "Two", None)


def test_final_review_and_only_latest_run() -> None:
    events = [
        ev(1, "worker.started", task="t1"),
        ev(2, "final_review.started", plan_version=1),
    ]
    a = current_activity(events)
    assert a is not None
    assert (a.phase, a.task_title, a.action_key) == ("reviewing", None, None)
    events.append(ev(3, "orchestrator.started"))
    assert current_activity(events) is None


def test_long_names_are_shortened() -> None:
    events = [
        ev(1, "worker.started", task="t1"),
        ev(2, "tool.call", tool="write_file", ok=True, path="d/" + "x" * 300),
    ]
    a = current_activity(events, {"t1": "y" * 300})
    assert a is not None
    assert len(a.action_arg) == 80 and a.action_arg.endswith("…")
    assert a.task_title is not None and len(a.task_title) == 80


def test_every_action_key_is_translated() -> None:
    from automa.webui.activity import _ACTIONS

    keys = {k for k, _ in _ACTIONS.values()} | {
        "activity.heading",
        "activity.working",
        "activity.working_any",
        "activity.checking",
        "activity.checking_any",
        "activity.reviewing",
        "activity.last",
    }
    for key in keys:
        assert set(MESSAGES[key]) == {"en", "fa"}, key
