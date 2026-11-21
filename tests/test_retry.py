"""Tests for retrying failed tasks (automa.retry, D156) and the deliverable (D155).

FakeProvider behind the real gateway/guard/agents (Rig from test_orchestrator);
no network, no real model calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from automa.budget import CostLedger
from automa.budget.brake import BRAKE_CONTINUED, TaskBrake, TaskBrakeStop
from automa.config import Tier
from automa.events import Event
from automa.llm.gateway import CallContext, CallEstimate
from automa.orchestrator import StopReason
from automa.plan import Plan, TaskStatus
from automa.plan.models import MAX_TASK_ATTEMPTS, Deliverable
from automa.project import ProjectStatus
from automa.retry import (
    RETRY_GRANTED,
    RetryError,
    brake_inputs,
    can_retry,
    failed_tasks,
    grant_retry,
    retry_grants,
    strongest_tier,
)
from test_orchestrator import PID, Rig, make_plan, task_dict, write

T0 = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)


def _ev(type_: str, **data: object) -> Event:
    return Event(ts=T0, type=type_, project_id=PID, data=dict(data))


def failed_rig(tmp_path: Path, extra_script: list[object]) -> Rig:
    plan = make_plan(task_dict("a"), task_dict("b", deps=["a"]), task_dict("d"))
    script = [*write("a", "bad"), *write("a", "bad"), *write("d"), *extra_script]
    rig = Rig(tmp_path, script, plan)
    report = rig.run(max_attempts=2)
    assert report.status is ProjectStatus.FAILED
    return rig


# --- retry ---------------------------------------------------------------------


def test_retry_turns_a_failed_project_into_a_product(tmp_path: Path) -> None:
    rig = failed_rig(tmp_path, [*write("a", "ok"), *write("b")])
    assert can_retry(rig.store, PID)
    assert failed_tasks(rig.store, PID) == ["a"]
    granted = grant_retry(rig.store, PID, rig.cfg, via="test", spent_nusd={"a": 123})
    assert granted == ["a"]
    assert rig.project_status() is ProjectStatus.PAUSED
    assert rig.state("a").status is TaskStatus.PENDING
    assert rig.state("a").tier is Tier.LARGE  # cheaper tiers already failed
    assert rig.state("b").status is TaskStatus.PENDING  # was blocked
    assert rig.state("d").status is TaskStatus.PASSED  # untouched
    report = rig.run(max_attempts=2)
    assert report.status is ProjectStatus.DONE
    assert report.passed == ["a", "b", "d"]
    assert rig.store.attempt_count(PID, "a") == 3
    assert rig.models()[-4] == "fake-large"  # the retried attempt ran on the large tier
    granted_events = [e for e in rig.store.events(PID).read() if e.type == RETRY_GRANTED]
    assert len(granted_events) == 1
    assert granted_events[0].data == {
        "task": "a",
        "attempts": 2,
        "via": "test",
        "spent_nusd": 123,
        "tier": "large",
    }


def test_retry_feedback_reaches_the_new_attempt(tmp_path: Path) -> None:
    rig = failed_rig(tmp_path, [*write("a", "ok"), *write("b")])
    grant_retry(rig.store, PID, rig.cfg, via="test")
    n = len(rig.provider.requests)
    rig.run(max_attempts=2)
    first = rig.provider.requests[n]
    text = "\n".join(m.content or "" for m in first.messages)
    assert "ok" in text  # the verifier's issue about the missing text is fed back


def test_retry_attempts_are_bounded_and_fail_again(tmp_path: Path) -> None:
    script = [*write("a", "bad"), *write("a", "bad")]
    rig = failed_rig(tmp_path, script)
    grant_retry(rig.store, PID, rig.cfg, via="test")
    report = rig.run(max_attempts=2)
    assert report.status is ProjectStatus.FAILED
    assert rig.store.attempt_count(PID, "a") == 4  # 2 + 2 granted, not more
    assert rig.provider.remaining == 0
    # a second retry is possible and grants two more
    grant_retry(rig.store, PID, rig.cfg, via="test")
    assert retry_grants(rig.store.events(PID).read()) == {"a": 4}


def test_retry_refused_unless_failed(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [*write("a")], make_plan(task_dict("a")))
    with pytest.raises(RetryError, match="only a failed project"):
        grant_retry(rig.store, PID, rig.cfg, via="test")
    rig.run()
    assert rig.project_status() is ProjectStatus.DONE
    assert not can_retry(rig.store, PID)
    with pytest.raises(RetryError):
        grant_retry(rig.store, PID, rig.cfg, via="test")
    assert not [e for e in rig.store.events(PID).read() if e.type == RETRY_GRANTED]


def test_retry_attempt_bounds(tmp_path: Path) -> None:
    rig = failed_rig(tmp_path, [])
    for bad in (0, 11):
        with pytest.raises(RetryError, match="attempts"):
            grant_retry(rig.store, PID, rig.cfg, via="test", attempts=bad)
    real = rig.store.attempt_count
    rig.store.attempt_count = lambda pid, tid: MAX_TASK_ATTEMPTS - 1  # type: ignore[method-assign]
    with pytest.raises(RetryError, match="at most"):
        grant_retry(rig.store, PID, rig.cfg, via="test")
    rig.store.attempt_count = real  # type: ignore[method-assign]
    assert rig.project_status() is ProjectStatus.FAILED  # nothing changed


def test_plain_rerun_of_failed_project_still_spends_nothing(tmp_path: Path) -> None:
    rig = failed_rig(tmp_path, [])
    n = len(rig.provider.requests)
    assert rig.run(max_attempts=2).reason is StopReason.TASKS_FAILED
    assert len(rig.provider.requests) == n


def test_retry_grants_ignores_malformed_events() -> None:
    events = [
        _ev(RETRY_GRANTED, task="a", attempts=2),
        _ev(RETRY_GRANTED, task="a", attempts=True),
        _ev(RETRY_GRANTED, task="a", attempts=-1),
        _ev(RETRY_GRANTED, task=3, attempts=2),
        _ev("other.event", task="a", attempts=5),
        _ev(RETRY_GRANTED, task="b", attempts=1),
    ]
    assert retry_grants(events) == {"a": 2, "b": 1}


def test_strongest_tier(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [], tiers={"small": "s", "medium": "m"})
    assert strongest_tier(rig.cfg) is Tier.MEDIUM


# --- brake after a retry ----------------------------------------------------------


def test_brake_inputs_restart_after_retry() -> None:
    events = [
        _ev(BRAKE_CONTINUED, task="a"),
        _ev(BRAKE_CONTINUED, task="b"),
        _ev(RETRY_GRANTED, task="a", attempts=2, spent_nusd=500),
        _ev(BRAKE_CONTINUED, task="a"),
        _ev(RETRY_GRANTED, task="c", attempts=2, spent_nusd="x"),
    ]
    continues, offsets = brake_inputs(events)
    assert continues == {"a": 1, "b": 1}
    assert offsets == {"a": 500, "c": 0}


class _Inner:
    def __init__(self) -> None:
        self.authorized = 0

    def authorize(self, estimate: CallEstimate) -> None:
        self.authorized += 1

    def record(self, estimate: CallEstimate, result: object) -> object:
        return None

    def release(self, estimate: CallEstimate) -> None:
        return None


def _estimate(task: str) -> CallEstimate:
    return CallEstimate(
        model="m",
        tier=Tier.SMALL,
        max_input_tokens=10,
        max_output_tokens=10,
        max_cost_usd=0.001,
        context=CallContext(project_id=PID, task_id=task),
    )


def test_brake_offset_counts_fresh_steps_only(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "l.sqlite3")
    inner = _Inner()
    spent = {"a": 1_000}
    ledger.spent_nusd = lambda **kw: spent[kw["task_id"]]  # type: ignore[method-assign]
    braked = TaskBrake(inner, ledger, PID, {"a": 400}, default_step_nusd=400)
    with pytest.raises(TaskBrakeStop):
        braked.authorize(_estimate("a"))
    retried = TaskBrake(inner, ledger, PID, {"a": 400}, default_step_nusd=400, offsets={"a": 900})
    assert retried.limit_nusd("a") == 1_300
    retried.authorize(_estimate("a"))
    assert inner.authorized == 1
    spent["a"] = 1_301
    with pytest.raises(TaskBrakeStop):
        retried.authorize(_estimate("a"))
    with pytest.raises(ValueError, match="offsets"):
        TaskBrake(inner, ledger, PID, {}, default_step_nusd=1, offsets={"a": -1})


# --- deliverable (D155) -------------------------------------------------------------


def _plan_with(deliverable: dict) -> Plan:
    return Plan.model_validate({"goal": "g", "tasks": [task_dict("a")], "deliverable": deliverable})


def test_deliverable_main_files_must_be_checked() -> None:
    plan = _plan_with({"summary": "a page", "main_files": ["a.txt"], "how_to_use": "open it"})
    assert plan.deliverable == Deliverable(
        summary="a page", main_files=["a.txt"], how_to_use="open it"
    )
    assert plan.task_for_file("a.txt") is not None
    assert plan.task_for_file("zzz") is None
    with pytest.raises(ValidationError, match="not checked by any task"):
        _plan_with({"summary": "x", "main_files": ["other.html"]})
    with pytest.raises(ValidationError, match="duplicate"):
        _plan_with({"summary": "x", "main_files": ["a.txt", "a.txt"]})
    with pytest.raises(ValidationError):
        _plan_with({"summary": "  ", "main_files": []})
    with pytest.raises(ValidationError):
        _plan_with({"summary": "x", "main_files": ["../etc/passwd"]})


def test_plan_without_deliverable_still_loads() -> None:
    plan = make_plan(task_dict("a"))
    assert plan.deliverable is None
    again = Plan.model_validate(plan.model_dump(mode="json"))
    assert again == plan
