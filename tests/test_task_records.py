"""Tests for per-task attempt history and state in ``ProjectStore`` (``tasks/<id>/``)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from automa.config import Tier
from automa.events import JsonlEventLog
from automa.plan import (
    MAX_TASK_ATTEMPTS,
    AcceptanceCriterion,
    CheckResult,
    Issue,
    Plan,
    ResultStatus,
    Task,
    TaskResult,
    TaskState,
    TaskStatus,
    Verdict,
    VerdictOutcome,
)
from automa.project import ProjectConflict, ProjectNotFound, ProjectStore, ProjectStoreError

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def _task(tid: str, deps: list[str] | None = None) -> Task:
    return Task(
        id=tid,
        title=f"task {tid}",
        role="writer",
        depends_on=deps or [],
        acceptance_criteria=[AcceptanceCriterion(description="it is good")],
    )


def _plan(version: int = 1) -> Plan:
    return Plan(version=version, goal="g", tasks=[_task("a"), _task("b", ["a"])])


def _result(tid: str = "a", attempt: int = 1, status: ResultStatus = ResultStatus.COMPLETED):
    return TaskResult(
        task_id=tid,
        attempt=attempt,
        status=status,
        summary="did it",
        files=["out.md"],
        tier=Tier.MEDIUM,
        model="fake-medium",
        steps=2,
        cost_nusd=123,
        error="boom" if status is ResultStatus.FAILED else None,
        started_at=T0,
        finished_at=T0 + timedelta(seconds=5),
    )


def _verdict(tid: str = "a", attempt: int = 1, passed: bool = True, criterion: int = 0):
    if passed:
        return Verdict(task_id=tid, attempt=attempt, outcome=VerdictOutcome.PASS, created_at=T0)
    return Verdict(
        task_id=tid,
        attempt=attempt,
        outcome=VerdictOutcome.FAIL,
        issues=[Issue(description="not good", criterion=criterion)],
        created_at=T0,
    )


def _state(tid: str = "a", **kw) -> TaskState:
    kw.setdefault("updated_at", T0)
    return TaskState(task_id=tid, **kw)


@pytest.fixture
def store(tmp_path: Path) -> ProjectStore:
    s = ProjectStore.for_workspace(tmp_path / "ws")
    s.create("goal", project_id="p1")
    s.save_plan("p1", _plan())
    return s


def _events(store: ProjectStore, kind: str) -> list[dict]:
    log = JsonlEventLog(store.project_dir("p1") / "events.jsonl")
    return [e.data for e in log.read() if e.type == kind]


# --- TaskState model ----------------------------------------------------------


def test_task_state_defaults_and_limits() -> None:
    s = _state()
    assert s.status is TaskStatus.PENDING and s.attempts == 0 and s.tier is None
    assert s.revision == 0 and s.plan_version == 1
    with pytest.raises(ValidationError):
        _state(attempts=MAX_TASK_ATTEMPTS + 1)
    with pytest.raises(ValidationError):
        _state(attempts=-1)
    with pytest.raises(ValidationError):
        _state(tid="../x")
    with pytest.raises(ValidationError):
        TaskState.model_validate({"task_id": "a", "updated_at": T0, "extra": 1})
    with pytest.raises(ValidationError):
        _state(schema_version=99)
    with pytest.raises(ValidationError):
        _state(updated_at=datetime(2026, 1, 1))  # naive


def test_task_state_consistency_rules() -> None:
    with pytest.raises(ValidationError):
        _state(status=TaskStatus.PASSED, attempts=0)
    for status in (TaskStatus.FAILED, TaskStatus.BLOCKED):
        with pytest.raises(ValidationError):
            _state(status=status)
        with pytest.raises(ValidationError):
            _state(status=status, error="   ")
        assert _state(status=status, error="why").error == "why"


def test_task_state_is_frozen() -> None:
    s = _state()
    with pytest.raises(ValidationError):
        s.status = TaskStatus.PASSED  # type: ignore[misc]


# --- results ------------------------------------------------------------------


def test_result_roundtrip_and_event(store: ProjectStore) -> None:
    assert store.attempt_count("p1", "a") == 0
    assert store.task_history("p1", "a") == []
    saved = store.save_task_result("p1", _result())
    assert store.attempt_count("p1", "a") == 1
    assert store.load_task_result("p1", "a", 1) == saved
    path = store.task_dir("p1", "a") / "result-1.json"
    assert json.loads(path.read_text(encoding="utf-8"))["summary"] == "did it"
    ev = _events(store, "task.result_saved")
    assert ev == [{"task_id": "a", "attempt": 1, "status": "completed"}]
    assert "did it" not in json.dumps(ev)  # no content in events


def test_result_is_write_once(store: ProjectStore) -> None:
    store.save_task_result("p1", _result())
    with pytest.raises(ProjectConflict):
        store.save_task_result("p1", _result())
    assert store.attempt_count("p1", "a") == 1


def test_result_attempts_must_be_sequential(store: ProjectStore) -> None:
    with pytest.raises(ProjectConflict):
        store.save_task_result("p1", _result(attempt=2))
    store.save_task_result("p1", _result(attempt=1))
    # next attempt needs a verdict for the previous one first
    with pytest.raises(ProjectConflict):
        store.save_task_result("p1", _result(attempt=2))
    store.save_verdict("p1", _verdict(passed=False))
    with pytest.raises(ProjectConflict):
        store.save_task_result("p1", _result(attempt=3))
    store.save_task_result("p1", _result(attempt=2))
    assert store.attempt_count("p1", "a") == 2


def test_result_for_unknown_task_or_project(store: ProjectStore) -> None:
    with pytest.raises(ProjectStoreError, match="not in the plan"):
        store.save_task_result("p1", _result(tid="zzz"))
    assert not (store.tasks_dir("p1") / "zzz").exists()
    with pytest.raises(ProjectNotFound):
        store.save_task_result("nope", _result())


def test_result_needs_a_plan(tmp_path: Path) -> None:
    s = ProjectStore.for_workspace(tmp_path / "ws")
    s.create("goal", project_id="p2")
    with pytest.raises(ProjectStoreError, match="no plan"):
        s.save_task_result("p2", _result())


def test_result_is_revalidated(store: ProjectStore) -> None:
    bad = _result().model_copy(update={"files": ["../escape"]})
    with pytest.raises(ProjectStoreError):
        store.save_task_result("p1", bad)
    bad = TaskResult.model_construct(**{**_result().model_dump(), "attempt": 0})
    with pytest.raises(ProjectStoreError):
        store.save_task_result("p1", bad)
    assert store.attempt_count("p1", "a") == 0


def test_no_temp_files_left(store: ProjectStore) -> None:
    store.save_task_result("p1", _result())
    with pytest.raises(ProjectConflict):
        store.save_task_result("p1", _result())
    names = sorted(p.name for p in store.task_dir("p1", "a").iterdir())
    assert names == ["result-1.json"]


def test_corrupt_or_mismatched_result_file(store: ProjectStore) -> None:
    store.save_task_result("p1", _result())
    path = store.task_dir("p1", "a") / "result-1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, "attempt": 7}), encoding="utf-8")
    with pytest.raises(ProjectStoreError, match="do not match"):
        store.load_task_result("p1", "a", 1)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProjectStoreError, match="unreadable"):
        store.load_task_result("p1", "a", 1)
    path.write_text(json.dumps({**data, "status": "great"}), encoding="utf-8")
    with pytest.raises(ProjectStoreError, match="invalid"):
        store.load_task_result("p1", "a", 1)
    with pytest.raises(ProjectStoreError, match="missing"):
        store.load_task_result("p1", "a", 2)


def test_symlinked_result_file_is_rejected(store: ProjectStore, tmp_path: Path) -> None:
    store.save_task_result("p1", _result())
    path = store.task_dir("p1", "a") / "result-1.json"
    outside = tmp_path / "outside.json"
    outside.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ProjectStoreError, match="symlink"):
        store.load_task_result("p1", "a", 1)
    # a symlink still counts as an occupied slot: it can't be bypassed by re-saving
    assert store.attempt_count("p1", "a") == 1
    with pytest.raises(ProjectConflict):
        store.save_task_result("p1", _result())


# --- task directory safety ------------------------------------------------------


@pytest.mark.parametrize("tid", ["../a", "A", "", "a/b", ".", "a\x00", "-a"])
def test_invalid_task_ids(store: ProjectStore, tid: str) -> None:
    with pytest.raises(ProjectStoreError, match="invalid task id"):
        store.task_dir("p1", tid)


def test_symlinked_task_dir_is_rejected(store: ProjectStore, tmp_path: Path) -> None:
    outside = tmp_path / "evil"
    outside.mkdir()
    store.tasks_dir("p1").mkdir(exist_ok=True)
    (store.tasks_dir("p1") / "a").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProjectStoreError, match="escapes"):
        store.save_task_result("p1", _result())
    assert list(outside.iterdir()) == []


def test_symlinked_tasks_dir_is_rejected(store: ProjectStore, tmp_path: Path) -> None:
    outside = tmp_path / "evil"
    outside.mkdir()
    tasks = store.tasks_dir("p1")
    os.rmdir(tasks)
    tasks.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProjectStoreError, match="symlink"):
        store.save_task_result("p1", _result())
    assert list(outside.iterdir()) == []


def test_task_path_that_is_a_file_is_rejected(store: ProjectStore) -> None:
    (store.tasks_dir("p1") / "a").write_text("x", encoding="utf-8")
    with pytest.raises(ProjectStoreError):
        store.save_task_result("p1", _result())


# --- verdicts -------------------------------------------------------------------


def test_verdict_roundtrip_and_event(store: ProjectStore) -> None:
    store.save_task_result("p1", _result())
    assert store.load_verdict("p1", "a", 1) is None
    v = store.save_verdict("p1", _verdict(passed=False))
    assert store.load_verdict("p1", "a", 1) == v
    history = store.task_history("p1", "a")
    assert len(history) == 1 and history[0][1] == v
    ev = _events(store, "task.verdict_saved")
    assert ev == [{"task_id": "a", "attempt": 1, "outcome": "fail"}]
    assert "not good" not in json.dumps(ev)


def test_verdict_needs_stored_result(store: ProjectStore) -> None:
    with pytest.raises(ProjectConflict, match="no stored result"):
        store.save_verdict("p1", _verdict())
    store.save_task_result("p1", _result())
    with pytest.raises(ProjectConflict):
        store.save_verdict("p1", _verdict(attempt=2))


def test_verdict_is_write_once(store: ProjectStore) -> None:
    store.save_task_result("p1", _result())
    store.save_verdict("p1", _verdict(passed=False))
    with pytest.raises(ProjectConflict):
        store.save_verdict("p1", _verdict(passed=True))  # can't flip a fail into a pass
    assert not store.load_verdict("p1", "a", 1).passed  # type: ignore[union-attr]


def test_verdict_must_match_task(store: ProjectStore) -> None:
    store.save_task_result("p1", _result())
    # task "a" has one criterion; criterion 3 does not exist
    with pytest.raises(ProjectStoreError, match="does not match"):
        store.save_verdict("p1", _verdict(passed=False, criterion=3))
    bad = Verdict(
        task_id="a",
        attempt=1,
        outcome=VerdictOutcome.FAIL,
        checks=[CheckResult(criterion=5, passed=False)],
        created_at=T0,
    )
    with pytest.raises(ProjectStoreError, match="does not match"):
        store.save_verdict("p1", bad)
    assert store.load_verdict("p1", "a", 1) is None


def test_verdict_is_revalidated(store: ProjectStore) -> None:
    store.save_task_result("p1", _result())
    # a "pass" with an issue is inconsistent; model_construct skips validation
    bad = Verdict.model_construct(
        task_id="a",
        attempt=1,
        outcome=VerdictOutcome.PASS,
        checks=[],
        issues=[Issue(description="x")],
        model_evaluated=False,
        created_at=T0,
    )
    with pytest.raises(ProjectStoreError):
        store.save_verdict("p1", bad)
    assert store.load_verdict("p1", "a", 1) is None


def test_symlinked_verdict_file_is_rejected(store: ProjectStore, tmp_path: Path) -> None:
    store.save_task_result("p1", _result())
    outside = tmp_path / "v.json"
    outside.write_text(_verdict().model_dump_json(), encoding="utf-8")
    (store.task_dir("p1", "a") / "verdict-1.json").symlink_to(outside)
    with pytest.raises(ProjectStoreError, match="symlink"):
        store.load_verdict("p1", "a", 1)
    with pytest.raises(ProjectConflict):
        store.save_verdict("p1", _verdict())


# --- task state -------------------------------------------------------------------


def test_state_absent_then_saved(store: ProjectStore) -> None:
    assert store.load_task_state("p1", "a") is None
    stored = store.save_task_state("p1", _state(status=TaskStatus.RUNNING))
    assert stored.revision == 1
    assert store.load_task_state("p1", "a") == stored
    ev = _events(store, "task.status")
    assert ev == [{"task_id": "a", "from": None, "to": "running"}]


def test_state_optimistic_concurrency(store: ProjectStore) -> None:
    s1 = store.save_task_state("p1", _state(status=TaskStatus.RUNNING))
    with pytest.raises(ProjectConflict):
        store.save_task_state("p1", _state(status=TaskStatus.RUNNING))  # revision 0 again
    s2 = store.save_task_state("p1", s1.model_copy(update={"tier": Tier.LARGE}))
    assert s2.revision == 2 and s2.tier is Tier.LARGE
    with pytest.raises(ProjectConflict):
        store.save_task_state("p1", s1)  # stale
    # no status change -> no extra task.status event
    assert len(_events(store, "task.status")) == 1


def test_state_attempts_must_match_history(store: ProjectStore) -> None:
    with pytest.raises(ProjectStoreError, match="attempt"):
        store.save_task_state("p1", _state(attempts=1))
    store.save_task_result("p1", _result())
    with pytest.raises(ProjectStoreError, match="attempt"):
        store.save_task_state("p1", _state(attempts=0))
    assert store.save_task_state("p1", _state(attempts=1)).attempts == 1


def test_passed_requires_stored_passing_verdict(store: ProjectStore) -> None:
    store.save_task_result("p1", _result())
    passed = _state(status=TaskStatus.PASSED, attempts=1)
    with pytest.raises(ProjectStoreError, match="passing verdict"):
        store.save_task_state("p1", passed)  # no verdict yet
    store.save_verdict("p1", _verdict(passed=False))
    with pytest.raises(ProjectStoreError, match="passing verdict"):
        store.save_task_state("p1", passed)  # verdict is a fail
    store.save_task_result("p1", _result(attempt=2))
    store.save_verdict("p1", _verdict(attempt=2, passed=True))
    with pytest.raises(ProjectStoreError):
        store.save_task_state("p1", passed)  # attempts=1 but 2 stored
    ok = store.save_task_state("p1", _state(status=TaskStatus.PASSED, attempts=2))
    assert ok.status is TaskStatus.PASSED
    assert store.passed_tasks("p1") == {"a"}


def test_passed_cannot_use_an_earlier_pass(store: ProjectStore) -> None:
    store.save_task_result("p1", _result())
    store.save_verdict("p1", _verdict(passed=True))
    store.save_task_result("p1", _result(attempt=2))  # newer attempt, not judged yet
    with pytest.raises(ProjectStoreError, match="passing verdict"):
        store.save_task_state("p1", _state(status=TaskStatus.PASSED, attempts=2))


def test_passed_construct_bypass_is_revalidated(store: ProjectStore) -> None:
    bad = TaskState.model_construct(
        schema_version=1,
        task_id="a",
        status=TaskStatus.PASSED,
        attempts=0,
        tier=None,
        plan_version=1,
        error=None,
        updated_at=T0,
        revision=0,
    )
    with pytest.raises(ProjectStoreError):
        store.save_task_state("p1", bad)
    assert store.load_task_state("p1", "a") is None


def test_state_for_unknown_task_or_old_plan(store: ProjectStore) -> None:
    with pytest.raises(ProjectStoreError, match="not in the plan"):
        store.save_task_state("p1", _state(tid="zzz"))
    with pytest.raises(ProjectStoreError, match="plan version"):
        store.save_task_state("p1", _state(plan_version=2))


def test_corrupt_state_file(store: ProjectStore) -> None:
    store.save_task_state("p1", _state())
    path = store.task_dir("p1", "a") / "state.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, "task_id": "b"}), encoding="utf-8")
    with pytest.raises(ProjectStoreError, match="does not match"):
        store.load_task_state("p1", "a")
    path.write_text(json.dumps({**data, "status": "weird"}), encoding="utf-8")
    with pytest.raises(ProjectStoreError, match="invalid"):
        store.load_task_state("p1", "a")
    path.write_text("[", encoding="utf-8")
    with pytest.raises(ProjectStoreError, match="unreadable"):
        store.load_task_state("p1", "a")


def test_symlinked_state_file_is_rejected(store: ProjectStore, tmp_path: Path) -> None:
    store.save_task_state("p1", _state())
    path = store.task_dir("p1", "a") / "state.json"
    outside = tmp_path / "s.json"
    outside.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ProjectStoreError, match="symlink"):
        store.load_task_state("p1", "a")


def test_state_updated_at_never_goes_back(tmp_path: Path) -> None:
    times = iter([T0, T0, T0 + timedelta(hours=1)] + [T0] * 50)
    s = ProjectStore.for_workspace(tmp_path / "ws", clock=lambda: next(times))
    s.create("goal", project_id="p1")
    s.save_plan("p1", _plan())
    first = s.save_task_state("p1", _state())
    second = s.save_task_state("p1", first)
    assert second.updated_at >= first.updated_at


# --- aggregate views ---------------------------------------------------------------


def test_task_states_defaults_and_passed(store: ProjectStore) -> None:
    states = store.task_states("p1")
    assert list(states) == ["a", "b"]
    assert all(s.status is TaskStatus.PENDING and s.revision == 0 for s in states.values())
    assert store.passed_tasks("p1") == set()
    store.save_task_result("p1", _result())
    store.save_verdict("p1", _verdict())
    store.save_task_state("p1", _state(status=TaskStatus.PASSED, attempts=1))
    assert store.passed_tasks("p1") == {"a"}
    assert [t.id for t in store.load_plan("p1").ready_tasks(store.passed_tasks("p1"))] == ["b"]
    # a default state for "b" can be saved as-is (revision 0, attempts 0)
    saved = store.save_task_state("p1", states["b"])
    assert saved.revision == 1


def test_task_states_after_replan_reset_status(store: ProjectStore) -> None:
    store.save_task_result("p1", _result())
    store.save_verdict("p1", _verdict())
    s = store.save_task_state("p1", _state(status=TaskStatus.PASSED, attempts=1))
    store.save_plan("p1", _plan(version=2))
    states = store.task_states("p1")
    # old-plan state is not trusted as passed, but history and revision carry over
    assert states["a"].status is TaskStatus.PENDING
    assert states["a"].plan_version == 2 and states["a"].attempts == 1
    assert states["a"].revision == s.revision
    assert store.passed_tasks("p1") == set()
    store.save_task_state("p1", states["a"])  # saveable without conflict
    with pytest.raises(ProjectStoreError, match="plan version"):
        store.save_task_state("p1", s)


def test_tasks_dir_recreated_if_missing(store: ProjectStore) -> None:
    os.rmdir(store.tasks_dir("p1"))
    store.save_task_result("p1", _result())
    assert store.attempt_count("p1", "a") == 1


def test_failed_result_and_blocked_state(store: ProjectStore) -> None:
    store.save_task_result("p1", _result(status=ResultStatus.FAILED))
    store.save_verdict("p1", _verdict(passed=False))
    st = store.save_task_state(
        "p1", _state(status=TaskStatus.FAILED, attempts=1, error="gave up after 1 attempt")
    )
    assert st.error == "gave up after 1 attempt"
    blocked = store.save_task_state(
        "p1", _state(tid="b", status=TaskStatus.BLOCKED, error="dependency a failed")
    )
    assert blocked.status is TaskStatus.BLOCKED
    assert store.passed_tasks("p1") == set()
