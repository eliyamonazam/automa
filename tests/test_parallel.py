"""Tests for parallel execution of independent tasks (Orchestrator ``max_parallel``, D35).

Local only: FakeProvider behind the real LLMGateway + BudgetGuard + CostLedger, real
agents, tools and ProjectStore (the ``Rig`` of ``test_orchestrator``). Responses are
routed per task, so the tests do not depend on thread scheduling order. No network,
no real model calls.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from automa.events import Event, JsonlEventLog
from automa.llm import ChatRequest, ChatResponse, FakeProvider, ProviderError
from automa.orchestrator import MAX_PARALLEL, OrchestratorSettings, StopReason
from automa.plan import TaskStatus
from automa.project import ProjectStatus
from test_orchestrator import PID, Rig, calls, make_plan, task_dict, write

WAIT = 5.0  # seconds; a broken implementation fails instead of hanging


def task_of(request: ChatRequest) -> str:
    text = request.messages[1].content or ""
    start = text.index("Task `") + len("Task `")
    return text[start : text.index("`", start)]


class Router:
    """Answers each request from the script of the task it belongs to."""

    def __init__(self, scripts: dict[str, list[Any]]):
        self.scripts = {tid: deque(items) for tid, items in scripts.items()}
        self.lock = threading.Lock()
        self.order: list[str] = []

    def __call__(self, request: ChatRequest) -> ChatResponse | str:
        tid = task_of(request)
        with self.lock:
            self.order.append(tid)
            item = self.scripts[tid].popleft()
        if isinstance(item, BaseException):
            raise item
        if callable(item) and not isinstance(item, ChatResponse):
            return item(request)
        return item

    def total(self) -> int:
        return sum(len(q) for q in self.scripts.values())


def routed_rig(tmp_path: Path, plan: Any, scripts: dict[str, list[Any]], **limits: float) -> Rig:
    router = Router(scripts)
    rig = Rig(tmp_path, [router] * router.total(), plan, **limits)
    rig.router = router  # type: ignore[attr-defined]
    return rig


def statuses(rig: Rig) -> dict[str, TaskStatus]:
    return {tid: s.status for tid, s in rig.store.task_states(PID).items()}


# --- settings -------------------------------------------------------------------


def test_max_parallel_default_and_bounds() -> None:
    assert OrchestratorSettings().max_parallel == 1
    assert OrchestratorSettings(max_parallel=MAX_PARALLEL).max_parallel == MAX_PARALLEL
    for bad in (0, -1, MAX_PARALLEL + 1):
        with pytest.raises(ValueError):
            OrchestratorSettings(max_parallel=bad)


def test_sequential_mode_unchanged(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [*write("a"), *write("b")], make_plan(task_dict("a"), task_dict("b")))
    report = rig.run()
    assert report.status is ProjectStatus.DONE
    types = rig.event_types()
    assert "orchestrator.scheduled" not in types
    started = [e for e in rig.store.events(PID).read() if e.type == "orchestrator.started"]
    assert "max_parallel" not in started[0].data


# --- concurrency ------------------------------------------------------------------


def test_independent_tasks_run_at_the_same_time(tmp_path: Path) -> None:
    barrier = threading.Barrier(2, timeout=WAIT)

    def first_call(tid: str) -> Any:
        def respond(_: ChatRequest) -> ChatResponse:
            barrier.wait()  # only passes if both tasks are in a model call at once
            return calls(("write_file", {"path": f"{tid}.txt", "content": "ok"}))

        return respond

    plan = make_plan(task_dict("a"), task_dict("b"))
    rig = routed_rig(
        tmp_path, plan, {"a": [first_call("a"), "done"], "b": [first_call("b"), "done"]}
    )
    report = rig.run(max_parallel=2)
    assert report.status is ProjectStatus.DONE and report.reason is StopReason.COMPLETED
    assert report.passed == ["a", "b"] and report.attempts_run == 2
    assert rig.project_status() is ProjectStatus.DONE
    events = rig.store.events(PID).read()
    sched = [e.data["task"] for e in events if e.type == "orchestrator.scheduled"]
    assert sorted(sched) == ["a", "b"]
    started = [e for e in events if e.type == "orchestrator.started"]
    assert started[0].data["max_parallel"] == 2


def test_dependencies_are_respected(tmp_path: Path) -> None:
    seen: dict[str, dict[str, TaskStatus]] = {}
    holder: dict[str, Rig] = {}

    def check_deps(_: ChatRequest) -> ChatResponse:
        seen["c"] = statuses(holder["rig"])
        return calls(("write_file", {"path": "c.txt", "content": "ok"}))

    plan = make_plan(task_dict("c", deps=["a", "b"]), task_dict("a"), task_dict("b"))
    rig = routed_rig(tmp_path, plan, {"a": write("a"), "b": write("b"), "c": [check_deps, "done"]})
    holder["rig"] = rig
    report = rig.run(max_parallel=3)
    assert report.status is ProjectStatus.DONE
    assert seen["c"]["a"] is TaskStatus.PASSED and seen["c"]["b"] is TaskStatus.PASSED
    order = rig.router.order  # type: ignore[attr-defined]
    assert order.index("c") > max(i for i, t in enumerate(order) if t in "ab")


def test_concurrency_never_exceeds_max_parallel(tmp_path: Path) -> None:
    lock = threading.Lock()
    active = {"now": 0, "max": 0}

    def enter(tid: str) -> Any:
        def respond(_: ChatRequest) -> ChatResponse:
            with lock:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            threading.Event().wait(0.02)  # give the other threads a chance to overlap
            return calls(("write_file", {"path": f"{tid}.txt", "content": "ok"}))

        return respond

    def leave(_: ChatRequest) -> str:
        with lock:
            active["now"] -= 1
        return "done"

    ids = ["a", "b", "c", "d", "e"]
    plan = make_plan(*(task_dict(t) for t in ids))
    rig = routed_rig(tmp_path, plan, {t: [enter(t), leave] for t in ids})
    report = rig.run(max_parallel=2)
    assert report.status is ProjectStatus.DONE and report.passed == ids
    assert 1 <= active["max"] <= 2


def test_retries_and_escalation_work_in_parallel(tmp_path: Path) -> None:
    plan = make_plan(task_dict("a"), task_dict("b"))
    scripts = {"a": [*write("a", "bad"), *write("a", "bad"), *write("a")], "b": write("b")}
    rig = routed_rig(tmp_path, plan, scripts)
    report = rig.run(max_parallel=2)
    assert report.status is ProjectStatus.DONE and report.attempts_run == 4
    assert rig.state("a").attempts == 3 and rig.state("b").attempts == 1
    assert rig.state("a").tier is not None and rig.state("a").tier.value == "large"


def test_failed_task_blocks_dependents_but_not_independent_tasks(tmp_path: Path) -> None:
    plan = make_plan(task_dict("a"), task_dict("b", deps=["a"]), task_dict("c"))
    rig = routed_rig(tmp_path, plan, {"a": write("a", "bad"), "b": [], "c": write("c")})
    report = rig.run(max_parallel=3, max_attempts=1)
    assert report.status is ProjectStatus.FAILED and report.reason is StopReason.TASKS_FAILED
    assert report.failed == ["a"] and report.blocked == ["b"] and report.passed == ["c"]
    assert "b" not in rig.router.order  # type: ignore[attr-defined]


# --- stops ------------------------------------------------------------------------


def test_stop_lets_running_task_finish_its_step_and_schedules_nothing_new(
    tmp_path: Path,
) -> None:
    a_failed = threading.Event()

    def a_first(_: ChatRequest) -> ChatResponse:
        a_failed.set()
        raise ProviderError("503 from provider")

    def b_first(_: ChatRequest) -> ChatResponse:
        assert a_failed.wait(WAIT)  # b is mid-step when the stop happens
        return calls(("write_file", {"path": "b.txt", "content": "ok"}))

    plan = make_plan(task_dict("a"), task_dict("b"), task_dict("c"))
    rig = routed_rig(tmp_path, plan, {"a": [a_first], "b": [b_first, "done"], "c": write("c")})
    report = rig.run(max_parallel=2)
    assert report.status is ProjectStatus.PAUSED and report.reason is StopReason.PROVIDER
    assert report.task_id == "a" and "503" in report.detail
    assert rig.project_status() is ProjectStatus.PAUSED
    st = statuses(rig)
    assert st["b"] is TaskStatus.PASSED  # finished its step: result stored and verified
    assert st["a"] is TaskStatus.PENDING and rig.store.attempt_count(PID, "a") == 0
    assert st["c"] is TaskStatus.PENDING  # never scheduled after the stop
    assert "c" not in rig.router.order  # type: ignore[attr-defined]
    paused = [e for e in rig.store.events(PID).read() if e.type == "orchestrator.paused"]
    assert len(paused) == 1 and paused[0].data["task"] == "a"

    # resume: same command again, now sequentially or in parallel
    rig.router.scripts["a"].extend(write("a"))  # type: ignore[attr-defined]
    rig.provider.push(*[rig.router] * 4)  # type: ignore[attr-defined]
    again = rig.run(max_parallel=2)
    assert again.status is ProjectStatus.DONE and again.passed == ["a", "b", "c"]
    assert rig.store.attempt_count(PID, "b") == 1  # b was not redone


def test_budget_stop_in_parallel_spends_nothing(tmp_path: Path) -> None:
    plan = make_plan(task_dict("a"), task_dict("b"))
    rig = routed_rig(
        tmp_path,
        plan,
        {"a": write("a"), "b": write("b")},
        project_usd=0.004,
        per_call_usd=0.004,
    )
    report = rig.run(max_parallel=2)
    assert report.status is ProjectStatus.PAUSED and report.reason is StopReason.BUDGET
    assert report.task_id in ("a", "b")
    assert rig.provider.requests == []
    assert rig.ledger.total_cost() == 0
    for tid in ("a", "b"):
        assert rig.store.attempt_count(PID, tid) == 0
        assert rig.state(tid).status is TaskStatus.PENDING
    events = rig.store.events(PID).read()
    assert len([e for e in events if e.type == "orchestrator.paused"]) == 1
    other = [e for e in events if e.type == "orchestrator.task_stopped"]
    assert len(other) == 1 and other[0].data["reason"] == "budget"
    assert other[0].data["task"] != report.task_id


def test_concurrent_calls_cannot_overspend_project_cap(tmp_path: Path) -> None:
    """Each in-flight call holds its worst-case reservation, so parallel tasks can't
    jointly overrun the project cap: concurrent calls beyond it are refused."""
    ids = ["a", "b", "c", "d"]
    plan = make_plan(*(task_dict(t) for t in ids))
    cap = 0.01  # about one worst-case reservation
    rig = routed_rig(
        tmp_path, plan, {t: write(t) * 3 for t in ids}, project_usd=cap, per_call_usd=cap
    )
    report = rig.run(max_parallel=4)
    assert report.status is ProjectStatus.PAUSED and report.reason is StopReason.BUDGET
    assert 0 < rig.ledger.total_cost() <= cap
    assert len(rig.provider.requests) < 8  # not every task got to run
    stopped = [e for e in rig.store.events(PID).read() if e.type == "orchestrator.task_stopped"]
    assert all(e.data["reason"] == "budget" for e in stopped)
    for _ in range(2):  # later runs stay within the cap too
        rig.run(max_parallel=4)
    assert rig.ledger.total_cost() <= cap


def test_unexpected_error_pauses_and_propagates_after_draining(tmp_path: Path) -> None:
    a_failed = threading.Event()

    def a_first(_: ChatRequest) -> ChatResponse:
        a_failed.set()
        raise RuntimeError("bug in a")

    def b_first(_: ChatRequest) -> ChatResponse:
        assert a_failed.wait(WAIT)
        return calls(("write_file", {"path": "b.txt", "content": "ok"}))

    plan = make_plan(task_dict("a"), task_dict("b"))
    rig = routed_rig(tmp_path, plan, {"a": [a_first], "b": [b_first, "done"]})
    with pytest.raises(RuntimeError, match="bug in a"):
        rig.run(max_parallel=2)
    assert rig.project_status() is ProjectStatus.PAUSED
    assert rig.state("b").status is TaskStatus.PASSED
    assert rig.state("a").status is TaskStatus.PENDING


# --- shared layers ----------------------------------------------------------------


def test_event_log_stays_valid_with_many_parallel_tasks(tmp_path: Path) -> None:
    ids = [f"t{i}" for i in range(8)]
    plan = make_plan(*(task_dict(t) for t in ids))
    rig = routed_rig(tmp_path, plan, {t: write(t, "ok" + "x" * 20_000) for t in ids})
    report = rig.run(max_parallel=MAX_PARALLEL)
    assert report.status is ProjectStatus.DONE and report.passed == ids
    path = rig.store.project_dir(PID) / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line)["project_id"] == PID for line in lines)
    passed = [
        e
        for e in rig.store.events(PID).read()
        if e.type == "task.status" and e.data["to"] == "passed"
    ]
    assert sorted(e.data["task_id"] for e in passed) == ids


def test_jsonl_logs_on_one_path_share_a_lock(tmp_path: Path) -> None:
    one = JsonlEventLog(tmp_path / "e.jsonl")
    two = JsonlEventLog(tmp_path / "e.jsonl")
    other = JsonlEventLog(tmp_path / "f.jsonl")
    assert one._lock is two._lock  # noqa: SLF001
    assert one._lock is not other._lock  # noqa: SLF001


def test_jsonl_concurrent_writers_never_interleave(tmp_path: Path) -> None:
    path = tmp_path / "e.jsonl"
    from datetime import UTC, datetime

    def writer(n: int) -> None:
        log = JsonlEventLog(path)  # separate instance per thread, like store.events()
        for i in range(30):
            log.emit(
                Event(
                    ts=datetime(2026, 11, 11, tzinfo=UTC),
                    type="test.line",
                    data={"n": n, "i": i, "pad": "y" * 30_000},
                )
            )

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT * 4)
    events = JsonlEventLog(path).read()
    assert len(events) == 180
    assert {(e.data["n"], e.data["i"]) for e in events} == {
        (n, i) for n in range(6) for i in range(30)
    }


def test_fake_provider_hands_each_item_out_once_under_concurrency() -> None:
    from automa.llm import Message
    from automa.llm.types import ChatRequest as Req

    provider = FakeProvider([str(i) for i in range(400)])
    got: list[str] = []
    lock = threading.Lock()

    def caller() -> None:
        for _ in range(50):
            r = provider.complete(Req(model="m", messages=(Message.user("hi"),), max_tokens=10))
            with lock:
                got.append(r.message.content or "")

    threads = [threading.Thread(target=caller) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT * 4)
    assert sorted(got, key=int) == [str(i) for i in range(400)]
    assert len(provider.requests) == 400 and provider.remaining == 0
