"""Tests for file conflicts of parallel tasks (``PathClaims``, declared paths; D37).

Local only: FakeProvider behind the real gateway/guard/ledger (the ``Rig`` of
``test_orchestrator``), real tools and workspace. No network, no real model calls.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from automa.events import MemoryEventSink
from automa.llm import ChatRequest, ChatResponse
from automa.orchestrator import declared_paths
from automa.plan import Task, TaskStatus
from automa.project import ProjectStatus
from automa.tools import (
    DeleteFile,
    ListDir,
    PermissionPolicy,
    ReadFile,
    RiskLevel,
    RunCommand,
    ToolExecutor,
    Workspace,
    WorkspaceError,
    WriteFile,
    default_tools,
)
from automa.tools.approvals import ApprovalStore
from automa.tools.claims import PathClaims, paths_overlap
from automa.tools.files import DeleteFileArgs, ReadFileArgs, WriteFileArgs
from test_orchestrator import PID, Rig, calls, make_plan, task_dict, write
from test_parallel import WAIT, routed_rig

# --- paths_overlap / PathClaims ----------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("a.txt", "a.txt", True),
        ("src", "src/x.py", True),
        ("src/x.py", "src", True),
        ("src/x.py", "src/y.py", False),
        ("src", "src2/x.py", False),
        ("a.txt", "a.txt.bak", False),
    ],
)
def test_paths_overlap(a: str, b: str, expected: bool) -> None:
    assert paths_overlap(a, b) is expected


def test_claim_is_per_owner_and_released() -> None:
    c = PathClaims()
    assert c.claim(["out.txt"], "a") is None
    assert c.claim(["./out.txt"], "a") is None  # same owner, normalized spelling
    assert c.claim(["out.txt"], "b") == ("out.txt", "a")
    assert c.release("b") == 0
    assert c.release("a") == 1
    assert c.claim(["out.txt"], "b") is None
    assert c.snapshot() == {"out.txt": "b"}


def test_claim_conflicts_with_parent_and_child_paths() -> None:
    c = PathClaims()
    assert c.claim(["src/x.py"], "a") is None
    assert c.claim(["src"], "b") == ("src", "a")
    assert c.claim(["src/x.py/inner"], "b") == ("src/x.py/inner", "a")
    assert c.claim(["src/y.py"], "b") is None
    assert c.claim(["src/z.py"], "a") is None  # own paths never conflict


def test_claim_is_all_or_nothing() -> None:
    c = PathClaims()
    assert c.claim(["held.txt"], "a") is None
    assert c.claim(["free.txt", "held.txt"], "b") == ("held.txt", "a")
    assert c.owned_by("b") == []
    assert c.claim(["free.txt"], "c") is None  # nothing was claimed by the failed call


def test_claim_rejects_invalid_paths_and_owner() -> None:
    c = PathClaims()
    for bad in ("../x", "/etc/passwd", ""):
        with pytest.raises(WorkspaceError):
            c.claim([bad], "a")
    with pytest.raises(ValueError):
        c.claim(["x"], "")
    assert c.snapshot() == {}


def test_claim_is_thread_safe() -> None:
    c = PathClaims()
    barrier = threading.Barrier(16, timeout=WAIT)
    winners: list[str] = []
    lock = threading.Lock()

    def contend(owner: str) -> None:
        barrier.wait()
        if c.claim(["same.txt"], owner) is None:
            with lock:
                winners.append(owner)

    threads = [threading.Thread(target=contend, args=(f"t{i}",)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT)
    assert len(winners) == 1
    assert c.snapshot() == {"same.txt": winners[0]}


# --- Tool.writes --------------------------------------------------------------------


def test_only_writing_tools_declare_paths() -> None:
    assert WriteFile().writes(WriteFileArgs(path="./a/b.txt", content="")) == ("a/b.txt",)
    assert DeleteFile().writes(DeleteFileArgs(path="a/b.txt")) == ("a/b.txt",)
    assert WriteFile().writes(WriteFileArgs(path="../x", content="")) == ()  # tool rejects it
    assert ReadFile().writes(ReadFileArgs(path="a.txt")) == ()
    tool = ListDir()
    assert tool.writes(tool.args_model.model_validate({})) == ()
    cmd = RunCommand()
    assert cmd.writes(cmd.args_model.model_validate({"argv": ["true"]})) == ()


# --- ToolExecutor with claims -----------------------------------------------------


def executor(
    tmp_path: Path, claims: PathClaims | None, task_id: str | None, *, store: bool = False
) -> tuple[ToolExecutor, MemoryEventSink, ApprovalStore | None]:
    files = tmp_path / "files"
    files.mkdir(exist_ok=True)
    sink = MemoryEventSink()
    approvals = ApprovalStore(tmp_path / "approvals", PID) if store else None
    ex = ToolExecutor(
        default_tools(),
        PermissionPolicy(
            allowed_tools=frozenset({"read_file", "write_file", "delete_file", "list_dir"}),
            max_risk=RiskLevel.WRITE,
        ),
        Workspace(files),
        sink,
        PID,
        approvals=approvals,
        task_id=task_id,
        claims=claims,
    )
    return ex, sink, approvals


def test_executor_claims_need_a_task_and_the_right_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="task_id"):
        executor(tmp_path, PathClaims(), None)
    with pytest.raises(TypeError):
        executor(tmp_path, object(), "a")  # type: ignore[arg-type]


def test_second_task_cannot_write_a_claimed_file(tmp_path: Path) -> None:
    claims = PathClaims()
    a, _, _ = executor(tmp_path, claims, "a")
    b, sink, _ = executor(tmp_path, claims, "b")
    assert a.execute("write_file", {"path": "shared.txt", "content": "from a"}).ok
    res = b.execute("write_file", {"path": "./shared.txt", "content": "from b"})
    assert not res.ok and not res.denied
    assert "parallel task 'a'" in (res.error or "") and "different path" in (res.error or "")
    assert (tmp_path / "files" / "shared.txt").read_text() == "from a"
    conflict = [e for e in sink.events if e.type == "tool.path_conflict"]
    assert conflict and conflict[0].data == {
        "tool": "write_file",
        "path": "shared.txt",
        "owner": "a",
    }
    call = [e for e in sink.events if e.type == "tool.call"][-1]
    assert call.data["ok"] is False and call.data["path"] == "./shared.txt"
    # Reading, and writing elsewhere, are still fine; a's own rewrites too.
    assert b.execute("read_file", {"path": "shared.txt"}).content == "from a"
    assert b.execute("write_file", {"path": "b.txt", "content": "x"}).ok
    assert a.execute("write_file", {"path": "shared.txt", "content": "again"}).ok


def test_writing_into_a_claimed_directory_conflicts(tmp_path: Path) -> None:
    claims = PathClaims()
    a, _, _ = executor(tmp_path, claims, "a")
    b, _, _ = executor(tmp_path, claims, "b")
    assert a.execute("write_file", {"path": "pkg/mod.py", "content": "x"}).ok
    assert not b.execute("write_file", {"path": "pkg/mod.py/x", "content": "y"}).ok
    assert b.execute("write_file", {"path": "pkg/other.py", "content": "y"}).ok


def test_released_claim_lets_another_task_write(tmp_path: Path) -> None:
    claims = PathClaims()
    a, _, _ = executor(tmp_path, claims, "a")
    b, _, _ = executor(tmp_path, claims, "b")
    assert a.execute("write_file", {"path": "x.txt", "content": "a"}).ok
    claims.release("a")
    assert b.execute("write_file", {"path": "x.txt", "content": "b"}).ok
    assert claims.snapshot() == {"x.txt": "b"}


def test_conflicting_delete_fails_before_the_approval_gate(tmp_path: Path) -> None:
    claims = PathClaims()
    a, _, _ = executor(tmp_path, claims, "a", store=True)
    b, sink, approvals = executor(tmp_path, claims, "b", store=True)
    assert a.execute("write_file", {"path": "keep.txt", "content": "a"}).ok
    res = b.execute("delete_file", {"path": "keep.txt"})
    assert not res.ok and "parallel task 'a'" in (res.error or "")
    assert approvals is not None and approvals.list() == []  # the user is never asked
    assert "tool.approval_required" not in [e.type for e in sink.events]
    assert (tmp_path / "files" / "keep.txt").exists()


def test_denied_tool_is_denied_before_any_claim(tmp_path: Path) -> None:
    claims = PathClaims()
    files = tmp_path / "files"
    files.mkdir()
    ex = ToolExecutor(
        default_tools(),
        PermissionPolicy(allowed_tools=frozenset({"read_file"}), max_risk=RiskLevel.READ),
        Workspace(files),
        MemoryEventSink(),
        PID,
        task_id="a",
        claims=claims,
    )
    res = ex.execute("write_file", {"path": "x.txt", "content": "x"})
    assert res.denied and claims.snapshot() == {}


def test_invalid_path_is_not_claimed(tmp_path: Path) -> None:
    claims = PathClaims()
    a, _, _ = executor(tmp_path, claims, "a")
    res = a.execute("write_file", {"path": "../escape.txt", "content": "x"})
    assert not res.ok and claims.snapshot() == {}


def test_without_claims_last_write_wins_as_before(tmp_path: Path) -> None:
    a, _, _ = executor(tmp_path, None, "a")
    b, _, _ = executor(tmp_path, None, "b")
    assert a.execute("write_file", {"path": "s.txt", "content": "a"}).ok
    assert b.execute("write_file", {"path": "s.txt", "content": "b"}).ok


# --- declared paths -----------------------------------------------------------------


def test_declared_paths_come_from_file_checks_only() -> None:
    task = Task.model_validate(
        task_dict(
            "a",
            acceptance_criteria=[
                {"description": "x", "check": {"type": "file_exists", "path": "./out/x.txt"}},
                {
                    "description": "y",
                    "check": {"type": "file_contains", "path": "y.txt", "text": "ok"},
                },
                {"description": "z", "check": {"type": "command", "command": "true"}},
                {"description": "free text only"},
            ],
        )
    )
    assert declared_paths(task) == frozenset({"out/x.txt", "y.txt"})


# --- orchestrator -------------------------------------------------------------------


def shared_task(tid: str, path: str) -> dict[str, Any]:
    return task_dict(
        tid,
        acceptance_criteria=[
            {
                "description": f"{path} exists",
                "check": {"type": "file_exists", "path": path},
            }
        ],
    )


def test_tasks_with_overlapping_declared_paths_never_run_together(tmp_path: Path) -> None:
    lock = threading.Lock()
    active = {"now": 0, "max": 0}

    def enter(tid: str) -> Any:
        def respond(_: ChatRequest) -> ChatResponse:
            with lock:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            threading.Event().wait(0.05)
            return calls(("write_file", {"path": "out/report.md", "content": tid}))

        return respond

    def leave(_: ChatRequest) -> str:
        with lock:
            active["now"] -= 1
        return "done"

    plan = make_plan(shared_task("a", "out/report.md"), shared_task("b", "out"))
    rig = routed_rig(tmp_path, plan, {"a": [enter("a"), leave], "b": [enter("b"), leave]})
    report = rig.run(max_parallel=2)
    assert report.status is ProjectStatus.DONE and report.passed == ["a", "b"]
    assert active["max"] == 1
    deferred = [e.data for e in rig.store.events(PID).read() if e.type == "orchestrator.deferred"]
    assert deferred == [{"task": "b", "waits_for": "a", "path": "out"}]
    # Run one after the other, the later write wins, exactly as in sequential mode.
    assert (rig.store.files_dir(PID) / "out" / "report.md").read_text() == "b"


def test_undeclared_shared_file_is_not_overwritten_by_a_parallel_task(tmp_path: Path) -> None:
    a_wrote = threading.Event()
    b_tried = threading.Event()
    seen: dict[str, str] = {}

    def a_first(_: ChatRequest) -> ChatResponse:
        return calls(
            ("write_file", {"path": "shared.txt", "content": "from a"}),
            ("write_file", {"path": "a.txt", "content": "ok"}),
        )

    def a_second(_: ChatRequest) -> str:
        a_wrote.set()
        assert b_tried.wait(WAIT)  # a is still running while b tries
        return "done"

    def b_first(_: ChatRequest) -> ChatResponse:
        assert a_wrote.wait(WAIT)
        return calls(
            ("write_file", {"path": "shared.txt", "content": "from b"}),
            ("write_file", {"path": "b.txt", "content": "ok"}),
        )

    def b_second(request: ChatRequest) -> str:
        seen["tool"] = "\n".join(m.content or "" for m in request.messages if m.content)
        b_tried.set()
        return "done"

    plan = make_plan(task_dict("a"), task_dict("b"))
    rig = routed_rig(tmp_path, plan, {"a": [a_first, a_second], "b": [b_first, b_second]})
    report = rig.run(max_parallel=2)
    assert report.status is ProjectStatus.DONE and report.passed == ["a", "b"]
    assert (rig.store.files_dir(PID) / "shared.txt").read_text() == "from a"
    assert "being written by the parallel task 'a'" in seen["tool"]
    conflicts = [e.data for e in rig.store.events(PID).read() if e.type == "tool.path_conflict"]
    assert conflicts == [{"tool": "write_file", "path": "shared.txt", "owner": "a"}]


def test_dependent_task_may_rewrite_a_file_of_a_finished_task(tmp_path: Path) -> None:
    plan = make_plan(task_dict("a"), task_dict("b", deps=["a"]))
    scripts = {
        "a": [calls(("write_file", {"path": "a.txt", "content": "ok"})), "done"],
        "b": [
            calls(
                ("write_file", {"path": "a.txt", "content": "ok, edited"}),
                ("write_file", {"path": "b.txt", "content": "ok"}),
            ),
            "done",
        ],
    }
    rig = routed_rig(tmp_path, plan, scripts)
    report = rig.run(max_parallel=2)
    assert report.status is ProjectStatus.DONE
    assert (rig.store.files_dir(PID) / "a.txt").read_text() == "ok, edited"
    assert "tool.path_conflict" not in rig.event_types()


def test_claims_are_released_when_a_task_stops_being_driven(tmp_path: Path) -> None:
    plan = make_plan(task_dict("a"))
    rig = Rig(tmp_path, write("a"), plan)
    assert rig.run().status is ProjectStatus.DONE
    orch = rig.orch(max_parallel=2)
    claims = PathClaims()
    assert claims.claim(["a.txt"], "a") is None
    orch._drive(PID, plan, plan.task("a"), threading.Event(), claims)  # finished: returns at once
    assert claims.snapshot() == {}


def test_claims_are_released_after_a_failed_task(tmp_path: Path) -> None:
    # a writes shared.txt and fails; independent c (scheduled once a slot frees up)
    # may then write shared.txt — nothing still running owns it.
    plan = make_plan(task_dict("a"), task_dict("c"))
    scripts = {
        "a": [
            calls(
                ("write_file", {"path": "shared.txt", "content": "a"}),
                ("write_file", {"path": "a.txt", "content": "bad"}),
            ),
            "done",
        ],
        "c": [
            calls(
                ("write_file", {"path": "shared.txt", "content": "c"}),
                ("write_file", {"path": "c.txt", "content": "ok"}),
            ),
            "done",
        ],
    }
    rig = routed_rig(tmp_path, plan, scripts)
    # Force a and c to run one after the other while still using the parallel path.
    orch = rig.orch(max_parallel=2, max_attempts=1)
    original = orch._ready_tasks

    def one_at_a_time(p: Any, states: Any, busy: set[str]) -> list[Task]:
        return [] if busy else original(p, states, busy)[:1]

    orch._ready_tasks = one_at_a_time  # type: ignore[method-assign]
    report = orch.run(PID)
    assert report.failed == ["a"] and report.passed == ["c"]
    assert (rig.store.files_dir(PID) / "shared.txt").read_text() == "c"
    assert "tool.path_conflict" not in rig.event_types()


def test_sequential_mode_has_no_claims(tmp_path: Path) -> None:
    plan = make_plan(task_dict("a"), task_dict("b"))
    scripts = [
        calls(
            ("write_file", {"path": "shared.txt", "content": "a"}),
            ("write_file", {"path": "a.txt", "content": "ok"}),
        ),
        "done",
        calls(
            ("write_file", {"path": "shared.txt", "content": "b"}),
            ("write_file", {"path": "b.txt", "content": "ok"}),
        ),
        "done",
    ]
    rig = Rig(tmp_path, scripts, plan)
    report = rig.run()
    assert report.status is ProjectStatus.DONE
    assert (rig.store.files_dir(PID) / "shared.txt").read_text() == "b"
    types = rig.event_types()
    assert "tool.path_conflict" not in types and "orchestrator.deferred" not in types
    assert rig.state("b").status is TaskStatus.PASSED


def test_budget_still_stops_parallel_tasks_with_shared_paths(tmp_path: Path) -> None:
    plan = make_plan(shared_task("a", "s.txt"), shared_task("b", "s.txt"))
    rig = routed_rig(tmp_path, plan, {"a": write("a"), "b": write("b")}, per_call_usd=1e-9)
    report = rig.run(max_parallel=2)
    assert report.status is ProjectStatus.PAUSED
    assert rig.provider.requests == []
