"""Tests for the Approval Gate store and its use in the ToolExecutor.

Local only: temporary directories, no model calls.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest

from automa.events import MemoryEventSink
from automa.project import ProjectStore
from automa.tools import (
    ApprovalRequired,
    PermissionPolicy,
    RiskLevel,
    Tool,
    ToolArgs,
    ToolExecutor,
    ToolLayerError,
    Workspace,
)
from automa.tools.approvals import (
    ApprovalError,
    ApprovalNotFound,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalStore,
    fingerprint,
)

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
PID = "proj"


def clock() -> datetime:
    return T0


@pytest.fixture
def store(tmp_path: Path) -> ApprovalStore:
    return ApprovalStore(tmp_path / "approvals", PID, clock=clock)


def req(store: ApprovalStore, text: str = "hi", task: str | None = "a") -> ApprovalRequest:
    return store.request(
        tool="publish",
        risk=RiskLevel.EXTERNAL,
        arguments={"text": text},
        reason="needs approval",
        task_id=task,
    )


# --- store ---------------------------------------------------------------------


def test_fingerprint_ignores_key_order() -> None:
    assert fingerprint("t", {"a": 1, "b": 2}) == fingerprint("t", {"b": 2, "a": 1})
    assert fingerprint("t", {"a": 1}) != fingerprint("u", {"a": 1})
    assert fingerprint("t", {"a": 1}) != fingerprint("t", {"a": 2})


def test_request_is_recorded_and_listed(store: ApprovalStore) -> None:
    assert store.list() == []
    r = req(store)
    assert r.status is ApprovalStatus.PENDING and r.decided_at is None
    assert store.get(r.id) == r
    assert store.list() == [r]
    assert store.list(ApprovalStatus.APPROVED) == []
    data = json.loads((store.directory / f"{r.id}.json").read_text())
    assert data["arguments"] == {"text": "hi"} and data["task_id"] == "a"


def test_identical_pending_request_is_reused(store: ApprovalStore) -> None:
    r = req(store)
    assert req(store).id == r.id
    assert req(store, text="other").id != r.id
    assert req(store, task="b").id != r.id
    assert len(store.list()) == 3


def test_decide_once(store: ApprovalStore) -> None:
    r = req(store)
    d = store.decide(r.id, approve=True, note="ok")
    assert d.status is ApprovalStatus.APPROVED and d.decided_at == T0 and d.note == "ok"
    assert store.get(r.id) == d
    with pytest.raises(ApprovalError, match="already approved"):
        store.decide(r.id, approve=False)
    r2 = req(store)  # the decided one is not pending any more: a new request
    assert r2.id != r.id
    assert store.decide(r2.id, approve=False).status is ApprovalStatus.REJECTED


def test_decide_rejects_long_note(store: ApprovalStore) -> None:
    r = req(store)
    with pytest.raises(ApprovalError, match="note"):
        store.decide(r.id, approve=True, note="x" * 5000)
    assert store.get(r.id).status is ApprovalStatus.PENDING


@pytest.mark.parametrize("bad", ["apr-000000000000", "../x", "apr-ZZ", "", "apr-0123456789ab.json"])
def test_unknown_or_invalid_id(store: ApprovalStore, bad: str) -> None:
    with pytest.raises(ApprovalNotFound):
        store.get(bad)
    with pytest.raises(ApprovalNotFound):
        store.decide(bad, approve=True)


def test_consume_at_most_once(store: ApprovalStore) -> None:
    r = req(store)
    with pytest.raises(ApprovalError, match="not decided"):
        store.consume(r.id)
    store.decide(r.id, approve=True)
    assert not store.is_used(r.id)
    assert store.consume(r.id) is True
    assert store.consume(r.id) is False
    assert store.is_used(r.id)


def test_find_decision_matches_exactly(store: ApprovalStore) -> None:
    r = req(store)

    def find(text: str = "hi", task: str | None = "a", tool: str = "publish") -> str | None:
        found = store.find_decision(tool=tool, arguments={"text": text}, task_id=task)
        return found.id if found else None

    assert find() is None  # pending is not a decision
    store.decide(r.id, approve=True)
    assert find() == r.id
    assert find(text="other") is None
    assert find(task="b") is None
    assert find(task=None) is None
    assert find(tool="publish2") is None
    store.consume(r.id)
    assert find() is None


def test_corrupt_file_fails_closed(store: ApprovalStore) -> None:
    r = req(store)
    (store.directory / f"{r.id}.json").write_text("{not json")
    with pytest.raises(ApprovalError):
        store.get(r.id)
    with pytest.raises(ApprovalError):
        store.list()


def test_tampered_arguments_fail_closed(store: ApprovalStore) -> None:
    r = req(store)
    path = store.directory / f"{r.id}.json"
    data = json.loads(path.read_text())
    data["arguments"] = {"text": "something else"}
    data["status"] = "approved"
    data["decided_at"] = T0.isoformat()
    path.write_text(json.dumps(data))
    with pytest.raises(ApprovalError):
        store.get(r.id)


def test_inconsistent_decision_fails_closed(store: ApprovalStore) -> None:
    r = req(store)
    path = store.directory / f"{r.id}.json"
    data = json.loads(path.read_text())
    data["status"] = "approved"  # decided_at missing
    path.write_text(json.dumps(data))
    with pytest.raises(ApprovalError):
        store.get(r.id)


def test_file_of_another_project_is_rejected(tmp_path: Path, store: ApprovalStore) -> None:
    r = req(store)
    other = ApprovalStore(store.directory, "other", clock=clock)
    with pytest.raises(ApprovalError, match="does not belong"):
        other.get(r.id)


def test_symlinked_file_is_rejected(tmp_path: Path, store: ApprovalStore) -> None:
    r = req(store)
    path = store.directory / f"{r.id}.json"
    real = tmp_path / "real.json"
    real.write_text(path.read_text())
    path.unlink()
    os.symlink(real, path)
    with pytest.raises(ApprovalError, match="symlink"):
        store.get(r.id)


def test_symlinked_directory_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    os.symlink(real, tmp_path / "approvals")
    s = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    with pytest.raises(ApprovalError, match="symlink"):
        s.list()
    with pytest.raises(ApprovalError, match="symlink"):
        req(s)


def test_naive_clock_is_rejected(tmp_path: Path) -> None:
    s = ApprovalStore(tmp_path / "a", PID, clock=lambda: datetime(2026, 1, 1))  # noqa: DTZ001
    with pytest.raises(ApprovalError, match="naive"):
        req(s)


def test_project_store_gives_per_project_store(tmp_path: Path) -> None:
    ps = ProjectStore.for_workspace(tmp_path / "ws", clock=clock)
    ps.create("goal", project_id=PID)
    a = ps.approvals(PID)
    assert a.project_id == PID
    assert a.directory == ps.project_dir(PID) / "approvals"
    assert a.list() == []


# --- executor gate -----------------------------------------------------------------


class PubArgs(ToolArgs):
    text: str


class Publish(Tool):
    name: ClassVar[str] = "publish"
    description: ClassVar[str] = "Publish."
    risk: ClassVar[RiskLevel] = RiskLevel.EXTERNAL
    args_model: ClassVar[type[ToolArgs]] = PubArgs
    runs: ClassVar[int] = 0

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        Publish.runs += 1
        return "published"


def executor(
    tmp_path: Path,
    approvals: ApprovalStore | None,
    *,
    task_id: str | None = "a",
    allowed: frozenset[str] = frozenset({"publish"}),
) -> tuple[ToolExecutor, MemoryEventSink]:
    Publish.runs = 0
    ws = tmp_path / "files"
    ws.mkdir(exist_ok=True)
    sink = MemoryEventSink()
    ex = ToolExecutor(
        [Publish()],
        PermissionPolicy(allowed_tools=allowed, max_risk=RiskLevel.WRITE),
        Workspace(ws),
        sink,
        PID,
        approvals=approvals,
        task_id=task_id,
    )
    return ex, sink


def test_gate_records_request_and_raises(tmp_path: Path, store: ApprovalStore) -> None:
    ex, sink = executor(tmp_path, store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("publish", '{"text": "hi"}')
    assert Publish.runs == 0
    r = store.get(ei.value.request_id or "")
    assert (r.tool, r.arguments, r.task_id) == ("publish", {"text": "hi"}, "a")
    assert sink.of_type("tool.approval_required")[0].data["request"] == r.id


def test_without_store_nothing_can_be_approved(tmp_path: Path) -> None:
    ex, _ = executor(tmp_path, None)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("publish", {"text": "hi"})
    assert ei.value.request_id is None and Publish.runs == 0


def test_invalid_arguments_never_reach_the_gate(tmp_path: Path, store: ApprovalStore) -> None:
    ex, _ = executor(tmp_path, store)
    result = ex.execute("publish", {"wrong": 1})
    assert not result.ok and "invalid arguments" in (result.error or "")
    assert store.list() == [] and Publish.runs == 0


def test_unlisted_external_tool_is_denied(tmp_path: Path, store: ApprovalStore) -> None:
    ex, sink = executor(tmp_path, store, allowed=frozenset())
    result = ex.execute("publish", {"text": "hi"})
    assert result.denied and store.list() == [] and Publish.runs == 0
    assert sink.of_type("tool.denied")


def test_approved_runs_once_then_needs_new_approval(tmp_path: Path, store: ApprovalStore) -> None:
    ex, sink = executor(tmp_path, store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("publish", {"text": "hi"})
    first = ei.value.request_id or ""
    store.decide(first, approve=True)
    # Same arguments, different JSON spelling: same fingerprint.
    assert ex.execute("publish", '{ "text" : "hi" }').ok
    assert Publish.runs == 1 and store.is_used(first)
    assert sink.of_type("tool.approved")[0].data["request"] == first
    with pytest.raises(ApprovalRequired) as ei2:
        ex.execute("publish", {"text": "hi"})
    assert ei2.value.request_id != first and Publish.runs == 1


def test_rejected_is_reported_to_model(tmp_path: Path, store: ApprovalStore) -> None:
    ex, sink = executor(tmp_path, store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("publish", {"text": "hi"})
    store.decide(ei.value.request_id or "", approve=False, note="no")
    result = ex.execute("publish", {"text": "hi"})
    assert not result.ok and result.denied
    assert "rejected" in (result.error or "") and "no" in (result.error or "")
    assert Publish.runs == 0 and sink.of_type("tool.rejected")
    assert sink.of_type("tool.call")[-1].data["ok"] is False


def test_decision_is_bound_to_the_task(tmp_path: Path, store: ApprovalStore) -> None:
    ex, _ = executor(tmp_path, store, task_id="a")
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("publish", {"text": "hi"})
    store.decide(ei.value.request_id or "", approve=True)
    other, _ = executor(tmp_path, store, task_id="b")
    with pytest.raises(ApprovalRequired):
        other.execute("publish", {"text": "hi"})
    assert Publish.runs == 0


def test_broken_store_fails_closed(tmp_path: Path, store: ApprovalStore) -> None:
    r = req(store, text="hi", task="a")
    store.decide(r.id, approve=True)
    (store.directory / f"{r.id}.json").write_text("garbage")
    ex, _ = executor(tmp_path, store)
    with pytest.raises(ToolLayerError, match="approval gate"):
        ex.execute("publish", {"text": "hi"})
    assert Publish.runs == 0


def test_store_of_another_project_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="another project"):
        executor(tmp_path, ApprovalStore(tmp_path / "x", "other"))
    with pytest.raises(TypeError):
        executor(tmp_path, object())  # type: ignore[arg-type]
