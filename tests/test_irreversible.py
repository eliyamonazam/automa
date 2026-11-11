"""Irreversible tools: always gated by the Approval Gate, never more available."""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import pytest

from automa.events import MemoryEventSink
from automa.tools import (
    ApprovalRequired,
    Decision,
    DeleteFile,
    PermissionPolicy,
    RiskLevel,
    Tool,
    ToolArgs,
    ToolExecutor,
    Workspace,
    WorkspaceError,
    default_tools,
)
from automa.tools.approvals import ApprovalStatus, ApprovalStore

PID = "proj"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    files = tmp_path / "files"
    files.mkdir()
    return files


@pytest.fixture
def ws(root: Path) -> Workspace:
    return Workspace(root)


def make(
    tmp_path: Path,
    *,
    allowed: frozenset[str] = frozenset({"delete_file", "read_file"}),
    max_risk: RiskLevel = RiskLevel.WRITE,
    store: bool = True,
    task_id: str | None = "a",
) -> tuple[ToolExecutor, MemoryEventSink, ApprovalStore | None]:
    approvals = ApprovalStore(tmp_path / "approvals", PID) if store else None
    sink = MemoryEventSink()
    ex = ToolExecutor(
        default_tools(),
        PermissionPolicy(allowed_tools=allowed, max_risk=max_risk),
        Workspace(tmp_path / "files"),
        sink,
        PID,
        approvals=approvals,
        task_id=task_id,
    )
    return ex, sink, approvals


# -- Tool definition -------------------------------------------------------------


def test_flag_defaults_to_false() -> None:
    assert {t.name for t in default_tools() if t.irreversible} == {"delete_file"}
    assert DeleteFile.irreversible is True and DeleteFile.risk is RiskLevel.WRITE


@pytest.mark.parametrize("bad", [1, "yes", None])
def test_flag_must_be_a_bool(bad: object) -> None:
    with pytest.raises(TypeError, match="irreversible"):

        class Bad(Tool):
            name: ClassVar[str] = "bad_tool"
            description: ClassVar[str] = "Bad."
            risk: ClassVar[RiskLevel] = RiskLevel.READ
            irreversible = bad  # type: ignore[assignment]
            args_model: ClassVar[type[ToolArgs]] = ToolArgs

            def run(self, args: ToolArgs, workspace: Workspace) -> str:
                return ""


# -- Permission Policy -----------------------------------------------------------


@pytest.mark.parametrize("max_risk", [RiskLevel.WRITE, RiskLevel.EXECUTE])
def test_policy_gates_irreversible_at_any_level(max_risk: RiskLevel) -> None:
    p = PermissionPolicy(allowed_tools={"rm"}, max_risk=max_risk)
    d = p.decide("rm", RiskLevel.WRITE, irreversible=True)
    assert d.decision is Decision.NEEDS_APPROVAL and "irreversible" in d.reason
    assert p.decide("rm", RiskLevel.WRITE).decision is Decision.ALLOW


def test_flag_never_widens_access() -> None:
    p = PermissionPolicy(allowed_tools={"rm"}, max_risk=RiskLevel.READ)
    # Above the role's risk limit: denied, not offered for approval.
    assert p.decide("rm", RiskLevel.WRITE, irreversible=True).decision is Decision.DENY
    # Not in the role's tools: denied.
    assert p.decide("other", RiskLevel.READ, irreversible=True).decision is Decision.DENY


@pytest.mark.parametrize("bad", [1, "true", None])
def test_policy_rejects_non_bool_flag(bad: object) -> None:
    p = PermissionPolicy(allowed_tools={"rm"}, max_risk=RiskLevel.WRITE)
    assert p.decide("rm", RiskLevel.READ, irreversible=bad).decision is Decision.DENY  # type: ignore[arg-type]


def test_external_still_needs_approval_regardless_of_flag() -> None:
    p = PermissionPolicy(allowed_tools={"pub"}, max_risk=RiskLevel.WRITE)
    for flag in (False, True):
        d = p.decide("pub", RiskLevel.EXTERNAL, irreversible=flag)
        assert d.decision is Decision.NEEDS_APPROVAL


# -- Executor --------------------------------------------------------------------


def test_delete_pauses_and_records_request(tmp_path: Path, root: Path) -> None:
    (root / "old.txt").write_text("x")
    ex, sink, approvals = make(tmp_path)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("delete_file", {"path": "old.txt"})
    assert (root / "old.txt").exists()
    assert approvals is not None
    req = approvals.get(ei.value.request_id or "")
    assert (req.tool, req.arguments, req.status) == (
        "delete_file",
        {"path": "old.txt"},
        ApprovalStatus.PENDING,
    )
    ev = sink.of_type("tool.approval_required")[-1].data
    assert ev["irreversible"] is True and ev["request"] == req.id


def test_delete_without_store_never_runs(tmp_path: Path, root: Path) -> None:
    (root / "old.txt").write_text("x")
    ex, _, _ = make(tmp_path, store=False)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("delete_file", {"path": "old.txt"})
    assert ei.value.request_id is None and (root / "old.txt").exists()


def test_approved_delete_runs_exactly_once(tmp_path: Path, root: Path) -> None:
    (root / "old.txt").write_text("x")
    ex, sink, approvals = make(tmp_path)
    assert approvals is not None
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("delete_file", {"path": "old.txt"})
    approvals.decide(ei.value.request_id or "", approve=True)
    result = ex.execute("delete_file", {"path": "old.txt"})
    assert result.ok and not (root / "old.txt").exists()
    assert sink.of_type("tool.approved")[-1].data["irreversible"] is True
    (root / "old.txt").write_text("y")
    with pytest.raises(ApprovalRequired):  # the approval is spent
        ex.execute("delete_file", {"path": "old.txt"})
    assert (root / "old.txt").read_text() == "y"


def test_approval_covers_only_the_approved_path(tmp_path: Path, root: Path) -> None:
    (root / "a.txt").write_text("a")
    (root / "b.txt").write_text("b")
    ex, _, approvals = make(tmp_path)
    assert approvals is not None
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("delete_file", {"path": "a.txt"})
    approvals.decide(ei.value.request_id or "", approve=True)
    with pytest.raises(ApprovalRequired):
        ex.execute("delete_file", {"path": "b.txt"})
    assert (root / "a.txt").exists() and (root / "b.txt").exists()


def test_rejected_delete_is_reported(tmp_path: Path, root: Path) -> None:
    (root / "old.txt").write_text("x")
    ex, _, approvals = make(tmp_path)
    assert approvals is not None
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("delete_file", {"path": "old.txt"})
    approvals.decide(ei.value.request_id or "", approve=False, note="keep it")
    result = ex.execute("delete_file", {"path": "old.txt"})
    assert result.denied and "keep it" in (result.error or "")
    assert (root / "old.txt").exists()


def test_delete_above_risk_limit_is_denied_not_gated(tmp_path: Path, root: Path) -> None:
    (root / "old.txt").write_text("x")
    ex, sink, approvals = make(tmp_path, max_risk=RiskLevel.READ)
    result = ex.execute("delete_file", {"path": "old.txt"})
    assert result.denied and approvals is not None and approvals.list() == []
    assert sink.of_type("tool.denied") and (root / "old.txt").exists()


def test_unlisted_delete_is_denied(tmp_path: Path, root: Path) -> None:
    (root / "old.txt").write_text("x")
    ex, _, approvals = make(tmp_path, allowed=frozenset({"read_file"}))
    assert ex.execute("delete_file", {"path": "old.txt"}).denied
    assert approvals is not None and approvals.list() == []


def test_invalid_arguments_never_reach_the_gate(tmp_path: Path, root: Path) -> None:
    ex, _, approvals = make(tmp_path)
    result = ex.execute("delete_file", {"path": "../escape"})
    assert not result.ok
    assert approvals is not None and approvals.list() == []


def test_specs_offer_gated_tools_only_with_a_store(tmp_path: Path, root: Path) -> None:
    with_store, _, _ = make(tmp_path)
    assert {s["function"]["name"] for s in with_store.specs()} == {"delete_file", "read_file"}
    without, _, _ = make(tmp_path, store=False)
    assert {s["function"]["name"] for s in without.specs()} == {"read_file"}
    too_low, _, _ = make(tmp_path, max_risk=RiskLevel.READ)
    assert {s["function"]["name"] for s in too_low.specs()} == {"read_file"}


def test_builtin_roles_do_not_get_delete_file() -> None:
    from automa.roles import RoleRegistry

    for role in RoleRegistry.builtin():
        assert "delete_file" not in role.tools


# -- Workspace.delete_file -------------------------------------------------------


def test_workspace_delete_regular_file(ws: Workspace, root: Path) -> None:
    (root / "d").mkdir()
    (root / "d" / "f.txt").write_text("hello")
    assert ws.delete_file("d/f.txt") == 5
    assert not (root / "d" / "f.txt").exists() and (root / "d").is_dir()


def test_workspace_delete_missing_file(ws: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="not found"):
        ws.delete_file("nope.txt")


def test_workspace_refuses_directory(ws: Workspace, root: Path) -> None:
    (root / "d").mkdir()
    with pytest.raises(WorkspaceError, match="non-regular"):
        ws.delete_file("d")
    assert (root / "d").is_dir()


def test_workspace_refuses_symlink_and_keeps_target(
    ws: Workspace, root: Path, tmp_path: Path
) -> None:
    target = tmp_path / "outside.txt"
    target.write_text("secret")
    os.symlink(target, root / "link.txt")
    with pytest.raises(WorkspaceError, match="non-regular"):
        ws.delete_file("link.txt")
    assert target.read_text() == "secret" and (root / "link.txt").is_symlink()


def test_workspace_refuses_symlinked_parent(ws: Workspace, root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f.txt").write_text("secret")
    os.symlink(outside, root / "sub")
    with pytest.raises(WorkspaceError):
        ws.delete_file("sub/f.txt")
    assert (outside / "f.txt").exists()


@pytest.mark.parametrize("bad", ["../x", "/etc/passwd", "", "a/../../b"])
def test_workspace_rejects_unsafe_paths(ws: Workspace, bad: str) -> None:
    with pytest.raises(WorkspaceError):
        ws.delete_file(bad)


def test_equivalent_path_spellings_share_one_approval(tmp_path: Path, root: Path) -> None:
    (root / "d").mkdir()
    (root / "d" / "f.txt").write_text("x")
    ex, _, approvals = make(tmp_path)
    assert approvals is not None
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("delete_file", {"path": "./d//f.txt"})
    req = approvals.get(ei.value.request_id or "")
    assert req.arguments == {"path": "d/f.txt"}
    approvals.decide(req.id, approve=True)
    assert ex.execute("delete_file", {"path": "d/f.txt"}).ok
    assert not (root / "d" / "f.txt").exists()
