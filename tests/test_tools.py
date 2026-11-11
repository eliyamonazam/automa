"""Tool Layer and Permission Policy: workspace confinement, policy rules, executor.

No model is called here; everything runs on temporary directories.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from automa.events import Event, MemoryEventSink
from automa.project import ProjectStore
from automa.tools import (
    ApprovalRequired,
    Decision,
    PermissionPolicy,
    RiskLevel,
    Tool,
    ToolArgs,
    ToolError,
    ToolExecutor,
    ToolLayerError,
    Workspace,
    WorkspaceError,
    default_tools,
)
from automa.tools.base import MAX_TOOL_OUTPUT_CHARS
from automa.tools.workspace import MAX_WRITE_BYTES

SRC = Path(__file__).resolve().parents[1] / "src" / "automa"
ALL_FILE_TOOLS = {"read_file", "write_file", "list_dir"}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "files"
    r.mkdir()
    return r


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    o = tmp_path / "outside"
    o.mkdir()
    (o / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    return o


@pytest.fixture
def ws(root: Path) -> Workspace:
    return Workspace(root)


def make_executor(
    root: Path,
    *,
    allowed: set[str] | None = None,
    max_risk: RiskLevel = RiskLevel.WRITE,
    tools: list[Tool] | None = None,
    events: Any = None,
) -> tuple[ToolExecutor, MemoryEventSink]:
    sink = events if events is not None else MemoryEventSink()
    policy = PermissionPolicy(
        allowed_tools=ALL_FILE_TOOLS if allowed is None else allowed, max_risk=max_risk
    )
    ex = ToolExecutor(
        default_tools() if tools is None else tools, policy, Workspace(root), sink, "proj"
    )
    return ex, sink


# --------------------------------------------------------------------------- workspace


class TestWorkspacePaths:
    @pytest.mark.parametrize(
        "bad",
        [
            "../x",
            "a/../../x",
            "/etc/passwd",
            "C:/x",
            "a\\b",
            "a\x00b",
            "",
            ".",
            "./",
            "a/" * 40 + "x",
            "x" * 300,
        ],
    )
    def test_rejects_unsafe_paths(self, ws: Workspace, bad: str) -> None:
        with pytest.raises(WorkspaceError):
            ws.read_text(bad)
        with pytest.raises(WorkspaceError):
            ws.write_text(bad, "x")

    def test_root_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(WorkspaceError):
            Workspace(tmp_path / "nope")

    def test_root_must_not_be_symlink(self, tmp_path: Path, root: Path) -> None:
        link = tmp_path / "link"
        link.symlink_to(root)
        with pytest.raises(WorkspaceError):
            Workspace(link)

    def test_root_must_be_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "f"
        f.write_text("x")
        with pytest.raises(WorkspaceError):
            Workspace(f)

    def test_root_replaced_by_symlink_after_creation(
        self, tmp_path: Path, root: Path, outside: Path
    ) -> None:
        ws = Workspace(root)
        root.rmdir()
        root.symlink_to(outside)
        with pytest.raises(WorkspaceError):
            ws.read_text("secret.txt")
        with pytest.raises(WorkspaceError):
            ws.write_text("new.txt", "x")
        assert not (outside / "new.txt").exists()


class TestWorkspaceRead:
    def test_read_roundtrip(self, ws: Workspace, root: Path) -> None:
        (root / "a.txt").write_text("سلام", encoding="utf-8")
        assert ws.read_text("a.txt") == ("سلام", False)
        assert ws.read_text("./a.txt") == ("سلام", False)

    def test_missing_file(self, ws: Workspace) -> None:
        with pytest.raises(WorkspaceError, match="not found"):
            ws.read_text("missing.txt")

    def test_missing_dir(self, ws: Workspace) -> None:
        with pytest.raises(WorkspaceError, match="not found"):
            ws.read_text("nodir/a.txt")

    def test_read_directory_refused(self, ws: Workspace, root: Path) -> None:
        (root / "d").mkdir()
        with pytest.raises(WorkspaceError):
            ws.read_text("d")

    def test_symlink_file_refused(self, ws: Workspace, root: Path, outside: Path) -> None:
        (root / "s.txt").symlink_to(outside / "secret.txt")
        with pytest.raises(WorkspaceError):
            ws.read_text("s.txt")

    def test_symlink_inside_root_refused(self, ws: Workspace, root: Path) -> None:
        (root / "real.txt").write_text("x")
        (root / "alias.txt").symlink_to(root / "real.txt")
        with pytest.raises(WorkspaceError):
            ws.read_text("alias.txt")

    def test_symlinked_directory_refused(self, ws: Workspace, root: Path, outside: Path) -> None:
        (root / "d").symlink_to(outside)
        with pytest.raises(WorkspaceError):
            ws.read_text("d/secret.txt")

    def test_deep_symlinked_directory_refused(
        self, ws: Workspace, root: Path, outside: Path
    ) -> None:
        (root / "a" / "b").mkdir(parents=True)
        (root / "a" / "b" / "c").symlink_to(outside)
        with pytest.raises(WorkspaceError):
            ws.read_text("a/b/c/secret.txt")

    def test_binary_refused(self, ws: Workspace, root: Path) -> None:
        (root / "b.bin").write_bytes(b"\xff\xfe\x00\x01")
        with pytest.raises(WorkspaceError, match="UTF-8"):
            ws.read_text("b.bin")

    def test_nul_refused(self, ws: Workspace, root: Path) -> None:
        (root / "n.txt").write_bytes(b"ab\x00cd")
        with pytest.raises(WorkspaceError, match="binary"):
            ws.read_text("n.txt")

    def test_truncation(self, ws: Workspace, root: Path) -> None:
        (root / "big.txt").write_text("a" * 100)
        assert ws.read_text("big.txt", max_bytes=10) == ("a" * 10, True)

    def test_truncation_inside_multibyte_char(self, ws: Workspace, root: Path) -> None:
        (root / "fa.txt").write_text("سلام", encoding="utf-8")  # 8 bytes
        text, truncated = ws.read_text("fa.txt", max_bytes=3)
        assert truncated and text == "س"

    def test_truncated_but_invalid_early_is_refused(self, ws: Workspace, root: Path) -> None:
        (root / "x.txt").write_bytes(b"a\xffbcdefghij")
        with pytest.raises(WorkspaceError):
            ws.read_text("x.txt", max_bytes=5)

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
    def test_fifo_refused(self, ws: Workspace, root: Path) -> None:
        os.mkfifo(root / "fifo")
        with pytest.raises(WorkspaceError):
            ws.write_text("fifo", "x")


class TestWorkspaceWrite:
    def test_write_creates_parents(self, ws: Workspace, root: Path) -> None:
        assert ws.write_text("a/b/c.txt", "hello") == 5
        assert (root / "a" / "b" / "c.txt").read_text() == "hello"

    def test_overwrite(self, ws: Workspace, root: Path) -> None:
        ws.write_text("f.txt", "one")
        ws.write_text("f.txt", "two")
        assert (root / "f.txt").read_text() == "two"

    def test_no_temp_files_left(self, ws: Workspace, root: Path) -> None:
        ws.write_text("f.txt", "x")
        assert sorted(p.name for p in root.iterdir()) == ["f.txt"]

    def test_write_over_symlink_refused_and_target_untouched(
        self, ws: Workspace, root: Path, outside: Path
    ) -> None:
        (root / "s.txt").symlink_to(outside / "secret.txt")
        with pytest.raises(WorkspaceError):
            ws.write_text("s.txt", "pwned")
        assert (outside / "secret.txt").read_text() == "TOP SECRET"
        assert (root / "s.txt").is_symlink()

    def test_dangling_symlink_refused(self, ws: Workspace, root: Path, outside: Path) -> None:
        (root / "d.txt").symlink_to(outside / "new.txt")
        with pytest.raises(WorkspaceError):
            ws.write_text("d.txt", "x")
        assert not (outside / "new.txt").exists()

    def test_write_through_symlinked_dir_refused(
        self, ws: Workspace, root: Path, outside: Path
    ) -> None:
        (root / "d").symlink_to(outside)
        with pytest.raises(WorkspaceError):
            ws.write_text("d/new.txt", "x")
        assert not (outside / "new.txt").exists()

    def test_write_over_directory_refused(self, ws: Workspace, root: Path) -> None:
        (root / "d").mkdir()
        with pytest.raises(WorkspaceError):
            ws.write_text("d", "x")
        assert (root / "d").is_dir()

    def test_parent_is_file_refused(self, ws: Workspace, root: Path) -> None:
        (root / "f").write_text("x")
        with pytest.raises(WorkspaceError):
            ws.write_text("f/g.txt", "x")

    def test_size_limit(self, ws: Workspace, root: Path) -> None:
        with pytest.raises(WorkspaceError, match="larger"):
            ws.write_text("big.txt", "a" * (MAX_WRITE_BYTES + 1))
        assert not (root / "big.txt").exists()

    def test_size_limit_counts_bytes(self, ws: Workspace) -> None:
        with pytest.raises(WorkspaceError):
            ws.write_text("fa.txt", "س" * 6, max_bytes=10)

    def test_nul_content_refused(self, ws: Workspace) -> None:
        with pytest.raises(WorkspaceError):
            ws.write_text("n.txt", "a\x00b")

    def test_failed_write_keeps_old_content_and_no_temp(
        self, ws: Workspace, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws.write_text("f.txt", "old")

        def boom(*a: Any, **k: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(os, "fsync", boom)
        with pytest.raises(OSError):
            ws.write_text("f.txt", "new")
        assert (root / "f.txt").read_text() == "old"
        assert sorted(p.name for p in root.iterdir()) == ["f.txt"]


class TestWorkspaceList:
    def test_list_sorted_with_kinds(self, ws: Workspace, root: Path, outside: Path) -> None:
        (root / "b.txt").write_text("12345")
        (root / "a").mkdir()
        (root / "link").symlink_to(outside)
        entries, total = ws.list_dir()
        assert total == 3
        assert [(e.name, e.kind, e.size) for e in entries] == [
            ("a", "dir", 0),
            ("b.txt", "file", 5),
            ("link", "symlink", 0),
        ]

    def test_list_subdir(self, ws: Workspace, root: Path) -> None:
        ws.write_text("a/x.txt", "1")
        entries, _ = ws.list_dir("a")
        assert [e.name for e in entries] == ["x.txt"]

    def test_list_symlinked_dir_refused(self, ws: Workspace, root: Path, outside: Path) -> None:
        (root / "link").symlink_to(outside)
        with pytest.raises(WorkspaceError):
            ws.list_dir("link")

    def test_list_traversal_refused(self, ws: Workspace) -> None:
        with pytest.raises(WorkspaceError):
            ws.list_dir("..")

    def test_list_limit(self, ws: Workspace, root: Path) -> None:
        for i in range(12):
            (root / f"f{i:02}").write_text("")
        entries, total = ws.list_dir(max_entries=5)
        assert len(entries) == 5 and total == 12


# --------------------------------------------------------------------------- policy


class TestPolicy:
    def test_allow(self) -> None:
        p = PermissionPolicy(allowed_tools={"read_file"}, max_risk=RiskLevel.READ)
        assert p.decide("read_file", RiskLevel.READ).allowed

    def test_default_max_risk_is_read(self) -> None:
        p = PermissionPolicy(allowed_tools={"write_file"})
        assert p.max_risk is RiskLevel.READ
        assert p.decide("write_file", RiskLevel.WRITE).decision is Decision.DENY

    def test_not_listed_denied(self) -> None:
        p = PermissionPolicy(allowed_tools={"read_file"}, max_risk=RiskLevel.EXECUTE)
        d = p.decide("write_file", RiskLevel.WRITE)
        assert d.decision is Decision.DENY and "not allowed" in d.reason

    @pytest.mark.parametrize(
        ("max_risk", "risk", "ok"),
        [
            (RiskLevel.READ, RiskLevel.READ, True),
            (RiskLevel.READ, RiskLevel.WRITE, False),
            (RiskLevel.READ, RiskLevel.EXECUTE, False),
            (RiskLevel.WRITE, RiskLevel.WRITE, True),
            (RiskLevel.WRITE, RiskLevel.EXECUTE, False),
            (RiskLevel.EXECUTE, RiskLevel.EXECUTE, True),
            (RiskLevel.EXECUTE, RiskLevel.READ, True),
        ],
    )
    def test_risk_ceiling(self, max_risk: RiskLevel, risk: RiskLevel, ok: bool) -> None:
        p = PermissionPolicy(allowed_tools={"t"}, max_risk=max_risk)
        assert p.decide("t", risk).allowed is ok

    @pytest.mark.parametrize("max_risk", [RiskLevel.READ, RiskLevel.WRITE, RiskLevel.EXECUTE])
    def test_external_always_needs_approval(self, max_risk: RiskLevel) -> None:
        p = PermissionPolicy(allowed_tools={"publish"}, max_risk=max_risk)
        assert p.decide("publish", RiskLevel.EXTERNAL).decision is Decision.NEEDS_APPROVAL
        # Unlisted external tools are denied: an approval can never widen the tool list.
        q = PermissionPolicy(allowed_tools=set(), max_risk=max_risk)
        assert q.decide("publish", RiskLevel.EXTERNAL).decision is Decision.DENY

    def test_max_risk_external_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PermissionPolicy(allowed_tools={"x"}, max_risk=RiskLevel.EXTERNAL)

    def test_unknown_risk_denied(self) -> None:
        p = PermissionPolicy(allowed_tools={"x"}, max_risk=RiskLevel.EXECUTE)
        assert p.decide("x", "read").decision is Decision.DENY  # type: ignore[arg-type]

    def test_string_allowed_tools_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PermissionPolicy(allowed_tools="read_file")  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ["Read", "a b", "", "1x", "x" * 70])
    def test_invalid_tool_names_rejected(self, name: str) -> None:
        with pytest.raises(ValidationError):
            PermissionPolicy(allowed_tools={name})

    def test_immutable(self) -> None:
        p = PermissionPolicy(allowed_tools={"read_file"})
        with pytest.raises(ValidationError):
            p.max_risk = RiskLevel.EXECUTE  # type: ignore[misc]
        with pytest.raises(AttributeError):
            p.allowed_tools.add("write_file")  # type: ignore[attr-defined]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PermissionPolicy(allowed_tools=set(), approved=True)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- tools


class TestToolDefinition:
    def test_invalid_name(self) -> None:
        with pytest.raises(TypeError):

            class Bad(Tool):  # noqa: F841
                name: ClassVar[str] = "Bad Name"
                description: ClassVar[str] = "x"
                risk: ClassVar[RiskLevel] = RiskLevel.READ
                args_model: ClassVar[type[ToolArgs]] = ToolArgs

                def run(self, args: ToolArgs, workspace: Workspace) -> str:
                    return ""

    def test_risk_required(self) -> None:
        with pytest.raises(TypeError):

            class Bad(Tool):  # noqa: F841
                name: ClassVar[str] = "bad"
                description: ClassVar[str] = "x"
                risk = "read"  # type: ignore[assignment]
                args_model: ClassVar[type[ToolArgs]] = ToolArgs

                def run(self, args: ToolArgs, workspace: Workspace) -> str:
                    return ""

    def test_args_model_required(self) -> None:
        with pytest.raises(TypeError):

            class Bad(Tool):  # noqa: F841
                name: ClassVar[str] = "bad"
                description: ClassVar[str] = "x"
                risk: ClassVar[RiskLevel] = RiskLevel.READ
                args_model = dict  # type: ignore[assignment]

                def run(self, args: ToolArgs, workspace: Workspace) -> str:
                    return ""

    def test_builtin_risk_levels(self) -> None:
        risks = {t.name: t.risk for t in default_tools()}
        assert risks == {
            "read_file": RiskLevel.READ,
            "write_file": RiskLevel.WRITE,
            "list_dir": RiskLevel.READ,
            "run_command": RiskLevel.EXECUTE,
            "delete_file": RiskLevel.WRITE,
        }
        irreversible = {t.name for t in default_tools() if t.irreversible}
        assert irreversible == {"delete_file"}

    def test_specs_are_openai_functions(self) -> None:
        for t in default_tools():
            spec = t.spec()
            assert spec["type"] == "function"
            assert spec["function"]["name"] == t.name
            params = spec["function"]["parameters"]
            assert params["type"] == "object"
            assert params.get("additionalProperties") is False
            json.dumps(spec)

    def test_run_rejects_unvalidated_args(self, ws: Workspace) -> None:
        for t in default_tools():
            with pytest.raises(TypeError):
                t.run(ToolArgs(), ws)


class EchoArgs(ToolArgs):
    text: str = ""


class Publish(Tool):
    name: ClassVar[str] = "publish"
    description: ClassVar[str] = "Publish something to the internet."
    risk: ClassVar[RiskLevel] = RiskLevel.EXTERNAL
    args_model: ClassVar[type[ToolArgs]] = EchoArgs
    calls = 0

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        Publish.calls += 1
        return "published"


class Exec(Tool):
    name: ClassVar[str] = "exec_thing"
    description: ClassVar[str] = "Execute."
    risk: ClassVar[RiskLevel] = RiskLevel.EXECUTE
    args_model: ClassVar[type[ToolArgs]] = EchoArgs

    def __init__(self) -> None:
        self.calls = 0

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        self.calls += 1
        return "ran"


class Weird(Tool):
    name: ClassVar[str] = "weird"
    description: ClassVar[str] = "Misbehaving tool."
    risk: ClassVar[RiskLevel] = RiskLevel.READ
    args_model: ClassVar[type[ToolArgs]] = EchoArgs

    def __init__(self, behaviour: str) -> None:
        self.behaviour = behaviour

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        if self.behaviour == "oserror":
            raise PermissionError(13, "Permission denied")
        if self.behaviour == "toolerror":
            raise ToolError("nope")
        if self.behaviour == "big":
            return "x" * (MAX_TOOL_OUTPUT_CHARS * 2)
        if self.behaviour == "notstr":
            return 42  # type: ignore[return-value]
        raise RuntimeError("bug")


# --------------------------------------------------------------------------- executor


class TestExecutor:
    def test_write_read_list_flow(self, root: Path) -> None:
        ex, sink = make_executor(root)
        r = ex.execute("write_file", json.dumps({"path": "src/main.py", "content": "print(1)"}))
        assert r.ok and "8 bytes" in r.content
        r = ex.execute("read_file", {"path": "src/main.py"})
        assert r.ok and r.content == "print(1)"
        r = ex.execute("list_dir", None)
        assert r.ok and r.content == "src/"
        r = ex.execute("list_dir", '{"path": "src"}')
        assert r.content == "main.py\t8"
        calls = sink.of_type("tool.call")
        assert [e.data["tool"] for e in calls] == [
            "write_file",
            "read_file",
            "list_dir",
            "list_dir",
        ]
        assert all(e.project_id == "proj" and e.data["ok"] for e in calls)

    def test_file_content_not_logged(self, root: Path) -> None:
        ex, sink = make_executor(root)
        ex.execute("write_file", {"path": "k.txt", "content": "VERY-PRIVATE"})
        assert "VERY-PRIVATE" not in json.dumps([e.model_dump(mode="json") for e in sink.events])

    def test_empty_dir_listing(self, root: Path) -> None:
        ex, _ = make_executor(root)
        assert ex.execute("list_dir", "").content == "(empty)"

    def test_unknown_tool(self, root: Path) -> None:
        ex, sink = make_executor(root)
        r = ex.execute("rm_rf", {})
        assert not r.ok and r.denied and "unknown tool" in (r.error or "")
        assert sink.of_type("tool.denied")[0].data["reason"] == "unknown tool"

    def test_non_string_name(self, root: Path) -> None:
        ex, _ = make_executor(root)
        assert not ex.execute(None, {}).ok  # type: ignore[arg-type]

    def test_denied_not_in_role(self, root: Path) -> None:
        ex, sink = make_executor(root, allowed={"read_file"}, max_risk=RiskLevel.WRITE)
        r = ex.execute("write_file", {"path": "a.txt", "content": "x"})
        assert not r.ok and r.denied
        assert not (root / "a.txt").exists()
        assert sink.of_type("tool.denied") and not sink.of_type("tool.call")

    def test_denied_above_risk(self, root: Path) -> None:
        ex, _ = make_executor(root, allowed=ALL_FILE_TOOLS, max_risk=RiskLevel.READ)
        r = ex.execute("write_file", {"path": "a.txt", "content": "x"})
        assert r.denied and "risk" in (r.error or "")
        assert not (root / "a.txt").exists()

    def test_denied_before_argument_validation(self, root: Path) -> None:
        ex, _ = make_executor(root, allowed=set())
        r = ex.execute("write_file", "not json")
        assert r.denied

    def test_execute_risk_denied_for_write_role(self, root: Path) -> None:
        tool = Exec()
        ex, _ = make_executor(root, allowed={"exec_thing"}, max_risk=RiskLevel.WRITE, tools=[tool])
        assert ex.execute("exec_thing", {}).denied
        assert tool.calls == 0

    def test_execute_allowed_with_execute_ceiling(self, root: Path) -> None:
        tool = Exec()
        ex, _ = make_executor(
            root, allowed={"exec_thing"}, max_risk=RiskLevel.EXECUTE, tools=[tool]
        )
        assert ex.execute("exec_thing", {}).ok and tool.calls == 1

    def test_external_raises_approval_required(self, root: Path) -> None:
        Publish.calls = 0
        ex, sink = make_executor(
            root, allowed={"publish"}, max_risk=RiskLevel.EXECUTE, tools=[Publish()]
        )
        with pytest.raises(ApprovalRequired) as ei:
            ex.execute("publish", {"text": "hi"})
        assert ei.value.tool == "publish"
        assert Publish.calls == 0
        assert sink.of_type("tool.approval_required")

    def test_external_not_in_specs(self, root: Path) -> None:
        ex, _ = make_executor(
            root,
            allowed={"publish", "read_file"},
            max_risk=RiskLevel.EXECUTE,
            tools=[Publish(), *default_tools()],
        )
        names = [s["function"]["name"] for s in ex.specs()]
        assert names == ["read_file"]

    def test_specs_only_allowed(self, root: Path) -> None:
        ex, _ = make_executor(root, allowed=ALL_FILE_TOOLS, max_risk=RiskLevel.READ)
        assert sorted(s["function"]["name"] for s in ex.specs()) == ["list_dir", "read_file"]

    @pytest.mark.parametrize(
        "args",
        [
            "{not json",
            "[1, 2]",
            '"path"',
            "42",
            {"path": "a.txt"},
            {"path": "a.txt", "content": "x", "mode": "777"},
            {"path": 5, "content": "x"},
            {"path": "", "content": "x"},
        ],
    )
    def test_invalid_arguments(self, root: Path, args: Any) -> None:
        ex, sink = make_executor(root)
        r = ex.execute("write_file", args)
        assert not r.ok and not r.denied and r.error
        assert list(root.iterdir()) == []
        assert sink.of_type("tool.call")[-1].data["ok"] is False

    def test_oversized_arguments(self, root: Path) -> None:
        ex, _ = make_executor(root)
        r = ex.execute("write_file", " " * (2 * MAX_WRITE_BYTES + 1))
        assert not r.ok and "too large" in (r.error or "")

    def test_unsafe_path_is_error_result(self, root: Path, outside: Path) -> None:
        ex, sink = make_executor(root)
        r = ex.execute("read_file", {"path": "../outside/secret.txt"})
        assert not r.ok and "TOP SECRET" not in r.as_text()
        assert sink.of_type("tool.call")[-1].data["error"]

    def test_symlink_escape_is_error_result(self, root: Path, outside: Path) -> None:
        (root / "s").symlink_to(outside)
        ex, _ = make_executor(root)
        assert not ex.execute("read_file", {"path": "s/secret.txt"}).ok
        assert not ex.execute("write_file", {"path": "s/x.txt", "content": "x"}).ok
        assert not ex.execute("list_dir", {"path": "s"}).ok
        assert not (outside / "x.txt").exists()

    def test_read_file_truncates(self, root: Path) -> None:
        (root / "big.txt").write_text("a" * (MAX_TOOL_OUTPUT_CHARS + 50))
        ex, _ = make_executor(root)
        r = ex.execute("read_file", {"path": "big.txt"})
        assert r.ok and r.content.endswith("[... truncated]")
        assert len(r.content) <= MAX_TOOL_OUTPUT_CHARS + 20

    def test_tool_oserror_becomes_result(self, root: Path) -> None:
        ex, _ = make_executor(root, allowed={"weird"}, tools=[Weird("oserror")])
        r = ex.execute("weird", {})
        assert not r.ok and "Permission denied" in (r.error or "")

    def test_tool_error_becomes_result(self, root: Path) -> None:
        ex, _ = make_executor(root, allowed={"weird"}, tools=[Weird("toolerror")])
        assert ex.execute("weird", {}).error == "nope"

    def test_big_output_truncated(self, root: Path) -> None:
        ex, _ = make_executor(root, allowed={"weird"}, tools=[Weird("big")])
        r = ex.execute("weird", {})
        assert r.ok and r.content.endswith("[... truncated]")

    def test_non_string_output_is_layer_error(self, root: Path) -> None:
        ex, _ = make_executor(root, allowed={"weird"}, tools=[Weird("notstr")])
        with pytest.raises(ToolLayerError):
            ex.execute("weird", {})

    def test_bug_propagates(self, root: Path) -> None:
        ex, _ = make_executor(root, allowed={"weird"}, tools=[Weird("bug")])
        with pytest.raises(RuntimeError):
            ex.execute("weird", {})

    def test_result_as_text(self, root: Path) -> None:
        ex, _ = make_executor(root)
        assert ex.execute("read_file", {"path": "nope"}).as_text().startswith("ERROR: ")

    def test_long_error_shortened(self, root: Path) -> None:
        ex, _ = make_executor(root)
        r = ex.execute("nope" * 100, {})
        assert len(r.error or "") < 2000


class BrokenSink:
    def emit(self, event: Event) -> None:
        raise OSError("disk full")


class TestExecutorFailClosed:
    def test_event_failure_on_call(self, root: Path) -> None:
        ex, _ = make_executor(root, events=BrokenSink())
        with pytest.raises(ToolLayerError):
            ex.execute("list_dir", {})

    def test_event_failure_on_denied(self, root: Path) -> None:
        ex, _ = make_executor(root, allowed=set(), events=BrokenSink())
        with pytest.raises(ToolLayerError):
            ex.execute("write_file", {"path": "a", "content": "b"})

    def test_event_failure_on_approval(self, root: Path) -> None:
        ex, _ = make_executor(root, allowed={"publish"}, tools=[Publish()], events=BrokenSink())
        with pytest.raises(ToolLayerError):
            ex.execute("publish", {})


class TestExecutorConstruction:
    def test_duplicate_names(self, root: Path) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            make_executor(root, tools=[*default_tools(), *default_tools()])

    def test_non_tool(self, root: Path) -> None:
        with pytest.raises(TypeError):
            make_executor(root, tools=[object()])  # type: ignore[list-item]

    def test_policy_type(self, root: Path) -> None:
        with pytest.raises(TypeError):
            ToolExecutor([], {"allowed_tools": []}, Workspace(root), MemoryEventSink(), "p")  # type: ignore[arg-type]

    def test_workspace_type(self, root: Path) -> None:
        p = PermissionPolicy(allowed_tools=set())
        with pytest.raises(TypeError):
            ToolExecutor([], p, root, MemoryEventSink(), "p")  # type: ignore[arg-type]

    def test_events_required(self, root: Path) -> None:
        p = PermissionPolicy(allowed_tools=set())
        with pytest.raises(TypeError):
            ToolExecutor([], p, Workspace(root), None, "p")  # type: ignore[arg-type]

    @pytest.mark.parametrize("pid", ["", "  ", None])
    def test_project_id_required(self, root: Path, pid: Any) -> None:
        p = PermissionPolicy(allowed_tools=set())
        with pytest.raises(ValueError):
            ToolExecutor([], p, Workspace(root), MemoryEventSink(), pid)


# --------------------------------------------------------------------------- integration


def test_with_project_store(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "ws")
    project = store.create("build a site", project_id="site")
    policy = PermissionPolicy(allowed_tools=ALL_FILE_TOOLS, max_risk=RiskLevel.WRITE)
    ex = ToolExecutor(
        default_tools(),
        policy,
        Workspace(store.files_dir(project.id)),
        store.events(project.id),
        project.id,
    )
    assert ex.execute("write_file", {"path": "index.html", "content": "<h1>hi</h1>"}).ok
    assert (store.files_dir("site") / "index.html").read_text() == "<h1>hi</h1>"
    assert not ex.execute("read_file", {"path": "../project.json"}).ok
    types = [e.type for e in store.events("site").read()]
    assert "tool.call" in types


# --------------------------------------------------------------------------- structure


def _modules() -> list[tuple[Path, ast.Module]]:
    return [(p, ast.parse(p.read_text(encoding="utf-8"))) for p in sorted(SRC.rglob("*.py"))]


def test_only_executor_runs_tools() -> None:
    """``<tool>.run(args, workspace)`` is called only in ``ToolExecutor.execute``."""
    offenders = []
    for path, tree in _modules():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
                and len(node.args) == 2
            ):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert offenders == [f"tools/executor.py:{_executor_run_line()}"]


def _executor_run_line() -> int:
    tree = ast.parse((SRC / "tools" / "executor.py").read_text(encoding="utf-8"))
    lines = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "run"
    ]
    assert len(lines) == 1
    return lines[0]


def test_executor_decides_before_running() -> None:
    tree = ast.parse((SRC / "tools" / "executor.py").read_text(encoding="utf-8"))
    execute = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute"
    )
    decide = [
        n.lineno
        for n in ast.walk(execute)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "decide"
    ]
    run = [
        n.lineno
        for n in ast.walk(execute)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "run"
    ]
    assert decide and run and max(decide) < min(run)


def test_file_tools_never_touch_filesystem_directly() -> None:
    """File tools go through ``Workspace``; no ``open``/``os``/``Path`` I/O in tools/files.py."""
    tree = ast.parse((SRC / "tools" / "files.py").read_text(encoding="utf-8"))
    banned_names = {"open", "os", "Path", "shutil", "subprocess"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in banned_names, node.id
        if isinstance(node, ast.Import | ast.ImportFrom):
            mod = getattr(node, "module", None) or ""
            names = {a.name for a in node.names}
            assert not ({mod} | names) & {"os", "pathlib", "shutil", "subprocess"}
