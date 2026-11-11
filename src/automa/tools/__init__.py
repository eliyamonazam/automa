"""Tool Layer and Permission Policy. See docs/ARCHITECTURE.md ("Tool Layer")."""

from automa.tools.base import (
    MAX_TOOL_OUTPUT_CHARS,
    RiskLevel,
    Tool,
    ToolArgs,
    ToolError,
    ToolResult,
)
from automa.tools.command import RunCommand, RunCommandArgs, clean_env
from automa.tools.executor import ApprovalRequired, ToolExecutor, ToolLayerError
from automa.tools.files import FILE_TOOLS, DeleteFile, ListDir, ReadFile, WriteFile
from automa.tools.policy import Decision, PermissionPolicy, PolicyDecision
from automa.tools.workspace import DirEntry, Workspace, WorkspaceError


def default_tools() -> list[Tool]:
    """Fresh instances of the built-in tools."""
    return [*(cls() for cls in FILE_TOOLS), RunCommand()]


__all__ = [
    "FILE_TOOLS",
    "MAX_TOOL_OUTPUT_CHARS",
    "ApprovalRequired",
    "Decision",
    "DeleteFile",
    "DirEntry",
    "ListDir",
    "PermissionPolicy",
    "PolicyDecision",
    "ReadFile",
    "RunCommand",
    "RunCommandArgs",
    "RiskLevel",
    "Tool",
    "ToolArgs",
    "ToolError",
    "ToolExecutor",
    "ToolLayerError",
    "ToolResult",
    "Workspace",
    "WorkspaceError",
    "WriteFile",
    "clean_env",
    "default_tools",
]
