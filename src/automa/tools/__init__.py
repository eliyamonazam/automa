"""Tool Layer and Permission Policy. See docs/ARCHITECTURE.md ("Tool Layer")."""

from automa.tools.base import (
    MAX_TOOL_OUTPUT_CHARS,
    RiskLevel,
    Tool,
    ToolArgs,
    ToolError,
    ToolResult,
)
from automa.tools.executor import ApprovalRequired, ToolExecutor, ToolLayerError
from automa.tools.files import FILE_TOOLS, ListDir, ReadFile, WriteFile
from automa.tools.policy import Decision, PermissionPolicy, PolicyDecision
from automa.tools.workspace import DirEntry, Workspace, WorkspaceError


def default_tools() -> list[Tool]:
    """Fresh instances of the built-in tools."""
    return [cls() for cls in FILE_TOOLS]


__all__ = [
    "FILE_TOOLS",
    "MAX_TOOL_OUTPUT_CHARS",
    "ApprovalRequired",
    "Decision",
    "DirEntry",
    "ListDir",
    "PermissionPolicy",
    "PolicyDecision",
    "ReadFile",
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
    "default_tools",
]
