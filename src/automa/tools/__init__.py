"""Tool Layer and Permission Policy. See docs/ARCHITECTURE.md ("Tool Layer")."""

from collections.abc import Callable

from automa.tools.base import (
    MAX_TOOL_OUTPUT_CHARS,
    RiskLevel,
    Tool,
    ToolArgs,
    ToolError,
    ToolResult,
)
from automa.tools.command import RunCommand, RunCommandArgs, check_docker, clean_env
from automa.tools.executor import ApprovalRequired, ToolExecutor, ToolLayerError
from automa.tools.files import FILE_TOOLS, DeleteFile, ListDir, ReadFile, WriteFile
from automa.tools.policy import Decision, PermissionPolicy, PolicyDecision
from automa.tools.sandbox import DockerSandbox, SandboxError, SandboxMode
from automa.tools.workspace import DirEntry, Workspace, WorkspaceError


def default_tools(sandbox: DockerSandbox | None = None) -> list[Tool]:
    """Fresh instances of the built-in tools.

    With ``sandbox``, ``run_command`` runs every call in that Docker sandbox (D32, D33).
    """
    return [*(cls() for cls in FILE_TOOLS), RunCommand(sandbox=sandbox)]


def tools_factory(sandbox: DockerSandbox | None = None) -> Callable[[], list[Tool]]:
    """A factory of fresh built-in tools bound to ``sandbox`` (for the Orchestrator)."""

    def make() -> list[Tool]:
        return default_tools(sandbox)

    return make


__all__ = [
    "FILE_TOOLS",
    "MAX_TOOL_OUTPUT_CHARS",
    "ApprovalRequired",
    "Decision",
    "DeleteFile",
    "DirEntry",
    "DockerSandbox",
    "ListDir",
    "PermissionPolicy",
    "PolicyDecision",
    "ReadFile",
    "RunCommand",
    "RunCommandArgs",
    "RiskLevel",
    "SandboxError",
    "SandboxMode",
    "Tool",
    "ToolArgs",
    "ToolError",
    "ToolExecutor",
    "ToolLayerError",
    "ToolResult",
    "Workspace",
    "WorkspaceError",
    "WriteFile",
    "check_docker",
    "clean_env",
    "default_tools",
    "tools_factory",
]
