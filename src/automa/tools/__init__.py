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
from automa.tools.github import (
    GitHubRead,
    GitHubReadArgs,
    GitHubRequestView,
    describe_github_request,
    github_api_url,
)
from automa.tools.images import GenerateImage, GenerateImageArgs, image_info
from automa.tools.policy import Decision, PermissionPolicy, PolicyDecision
from automa.tools.sandbox import DockerSandbox, SandboxError, SandboxMode
from automa.tools.search import (
    FakeSearchProvider,
    SearchHit,
    SearchProvider,
    SearchProviderError,
    WebSearch,
    WebSearchArgs,
)
from automa.tools.web import FetchRequestView, FetchUrl, FetchUrlArgs, describe_fetch_request
from automa.tools.workspace import DirEntry, Workspace, WorkspaceError


def default_tools(sandbox: DockerSandbox | None = None) -> list[Tool]:
    """Fresh instances of the built-in tools.

    With ``sandbox``, ``run_command`` runs every call in that Docker sandbox (D32, D33).
    ``fetch_url`` is ``external``: only the ``researcher`` role lists it and every
    call needs the user's approval (D72, D73). ``web_search`` is not included: no
    real search service is chosen yet (D74). ``github_read`` is ``external`` too:
    only the ``researcher`` role lists it and every call needs approval (D75, D76).
    ``generate_image`` is included *unconfigured* (it refuses every call) so role
    files can list it; the Orchestrator replaces it per task attempt with an
    instance bound to the ``ImageGateway``, the project/task context and the
    ``[images]`` limits (D100, D101). Only the ``designer`` role lists it.
    """
    return [
        *(cls() for cls in FILE_TOOLS),
        RunCommand(sandbox=sandbox),
        FetchUrl(),
        GitHubRead(),
        GenerateImage(),
    ]


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
    "FetchRequestView",
    "FetchUrl",
    "FetchUrlArgs",
    "FakeSearchProvider",
    "GenerateImage",
    "GenerateImageArgs",
    "GitHubRead",
    "GitHubReadArgs",
    "GitHubRequestView",
    "ListDir",
    "PermissionPolicy",
    "PolicyDecision",
    "ReadFile",
    "RunCommand",
    "RunCommandArgs",
    "RiskLevel",
    "SandboxError",
    "SandboxMode",
    "SearchHit",
    "SearchProvider",
    "SearchProviderError",
    "Tool",
    "ToolArgs",
    "ToolError",
    "ToolExecutor",
    "ToolLayerError",
    "ToolResult",
    "WebSearch",
    "WebSearchArgs",
    "Workspace",
    "WorkspaceError",
    "WriteFile",
    "check_docker",
    "clean_env",
    "default_tools",
    "describe_fetch_request",
    "describe_github_request",
    "github_api_url",
    "image_info",
    "tools_factory",
]
