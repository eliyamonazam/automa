"""ToolExecutor: the only code path that runs a tool.

For every call it (1) finds the tool, (2) asks the ``PermissionPolicy``,
(3) validates the arguments against the tool's model, (4) runs it inside the
``Workspace`` and (5) logs a ``tool.*`` event. Problems the model can fix
(unknown tool, bad arguments, denied, tool error) come back as a failed
``ToolResult``; an external action raises ``ApprovalRequired`` so the
orchestrator can pause the project. Event-log failures fail closed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from automa.events import Event, EventSink
from automa.tools.base import MAX_TOOL_OUTPUT_CHARS, Tool, ToolError, ToolResult
from automa.tools.policy import Decision, PermissionPolicy
from automa.tools.workspace import MAX_WRITE_BYTES, Workspace

MAX_ARGUMENTS_CHARS = 2 * MAX_WRITE_BYTES
MAX_ERROR_CHARS = 1_000


class ToolLayerError(Exception):
    """The tool layer itself failed (e.g. the event log); the call must not count."""


class ApprovalRequired(Exception):  # noqa: N818 - a control-flow signal, not an error
    """The call needs the user's approval (Approval Gate); the project should pause."""

    def __init__(self, tool: str, reason: str) -> None:
        super().__init__(reason)
        self.tool = tool
        self.reason = reason


def _short(text: str, limit: int = MAX_ERROR_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + " [...]"


def _format_validation(e: ValidationError) -> str:
    parts = []
    for err in e.errors()[:5]:
        loc = ".".join(str(p) for p in err["loc"]) or "(arguments)"
        parts.append(f"{loc}: {err['msg']}")
    return "invalid arguments: " + "; ".join(parts)


class ToolExecutor:
    def __init__(
        self,
        tools: Iterable[Tool],
        policy: PermissionPolicy,
        workspace: Workspace,
        events: EventSink,
        project_id: str,
    ) -> None:
        if not isinstance(policy, PermissionPolicy):
            raise TypeError("policy must be a PermissionPolicy")
        if not isinstance(workspace, Workspace):
            raise TypeError("workspace must be a Workspace")
        if not callable(getattr(events, "emit", None)):
            raise TypeError("events must be an EventSink")
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("project_id is required")
        registry: dict[str, Tool] = {}
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError(f"not a Tool: {tool!r}")
            if tool.name in registry:
                raise ValueError(f"duplicate tool name: {tool.name!r}")
            registry[tool.name] = tool
        self._tools = registry
        self._policy = policy
        self._workspace = workspace
        self._events = events
        self._project_id = project_id

    @property
    def policy(self) -> PermissionPolicy:
        return self._policy

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        """Schemas of the tools this policy allows outright (what the model is offered)."""
        return [
            t.spec()
            for name, t in sorted(self._tools.items())
            if self._policy.decide(name, t.risk).decision is Decision.ALLOW
        ]

    def _emit(self, type_: str, data: dict[str, Any]) -> None:
        try:
            self._events.emit(
                Event(ts=datetime.now(UTC), type=type_, project_id=self._project_id, data=data)
            )
        except Exception as e:
            raise ToolLayerError(f"could not log {type_} event") from e

    def _fail(self, name: str, error: str, *, denied: bool = False) -> ToolResult:
        return ToolResult(tool=name, ok=False, error=_short(error), denied=denied)

    def execute(self, name: str, arguments: str | Mapping[str, Any] | None) -> ToolResult:
        tool = self._tools.get(name) if isinstance(name, str) else None
        if tool is None:
            label = name if isinstance(name, str) else repr(name)
            self._emit("tool.denied", {"tool": _short(label, 100), "reason": "unknown tool"})
            known = ", ".join(self.tool_names()) or "(none)"
            return self._fail(_short(label, 100), f"unknown tool; available: {known}", denied=True)

        decision = self._policy.decide(tool.name, tool.risk)
        if decision.decision is Decision.NEEDS_APPROVAL:
            self._emit("tool.approval_required", {"tool": tool.name, "risk": tool.risk.value})
            raise ApprovalRequired(tool.name, decision.reason)
        if decision.decision is not Decision.ALLOW:
            self._emit(
                "tool.denied",
                {"tool": tool.name, "risk": tool.risk.value, "reason": decision.reason},
            )
            return self._fail(tool.name, decision.reason, denied=True)

        try:
            if arguments is None:
                raw: Any = {}
            elif isinstance(arguments, str):
                if len(arguments) > MAX_ARGUMENTS_CHARS:
                    raise ToolError("arguments are too large")
                raw = json.loads(arguments) if arguments.strip() else {}
            else:
                raw = dict(arguments)
            if not isinstance(raw, dict):
                raise ToolError("arguments must be a JSON object")
            args = tool.args_model.model_validate(raw)
        except ValidationError as e:
            return self._finish(tool, None, self._fail(tool.name, _format_validation(e)))
        except (ValueError, TypeError) as e:  # JSONDecodeError is a ValueError
            return self._finish(tool, None, self._fail(tool.name, f"invalid arguments: {e}"))
        except ToolError as e:
            return self._finish(tool, None, self._fail(tool.name, str(e)))

        try:
            content = tool.run(args, self._workspace)
        except ToolError as e:
            result = self._fail(tool.name, str(e))
        except OSError as e:
            result = self._fail(tool.name, f"{type(e).__name__}: {e.strerror or e}")
        else:
            if not isinstance(content, str):
                raise ToolLayerError(f"tool {tool.name!r} returned {type(content).__name__}")
            if len(content) > MAX_TOOL_OUTPUT_CHARS:
                content = content[:MAX_TOOL_OUTPUT_CHARS] + "\n[... truncated]"
            result = ToolResult(tool=tool.name, ok=True, content=content)
        return self._finish(tool, args, result)

    def _finish(self, tool: Tool, args: Any, result: ToolResult) -> ToolResult:
        data: dict[str, Any] = {"tool": tool.name, "risk": tool.risk.value, "ok": result.ok}
        path = getattr(args, "path", None)
        if isinstance(path, str):
            data["path"] = _short(path, 300)
        if result.error:
            data["error"] = _short(result.error, 300)
        self._emit("tool.call", data)
        return result
