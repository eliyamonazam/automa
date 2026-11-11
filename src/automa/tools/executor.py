"""ToolExecutor: the only code path that runs a tool.

For every call it (1) finds the tool, (2) asks the ``PermissionPolicy``,
(3) validates the arguments against the tool's model, (4) runs it inside the
``Workspace`` and (5) logs a ``tool.*`` event. Problems the model can fix
(unknown tool, bad arguments, denied, tool error) come back as a failed
``ToolResult``; an external or irreversible action goes through the Approval Gate (see
``_gate``): without a matching decision it records a request and raises
``ApprovalRequired`` so the orchestrator can pause the project. With
``PathClaims`` (parallel runs, D37) a call that writes a path another running task
has claimed fails before the gate and before it runs. Event-log failures fail closed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from automa.events import Event, EventSink
from automa.tools.approvals import ApprovalError, ApprovalStatus, ApprovalStore
from automa.tools.base import MAX_TOOL_OUTPUT_CHARS, Tool, ToolError, ToolResult
from automa.tools.claims import PathClaims
from automa.tools.policy import Decision, PermissionPolicy, PolicyDecision
from automa.tools.workspace import MAX_WRITE_BYTES, Workspace

MAX_ARGUMENTS_CHARS = 2 * MAX_WRITE_BYTES
MAX_ERROR_CHARS = 1_000


class ToolLayerError(Exception):
    """The tool layer itself failed (e.g. the event log); the call must not count."""


class ApprovalRequired(Exception):  # noqa: N818 - a control-flow signal, not an error
    """The call needs the user's approval (Approval Gate); the project should pause."""

    def __init__(self, tool: str, reason: str, request_id: str | None = None) -> None:
        super().__init__(reason)
        self.tool = tool
        self.reason = reason
        self.request_id = request_id


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
        *,
        approvals: ApprovalStore | None = None,
        task_id: str | None = None,
        claims: PathClaims | None = None,
    ) -> None:
        if not isinstance(policy, PermissionPolicy):
            raise TypeError("policy must be a PermissionPolicy")
        if not isinstance(workspace, Workspace):
            raise TypeError("workspace must be a Workspace")
        if not callable(getattr(events, "emit", None)):
            raise TypeError("events must be an EventSink")
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("project_id is required")
        if approvals is not None:
            if not isinstance(approvals, ApprovalStore):
                raise TypeError("approvals must be an ApprovalStore")
            if approvals.project_id != project_id:
                raise ValueError("approvals belong to another project")
        if claims is not None:
            if not isinstance(claims, PathClaims):
                raise TypeError("claims must be PathClaims")
            if not task_id:
                raise ValueError("claims need a task_id (the owner of written paths)")
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
        self._approvals = approvals
        self._task_id = task_id
        self._claims = claims

    @property
    def policy(self) -> PermissionPolicy:
        return self._policy

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def _decide(self, tool: Tool) -> PolicyDecision:
        return self._policy.decide(tool.name, tool.risk, irreversible=tool.irreversible)

    def specs(self) -> list[dict[str, Any]]:
        """Schemas of the tools the model is offered.

        Tools the policy allows outright, plus gated (``needs_approval``) tools when
        this executor has an Approval Gate store; without a store a gated tool could
        never run, so it is not offered.
        """
        offered = {Decision.ALLOW}
        if self._approvals is not None:
            offered.add(Decision.NEEDS_APPROVAL)
        return [
            t.spec() for _, t in sorted(self._tools.items()) if self._decide(t).decision in offered
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

        decision = self._policy.decide(tool.name, tool.risk, irreversible=tool.irreversible)
        needs_approval = decision.decision is Decision.NEEDS_APPROVAL
        if needs_approval and self._approvals is None:
            # No Approval Gate store: nothing can ever be approved here.
            self._emit("tool.approval_required", self._gate_data(tool))
            raise ApprovalRequired(tool.name, decision.reason)
        if not needs_approval and decision.decision is not Decision.ALLOW:
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

        conflict = self._claim(tool, args)
        if conflict is not None:
            return self._finish(tool, args, conflict)

        if needs_approval:
            rejected = self._gate(tool, args, decision.reason)
            if rejected is not None:
                return self._finish(tool, args, rejected)

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

    def _claim(self, tool: Tool, args: Any) -> ToolResult | None:
        """Claim the paths this call writes; a failed result if another task holds one."""
        if self._claims is None or self._task_id is None:
            return None
        paths = tool.writes(args)
        if not paths:
            return None
        try:
            conflict = self._claims.claim(paths, self._task_id)
        except ToolError as e:
            return self._fail(tool.name, str(e))
        if conflict is None:
            return None
        path, other = conflict
        self._emit(
            "tool.path_conflict",
            {"tool": tool.name, "path": _short(path, 300), "owner": other},
        )
        return self._fail(
            tool.name,
            f"{path!r} is being written by the parallel task {other!r}; "
            "write your output to a different path (do not retry this one)",
        )

    @staticmethod
    def _gate_data(tool: Tool) -> dict[str, Any]:
        data: dict[str, Any] = {"tool": tool.name, "risk": tool.risk.value}
        if tool.irreversible:
            data["irreversible"] = True
        return data

    def _gate(self, tool: Tool, args: Any, reason: str) -> ToolResult | None:
        """Approval Gate for one validated call.

        Returns ``None`` if an approval for exactly this call was consumed (the
        tool may run once), a failed result if the user rejected it, and otherwise
        records a pending request and raises ``ApprovalRequired``.
        """
        if self._approvals is None:  # pragma: no cover - guarded by the caller
            raise ToolLayerError("approval gate has no store")
        arguments = args.model_dump(mode="json")
        try:
            found = self._approvals.find_decision(
                tool=tool.name, arguments=arguments, task_id=self._task_id
            )
            if found is not None and self._approvals.consume(found.id):
                data = {**self._gate_data(tool), "request": found.id}
                if found.status is ApprovalStatus.APPROVED:
                    self._emit("tool.approved", data)
                    return None
                self._emit("tool.rejected", data)
                note = f" (user note: {found.note})" if found.note else ""
                return self._fail(
                    tool.name,
                    "the user rejected this action; do not repeat it" + note,
                    denied=True,
                )
            req = self._approvals.request(
                tool=tool.name,
                risk=tool.risk,
                arguments=arguments,
                reason=reason,
                task_id=self._task_id,
            )
        except (ApprovalError, OSError) as e:
            raise ToolLayerError(f"approval gate failed: {e}") from e
        self._emit(
            "tool.approval_required",
            {**self._gate_data(tool), "request": req.id},
        )
        raise ApprovalRequired(tool.name, reason, request_id=req.id)

    def _finish(self, tool: Tool, args: Any, result: ToolResult) -> ToolResult:
        data: dict[str, Any] = {"tool": tool.name, "risk": tool.risk.value, "ok": result.ok}
        if args is not None:
            for key, value in tool.audit(args).items():
                if key not in data and isinstance(value, str):
                    data[key] = _short(value, 300)
        if result.error:
            data["error"] = _short(result.error, 300)
        self._emit("tool.call", data)
        return result
