"""Tests for WorkerAgent (automa.agents.worker).

Local only: FakeProvider behind a real LLMGateway + BudgetGuard + CostLedger, and a
real ToolExecutor over a temporary Workspace. No network, no real model calls.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest

from automa.agents import WorkerAgent, WorkerSettings, task_message
from automa.budget import BudgetExceeded, BudgetGuard, CostLedger
from automa.budget.ledger import usd_to_nusd
from automa.config import AutomaConfig, Tier
from automa.events import MemoryEventSink
from automa.llm import (
    ChatRequest,
    ChatResponse,
    FakeProvider,
    FinishReason,
    LLMGateway,
    Message,
    ProviderError,
    ToolCall,
    Usage,
)
from automa.llm import (
    Role as MsgRole,
)
from automa.plan import Issue, ResultStatus, Task
from automa.roles import Role, RoleRegistry
from automa.tools import (
    ApprovalRequired,
    PermissionPolicy,
    RiskLevel,
    Tool,
    ToolArgs,
    ToolExecutor,
    ToolLayerError,
    Workspace,
    default_tools,
)

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
PID = "proj-1"


def clock() -> datetime:
    return T0


def config(**limits: float) -> AutomaConfig:
    data: dict[str, Any] = {
        "tiers": {"small": "fake-small", "medium": "fake-medium", "large": "fake-large"},
        "prices": {
            m: {"input_per_mtok": 1.0, "output_per_mtok": 2.0}
            for m in ("fake-small", "fake-medium", "fake-large")
        },
    }
    if limits:
        data["limits"] = limits
    return AutomaConfig.model_validate(data)


def make_task(role: str = "developer", **kw: Any) -> Task:
    d: dict[str, Any] = {
        "id": "t1",
        "title": "Write hello",
        "description": "Create hello.txt containing hi.",
        "role": role,
        "acceptance_criteria": [
            {"description": "file exists", "check": {"type": "file_exists", "path": "hello.txt"}},
            {"description": "is friendly"},
        ],
    }
    d.update(kw)
    return Task.model_validate(d)


def calls(*items: tuple[str, dict[str, Any] | str], content: str | None = None) -> ChatResponse:
    tcs = tuple(
        ToolCall(id=f"x{i}", name=n, arguments=a if isinstance(a, str) else json.dumps(a))
        for i, (n, a) in enumerate(items)
    )
    return ChatResponse(
        model="fake-medium",
        message=Message(role=MsgRole.ASSISTANT, content=content, tool_calls=tcs),
        usage=Usage(input_tokens=100, output_tokens=20),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def reply(text: str | None, finish: FinishReason = FinishReason.STOP) -> ChatResponse:
    return ChatResponse(
        model="fake-medium",
        message=Message(role=MsgRole.ASSISTANT, content=text or ""),
        usage=Usage(input_tokens=100, output_tokens=20),
        finish_reason=finish,
    )


class Rig:
    def __init__(self, tmp_path: Path, script: list[Any], **limits: float):
        self.cfg = config(**limits)
        self.provider = FakeProvider(script)
        self.ledger = CostLedger(tmp_path / "ledger.sqlite3")
        self.guard = BudgetGuard(self.cfg.limits, self.ledger)
        self.gateway = LLMGateway(self.cfg, self.provider, guard=self.guard)
        self.events = MemoryEventSink()
        self.tool_events = MemoryEventSink()
        root = tmp_path / "files"
        root.mkdir()
        self.root = root
        self.workspace = Workspace(root)
        self.roles = RoleRegistry.builtin(tools=default_tools())

    def executor(self, role: Role, tools: list[Tool] | None = None) -> ToolExecutor:
        return ToolExecutor(
            tools if tools is not None else default_tools(),
            role.policy(),
            self.workspace,
            self.tool_events,
            PID,
        )

    def agent(self, **kw: Any) -> WorkerAgent:
        kw.setdefault("events", self.events)
        return WorkerAgent(self.gateway, clock=clock, **kw)

    def run(self, role_name: str = "developer", **kw: Any) -> Any:
        role = kw.pop("role", None) or self.roles.get(role_name)
        task = kw.pop("task", None) or make_task(role.name)
        executor = kw.pop("executor", None) or self.executor(role)
        agent = self.agent(**kw.pop("agent_kw", {}))
        return agent.run(task, role, executor, project_id=PID, **kw)

    def types(self) -> list[str]:
        return [e.type for e in self.events.events]


# --- construction & argument checks -----------------------------------------


def test_requires_a_real_gateway(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [])
    for bad in (rig.provider, rig.guard, None):
        with pytest.raises(TypeError):
            WorkerAgent(bad)  # type: ignore[arg-type]


def test_settings_bounds() -> None:
    for kw in (
        {"max_steps": 0},
        {"max_steps": 201},
        {"step_max_tokens": 0},
        {"max_tool_calls_per_step": 0},
        {"max_tool_result_chars": 10},
        {"temperature": 2},
        {"unknown": 1},
    ):
        with pytest.raises(ValueError):
            WorkerSettings(**kw)


def test_role_mismatch_rejected_without_model_call(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["done"])
    writer = rig.roles.get("writer")
    with pytest.raises(ValueError, match="needs role"):
        rig.run(role=writer, task=make_task("developer"), executor=rig.executor(writer))
    assert rig.provider.requests == []


def test_wider_executor_policy_rejected(tmp_path: Path) -> None:
    """A caller cannot run a role with permissions wider than the role's own policy."""
    rig = Rig(tmp_path, ["done"])
    writer = rig.roles.get("writer")
    wide = ToolExecutor(
        default_tools(),
        PermissionPolicy(
            allowed_tools=frozenset({*writer.tools, "run_command"}), max_risk=RiskLevel.EXECUTE
        ),
        rig.workspace,
        rig.tool_events,
        PID,
    )
    with pytest.raises(ValueError, match="policy"):
        rig.run(role=writer, task=make_task("writer"), executor=wide)
    assert rig.provider.requests == []


def test_narrower_executor_policy_also_rejected(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["done"])
    dev = rig.roles.get("developer")
    narrow = ToolExecutor(
        default_tools(), PermissionPolicy(allowed_tools=frozenset()), rig.workspace,
        rig.tool_events, PID,
    )  # fmt: skip
    with pytest.raises(ValueError):
        rig.run(role=dev, executor=narrow)


@pytest.mark.parametrize("bad", [{"project_id": ""}, {"project_id": "  "}, {"attempt": 0}])
def test_bad_run_arguments(tmp_path: Path, bad: dict[str, Any]) -> None:
    rig = Rig(tmp_path, ["done"])
    dev = rig.roles.get("developer")
    kw: dict[str, Any] = {"project_id": PID, **bad}
    with pytest.raises(ValueError):
        rig.agent().run(make_task(), dev, rig.executor(dev), **kw)
    assert rig.provider.requests == []


def test_wrong_types_rejected(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [])
    dev = rig.roles.get("developer")
    ex = rig.executor(dev)
    agent = rig.agent()
    with pytest.raises(TypeError):
        agent.run({"id": "t1"}, dev, ex, project_id=PID)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        agent.run(make_task(), "developer", ex, project_id=PID)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        agent.run(make_task(), dev, object(), project_id=PID)  # type: ignore[arg-type]


# --- happy path --------------------------------------------------------------


def test_writes_file_then_completes(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        [
            calls(("write_file", {"path": "hello.txt", "content": "hi"})),
            calls(("read_file", {"path": "hello.txt"})),
            "Created hello.txt with 'hi'.",
        ],
    )
    result = rig.run()
    assert result.status is ResultStatus.COMPLETED
    assert result.summary == "Created hello.txt with 'hi'."
    assert result.files == ["hello.txt"]
    assert result.steps == 3
    assert result.error is None
    assert result.tier is Tier.MEDIUM and result.model == "fake-medium"
    assert result.task_id == "t1" and result.attempt == 1
    assert result.started_at == T0 and result.finished_at == T0
    assert (rig.root / "hello.txt").read_text() == "hi"
    # The second call saw the tool result of the first one.
    second = rig.provider.requests[1].messages
    assert second[-1].role is MsgRole.TOOL and second[-1].tool_call_id == "call_1_0"
    third = rig.provider.requests[2].messages
    assert third[-1].content == "hi"
    assert rig.types() == ["worker.started", "worker.step", "worker.step", "worker.finished"]


def test_cost_matches_ledger(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(("list_dir", {})), "done"])
    result = rig.run()
    rows = rig.ledger.entries(project_id=PID)
    assert len(rows) == 2
    assert result.cost_nusd > 0
    assert result.cost_nusd == sum(r.cost_nusd for r in rows)
    assert all(r.task_id == "t1" for r in rows)
    # The first call: 100 in / 20 out at 1$/2$ per Mtok.
    assert rows[0].cost_nusd == usd_to_nusd((100 * 1.0 + 20 * 2.0) / 1_000_000)


def test_requests_carry_task_context_and_only_allowed_tools(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["done"])
    rig.run("writer", task=make_task("writer"))
    req: ChatRequest = rig.provider.requests[0]
    names = {t.name for t in req.tools}
    assert names == {"read_file", "list_dir", "write_file"}
    assert "run_command" not in names
    assert req.max_tokens == WorkerSettings().step_max_tokens
    assert req.model == "fake-medium"


def test_task_context_reaches_guard(tmp_path: Path) -> None:
    seen: list[Any] = []
    rig = Rig(tmp_path, ["done"])
    original = rig.guard.authorize

    def spy(estimate: Any) -> None:
        seen.append(estimate.context)
        original(estimate)

    rig.guard.authorize = spy  # type: ignore[method-assign]
    rig.run()
    assert seen and seen[0].project_id == PID and seen[0].task_id == "t1"


def test_system_prompt_has_role_instructions_and_protocol(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["done"])
    dev = rig.roles.get("developer")
    rig.run()
    system = rig.provider.requests[0].messages[0]
    assert system.role is MsgRole.SYSTEM
    assert dev.instructions.strip() in (system.content or "")
    assert "WITHOUT any tool" in (system.content or "")


def test_task_message_contents() -> None:
    t = make_task(inputs={"lang": "python"})
    text = task_message(t, [Issue(description="hello.txt is empty", criterion=0)])
    assert "t1" in text and "Write hello" in text and "Create hello.txt" in text
    assert "lang: python" in text
    assert "0. file exists" in text and "file_exists" in text
    assert "1. is friendly" in text
    assert "[criterion 0] hello.txt is empty" in text
    assert "rejected" not in task_message(t)


def test_feedback_is_sent_on_retry(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["fixed"])
    result = rig.run(attempt=2, feedback=[Issue(description="missing newline")])
    assert result.attempt == 2
    user = rig.provider.requests[0].messages[1].content or ""
    assert "missing newline" in user


@pytest.mark.parametrize(
    ("override", "task_tier", "expected"),
    [
        (None, None, Tier.MEDIUM),
        (None, Tier.SMALL, Tier.SMALL),
        (Tier.LARGE, Tier.SMALL, Tier.LARGE),
        (Tier.LARGE, None, Tier.LARGE),
    ],
)
def test_tier_precedence(
    tmp_path: Path, override: Tier | None, task_tier: Tier | None, expected: Tier
) -> None:
    rig = Rig(tmp_path, ["done"])
    result = rig.run(task=make_task(tier=task_tier), tier=override)
    assert result.tier is expected
    assert rig.provider.requests[0].model == f"fake-{expected.value}"


def test_files_deduplicated_and_normalized(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        [
            calls(
                ("write_file", {"path": "a/b.txt", "content": "1"}),
                ("write_file", {"path": "./a/b.txt", "content": "2"}),
                ("write_file", {"path": "c.txt", "content": "3"}),
            ),
            "done",
        ],
    )
    result = rig.run()
    assert result.files == ["a/b.txt", "c.txt"]


def test_failed_writes_are_not_listed(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        [
            calls(
                ("write_file", {"path": "../escape.txt", "content": "x"}),
                ("write_file", {"path": "ok.txt"}),  # missing content
                ("write_file", "{not json"),
            ),
            "gave up",
        ],
    )
    result = rig.run()
    assert result.files == []
    assert not (tmp_path / "escape.txt").exists()
    tool_msgs = [m for m in rig.provider.requests[1].messages if m.role is MsgRole.TOOL]
    assert len(tool_msgs) == 3
    assert all((m.content or "").startswith("ERROR:") for m in tool_msgs)


# --- policy is enforced by the executor, not the prompt ---------------------


def test_denied_tool_returns_error_to_model_and_does_not_run(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        [calls(("run_command", {"argv": ["touch", "pwned"]})), "ok, I cannot run commands"],
    )
    result = rig.run("writer", task=make_task("writer"))
    assert result.status is ResultStatus.COMPLETED
    assert not (rig.root / "pwned").exists()
    tool_msg = rig.provider.requests[1].messages[-1]
    assert "not allowed" in (tool_msg.content or "")
    assert any(e.type == "tool.denied" for e in rig.tool_events.events)


def test_unknown_tool_returns_error(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(("delete_everything", {})), "done"])
    rig.run()
    assert "unknown tool" in (rig.provider.requests[1].messages[-1].content or "")


class _Publish(Tool):
    name: ClassVar[str] = "publish"
    description: ClassVar[str] = "Publish to the internet."
    risk: ClassVar[RiskLevel] = RiskLevel.EXTERNAL
    args_model: ClassVar[type[ToolArgs]] = ToolArgs
    ran: ClassVar[bool] = False

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        _Publish.ran = True
        return "published"


def test_external_tool_raises_approval_required(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(("publish", {})), "done"])
    dev = rig.roles.get("developer")
    ex = rig.executor(dev, [*default_tools(), _Publish()])
    with pytest.raises(ApprovalRequired):
        rig.run(executor=ex)
    assert _Publish.ran is False
    assert len(rig.provider.requests) == 1


def test_tool_layer_error_propagates(tmp_path: Path) -> None:
    class Broken(MemoryEventSink):
        def emit(self, event: Any) -> None:
            raise OSError("disk full")

    rig = Rig(tmp_path, [calls(("list_dir", {})), "done"])
    dev = rig.roles.get("developer")
    ex = ToolExecutor(default_tools(), dev.policy(), rig.workspace, Broken(), PID)
    with pytest.raises(ToolLayerError):
        rig.run(executor=ex)


def test_too_many_tool_calls_are_not_run(tmp_path: Path) -> None:
    writes = [("write_file", {"path": f"f{i}.txt", "content": "x"}) for i in range(5)]
    rig = Rig(tmp_path, [calls(*writes), "done"])
    result = rig.run(agent_kw={"settings": WorkerSettings(max_tool_calls_per_step=2)})
    assert result.files == ["f0.txt", "f1.txt"]
    assert not (rig.root / "f2.txt").exists()
    tool_msgs = [m for m in rig.provider.requests[1].messages if m.role is MsgRole.TOOL]
    assert len(tool_msgs) == 5  # every call is answered
    assert all("not run" in (m.content or "") for m in tool_msgs[2:])


def test_tool_call_ids_are_reissued(tmp_path: Path) -> None:
    resp = ChatResponse(
        model="fake-medium",
        message=Message(
            role=MsgRole.ASSISTANT,
            tool_calls=(
                ToolCall(id="", name="list_dir", arguments="{}"),
                ToolCall(id="", name="list_dir", arguments="{}"),
            ),
        ),
        usage=Usage(input_tokens=10, output_tokens=5),
        finish_reason=FinishReason.TOOL_CALLS,
    )
    rig = Rig(tmp_path, [resp, "done"])
    rig.run()
    msgs = rig.provider.requests[1].messages
    assistant = msgs[2]
    ids = [tc.id for tc in assistant.tool_calls]
    assert ids == ["call_1_0", "call_1_1"]
    assert [m.tool_call_id for m in msgs[3:]] == ids


def test_tool_results_are_truncated_in_conversation(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(("read_file", {"path": "big.txt"})), "done"])
    (rig.root / "big.txt").write_text("x" * 15_000)
    rig.run(agent_kw={"settings": WorkerSettings(max_tool_result_chars=1_000)})
    content = rig.provider.requests[1].messages[-1].content or ""
    assert len(content) < 1_100 and content.endswith("[... truncated]")


def test_run_command_for_developer(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        [calls(("run_command", {"argv": ["python3", "-c", "print(6*7)"]})), "printed 42"],
    )
    result = rig.run()
    assert result.status is ResultStatus.COMPLETED
    assert "42" in (rig.provider.requests[1].messages[-1].content or "")


# --- termination -------------------------------------------------------------


def test_step_limit_from_role(tmp_path: Path) -> None:
    # Endless tool calls: the cap must stop it.
    rig = Rig(tmp_path, [calls(("list_dir", {})) for _ in range(50)])
    dev = rig.roles.get("developer")
    small = dev.model_copy(update={"max_steps": 3})
    result = rig.run(role=small, executor=rig.executor(small))
    assert result.status is ResultStatus.STEP_LIMIT
    assert result.steps == 3
    assert len(rig.provider.requests) == 3
    assert "step limit" in (result.error or "")
    assert rig.provider.remaining == 47


def test_step_limit_from_settings_is_min(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(("list_dir", {})) for _ in range(10)])
    result = rig.run(agent_kw={"settings": WorkerSettings(max_steps=2)})
    assert result.status is ResultStatus.STEP_LIMIT and result.steps == 2
    assert len(rig.provider.requests) == 2


def test_step_limit_records_files_written(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(("write_file", {"path": "p.txt", "content": "x"}))])
    result = rig.run(agent_kw={"settings": WorkerSettings(max_steps=1)})
    assert result.status is ResultStatus.STEP_LIMIT
    assert result.files == ["p.txt"]


def test_empty_reply_is_nudged_and_counts_as_step(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [reply(""), reply("   "), "done now"])
    result = rig.run()
    assert result.status is ResultStatus.COMPLETED and result.steps == 3
    last = rig.provider.requests[2].messages
    assert last[-1].role is MsgRole.USER and "empty" in (last[-1].content or "")


def test_empty_replies_cannot_loop_forever(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [reply("") for _ in range(10)])
    result = rig.run(agent_kw={"settings": WorkerSettings(max_steps=4)})
    assert result.status is ResultStatus.STEP_LIMIT
    assert len(rig.provider.requests) == 4


def test_truncated_reply_is_not_a_final_answer(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [reply("I did half of", FinishReason.LENGTH), "done"])
    result = rig.run()
    assert result.status is ResultStatus.COMPLETED and result.summary == "done"
    msgs = rig.provider.requests[1].messages
    assert "cut off" in (msgs[-1].content or "")


def test_content_filter_fails(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [reply("", FinishReason.CONTENT_FILTER), "never"])
    result = rig.run()
    assert result.status is ResultStatus.FAILED
    assert "content filter" in (result.error or "")
    assert rig.provider.remaining == 1


def test_long_summary_is_truncated(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["y" * 50_000])
    result = rig.run()
    assert result.status is ResultStatus.COMPLETED
    assert len(result.summary) <= 10_000


def test_text_alongside_tool_calls_is_not_final(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(("list_dir", {}), content="Let me look"), "done"])
    result = rig.run()
    assert result.summary == "done" and result.steps == 2


# --- gateway errors are never retried ---------------------------------------


def test_provider_error_propagates_without_retry(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [ProviderError("boom", status=500, retryable=True), "done"])
    with pytest.raises(ProviderError):
        rig.run()
    assert len(rig.provider.requests) == 1
    assert rig.provider.remaining == 1
    assert "worker.finished" not in rig.types()


def test_budget_stop_before_first_call(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["done"], per_call_usd=1e-9)
    with pytest.raises(BudgetExceeded):
        rig.run()
    assert rig.provider.requests == []


def test_budget_stop_mid_loop_is_not_retried(tmp_path: Path) -> None:
    # Each call reserves ~0.008 USD (4000 output tokens at 2$/Mtok + input). The first
    # call really costs 0.005 USD, so the second reservation no longer fits the cap.
    first = calls(("write_file", {"path": "a.txt", "content": "x"}))
    first = first.model_copy(
        update={"usage": Usage(input_tokens=100, output_tokens=20, reported_cost_usd=0.005)}
    )
    rig = Rig(
        tmp_path,
        [first, "done", "done"],
        project_usd=0.0101,
        per_call_usd=0.0101,
    )
    with pytest.raises(BudgetExceeded):
        rig.run()
    assert len(rig.provider.requests) == 1
    assert (rig.root / "a.txt").read_text() == "x"


def test_interrupt_propagates(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        rig.run()


# --- events ------------------------------------------------------------------


def test_events_have_no_content(tmp_path: Path) -> None:
    marker = "MARKER-CONTENT-123"
    rig = Rig(
        tmp_path,
        [calls(("write_file", {"path": "s.txt", "content": marker})), f"wrote {marker}"],
    )
    rig.run()
    dumped = json.dumps([e.model_dump(mode="json") for e in rig.events.events])
    assert marker not in dumped
    assert all(e.project_id == PID for e in rig.events.events)
    finished = rig.events.events[-1]
    assert finished.type == "worker.finished"
    assert finished.data["status"] == "completed" and finished.data["files"] == 1


def test_no_events_sink_is_fine(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["done"])
    dev = rig.roles.get("developer")
    result = WorkerAgent(rig.gateway).run(make_task(), dev, rig.executor(dev), project_id=PID)
    assert result.status is ResultStatus.COMPLETED


# --- every builtin role can run ----------------------------------------------


@pytest.mark.parametrize("name", ["researcher", "architect", "developer", "writer"])
def test_builtin_roles_run(tmp_path: Path, name: str) -> None:
    rig = Rig(tmp_path, [calls(("list_dir", {})), "done"])
    result = rig.run(name, task=make_task(name))
    assert result.status is ResultStatus.COMPLETED


# --- structure: the worker only reaches models via the gateway and tools via the executor


def test_worker_module_structure() -> None:
    import automa.agents.worker as mod

    tree = ast.parse(Path(mod.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    forbidden = {
        "automa.llm.provider",
        "automa.llm.fake",
        "automa.llm.openai_compat",
        "automa.llm.factory",
        "subprocess",
        "os",
        "httpx",
    }
    assert not imported & forbidden
    # Tools are never run directly: no `.run(` on anything but the agent itself.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"authorize", "record", "release"}
            if node.func.attr == "run":
                pytest.fail("worker must not call Tool.run directly")
