"""Tests for the Orchestrator loop (automa.orchestrator).

Local only: FakeProvider behind a real LLMGateway + BudgetGuard + CostLedger, real
WorkerAgent/VerifierAgent, real ToolExecutor and ProjectStore on a temporary
workspace. No network, no real model calls.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest

from automa.agents import VerifierAgent, WorkerAgent
from automa.budget import BudgetGuard, CostLedger
from automa.config import AutomaConfig, Tier
from automa.llm import (
    ChatResponse,
    FakeProvider,
    FinishReason,
    LLMGateway,
    Message,
    ProviderError,
    ToolCall,
    Usage,
)
from automa.llm import Role as MsgRole
from automa.orchestrator import (
    Orchestrator,
    OrchestratorError,
    OrchestratorSettings,
    RunReport,
    StopReason,
    next_tier,
    rejections_at_tier,
)
from automa.plan import Plan, TaskStatus
from automa.project import ProjectNotFound, ProjectStatus, ProjectStore
from automa.roles import Role, RoleRegistry
from automa.tools import RiskLevel, Tool, ToolArgs, Workspace, default_tools
from automa.tools.approvals import ApprovalStatus

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
PID = "proj-1"
ALL_TIERS = {"small": "fake-small", "medium": "fake-medium", "large": "fake-large"}


def clock() -> datetime:
    return T0


def config(tiers: dict[str, str] | None = None, **limits: float) -> AutomaConfig:
    tiers = ALL_TIERS if tiers is None else tiers
    data: dict[str, Any] = {
        "tiers": tiers,
        "prices": {m: {"input_per_mtok": 1.0, "output_per_mtok": 2.0} for m in tiers.values()},
    }
    if limits:
        data["limits"] = limits
    return AutomaConfig.model_validate(data)


def task_dict(tid: str, *, deps: list[str] | None = None, role: str = "writer", **kw: Any) -> dict:
    d: dict[str, Any] = {
        "id": tid,
        "title": f"Write {tid}",
        "role": role,
        "depends_on": deps or [],
        "acceptance_criteria": [
            {
                "description": f"{tid}.txt says ok",
                "check": {"type": "file_contains", "path": f"{tid}.txt", "text": "ok"},
            }
        ],
    }
    d.update(kw)
    return d


def make_plan(*tasks: dict) -> Plan:
    return Plan.model_validate({"goal": "test goal", "tasks": list(tasks)})


def calls(*items: tuple[str, dict[str, Any]]) -> ChatResponse:
    tcs = tuple(
        ToolCall(id=f"x{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(items)
    )
    return ChatResponse(
        model="fake",
        message=Message(role=MsgRole.ASSISTANT, content=None, tool_calls=tcs),
        usage=Usage(input_tokens=100, output_tokens=20),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def write(tid: str, content: str = "ok") -> list[Any]:
    """Worker script of one attempt: write ``<tid>.txt`` then finish."""
    return [calls(("write_file", {"path": f"{tid}.txt", "content": content})), "done"]


class Rig:
    def __init__(
        self,
        tmp_path: Path,
        script: list[Any],
        plan: Plan | None = None,
        *,
        tiers: dict[str, str] | None = None,
        roles: RoleRegistry | None = None,
        tools: Any = default_tools,
        **limits: float,
    ):
        self.tmp = tmp_path
        self.store = ProjectStore.for_workspace(tmp_path / "ws", clock=clock)
        self.store.create("test goal", project_id=PID)
        if plan is not None:
            self.store.save_plan(PID, plan)
        self.provider = FakeProvider(script)
        self.ledger = CostLedger(tmp_path / "ledger.sqlite3")
        self.roles = roles or RoleRegistry.builtin(tools=default_tools())
        self.tools = tools
        self.tiers = tiers
        self.rebuild(**limits)

    def rebuild(self, **limits: float) -> None:
        """New gateway/guard/agents (like a fresh process), same store, ledger and provider."""
        self.cfg = config(self.tiers, **limits)
        events = self.store.events(PID)
        self.guard = BudgetGuard(self.cfg.limits, self.ledger, events=events)
        self.gateway = LLMGateway(self.cfg, self.provider, guard=self.guard)
        self.worker = WorkerAgent(self.gateway, events=events, clock=clock)
        self.verifier = VerifierAgent(self.gateway, events=events, clock=clock)

    def orch(self, **settings: Any) -> Orchestrator:
        return Orchestrator(
            self.store,
            self.worker,
            self.verifier,
            self.roles,
            self.cfg,
            settings=OrchestratorSettings(**settings) if settings else None,
            tools=self.tools,
            clock=clock,
        )

    def run(self, **settings: Any) -> RunReport:
        return self.orch(**settings).run(PID)

    def state(self, tid: str) -> Any:
        return self.store.task_states(PID)[tid]

    def project_status(self) -> ProjectStatus:
        return self.store.load(PID).status

    def event_types(self) -> list[str]:
        return [e.type for e in self.store.events(PID).read()]

    def models(self) -> list[str]:
        return [r.model for r in self.provider.requests]


# --- construction ------------------------------------------------------------


def test_constructor_type_checks(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [])
    good = [rig.store, rig.worker, rig.verifier, rig.roles, rig.cfg]
    for i in range(len(good)):
        args = list(good)
        args[i] = object()
        with pytest.raises(TypeError):
            Orchestrator(*args)  # type: ignore[arg-type]


def test_settings_bounds() -> None:
    for kw in ({"max_attempts": 0}, {"max_attempts": 51}, {"escalate_after": 0}, {"x": 1}):
        with pytest.raises(ValueError):
            OrchestratorSettings(**kw)
    s = OrchestratorSettings()
    assert (s.max_attempts, s.escalate_after) == (4, 2)


def test_unknown_project_and_missing_plan(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [])
    with pytest.raises(ProjectNotFound):
        rig.orch().run("nope")
    with pytest.raises(OrchestratorError, match="no plan"):
        rig.run()
    assert rig.provider.requests == []
    assert rig.project_status() is ProjectStatus.NEW


# --- helpers -------------------------------------------------------------------


def test_next_tier() -> None:
    every = {Tier.SMALL, Tier.MEDIUM, Tier.LARGE}
    assert next_tier(Tier.SMALL, every) is Tier.MEDIUM
    assert next_tier(Tier.MEDIUM, every) is Tier.LARGE
    assert next_tier(Tier.LARGE, every) is None
    assert next_tier(Tier.SMALL, {Tier.SMALL, Tier.LARGE}) is Tier.LARGE
    assert next_tier(Tier.MEDIUM, {Tier.SMALL, Tier.MEDIUM}) is None


# --- happy paths ---------------------------------------------------------------


def test_single_task_passes_and_project_done(tmp_path: Path) -> None:
    rig = Rig(tmp_path, write("a"), make_plan(task_dict("a")))
    report = rig.run()
    assert report.status is ProjectStatus.DONE
    assert report.reason is StopReason.COMPLETED
    assert report.passed == ["a"] and report.failed == [] and report.attempts_run == 1
    assert rig.project_status() is ProjectStatus.DONE
    st = rig.state("a")
    assert st.status is TaskStatus.PASSED and st.attempts == 1
    [(result, verdict)] = rig.store.task_history(PID, "a")
    assert result.files == ["a.txt"] and verdict is not None and verdict.passed
    types = rig.event_types()
    for t in ("orchestrator.started", "orchestrator.attempt", "orchestrator.finished"):
        assert t in types
    # status went through running → verifying → passed
    statuses = [e.data["to"] for e in rig.store.events(PID).read() if e.type == "task.status"]
    assert statuses == ["running", "verifying", "passed"]


def test_dependencies_run_in_order(tmp_path: Path) -> None:
    plan = make_plan(task_dict("b", deps=["a"]), task_dict("a"))
    rig = Rig(tmp_path, [*write("a"), *write("b")], plan)
    report = rig.run()
    assert report.status is ProjectStatus.DONE
    assert report.passed == ["b", "a"]  # plan order in the report
    first_user = rig.provider.requests[0].messages[1].content or ""
    assert "Task `a`" in first_user
    assert "Task `b`" in (rig.provider.requests[2].messages[1].content or "")


def test_executors_use_role_and_verifier_policies(tmp_path: Path) -> None:
    """The writer role never sees run_command; the worker gets only its role's tools."""
    rig = Rig(tmp_path, write("a"), make_plan(task_dict("a", role="writer")))
    rig.run()
    names = {t.name for t in rig.provider.requests[0].tools}
    assert names == set(rig.roles.get("writer").tools)
    assert "run_command" not in names


# --- retry, feedback, escalation ------------------------------------------------


def test_rejected_attempt_is_retried_with_feedback(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [*write("a", "bad"), *write("a", "ok")], make_plan(task_dict("a")))
    report = rig.run()
    assert report.status is ProjectStatus.DONE and report.attempts_run == 2
    retry_msg = rig.provider.requests[2].messages[1].content or ""
    assert "rejected by the verifier" in retry_msg
    assert "[criterion 0]" in retry_msg
    history = rig.store.task_history(PID, "a")
    assert [v.passed for _, v in history if v is not None] == [False, True]
    assert rig.state("a").attempts == 2


def test_escalates_after_two_rejections_at_one_tier(tmp_path: Path) -> None:
    script = [*write("a", "bad"), *write("a", "bad"), *write("a", "ok")]
    rig = Rig(tmp_path, script, make_plan(task_dict("a")))
    report = rig.run()
    assert report.status is ProjectStatus.DONE
    # writer default tier is medium; 2 steps per attempt
    assert rig.models() == ["fake-medium"] * 4 + ["fake-large"] * 2
    tiers = [r.tier for r, _ in rig.store.task_history(PID, "a")]
    assert tiers == [Tier.MEDIUM, Tier.MEDIUM, Tier.LARGE]
    esc = [e for e in rig.store.events(PID).read() if e.type == "orchestrator.escalated"]
    assert [(e.data["from"], e.data["to"]) for e in esc] == [("medium", "large")]


def test_escalate_after_setting(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        [*write("a", "bad"), *write("a", "ok")],
        make_plan(task_dict("a", tier="small")),
    )
    rig.run(escalate_after=1)
    assert rig.models() == ["fake-small"] * 2 + ["fake-medium"] * 2


def test_escalation_skips_unconfigured_tier(tmp_path: Path) -> None:
    tiers = {"small": "fake-small", "large": "fake-large"}
    script = [*write("a", "bad"), *write("a", "ok")]
    rig = Rig(tmp_path, script, make_plan(task_dict("a", tier="small")), tiers=tiers)
    rig.run(escalate_after=1)
    assert rig.models() == ["fake-small"] * 2 + ["fake-large"] * 2


def test_no_tier_above_large_keeps_large(tmp_path: Path) -> None:
    script = [*write("a", "bad"), *write("a", "bad"), *write("a", "ok")]
    rig = Rig(tmp_path, script, make_plan(task_dict("a", tier="large")))
    report = rig.run(escalate_after=1)
    assert report.status is ProjectStatus.DONE
    assert set(rig.models()) == {"fake-large"}
    assert not [e for e in rig.event_types() if e == "orchestrator.escalated"]


def test_rejections_at_tier_counts_trailing_streak(tmp_path: Path) -> None:
    script = [*write("a", "bad"), *write("a", "bad"), *write("a", "ok")]
    rig = Rig(tmp_path, script, make_plan(task_dict("a")))
    rig.run()
    h = rig.store.task_history(PID, "a")
    assert rejections_at_tier(h[:2], Tier.MEDIUM) == 2
    assert rejections_at_tier(h[:2], Tier.LARGE) == 0
    assert rejections_at_tier(h, Tier.LARGE) == 0  # last one passed
    assert rejections_at_tier([], Tier.MEDIUM) == 0


# --- failure and blocking -------------------------------------------------------


def test_max_attempts_fails_task_and_blocks_dependents(tmp_path: Path) -> None:
    plan = make_plan(
        task_dict("a"),
        task_dict("b", deps=["a"]),
        task_dict("c", deps=["b"]),
        task_dict("d"),
    )
    script = [*write("a", "bad"), *write("a", "bad"), *write("d")]
    rig = Rig(tmp_path, script, plan)
    report = rig.run(max_attempts=2)
    assert report.status is ProjectStatus.FAILED
    assert report.reason is StopReason.TASKS_FAILED
    assert report.failed == ["a"]
    assert report.blocked == ["b", "c"]
    assert report.passed == ["d"]  # independent work still runs
    assert rig.project_status() is ProjectStatus.FAILED
    assert "2 attempt" in (rig.state("a").error or "")
    assert "'a'" in (rig.state("b").error or "")
    assert rig.provider.remaining == 0
    # rerun does nothing more and spends nothing
    n = len(rig.provider.requests)
    again = rig.run(max_attempts=2)
    assert again.status is ProjectStatus.FAILED and len(rig.provider.requests) == n


def test_lowering_max_attempts_on_resume_fails_task(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [*write("a", "bad"), ProviderError("down")], make_plan(task_dict("a")))
    assert rig.run().reason is StopReason.PROVIDER
    report = rig.run(max_attempts=1)
    assert report.failed == ["a"] and report.status is ProjectStatus.FAILED


def test_unknown_role_fails_task_without_model_call(tmp_path: Path) -> None:
    plan = make_plan(task_dict("a", role="poet"), task_dict("b", deps=["a"]))
    rig = Rig(tmp_path, [], plan)
    report = rig.run()
    assert report.failed == ["a"] and report.blocked == ["b"]
    assert "unknown role" in (rig.state("a").error or "")
    assert rig.provider.requests == []


def test_step_limit_counts_as_rejection(tmp_path: Path) -> None:
    loop = calls(("list_dir", {}))
    from automa.roles import Role

    base = RoleRegistry.builtin(tools=default_tools())
    roles = RoleRegistry(
        [
            r if r.name != "writer" else Role.model_validate({**r.model_dump(), "max_steps": 1})
            for r in base
        ]
    )
    rig = Rig(tmp_path, [loop, loop], make_plan(task_dict("a")), roles=roles)
    report = rig.run(max_attempts=2)
    assert report.failed == ["a"]
    assert [r.status.value for r, _ in rig.store.task_history(PID, "a")] == ["step_limit"] * 2


# --- pauses and resume --------------------------------------------------------


def test_budget_stop_pauses_without_spending_or_attempt(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path, write("a"), make_plan(task_dict("a")), project_usd=0.004, per_call_usd=0.004
    )
    report = rig.run()
    assert report.status is ProjectStatus.PAUSED and report.reason is StopReason.BUDGET
    assert report.task_id == "a"
    assert rig.provider.requests == []
    assert rig.project_status() is ProjectStatus.PAUSED
    st = rig.state("a")
    assert st.status is TaskStatus.PENDING and st.attempts == 0
    assert rig.store.attempt_count(PID, "a") == 0
    # the user raises the budget (new process); the run resumes at attempt 1
    rig.rebuild(project_usd=5.0, per_call_usd=0.25)
    again = rig.run()
    assert again.status is ProjectStatus.DONE
    assert rig.store.attempt_count(PID, "a") == 1


def test_budget_stop_is_not_retried_within_run(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path, write("a") * 3, make_plan(task_dict("a")), project_usd=0.004, per_call_usd=0.004
    )
    rig.run()
    rig.run()
    assert rig.provider.requests == []  # the guard refuses before the provider every time


def test_provider_error_pauses_and_resume_redoes_attempt(tmp_path: Path) -> None:
    script = [calls(("write_file", {"path": "a.txt", "content": "ok"})), ProviderError("503")]
    rig = Rig(tmp_path, script, make_plan(task_dict("a")))
    report = rig.run()
    assert report.reason is StopReason.PROVIDER and report.status is ProjectStatus.PAUSED
    assert rig.store.attempt_count(PID, "a") == 0
    assert "503" in (rig.state("a").error or "")
    rig.provider.push(*write("a"))
    assert rig.run().status is ProjectStatus.DONE
    assert rig.store.attempt_count(PID, "a") == 1


def test_escalated_tier_survives_restart(tmp_path: Path) -> None:
    script = [*write("a", "bad"), *write("a", "bad"), ProviderError("down")]
    rig = Rig(tmp_path, script, make_plan(task_dict("a")))
    assert rig.run().reason is StopReason.PROVIDER
    assert rig.state("a").tier is Tier.LARGE
    rig.rebuild()
    rig.provider.push(*write("a"))
    n = len(rig.provider.requests)
    assert rig.run().status is ProjectStatus.DONE
    assert set(rig.models()[n:]) == {"fake-large"}


class _PubArgs(ToolArgs):
    text: str = ""


class _Publish(Tool):
    name: ClassVar[str] = "publish"
    description: ClassVar[str] = "Publish to the internet."
    risk: ClassVar[RiskLevel] = RiskLevel.EXTERNAL
    args_model: ClassVar[type[ToolArgs]] = _PubArgs
    runs: ClassVar[list[str]] = []

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        assert isinstance(args, _PubArgs)
        _Publish.runs.append(args.text)
        return "published"


def _pub_tools() -> list[Tool]:
    return [*default_tools(), _Publish()]


def _pub_roles() -> RoleRegistry:
    publisher = Role(
        name="publisher",
        description="Writes and publishes.",
        tools=("write_file", "publish"),
        max_risk=RiskLevel.WRITE,
        instructions="Publish the text, then write the file.",
    )
    return RoleRegistry([publisher], tools=_pub_tools())


def pub_rig(tmp_path: Path, script: list[Any]) -> Rig:
    _Publish.runs = []
    return Rig(
        tmp_path,
        script,
        make_plan(task_dict("a", role="publisher")),
        roles=_pub_roles(),
        tools=_pub_tools,
    )


def publish(text: str = "hi") -> ChatResponse:
    return calls(("publish", {"text": text}))


def test_external_tool_pauses_for_approval(tmp_path: Path) -> None:
    rig = pub_rig(tmp_path, [publish()])
    report = rig.run()
    assert report.reason is StopReason.APPROVAL and report.status is ProjectStatus.PAUSED
    assert "publish" in report.detail
    assert _Publish.runs == []
    assert rig.store.attempt_count(PID, "a") == 0
    assert rig.state("a").status is TaskStatus.PENDING
    # The request is recorded with exactly what would run.
    pending = rig.store.approvals(PID).list(ApprovalStatus.PENDING)
    assert [r.id for r in pending] == [report.approval_id]
    req = pending[0]
    assert (req.task_id, req.tool, req.arguments) == ("a", "publish", {"text": "hi"})
    paused = [e for e in rig.store.events(PID).read() if e.type == "orchestrator.paused"]
    assert paused[-1].data["approval"] == req.id


def test_unlisted_external_tool_is_denied_not_gated(tmp_path: Path) -> None:
    _Publish.runs = []
    rig = Rig(
        tmp_path,
        [publish(), *write("a")],
        make_plan(task_dict("a")),  # writer: publish is not in its tools
        tools=_pub_tools,
    )
    assert rig.run().status is ProjectStatus.DONE
    assert _Publish.runs == []
    assert rig.store.approvals(PID).list() == []


def test_resume_without_decision_reuses_the_pending_request(tmp_path: Path) -> None:
    rig = pub_rig(tmp_path, [publish(), publish()])
    first = rig.run()
    second = rig.run()
    assert second.reason is StopReason.APPROVAL
    assert second.approval_id == first.approval_id
    assert len(rig.store.approvals(PID).list()) == 1
    assert _Publish.runs == []


def test_approved_call_runs_once_on_resume(tmp_path: Path) -> None:
    rig = pub_rig(tmp_path, [publish(), publish(), *write("a")])
    report = rig.run()
    approvals = rig.store.approvals(PID)
    assert report.approval_id is not None
    approvals.decide(report.approval_id, approve=True)
    assert rig.run().status is ProjectStatus.DONE
    assert _Publish.runs == ["hi"]
    assert approvals.is_used(report.approval_id)
    assert "tool.approved" in rig.event_types()


def test_approval_is_consumed_so_a_repeat_needs_a_new_one(tmp_path: Path) -> None:
    rig = pub_rig(tmp_path, [publish(), publish(), publish()])
    first = rig.run()
    assert first.approval_id is not None
    rig.store.approvals(PID).decide(first.approval_id, approve=True)
    second = rig.run()  # runs once, then the same call again needs a new approval
    assert _Publish.runs == ["hi"]
    assert second.reason is StopReason.APPROVAL
    assert second.approval_id not in (None, first.approval_id)


def test_approval_does_not_cover_other_arguments(tmp_path: Path) -> None:
    rig = pub_rig(tmp_path, [publish("hi"), publish("something else")])
    first = rig.run()
    assert first.approval_id is not None
    rig.store.approvals(PID).decide(first.approval_id, approve=True)
    second = rig.run()
    assert _Publish.runs == []
    assert second.reason is StopReason.APPROVAL
    assert second.approval_id != first.approval_id
    assert not rig.store.approvals(PID).is_used(first.approval_id)


def test_rejected_call_never_runs_and_the_model_is_told(tmp_path: Path) -> None:
    rig = pub_rig(tmp_path, [publish(), publish(), *write("a")])
    report = rig.run()
    assert report.approval_id is not None
    rig.store.approvals(PID).decide(report.approval_id, approve=False, note="not public yet")
    n = len(rig.provider.requests)
    assert rig.run().status is ProjectStatus.DONE
    assert _Publish.runs == []
    told = rig.provider.requests[n + 1].messages[-1].content or ""
    assert "rejected" in told and "not public yet" in told
    assert "tool.rejected" in rig.event_types()


def eval_reply(passed: bool, reason: str = "") -> str:
    return json.dumps({"criteria": [{"criterion": 0, "passed": passed, "reason": reason}]})


def judged_task(tid: str = "a") -> dict:
    return task_dict(tid, acceptance_criteria=[{"description": "reads well"}])


def test_verification_failure_pauses_and_resume_reverifies(tmp_path: Path) -> None:
    script = [*write("a"), "not json", "still not json"]
    rig = Rig(tmp_path, script, make_plan(judged_task()))
    report = rig.run()
    assert report.reason is StopReason.VERIFICATION and report.status is ProjectStatus.PAUSED
    assert rig.store.attempt_count(PID, "a") == 1
    assert rig.store.load_verdict(PID, "a", 1) is None
    assert rig.state("a").status is TaskStatus.VERIFYING
    # resume: the stored result is judged again; the worker does NOT run again
    rig.provider.push(eval_reply(True))
    n = len(rig.provider.requests)
    again = rig.run()
    assert again.status is ProjectStatus.DONE and again.attempts_run == 0
    assert len(rig.provider.requests) == n + 1
    assert rig.store.attempt_count(PID, "a") == 1


def test_model_rejection_feeds_reason_back(tmp_path: Path) -> None:
    script = [*write("a"), eval_reply(False, "too terse"), *write("a"), eval_reply(True)]
    rig = Rig(tmp_path, script, make_plan(judged_task()))
    assert rig.run().status is ProjectStatus.DONE
    retry_user = rig.provider.requests[3].messages[1].content or ""
    assert "too terse" in retry_user


def test_unexpected_error_pauses_project_and_propagates(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [KeyboardInterrupt()], make_plan(task_dict("a")))
    with pytest.raises(KeyboardInterrupt):
        rig.run()
    assert rig.project_status() is ProjectStatus.PAUSED
    assert rig.state("a").status is TaskStatus.PENDING
    rig.provider.push(*write("a"))
    assert rig.run().status is ProjectStatus.DONE


def test_config_error_for_unmapped_tier_pauses(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [], make_plan(task_dict("a", tier="small")), tiers={"medium": "fake-m"})
    report = rig.run()
    assert report.reason is StopReason.CONFIG and report.status is ProjectStatus.PAUSED
    assert rig.provider.requests == []


def test_done_project_rerun_is_noop(tmp_path: Path) -> None:
    rig = Rig(tmp_path, write("a"), make_plan(task_dict("a")))
    rig.run()
    n = len(rig.provider.requests)
    report = rig.run()
    assert report.status is ProjectStatus.DONE and report.attempts_run == 0
    assert len(rig.provider.requests) == n


def test_resume_after_crash_mid_attempt_state(tmp_path: Path) -> None:
    """A state left 'running' (process killed) is simply picked up again."""
    rig = Rig(tmp_path, write("a"), make_plan(task_dict("a")))
    st = rig.state("a")
    rig.store.save_task_state(PID, st.model_copy(update={"status": TaskStatus.RUNNING}))
    assert rig.run().status is ProjectStatus.DONE


def test_ledger_records_every_call_for_project(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [*write("a", "bad"), *write("a")], make_plan(task_dict("a")))
    rig.run()
    total = sum(r.cost_nusd for r, _ in rig.store.task_history(PID, "a"))
    assert total > 0
    assert rig.ledger.spent_nusd(project_id=PID) == total


# --- structure ------------------------------------------------------------------


def test_orchestrator_never_calls_models_or_tools_directly() -> None:
    import automa.orchestrator as mod

    tree = ast.parse(Path(mod.__file__).read_text())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "complete" not in attrs  # no direct model call
    assert "execute" not in attrs  # tools only via agents' executors
    assert "PermissionPolicy" not in {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }  # never builds its own policy
    assert "LLMGateway" not in names and "FakeProvider" not in names


# -- irreversible tools (delete_file) ------------------------------------------


def _cleaner_roles() -> RoleRegistry:
    cleaner = Role(
        name="cleaner",
        description="Removes a stale file, then writes the result.",
        tools=("write_file", "delete_file"),
        max_risk=RiskLevel.WRITE,
        instructions="Delete old.txt, then write the file.",
    )
    return RoleRegistry([cleaner], tools=default_tools())


def delete(path: str = "old.txt") -> ChatResponse:
    return calls(("delete_file", {"path": path}))


def cleaner_rig(tmp_path: Path, script: list[Any]) -> Rig:
    rig = Rig(tmp_path, script, make_plan(task_dict("a", role="cleaner")), roles=_cleaner_roles())
    old = rig.store.project_dir(PID) / "files" / "old.txt"
    old.write_text("keep me until approved")
    return rig


def test_irreversible_tool_pauses_even_within_risk_limit(tmp_path: Path) -> None:
    rig = cleaner_rig(tmp_path, [delete()])
    old = rig.store.project_dir(PID) / "files" / "old.txt"
    report = rig.run()
    assert report.reason is StopReason.APPROVAL and report.status is ProjectStatus.PAUSED
    assert old.exists()
    (req,) = rig.store.approvals(PID).list(ApprovalStatus.PENDING)
    assert (req.tool, req.arguments, req.risk) == ("delete_file", {"path": "old.txt"}, "write")
    assert "irreversible" in req.reason
    gated = [e for e in rig.store.events(PID).read() if e.type == "tool.approval_required"]
    assert gated[-1].data["irreversible"] is True


def test_irreversible_tool_runs_once_after_approval(tmp_path: Path) -> None:
    rig = cleaner_rig(tmp_path, [delete(), delete(), *write("a")])
    old = rig.store.project_dir(PID) / "files" / "old.txt"
    report = rig.run()
    assert report.approval_id is not None
    rig.store.approvals(PID).decide(report.approval_id, approve=True)
    assert rig.run().status is ProjectStatus.DONE
    assert not old.exists()
    assert rig.store.approvals(PID).is_used(report.approval_id)


def test_rejected_irreversible_call_keeps_the_file(tmp_path: Path) -> None:
    rig = cleaner_rig(tmp_path, [delete(), delete(), *write("a")])
    old = rig.store.project_dir(PID) / "files" / "old.txt"
    report = rig.run()
    assert report.approval_id is not None
    rig.store.approvals(PID).decide(report.approval_id, approve=False)
    assert rig.run().status is ProjectStatus.DONE
    assert old.read_text() == "keep me until approved"
