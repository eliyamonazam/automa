"""Tests for ExecutiveAgent (automa.agents.executive).

Local only: FakeProvider behind a real LLMGateway + BudgetGuard + CostLedger.
No network, no real model calls.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from automa.agents import (
    Clarification,
    ExecutiveAgent,
    ExecutiveError,
    ExecutiveSettings,
    PlanningFailed,
    parse_questions,
    plan_project,
)
from automa.agents.executive import MAX_ECHO_CHARS
from automa.budget import BudgetExceeded, BudgetGuard, CostLedger
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
    Role,
    Usage,
)
from automa.plan import PlanError
from automa.project import ProjectStatus, ProjectStore

T0 = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)


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


def task(tid: str, *deps: str, **kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": tid,
        "title": f"Task {tid}",
        "role": "developer",
        "depends_on": list(deps),
        "acceptance_criteria": [{"description": "works"}],
    }
    d.update(kw)
    return d


def plan_json(*tasks: dict[str, Any], **kw: Any) -> str:
    d: dict[str, Any] = {"goal": "whatever the model says", "tasks": list(tasks)}
    d.update(kw)
    return json.dumps(d)


VALID = plan_json(task("a"), task("b", "a"))


class Rig:
    def __init__(self, tmp_path: Path, script: list[Any], **limits: float):
        self.cfg = config(**limits)
        self.provider = FakeProvider(script)
        self.ledger = CostLedger(tmp_path / "ledger.sqlite3")
        self.guard = BudgetGuard(self.cfg.limits, self.ledger)
        self.gateway = LLMGateway(self.cfg, self.provider, guard=self.guard)
        self.events = MemoryEventSink()

    def agent(self, **kw: Any) -> ExecutiveAgent:
        kw.setdefault("events", self.events)
        return ExecutiveAgent(self.gateway, clock=clock, **kw)


@pytest.fixture
def store(tmp_path: Path) -> ProjectStore:
    return ProjectStore.for_workspace(tmp_path / "ws")


# --- construction ------------------------------------------------------------


def test_requires_a_real_gateway(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [])
    with pytest.raises(TypeError):
        ExecutiveAgent(rig.provider)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ExecutiveAgent(rig.guard)  # type: ignore[arg-type]


def test_empty_known_roles_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Rig(tmp_path, []).agent(known_roles=[])


def test_settings_bounds() -> None:
    with pytest.raises(ValueError):
        ExecutiveSettings(max_attempts=0)
    with pytest.raises(ValueError):
        ExecutiveSettings(max_attempts=6)
    with pytest.raises(ValueError):
        ExecutiveSettings(plan_max_tokens=0)
    assert ExecutiveSettings().tier is Tier.LARGE


@pytest.mark.parametrize("goal", ["", "   ", "\n\t"])
def test_blank_goal_rejected_without_model_call(tmp_path: Path, goal: str) -> None:
    rig = Rig(tmp_path, [VALID])
    agent = rig.agent()
    with pytest.raises(ValueError):
        agent.plan(goal, project_id="p1")
    with pytest.raises(ValueError):
        agent.clarify(goal, project_id="p1")
    assert rig.provider.requests == []


# --- planning ----------------------------------------------------------------


def test_plan_valid_first_try(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [VALID])
    plan = rig.agent().plan("  build a CLI  ", project_id="p1")
    assert plan.task_ids == ["a", "b"]
    assert plan.goal == "build a CLI"  # set by code, not by the model
    assert plan.version == 1
    assert len(rig.provider.requests) == 1
    req = rig.provider.requests[0]
    assert req.model == "fake-large"
    assert req.max_tokens == ExecutiveSettings().plan_max_tokens
    assert req.messages[0].role is Role.SYSTEM
    assert "build a CLI" in (req.messages[1].content or "")
    assert len(rig.ledger.entries()) == 1
    assert rig.ledger.entries()[0].project_id == "p1"
    ready = rig.events.of_type("executive.plan_ready")
    assert ready and ready[0].data == {"version": 1, "tasks": 2}


def test_plan_accepts_code_fence(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [f"```json\n{VALID}\n```"])
    assert rig.agent().plan("g", project_id="p1").task_ids == ["a", "b"]


def test_model_cannot_set_version(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [plan_json(task("a"), version=99)])
    assert rig.agent().plan("g", project_id="p1", version=2).version == 2


def test_version_must_be_positive(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [VALID])
    with pytest.raises(ValueError):
        rig.agent().plan("g", project_id="p1", version=0)
    assert rig.provider.requests == []


def test_custom_tier_and_tokens(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [VALID])
    s = ExecutiveSettings(tier=Tier.SMALL, plan_max_tokens=1234, temperature=0.0)
    rig.agent(settings=s).plan("g", project_id="p1")
    req = rig.provider.requests[0]
    assert (req.model, req.max_tokens, req.temperature) == ("fake-small", 1234, 0.0)


def test_clarifications_are_sent(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [VALID])
    rig.agent().plan(
        "g",
        project_id="p1",
        clarifications=[
            Clarification(question="Which language?", answer="Python"),
            Clarification(question="Deadline?", answer=""),
        ],
    )
    user = rig.provider.requests[0].messages[1].content or ""
    assert "Q: Which language?\nA: Python" in user
    assert "A: (no answer)" in user


def test_clarification_dicts_are_validated(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [VALID])
    with pytest.raises(ValueError):
        rig.agent().plan(
            "g",
            project_id="p1",
            clarifications=[{"question": "", "answer": "x"}],  # type: ignore[list-item]
        )
    assert rig.provider.requests == []


@pytest.mark.parametrize(
    "bad",
    [
        "not json at all",
        "[1, 2, 3]",
        plan_json(),  # no tasks
        plan_json(task("a", "b"), task("b", "a")),  # cycle
        plan_json(task("a", "zzz")),  # unknown dependency
        plan_json(task("a"), task("a")),  # duplicate id
        plan_json(task("../etc")),  # unsafe id
        plan_json(task("a", acceptance_criteria=[])),  # no criteria
        plan_json(
            task(
                "a",
                acceptance_criteria=[
                    {"description": "x", "check": {"type": "file_exists", "path": "../x"}}
                ],
            )
        ),
        plan_json(task("a"), extra_field=1),
        "",
    ],
)
def test_invalid_then_valid_is_retried_with_feedback(tmp_path: Path, bad: str) -> None:
    rig = Rig(tmp_path, [bad, VALID])
    plan = rig.agent().plan("g", project_id="p1")
    assert plan.task_ids == ["a", "b"]
    assert len(rig.provider.requests) == 2
    retry = rig.provider.requests[1].messages
    # system, user, the rejected reply, feedback
    assert [m.role for m in retry] == [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.USER]
    assert "rejected" in (retry[3].content or "")
    rejected = rig.events.of_type("executive.reply_rejected")
    assert len(rejected) == 1 and rejected[0].data["attempt"] == 1
    assert rejected[0].data["kind"] == "plan"
    assert rejected[0].data["details"]
    assert len(rig.ledger.entries()) == 2  # every attempt is paid for and recorded


def test_cycle_feedback_names_the_cycle(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [plan_json(task("a", "b"), task("b", "a")), VALID])
    rig.agent().plan("g", project_id="p1")
    assert "cycle" in (rig.provider.requests[1].messages[3].content or "")


def test_gives_up_after_max_attempts(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["nope"] * 10)
    with pytest.raises(PlanningFailed) as ei:
        rig.agent(settings=ExecutiveSettings(max_attempts=3)).plan("g", project_id="p1")
    assert ei.value.attempts == 3
    assert "invalid JSON" in ei.value.details
    assert len(rig.provider.requests) == 3  # never more calls than max_attempts
    assert rig.provider.remaining == 7
    assert len(rig.events.of_type("executive.reply_rejected")) == 3
    assert rig.events.of_type("executive.plan_ready") == []


def test_single_attempt_setting(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["nope", VALID])
    with pytest.raises(PlanningFailed):
        rig.agent(settings=ExecutiveSettings(max_attempts=1)).plan("g", project_id="p1")
    assert len(rig.provider.requests) == 1


def test_history_grows_per_attempt(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["x", "y", VALID])
    rig.agent().plan("g", project_id="p1")
    assert [len(r.messages) for r in rig.provider.requests] == [2, 4, 6]


def test_long_rejected_reply_is_truncated_in_echo(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["x" * (MAX_ECHO_CHARS * 3), VALID])
    rig.agent().plan("g", project_id="p1")
    echoed = rig.provider.requests[1].messages[2].content or ""
    assert len(echoed) < MAX_ECHO_CHARS + 50
    assert echoed.endswith("(truncated)")


def test_truncated_reply_is_rejected_even_if_valid(tmp_path: Path) -> None:
    cut = ChatResponse(
        model="fake-large",
        message=Message.assistant(VALID),
        usage=Usage(input_tokens=10, output_tokens=10),
        finish_reason=FinishReason.LENGTH,
    )
    rig = Rig(tmp_path, [cut, VALID])
    rig.agent().plan("g", project_id="p1")
    assert len(rig.provider.requests) == 2
    assert "cut off" in (rig.provider.requests[1].messages[3].content or "")


def test_known_roles_enforced(tmp_path: Path) -> None:
    bad = plan_json(task("a", role="hacker"))
    rig = Rig(tmp_path, [bad, VALID])
    plan = rig.agent(known_roles={"developer", "writer"}).plan("g", project_id="p1")
    assert plan.task_ids == ["a", "b"]
    first_system = rig.provider.requests[0].messages[0].content or ""
    assert "Allowed roles: developer, writer" in first_system
    feedback = rig.provider.requests[1].messages[3].content or ""
    assert "hacker" in feedback and "developer" in feedback


def test_known_roles_exhausted(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [plan_json(task("a", role="hacker"))] * 3)
    with pytest.raises(PlanningFailed) as ei:
        rig.agent(known_roles={"developer"}).plan("g", project_id="p1")
    assert "hacker" in ei.value.details


def test_any_role_allowed_without_known_roles(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [plan_json(task("a", role="custom-role"))])
    assert rig.agent().plan("g", project_id="p1").tasks[0].role == "custom-role"


def test_no_events_sink_is_fine(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["x", VALID])
    assert rig.agent(events=None).plan("g", project_id="p1").task_ids == ["a", "b"]


# --- gateway errors are never retried ----------------------------------------


def test_budget_stop_propagates_without_retry(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [VALID] * 3, per_call_usd=0.000001, daily_usd=1.0, project_usd=1.0)
    with pytest.raises(BudgetExceeded):
        rig.agent().plan("g", project_id="p1")
    assert rig.provider.requests == []  # refused before reaching the provider
    assert rig.ledger.entries() == []


def test_budget_stop_mid_retries_propagates(tmp_path: Path) -> None:
    # The first (invalid) reply is expensive; the retry's worst case no longer fits the cap.
    costly_bad = ChatResponse(
        model="fake-large",
        message=Message.assistant("bad"),
        usage=Usage(input_tokens=10, output_tokens=10, reported_cost_usd=0.02),
    )
    rig = Rig(tmp_path, [costly_bad, VALID], project_usd=0.03, daily_usd=1.0, per_call_usd=0.03)
    with pytest.raises(BudgetExceeded):
        rig.agent(settings=ExecutiveSettings(plan_max_tokens=8_000)).plan("g", project_id="p1")
    assert len(rig.provider.requests) == 1
    assert len(rig.ledger.entries()) == 1


def test_provider_error_propagates_without_retry(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [ProviderError("boom", retryable=True), VALID])
    with pytest.raises(ProviderError):
        rig.agent().plan("g", project_id="p1")
    assert len(rig.provider.requests) == 1
    assert rig.provider.remaining == 1


def test_provider_error_on_retry_propagates(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["bad", ProviderError("down"), VALID])
    with pytest.raises(ProviderError):
        rig.agent().plan("g", project_id="p1")
    assert len(rig.provider.requests) == 2


# --- clarifying questions ----------------------------------------------------


def test_clarify_returns_questions(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ['{"questions": ["  Which OS?  ", "Budget?"]}'])
    qs = rig.agent().clarify("make an app", project_id="p1")
    assert qs == ["Which OS?", "Budget?"]
    req = rig.provider.requests[0]
    assert req.max_tokens == ExecutiveSettings().questions_max_tokens
    assert "make an app" in (req.messages[1].content or "")
    assert rig.events.of_type("executive.questions")[0].data == {"count": 2}


def test_clarify_no_questions(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ['```json\n{"questions": []}\n```'])
    assert rig.agent().clarify("g", project_id="p1") == []


@pytest.mark.parametrize(
    "bad",
    [
        "Sure! Here are my questions",
        '["a"]',
        '{"questions": ["a", "b", "c", "d", "e", "f"]}',
        '{"questions": [""]}',
        '{"questions": "one"}',
        '{"questions": [], "extra": 1}',
        json.dumps({"questions": ["x" * 501]}),
    ],
)
def test_clarify_invalid_is_retried(tmp_path: Path, bad: str) -> None:
    rig = Rig(tmp_path, [bad, '{"questions": ["ok?"]}'])
    assert rig.agent().clarify("g", project_id="p1") == ["ok?"]
    assert len(rig.provider.requests) == 2
    assert rig.events.of_type("executive.reply_rejected")[0].data["kind"] == "questions"


def test_clarify_gives_up(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["no"] * 5)
    with pytest.raises(PlanningFailed):
        rig.agent(settings=ExecutiveSettings(max_attempts=2)).clarify("g", project_id="p1")
    assert len(rig.provider.requests) == 2


def test_parse_questions_details() -> None:
    with pytest.raises(PlanError) as ei:
        parse_questions('{"questions": [1]}')
    assert ei.value.details.startswith("- questions.0")


# --- plan_project (store integration) ----------------------------------------


def test_plan_project_writes_plan(tmp_path: Path, store: ProjectStore) -> None:
    pr = store.create("build a CLI", project_id="p1")
    rig = Rig(tmp_path, ["bad", VALID])
    agent = ExecutiveAgent(rig.gateway, events=store.events(pr.id), clock=clock)
    plan = plan_project(agent, store, pr.id)
    assert store.load_plan(pr.id) == plan
    assert plan.goal == "build a CLI"
    assert store.load(pr.id).status is ProjectStatus.PLANNING
    types = [e.type for e in store.events(pr.id).read()]
    assert "executive.reply_rejected" in types
    assert types.index("executive.plan_ready") < types.index("plan.saved")


def test_plan_project_with_budget_events_in_same_log(tmp_path: Path, store: ProjectStore) -> None:
    pr = store.create("g", project_id="p1")
    cfg = config()
    provider = FakeProvider([VALID])
    guard = BudgetGuard(cfg.limits, CostLedger(tmp_path / "l.sqlite3"), events=store.events(pr.id))
    agent = ExecutiveAgent(LLMGateway(cfg, provider, guard=guard), events=store.events(pr.id))
    plan_project(agent, store, pr.id)
    assert store.has_plan(pr.id)


def test_plan_project_failure_marks_failed(tmp_path: Path, store: ProjectStore) -> None:
    pr = store.create("g", project_id="p1")
    rig = Rig(tmp_path, ["bad"] * 3)
    with pytest.raises(PlanningFailed):
        plan_project(rig.agent(), store, pr.id)
    assert store.load(pr.id).status is ProjectStatus.FAILED
    assert not store.has_plan(pr.id)


def test_plan_project_budget_stop_pauses(tmp_path: Path, store: ProjectStore) -> None:
    pr = store.create("g", project_id="p1")
    rig = Rig(tmp_path, [VALID], per_call_usd=0.000001, daily_usd=1.0, project_usd=1.0)
    with pytest.raises(BudgetExceeded):
        plan_project(rig.agent(), store, pr.id)
    assert store.load(pr.id).status is ProjectStatus.PAUSED
    assert not store.has_plan(pr.id)


def test_plan_project_other_error_restores_status(tmp_path: Path, store: ProjectStore) -> None:
    pr = store.create("g", project_id="p1")
    rig = Rig(tmp_path, [ProviderError("down")])
    with pytest.raises(ProviderError):
        plan_project(rig.agent(), store, pr.id)
    assert store.load(pr.id).status is ProjectStatus.NEW


def test_plan_project_refuses_existing_plan(tmp_path: Path, store: ProjectStore) -> None:
    pr = store.create("g", project_id="p1")
    rig = Rig(tmp_path, [VALID, VALID])
    plan_project(rig.agent(), store, pr.id)
    with pytest.raises(ExecutiveError):
        plan_project(rig.agent(), store, pr.id)
    assert len(rig.provider.requests) == 1


def test_plan_project_unknown_project(tmp_path: Path, store: ProjectStore) -> None:
    rig = Rig(tmp_path, [VALID])
    with pytest.raises(Exception, match="p404"):
        plan_project(rig.agent(), store, "p404")
    assert rig.provider.requests == []


def test_plan_project_passes_clarifications(tmp_path: Path, store: ProjectStore) -> None:
    pr = store.create("g", project_id="p1")
    seen: list[ChatRequest] = []

    def reply(req: ChatRequest) -> str:
        seen.append(req)
        return VALID

    rig = Rig(tmp_path, [reply])
    plan_project(
        rig.agent(), store, pr.id, clarifications=[Clarification(question="Q1?", answer="yes")]
    )
    assert "Q: Q1?\nA: yes" in (seen[0].messages[1].content or "")
