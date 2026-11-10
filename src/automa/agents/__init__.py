"""Agents of the loop (see docs/ARCHITECTURE.md). Every agent calls models only
through the guarded ``LLMGateway``."""

from automa.agents.executive import (
    Clarification,
    ExecutiveAgent,
    ExecutiveError,
    ExecutiveSettings,
    PlanningFailed,
    Questions,
    parse_questions,
    plan_project,
)
from automa.agents.verifier import (
    Evaluation,
    VerificationFailed,
    VerifierAgent,
    VerifierError,
    VerifierSettings,
    parse_evaluation,
    run_checks,
    run_criteria_checks,
    verifier_policy,
)
from automa.agents.worker import WorkerAgent, WorkerSettings, task_message

__all__ = [
    "Clarification",
    "Evaluation",
    "ExecutiveAgent",
    "ExecutiveError",
    "ExecutiveSettings",
    "PlanningFailed",
    "Questions",
    "VerificationFailed",
    "VerifierAgent",
    "VerifierError",
    "VerifierSettings",
    "WorkerAgent",
    "WorkerSettings",
    "parse_evaluation",
    "parse_questions",
    "plan_project",
    "run_checks",
    "run_criteria_checks",
    "task_message",
    "verifier_policy",
]
