"""``ExecutiveAgent``: turns a user goal into clarifying questions and a valid ``Plan``.

Model output is untrusted. Every reply is parsed and validated in code
(``parse_plan`` / ``Questions``); an invalid reply is sent back to the model with a
short, precise error list and retried, at most ``max_attempts`` times. Retries are
only for *invalid output*: errors raised by the gateway (``BudgetExceeded``,
``ProviderError``, …) always propagate immediately, so a budget stop is never
worked around by asking again.

The agent never trusts fields that belong to the caller: the plan's ``goal`` and
``version`` are set by code, not by the model.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Sequence
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from automa.budget.guard import BudgetError
from automa.config import Tier
from automa.events import Event, EventSink
from automa.llm.gateway import CallContext, LLMGateway
from automa.llm.types import FinishReason, LLMResult, Message
from automa.plan.models import (
    Clarification,
    Plan,
    PlanError,
    QuestionText,
    parse_plan,
    strip_code_fence,
)
from automa.project.store import ProjectStatus, ProjectStore

MAX_QUESTIONS = 5
# How much of a rejected reply is echoed back to the model on retry (bounds cost).
MAX_ECHO_CHARS = 4_000

Clock = Callable[[], datetime]
T = TypeVar("T")


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ExecutiveError(Exception):
    """Base error of the Executive agent."""


class PlanningFailed(ExecutiveError):
    """The model did not produce valid output within ``max_attempts``."""

    def __init__(self, message: str, *, attempts: int, details: str):
        super().__init__(f"{message} after {attempts} attempt(s)")
        self.attempts = attempts
        self.details = details


class Questions(BaseModel):
    """Model reply to the clarification prompt. An empty list means "no questions"."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    questions: list[QuestionText] = Field(default_factory=list, max_length=MAX_QUESTIONS)


class ExecutiveSettings(BaseModel):
    """Bounds of the Executive. Spending itself is always bounded by the BudgetGuard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: Tier = Tier.LARGE
    max_attempts: int = Field(default=3, ge=1, le=5)
    plan_max_tokens: int = Field(default=8_000, gt=0, le=32_000)
    questions_max_tokens: int = Field(default=1_000, gt=0, le=8_000)
    temperature: float = Field(default=0.2, ge=0, le=1)


QUESTIONS_SYSTEM_PROMPT = f"""\
You are the Executive of an automation system. A user gives you a goal.
Decide whether you need clarifying questions before you can plan the work.
Ask only questions whose answers would change the plan; ask none if the goal is clear.
Reply with ONLY a JSON object: {{"questions": ["...", ...]}} (at most {MAX_QUESTIONS} \
questions, an empty list if none). No prose outside the JSON."""

PLAN_SYSTEM_PROMPT = """\
You are the Executive of an automation system. Break the user's goal into a plan of
small, independently verifiable tasks. Reply with ONLY one JSON object, no prose:

{
  "goal": "<the goal>",
  "tasks": [
    {
      "id": "lowercase-id",            // a-z, 0-9, '-' and '_'; unique
      "title": "short title",
      "description": "what to do",
      "role": "developer",             // a role name, see the allowed roles below
      "depends_on": ["other-id"],      // ids of tasks that must pass first; no cycles
      "inputs": {"key": "value"},      // optional
      "acceptance_criteria": [         // at least one per task
        {"description": "README.md exists",
         "check": {"type": "file_exists", "path": "README.md"}},
        {"description": "tests pass",
         "check": {"type": "command", "command": "pytest -q", "expect_exit_code": 0}},
        {"description": "explains installation"}
      ]
    }
  ]
}

Check types: file_exists {path}, file_contains {path, text}, command {command,
expect_exit_code, timeout_s}. Paths are relative to the project's files directory and
must not contain "..". Prefer deterministic checks wherever possible.
(The // comments above are explanations; JSON must not contain comments.)"""


def _goal_message(goal: str, clarifications: Sequence[Clarification]) -> str:
    parts = [f"Goal:\n{goal}"]
    if clarifications:
        qa = "\n".join(f"Q: {c.question}\nA: {c.answer or '(no answer)'}" for c in clarifications)
        parts.append(f"Clarifications from the user:\n{qa}")
    return "\n\n".join(parts)


def _echo(text: str) -> str:
    if len(text) <= MAX_ECHO_CHARS:
        return text
    return text[:MAX_ECHO_CHARS] + "\n…(truncated)"


def _retry_message(details: str) -> str:
    return (
        "Your previous reply was rejected:\n"
        f"{details}\n"
        "Reply again with ONLY the corrected JSON object."
    )


def parse_questions(text: str) -> Questions:
    """Parse the clarification reply. Raises ``PlanError`` with model-readable details."""
    raw = strip_code_fence(text)
    try:
        data: Any = json.loads(raw)
    except ValueError as e:
        raise PlanError("questions are not valid JSON", f"- <root>: invalid JSON ({e})") from e
    if not isinstance(data, dict):
        raise PlanError("questions must be a JSON object", "- <root>: expected a JSON object")
    try:
        return Questions.model_validate(data)
    except ValidationError as e:
        lines = []
        for err in e.errors(include_url=False, include_input=False)[:20]:
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            lines.append(f"- {loc}: {err['msg']}")
        raise PlanError("questions failed validation", "\n".join(lines)) from e


class ExecutiveAgent:
    """Plans a goal through the (guarded) ``LLMGateway``.

    ``known_roles``: if given, every task's role must be in it (otherwise the reply is
    rejected and retried). ``events``: optional sink for ``executive.*`` events.
    """

    def __init__(
        self,
        gateway: LLMGateway,
        *,
        settings: ExecutiveSettings | None = None,
        known_roles: Collection[str] | None = None,
        events: EventSink | None = None,
        clock: Clock = _utc_now,
    ):
        if not isinstance(gateway, LLMGateway):
            raise TypeError("ExecutiveAgent requires an LLMGateway (all calls are guarded)")
        if known_roles is not None and not known_roles:
            raise ValueError("known_roles must not be empty (use None to allow any role)")
        self._gateway = gateway
        self._settings = settings or ExecutiveSettings()
        self._roles = frozenset(known_roles) if known_roles is not None else None
        self._events = events
        self._clock = clock

    @property
    def settings(self) -> ExecutiveSettings:
        return self._settings

    # -- public API --------------------------------------------------------

    def clarify(self, goal: str, *, project_id: str) -> list[str]:
        """Return clarifying questions for ``goal`` (empty if none are needed)."""
        goal = self._check_goal(goal)
        result = self._run(
            system=QUESTIONS_SYSTEM_PROMPT,
            user=_goal_message(goal, ()),
            max_tokens=self._settings.questions_max_tokens,
            project_id=project_id,
            parse=parse_questions,
            kind="questions",
        )
        questions = list(result.questions)
        self._emit("executive.questions", project_id, {"count": len(questions)})
        return questions

    def plan(
        self,
        goal: str,
        *,
        project_id: str,
        clarifications: Sequence[Clarification] = (),
        version: int = 1,
    ) -> Plan:
        """Produce a validated ``Plan`` for ``goal``. Raises ``PlanningFailed``."""
        goal = self._check_goal(goal)
        if version < 1:
            raise ValueError("plan version must be >= 1")
        clarifications = [Clarification.model_validate(c) for c in clarifications]

        def parse(text: str) -> Plan:
            plan = parse_plan(text)
            if self._roles is not None:
                unknown = sorted({t.role for t in plan.tasks} - self._roles)
                if unknown:
                    allowed = ", ".join(sorted(self._roles))
                    raise PlanError(
                        "plan uses unknown roles",
                        f"- tasks.role: unknown role(s) {', '.join(unknown)}; allowed: {allowed}",
                    )
            # goal and version belong to the caller, never to the model.
            return Plan.model_validate(
                {**plan.model_dump(mode="json"), "goal": goal, "version": version}
            )

        system = PLAN_SYSTEM_PROMPT
        if self._roles is not None:
            system += f"\nAllowed roles: {', '.join(sorted(self._roles))}."
        plan = self._run(
            system=system,
            user=_goal_message(goal, clarifications),
            max_tokens=self._settings.plan_max_tokens,
            project_id=project_id,
            parse=parse,
            kind="plan",
        )
        self._emit(
            "executive.plan_ready",
            project_id,
            {"version": plan.version, "tasks": len(plan.tasks)},
        )
        return plan

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _check_goal(goal: str) -> str:
        goal = goal.strip()
        if not goal:
            raise ValueError("goal must not be empty")
        return goal

    def _run(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        project_id: str,
        parse: Callable[[str], T],
        kind: str,
    ) -> T:
        messages = [Message.system(system), Message.user(user)]
        context = CallContext(project_id=project_id)
        details = ""
        for attempt in range(1, self._settings.max_attempts + 1):
            # Gateway errors (budget stop, provider failure) propagate: never retried here.
            result: LLMResult = self._gateway.complete(
                messages,
                tier=self._settings.tier,
                max_tokens=max_tokens,
                temperature=self._settings.temperature,
                context=context,
            )
            text = result.text
            try:
                if result.response.finish_reason is FinishReason.LENGTH:
                    raise PlanError(
                        "reply truncated",
                        "- <root>: reply was cut off at the output limit; "
                        "reply with a shorter, complete JSON object",
                    )
                return parse(text)
            except PlanError as e:
                details = e.details or str(e)
                self._emit(
                    "executive.reply_rejected",
                    project_id,
                    {"kind": kind, "attempt": attempt, "details": details},
                )
                messages += [
                    Message.assistant(_echo(text) or "(empty reply)"),
                    Message.user(_retry_message(details)),
                ]
        raise PlanningFailed(
            f"no valid {kind} from the model",
            attempts=self._settings.max_attempts,
            details=details,
        )

    def _emit(self, type_: str, project_id: str, data: dict[str, Any]) -> None:
        if self._events is not None:
            self._events.emit(Event(ts=self._clock(), type=type_, project_id=project_id, data=data))


def plan_project(
    agent: ExecutiveAgent,
    store: ProjectStore,
    project_id: str,
    *,
    clarifications: Sequence[Clarification] = (),
) -> Plan:
    """Plan a stored project and write ``plan.json``.

    Status: ``planning`` while working and on success (the Orchestrator moves it on);
    ``failed`` if the model never produced a valid plan; ``paused`` on a budget stop
    (waiting for the user); restored to the previous status on any other error.
    A project that already has a plan is refused (re-planning is a separate step).
    """
    project = store.load(project_id)
    if store.has_plan(project_id):
        raise ExecutiveError(f"project {project_id!r} already has a plan")
    previous = project.status
    store.set_status(project_id, ProjectStatus.PLANNING)
    try:
        plan = agent.plan(project.goal, project_id=project_id, clarifications=clarifications)
        return store.save_plan(project_id, plan)
    except PlanningFailed:
        store.set_status(project_id, ProjectStatus.FAILED)
        raise
    except BudgetError:
        store.set_status(project_id, ProjectStatus.PAUSED)
        raise
    except BaseException:
        store.set_status(project_id, previous)
        raise
