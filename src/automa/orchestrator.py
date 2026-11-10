"""``Orchestrator``: the main loop — run, verify, retry with feedback, escalate tiers.

One ``run(project_id)`` drives a planned project until it is ``done``, ``failed`` or
``paused``:

1. pick the first task (plan order) that is not finished and whose dependencies
   all ``passed``;
2. if its last stored attempt has no verdict (e.g. a crash, or the verifier gave no
   valid evaluation), verify that stored result again — never re-run the worker for it;
3. otherwise run a new worker attempt (with the issues of the previous verdict as
   feedback), store the ``TaskResult``, verify it, store the ``Verdict``;
4. ``pass`` → task ``passed``; ``fail`` → retry, escalating to the next configured
   tier after ``escalate_after`` consecutive rejections at one tier; after
   ``max_attempts`` the task ``failed`` and every task depending on it is ``blocked``.

State is saved after every step (``tasks/<id>/state.json`` + write-once history), so
a run can be stopped at any point and resumed by calling ``run`` again.

What the orchestrator never does:

- mark a task passed by itself: only a stored ``pass`` verdict can (enforced by the
  ``ProjectStore``, D22);
- retry or work around a budget stop: ``BudgetError`` pauses the project at once;
- run a tool that needs approval without it: ``ApprovalRequired`` records a request
  in ``approvals/`` and pauses the project; only a user decision for exactly that
  call (same task, tool and arguments) lets it run once on a later run;
- hand an agent wider permissions: executors are built only from ``role.policy()``
  and ``verifier_policy()`` (and the agents re-check that).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from automa.agents.verifier import VerificationFailed, VerifierAgent, verifier_policy
from automa.agents.worker import WorkerAgent
from automa.budget.guard import BudgetError
from automa.config import AutomaConfig, ConfigError, Tier
from automa.events import Event
from automa.llm.provider import ProviderError
from automa.plan.models import (
    MAX_TASK_ATTEMPTS,
    Issue,
    Plan,
    Task,
    TaskResult,
    TaskState,
    TaskStatus,
    Verdict,
)
from automa.project.store import ProjectStatus, ProjectStore
from automa.roles.registry import RoleError, RoleRegistry
from automa.tools import default_tools
from automa.tools.base import Tool
from automa.tools.executor import ApprovalRequired, ToolExecutor, ToolLayerError
from automa.tools.policy import PermissionPolicy
from automa.tools.workspace import Workspace

Clock = Callable[[], datetime]

# Escalation order of tiers (lowest first).
TIER_ORDER: tuple[Tier, ...] = (Tier.SMALL, Tier.MEDIUM, Tier.LARGE)

MAX_ERROR_CHARS = 2_000


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _short(text: str, limit: int = MAX_ERROR_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + " [...]"


class OrchestratorError(Exception):
    """The orchestrator cannot run this project (no plan, inconsistent state, …)."""


class OrchestratorSettings(BaseModel):
    """Retry/escalation bounds. Spending itself is always bounded by the BudgetGuard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Total attempts per task (all tiers together) before it is marked failed.
    max_attempts: int = Field(default=4, ge=1, le=MAX_TASK_ATTEMPTS)
    # Consecutive rejections at one tier before moving to the next configured tier.
    escalate_after: int = Field(default=2, ge=1, le=MAX_TASK_ATTEMPTS)


class StopReason(StrEnum):
    COMPLETED = "completed"  # every task passed
    TASKS_FAILED = "tasks_failed"  # some task failed for good (dependents blocked)
    BUDGET = "budget"  # BudgetGuard stop (or budget layer failure)
    APPROVAL = "approval"  # a tool call needs the user's approval
    PROVIDER = "provider"  # model provider error (not retried here)
    VERIFICATION = "verification"  # verifier produced no valid evaluation
    TOOL_LAYER = "tool_layer"  # the tool layer itself failed
    CONFIG = "config"  # e.g. no model configured for the task's tier


class RunReport(BaseModel):
    """Outcome of one ``Orchestrator.run``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    status: ProjectStatus
    reason: StopReason
    detail: str = ""
    task_id: str | None = None  # the task being worked on when the run stopped
    approval_id: str | None = None  # pending Approval Gate request (reason ``approval``)
    attempts_run: int = 0  # worker attempts started in this run
    passed: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)


class _Stop(Exception):
    """Internal: stop the run and pause the project with ``reason``."""

    def __init__(
        self, reason: StopReason, detail: str, task_id: str, approval_id: str | None = None
    ) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.task_id = task_id
        self.approval_id = approval_id


_UNFINISHED = frozenset({TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.VERIFYING})


def next_tier(tier: Tier, available: frozenset[Tier] | set[Tier]) -> Tier | None:
    """The lowest configured tier above ``tier``, or ``None`` if there is none."""
    higher = TIER_ORDER[TIER_ORDER.index(tier) + 1 :]
    return next((t for t in higher if t in available), None)


def rejections_at_tier(history: Sequence[tuple[TaskResult, Verdict | None]], tier: Tier) -> int:
    """Consecutive rejected attempts at ``tier`` at the end of ``history``."""
    n = 0
    for result, verdict in reversed(history):
        if result.tier is not tier or verdict is None or verdict.passed:
            break
        n += 1
    return n


class Orchestrator:
    """Runs a planned project: worker → verifier → retry/escalate, state saved each step."""

    def __init__(
        self,
        store: ProjectStore,
        worker: WorkerAgent,
        verifier: VerifierAgent,
        roles: RoleRegistry,
        config: AutomaConfig,
        *,
        settings: OrchestratorSettings | None = None,
        tools: Callable[[], list[Tool]] = default_tools,
        clock: Clock = _utc_now,
    ):
        if not isinstance(store, ProjectStore):
            raise TypeError("store must be a ProjectStore")
        if not isinstance(worker, WorkerAgent):
            raise TypeError("worker must be a WorkerAgent")
        if not isinstance(verifier, VerifierAgent):
            raise TypeError("verifier must be a VerifierAgent")
        if not isinstance(roles, RoleRegistry):
            raise TypeError("roles must be a RoleRegistry")
        if not isinstance(config, AutomaConfig):
            raise TypeError("config must be an AutomaConfig")
        self._store = store
        self._worker = worker
        self._verifier = verifier
        self._roles = roles
        self._config = config
        self._settings = settings or OrchestratorSettings()
        self._tools = tools
        self._clock = clock

    @property
    def settings(self) -> OrchestratorSettings:
        return self._settings

    # -- public ------------------------------------------------------------

    def run(self, project_id: str) -> RunReport:
        """Drive ``project_id`` until done, failed or paused; resumable at any point."""
        store = self._store
        store.load(project_id)  # ProjectNotFound if missing
        if not store.has_plan(project_id):
            raise OrchestratorError(f"project {project_id!r} has no plan yet")
        plan = store.load_plan(project_id)
        self._set_project(project_id, ProjectStatus.RUNNING)
        self._emit(project_id, "orchestrator.started", {"plan_version": plan.version})

        attempts_run = 0
        # Every iteration stores a result, a verdict or a terminal state, so the loop
        # ends; the cap only turns a bug into an error instead of a hang.
        max_iterations = len(plan.tasks) * (2 * self._settings.max_attempts + 2) + 1
        try:
            for _ in range(max_iterations):
                states = store.task_states(project_id)
                self._block_dependents_of_failed(project_id, plan, states)
                states = store.task_states(project_id)
                task = self._next_task(plan, states)
                if task is None:
                    return self._finish(project_id, plan, states, attempts_run)
                attempts_run += self._step(project_id, plan, task, states[task.id])
            raise OrchestratorError("orchestrator made no progress (iteration cap reached)")
        except _Stop as stop:
            self._set_project(project_id, ProjectStatus.PAUSED)
            report = self._report(
                project_id,
                plan,
                ProjectStatus.PAUSED,
                stop.reason,
                detail=stop.detail,
                task_id=stop.task_id,
                attempts_run=attempts_run,
                approval_id=stop.approval_id,
            )
            self._emit(
                project_id,
                "orchestrator.paused",
                {"reason": stop.reason.value, "task": stop.task_id}
                | ({"approval": stop.approval_id} if stop.approval_id else {}),
            )
            return report
        except BaseException:
            # Unexpected (bug, interrupt): never leave the project marked running.
            try:
                self._set_project(project_id, ProjectStatus.PAUSED)
            except Exception:  # noqa: BLE001, S110 - keep the original error
                pass
            raise

    # -- one step ----------------------------------------------------------

    def _step(self, project_id: str, plan: Plan, task: Task, state: TaskState) -> int:
        """Advance ``task`` by one attempt (or one re-verification). Returns attempts started."""
        store = self._store
        s = self._settings
        try:
            role = self._roles.get(task.role)
        except RoleError as e:
            self._save_state(project_id, plan, task.id, TaskStatus.FAILED, None, str(e))
            return 0

        history = store.task_history(project_id, task.id)
        if history and history[-1][1] is None:
            # Last attempt was stored but never judged: judge it, don't redo the work.
            self._verify(project_id, plan, task, history[-1][0], history)
            return 0

        last_verdict = history[-1][1] if history else None
        if last_verdict is not None and not last_verdict.passed and len(history) >= s.max_attempts:
            self._fail(project_id, plan, task, len(history))
            return 0

        tier = state.tier or task.tier or role.tier
        attempt = len(history) + 1
        feedback: list[Issue] = list(last_verdict.issues) if last_verdict is not None else []
        self._save_state(project_id, plan, task.id, TaskStatus.RUNNING, tier, None)
        self._emit(
            project_id,
            "orchestrator.attempt",
            {"task": task.id, "attempt": attempt, "tier": tier.value},
        )
        executor = self._executor(project_id, role.policy(), task_id=task.id)
        try:
            result = self._worker.run(
                task,
                role,
                executor,
                project_id=project_id,
                attempt=attempt,
                feedback=feedback,
                tier=tier,
            )
        except BaseException as e:
            # Nothing stored for this attempt: the task goes back to pending.
            self._save_state(
                project_id, plan, task.id, TaskStatus.PENDING, tier, _short(f"worker: {e}")
            )
            stop = self._stop_for(e, task.id)
            if stop is None:
                raise
            raise stop from e
        stored = store.save_task_result(project_id, result)
        self._verify(project_id, plan, task, stored, [*history, (stored, None)])
        return 1

    def _verify(
        self,
        project_id: str,
        plan: Plan,
        task: Task,
        result: TaskResult,
        history: Sequence[tuple[TaskResult, Verdict | None]],
    ) -> None:
        store = self._store
        s = self._settings
        self._save_state(project_id, plan, task.id, TaskStatus.VERIFYING, result.tier, None)
        executor = self._executor(project_id, verifier_policy())
        try:
            verdict = self._verifier.verify(task, result, executor, project_id=project_id)
        except BaseException as e:
            # The result stays stored without a verdict; the next run re-verifies it.
            self._save_state(
                project_id,
                plan,
                task.id,
                TaskStatus.VERIFYING,
                result.tier,
                _short(f"verifier: {e}"),
            )
            stop = self._stop_for(e, task.id)
            if stop is None:
                raise
            raise stop from e
        verdict = store.save_verdict(project_id, verdict)
        history = [*history[:-1], (result, verdict)]
        attempts = len(history)

        if verdict.passed:
            self._save_state(project_id, plan, task.id, TaskStatus.PASSED, result.tier, None)
            return
        if attempts >= s.max_attempts:
            self._fail(project_id, plan, task, attempts)
            return
        tier = result.tier
        if rejections_at_tier(history, tier) >= s.escalate_after:
            higher = next_tier(tier, set(self._config.tiers))
            if higher is not None:
                self._emit(
                    project_id,
                    "orchestrator.escalated",
                    {"task": task.id, "from": tier.value, "to": higher.value},
                )
                tier = higher
        self._save_state(project_id, plan, task.id, TaskStatus.PENDING, tier, None)

    # -- helpers -----------------------------------------------------------

    def _fail(self, project_id: str, plan: Plan, task: Task, attempts: int) -> None:
        current = self._store.load_task_state(project_id, task.id)
        tier = current.tier if current is not None else None
        self._save_state(
            project_id,
            plan,
            task.id,
            TaskStatus.FAILED,
            tier,
            f"rejected by the verifier in {attempts} attempt(s)",
        )

    def _block_dependents_of_failed(
        self, project_id: str, plan: Plan, states: dict[str, TaskState]
    ) -> None:
        blocked: set[str] = set()
        for tid, state in states.items():
            if state.status not in (TaskStatus.FAILED, TaskStatus.BLOCKED):
                continue
            for dep in plan.dependents(tid):
                if dep.id in blocked or states[dep.id].status not in _UNFINISHED:
                    continue
                self._save_state(
                    project_id,
                    plan,
                    dep.id,
                    TaskStatus.BLOCKED,
                    states[dep.id].tier,
                    f"dependency {tid!r} is {state.status.value}",
                )
                blocked.add(dep.id)

    @staticmethod
    def _next_task(plan: Plan, states: dict[str, TaskState]) -> Task | None:
        passed = {tid for tid, s in states.items() if s.status is TaskStatus.PASSED}
        for task in plan.topological_order():
            if states[task.id].status in _UNFINISHED and all(d in passed for d in task.depends_on):
                return task
        return None

    def _finish(
        self, project_id: str, plan: Plan, states: dict[str, TaskState], attempts_run: int
    ) -> RunReport:
        if all(s.status is TaskStatus.PASSED for s in states.values()):
            status, reason = ProjectStatus.DONE, StopReason.COMPLETED
        else:
            status, reason = ProjectStatus.FAILED, StopReason.TASKS_FAILED
        self._set_project(project_id, status)
        self._emit(project_id, "orchestrator.finished", {"status": status.value})
        return self._report(project_id, plan, status, reason, attempts_run=attempts_run)

    def _report(
        self,
        project_id: str,
        plan: Plan,
        status: ProjectStatus,
        reason: StopReason,
        *,
        detail: str = "",
        task_id: str | None = None,
        attempts_run: int = 0,
        approval_id: str | None = None,
    ) -> RunReport:
        states = self._store.task_states(project_id)

        def ids(st: TaskStatus) -> list[str]:
            return [t.id for t in plan.tasks if states[t.id].status is st]

        return RunReport(
            project_id=project_id,
            status=status,
            reason=reason,
            detail=_short(detail),
            task_id=task_id,
            approval_id=approval_id,
            attempts_run=attempts_run,
            passed=ids(TaskStatus.PASSED),
            failed=ids(TaskStatus.FAILED),
            blocked=ids(TaskStatus.BLOCKED),
        )

    @staticmethod
    def _stop_for(e: BaseException, task_id: str) -> _Stop | None:
        """Map an agent error to a pause reason; anything else propagates unchanged."""
        if isinstance(e, BudgetError):
            return _Stop(StopReason.BUDGET, str(e), task_id)
        if isinstance(e, ApprovalRequired):
            return _Stop(
                StopReason.APPROVAL, f"{e.tool}: {e.reason}", task_id, approval_id=e.request_id
            )
        if isinstance(e, ProviderError):
            return _Stop(StopReason.PROVIDER, str(e), task_id)
        if isinstance(e, VerificationFailed):
            return _Stop(StopReason.VERIFICATION, str(e), task_id)
        if isinstance(e, ToolLayerError):
            return _Stop(StopReason.TOOL_LAYER, str(e), task_id)
        if isinstance(e, ConfigError):
            return _Stop(StopReason.CONFIG, str(e), task_id)
        return None

    def _executor(
        self, project_id: str, policy: PermissionPolicy, *, task_id: str | None = None
    ) -> ToolExecutor:
        workspace = Workspace(self._store.files_dir(project_id))
        return ToolExecutor(
            self._tools(),
            policy,
            workspace,
            self._store.events(project_id),
            project_id,
            approvals=self._store.approvals(project_id) if task_id is not None else None,
            task_id=task_id,
        )

    def _save_state(
        self,
        project_id: str,
        plan: Plan,
        task_id: str,
        status: TaskStatus,
        tier: Tier | None,
        error: str | None,
    ) -> TaskState:
        current = self._store.load_task_state(project_id, task_id)
        return self._store.save_task_state(
            project_id,
            TaskState(
                task_id=task_id,
                status=status,
                attempts=self._store.attempt_count(project_id, task_id),
                tier=tier,
                plan_version=plan.version,
                error=error,
                updated_at=self._clock(),
                revision=current.revision if current is not None else 0,
            ),
        )

    def _set_project(self, project_id: str, status: ProjectStatus) -> None:
        if self._store.load(project_id).status is not status:
            self._store.set_status(project_id, status)

    def _emit(self, project_id: str, type_: str, data: dict[str, Any]) -> None:
        self._store.events(project_id).emit(
            Event(ts=self._clock(), type=type_, project_id=project_id, data=data)
        )
