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

Parallel mode (``max_parallel`` > 1, D35): up to ``max_parallel`` *ready* tasks (all
dependencies passed) are driven at once, each in its own thread, by exactly the
same steps as above. Scheduling, blocking and project status stay in the calling
thread. The first stop (budget, approval, …) or unexpected error stops scheduling;
tasks already running finish their current step (never interrupted mid-call), then
the project is paused with the first stop. Spending stays bounded by the shared
``BudgetGuard``: every concurrent call reserves its worst case before it runs.
Shared ``files/`` (D37): tasks whose acceptance checks name overlapping paths are
never scheduled at the same time, and a worker write to a path another running
task has written fails (``PathClaims``), so two parallel tasks never write one file.

Data router (D59): with ``router=``, a task's *first* attempt starts at the tier the
``Router`` picks from quality statistics (event ``router.decision``); the choice is
saved in the task state like any tier, so resuming never routes again and
escalation continues from it. Without a router nothing changes.

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

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from automa.agents.verifier import VerificationFailed, VerifierAgent, verifier_policy
from automa.agents.worker import WorkerAgent
from automa.budget.guard import BudgetError
from automa.config import MAX_PARALLEL, AutomaConfig, ConfigError, SandboxMode, Tier
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
from automa.quality import QualityError, QualityLog, QualityRecord
from automa.roles.registry import RoleError, RoleRegistry
from automa.router import RouteDecision, Router
from automa.tools import RunCommand, tools_factory
from automa.tools.base import Tool
from automa.tools.claims import PathClaims, paths_overlap
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
    # Ready tasks driven at the same time; 1 keeps the sequential loop (default).
    max_parallel: int = Field(default=1, ge=1, le=MAX_PARALLEL)


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


def declared_paths(task: Task) -> frozenset[str]:
    """Paths in ``files/`` a task's deterministic checks name (its declared outputs)."""
    paths = set()
    for criterion in task.acceptance_criteria:
        path = getattr(criterion.check, "path", None)
        if isinstance(path, str):
            paths.add(path)
    return frozenset(paths)


def _path_conflict(
    paths: frozenset[str], taken: dict[str, frozenset[str]]
) -> tuple[str, str] | None:
    """``(task_id, path)`` of the first running task whose declared paths overlap."""
    for tid, theirs in sorted(taken.items()):
        for mine in sorted(paths):
            for other in sorted(theirs):
                if paths_overlap(mine, other):
                    return tid, mine
    return None


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
        tools: Callable[[], list[Tool]] | None = None,
        clock: Clock = _utc_now,
        quality: QualityLog | None = None,
        router: Router | None = None,
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
        if quality is not None and not isinstance(quality, QualityLog):
            raise TypeError("quality must be a QualityLog")
        if router is not None:
            if not isinstance(router, Router):
                raise TypeError("router must be a Router")
            if router.config != config:
                # Its tiers/prices must be the ones the gateway uses (D59).
                raise ValueError("router was built from a different configuration")
        self._store = store
        self._worker = worker
        self._verifier = verifier
        self._roles = roles
        self._config = config
        self._settings = settings or OrchestratorSettings()
        # Without an explicit factory the tools follow ``[sandbox]`` in the config.
        self._tools = tools or tools_factory(config.sandbox.docker_sandbox())
        self._clock = clock
        self._quality = quality
        self._router = router
        self._make_tools()  # a factory that ignores a required sandbox fails here

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
        self._emit(
            project_id,
            "orchestrator.started",
            {"plan_version": plan.version}
            | ({"max_parallel": self._settings.max_parallel} if self._parallel else {}),
        )
        if self._parallel:
            return self._run_parallel(project_id, plan)

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
            return self._pause(project_id, plan, stop, attempts_run)
        except BaseException:
            # Unexpected (bug, interrupt): never leave the project marked running.
            self._pause_quietly(project_id)
            raise

    # -- parallel mode -----------------------------------------------------

    @property
    def _parallel(self) -> bool:
        return self._settings.max_parallel > 1

    def _run_parallel(self, project_id: str, plan: Plan) -> RunReport:
        """Like the sequential loop, but up to ``max_parallel`` ready tasks at once."""
        store = self._store
        halt = threading.Event()  # set → running tasks stop after their current step
        attempts_run = 0
        first_stop: _Stop | None = None
        error: BaseException | None = None
        running: dict[Future[int], str] = {}
        claims = PathClaims()  # paths written by the tasks running right now
        deferred: set[str] = set()  # tasks already reported as waiting for a path
        pool = ThreadPoolExecutor(
            max_workers=self._settings.max_parallel, thread_name_prefix="automa-task"
        )
        try:
            while True:
                if first_stop is None and error is None:
                    states = store.task_states(project_id)
                    self._block_dependents_of_failed(project_id, plan, states)
                    states = store.task_states(project_id)
                    busy = set(running.values())
                    taken = {tid: declared_paths(plan.task(tid)) for tid in busy}
                    for task in self._ready_tasks(plan, states, busy):
                        if len(running) >= self._settings.max_parallel:
                            break
                        other = _path_conflict(declared_paths(task), taken)
                        if other is not None:
                            if task.id not in deferred:
                                deferred.add(task.id)
                                self._emit(
                                    project_id,
                                    "orchestrator.deferred",
                                    {"task": task.id, "waits_for": other[0], "path": other[1]},
                                )
                            continue
                        fut = pool.submit(self._drive, project_id, plan, task, halt, claims)
                        running[fut] = task.id
                        taken[task.id] = declared_paths(task)
                        self._emit(project_id, "orchestrator.scheduled", {"task": task.id})
                    if not running:
                        return self._finish(project_id, plan, states, attempts_run)
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in done:
                    running.pop(fut)
                    try:
                        attempts_run += fut.result()
                    except _Stop as stop:
                        halt.set()
                        if first_stop is None:
                            first_stop = stop
                        elif stop is not first_stop:
                            self._emit_stop(project_id, stop, "orchestrator.task_stopped")
                    except BaseException as e:  # noqa: BLE001 - re-raised after draining
                        halt.set()
                        if error is None:
                            error = e
        except BaseException:
            # Interrupt or bug in the scheduler itself: let running steps finish, pause.
            halt.set()
            pool.shutdown(wait=True, cancel_futures=True)
            self._pause_quietly(project_id)
            raise
        pool.shutdown(wait=True)
        if error is not None:
            self._pause_quietly(project_id)
            raise error
        if first_stop is None:  # pragma: no cover - the loop only breaks after a stop
            raise OrchestratorError("parallel run ended without a stop or an error")
        return self._pause(project_id, plan, first_stop, attempts_run)

    def _drive(
        self,
        project_id: str,
        plan: Plan,
        task: Task,
        halt: threading.Event,
        claims: PathClaims | None = None,
    ) -> int:
        """Advance one task (in a pool thread) until it is finished or the run halts.

        Paths the task claimed are released when it stops being driven, whatever
        the reason, so a later task may write them (as in a sequential run).
        """
        attempts = 0
        try:
            # Each step stores a result, a verdict or a terminal state (as in ``run``).
            for _ in range(2 * self._settings.max_attempts + 2):
                if halt.is_set():
                    return attempts
                state = self._store.task_states(project_id)[task.id]
                if state.status not in _UNFINISHED:
                    return attempts
                attempts += self._step(project_id, plan, task, state, claims=claims)
            raise OrchestratorError(f"task {task.id!r} made no progress (step cap reached)")
        finally:
            if claims is not None:
                claims.release(task.id)

    @staticmethod
    def _ready_tasks(plan: Plan, states: dict[str, TaskState], busy: set[str]) -> list[Task]:
        passed = {tid for tid, s in states.items() if s.status is TaskStatus.PASSED}
        return [
            t
            for t in plan.topological_order()
            if t.id not in busy
            and states[t.id].status in _UNFINISHED
            and all(d in passed for d in t.depends_on)
        ]

    # -- stopping ----------------------------------------------------------

    def _pause(self, project_id: str, plan: Plan, stop: _Stop, attempts_run: int) -> RunReport:
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
        self._emit_stop(project_id, stop, "orchestrator.paused")
        return report

    def _emit_stop(self, project_id: str, stop: _Stop, type_: str) -> None:
        self._emit(
            project_id,
            type_,
            {"reason": stop.reason.value, "task": stop.task_id}
            | ({"approval": stop.approval_id} if stop.approval_id else {}),
        )

    def _pause_quietly(self, project_id: str) -> None:
        try:
            self._set_project(project_id, ProjectStatus.PAUSED)
        except Exception:  # noqa: BLE001, S110 - keep the original error
            pass

    # -- one step ----------------------------------------------------------

    def _step(
        self,
        project_id: str,
        plan: Plan,
        task: Task,
        state: TaskState,
        *,
        claims: PathClaims | None = None,
    ) -> int:
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

        requested = task.tier or role.tier
        if state.tier is not None:
            tier = state.tier  # already chosen (routed or escalated): never re-route
        elif self._router is not None and not history:
            tier = self._route(self._router, project_id, task, requested)
        else:
            tier = requested
        attempt = len(history) + 1
        feedback: list[Issue] = list(last_verdict.issues) if last_verdict is not None else []
        self._save_state(project_id, plan, task.id, TaskStatus.RUNNING, tier, None)
        self._emit(
            project_id,
            "orchestrator.attempt",
            {"task": task.id, "attempt": attempt, "tier": tier.value},
        )
        executor = self._executor(project_id, role.policy(), task_id=task.id, claims=claims)
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

    def _route(self, router: Router, project_id: str, task: Task, requested: Tier) -> Tier:
        """Starting tier of a fresh task from the data router (D59).

        Only the *starting* tier changes; escalation, the gateway and the
        BudgetGuard work exactly as without a router. The choice is saved in the
        task state by the caller, so a resumed run never routes the task again.
        """
        try:
            decision: RouteDecision = router.route(task.role, requested)
        except ConfigError as e:
            raise _Stop(StopReason.CONFIG, str(e), task.id) from e
        best = next(e for e in decision.evidence if e.tier is decision.tier)
        self._emit(
            project_id,
            "router.decision",
            {
                "task": task.id,
                "role": task.role,
                "requested": decision.requested.value,
                "tier": decision.tier.value,
                "model": decision.model,
                "reason": decision.reason.value,
                "attempts": best.attempts,
                "passed": best.passed,
            },
        )
        return decision.tier

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
        self._record_quality(project_id, task, result, verdict)
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

    def _record_quality(
        self, project_id: str, task: Task, result: TaskResult, verdict: Verdict
    ) -> None:
        """Add the judged attempt to the quality log (advisory, D56).

        The stored verdict is the source of truth and ``sync_project`` can always
        re-import it, so a failing log never changes the run: it only emits
        ``quality.record_failed``.
        """
        if self._quality is None:
            return
        try:
            self._quality.record(QualityRecord.from_attempt(project_id, task, result, verdict))
        except QualityError as e:
            self._emit(
                project_id,
                "quality.record_failed",
                {"task": task.id, "attempt": result.attempt, "error": _short(str(e))},
            )

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

    def _make_tools(self) -> list[Tool]:
        """Fresh tools; with ``mode = "docker"`` every ``run_command`` must be sandboxed."""
        tools = self._tools()
        if self._config.sandbox.mode is SandboxMode.DOCKER:
            for tool in tools:
                if isinstance(tool, RunCommand) and tool.sandbox is None:
                    raise ConfigError(
                        "sandbox mode is docker but run_command has no sandbox; "
                        "refusing to run commands on the host"
                    )
        return tools

    def _executor(
        self,
        project_id: str,
        policy: PermissionPolicy,
        *,
        task_id: str | None = None,
        claims: PathClaims | None = None,
    ) -> ToolExecutor:
        workspace = Workspace(self._store.files_dir(project_id))
        return ToolExecutor(
            self._make_tools(),
            policy,
            workspace,
            self._store.events(project_id),
            project_id,
            approvals=self._store.approvals(project_id) if task_id is not None else None,
            task_id=task_id,
            claims=claims,
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
