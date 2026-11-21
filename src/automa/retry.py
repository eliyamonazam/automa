"""Retry failed tasks instead of ending a project without a product (D156).

A task that the verifier rejected ``max_attempts`` times is ``failed`` and every
task depending on it ``blocked``; the project ends ``failed``. Running it again
used to end the same way at once, so the user was left without a product. A
*retry* is an explicit user decision that gives every failed task more attempts:

* one ``task.retry_granted`` event per failed task records the extra attempts
  (default ``RETRY_ATTEMPTS``), the tier the next attempt starts at (the strongest
  configured one: the cheaper tiers already failed) and what the task had cost so
  far (``spent_nusd``);
* the failed tasks go back to ``pending`` at that tier and the blocked ones to
  ``pending`` (the orchestrator blocks them again if a dependency still fails);
  the project becomes ``paused``, so ``automa run`` (or the web UI) continues it;
* the orchestrator adds the granted attempts to ``max_attempts`` for that task
  (``retry_grants``); the last verdict's issues reach the new attempt as feedback
  like any other retry;
* the task spending brake starts counting fresh steps from ``spent_nusd``
  (``brake_inputs``): spending before the retry was accepted by the user.

Nothing here touches the ``BudgetGuard``: every call of a retried task is still
authorized against the per-call, daily and project caps.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from automa.budget.brake import BRAKE_CONTINUED
from automa.config import AutomaConfig, Tier
from automa.events import Event
from automa.plan.models import MAX_TASK_ATTEMPTS, TaskState, TaskStatus
from automa.project.store import ProjectStatus, ProjectStore

RETRY_GRANTED = "task.retry_granted"
RETRY_ATTEMPTS = 2
MAX_RETRY_ATTEMPTS = 10

_TIER_ORDER: tuple[Tier, ...] = (Tier.SMALL, Tier.MEDIUM, Tier.LARGE)


class RetryError(Exception):
    """The project cannot be retried (nothing failed, still running, …)."""


def strongest_tier(config: AutomaConfig) -> Tier | None:
    """The highest configured tier (``None`` if none is configured)."""
    return next((t for t in reversed(_TIER_ORDER) if t in config.tiers), None)


def retry_grants(events: Sequence[Event]) -> dict[str, int]:
    """Extra attempts granted to each task by retries (sum over all grants)."""
    extra: dict[str, int] = {}
    for e in events:
        if e.type != RETRY_GRANTED:
            continue
        task = e.data.get("task")
        n = e.data.get("attempts")
        if isinstance(task, str) and isinstance(n, int) and not isinstance(n, bool) and n > 0:
            extra[task] = extra.get(task, 0) + n
    return extra


def brake_inputs(events: Sequence[Event]) -> tuple[dict[str, int], dict[str, int]]:
    """``(continues, offsets)`` for the task brake, with retries taken into account.

    For a task that was retried, only brake continues *after* its latest retry
    count, and the brake starts from what the task had cost at that retry.
    Tasks never retried get the plain number of continues and no offset.
    """
    continues: dict[str, int] = {}
    offsets: dict[str, int] = {}
    for e in events:
        task = e.data.get("task")
        if not isinstance(task, str):
            continue
        if e.type == RETRY_GRANTED:
            spent = e.data.get("spent_nusd")
            offsets[task] = spent if isinstance(spent, int) and spent > 0 else 0
            continues.pop(task, None)
        elif e.type == BRAKE_CONTINUED:
            continues[task] = continues.get(task, 0) + 1
    return continues, offsets


def failed_tasks(store: ProjectStore, project_id: str) -> list[str]:
    """Ids of the current plan's failed tasks (plan order)."""
    states = store.task_states(project_id)
    plan = store.load_plan(project_id)
    return [t.id for t in plan.tasks if states[t.id].status is TaskStatus.FAILED]


def can_retry(store: ProjectStore, project_id: str) -> bool:
    """Whether a retry would do something now (project failed with a failed task)."""
    try:
        project = store.load(project_id)
        if project.status is not ProjectStatus.FAILED or not store.has_plan(project_id):
            return False
        return bool(failed_tasks(store, project_id))
    except Exception:  # noqa: BLE001 - a page must never break on this question
        return False


def grant_retry(
    store: ProjectStore,
    project_id: str,
    config: AutomaConfig,
    *,
    via: str,
    spent_nusd: dict[str, int] | None = None,
    attempts: int = RETRY_ATTEMPTS,
    now: datetime | None = None,
) -> list[str]:
    """Give every failed task ``attempts`` more attempts; returns their ids.

    ``spent_nusd``: what each task has cost so far (from the ledger), recorded so
    the spending brake counts fresh steps from there. Raises ``RetryError`` when
    the project is not ``failed``, has no failed task, or a task would go past
    the attempt limit.
    """
    if not 1 <= attempts <= MAX_RETRY_ATTEMPTS:
        raise RetryError(f"attempts must be between 1 and {MAX_RETRY_ATTEMPTS}")
    project = store.load(project_id)
    if project.status is not ProjectStatus.FAILED:
        raise RetryError(
            f"project {project_id} is {project.status.value}; only a failed project can be retried"
        )
    plan = store.load_plan(project_id)
    states = store.task_states(project_id)
    failed = [t.id for t in plan.tasks if states[t.id].status is TaskStatus.FAILED]
    if not failed:
        raise RetryError(f"project {project_id} has no failed task to retry")
    for tid in failed:
        used = store.attempt_count(project_id, tid)
        if used + attempts > MAX_TASK_ATTEMPTS:
            raise RetryError(
                f"task {tid!r} already has {used} attempt(s); at most {MAX_TASK_ATTEMPTS}"
            )
    tier = strongest_tier(config)
    ts = now or datetime.now(UTC)
    events = store.events(project_id)
    spent = spent_nusd or {}
    for tid in failed:
        events.emit(
            Event(
                ts=ts,
                type=RETRY_GRANTED,
                project_id=project_id,
                data={
                    "task": tid,
                    "attempts": attempts,
                    "via": via,
                    "spent_nusd": max(0, int(spent.get(tid, 0))),
                }
                | ({"tier": tier.value} if tier is not None else {}),
            )
        )
    for task in plan.tasks:
        state = states[task.id]
        if state.status is TaskStatus.FAILED:
            _reset(store, project_id, state, tier if tier is not None else state.tier)
        elif state.status is TaskStatus.BLOCKED:
            _reset(store, project_id, state, state.tier)
    store.set_status(project_id, ProjectStatus.PAUSED)
    return failed


def _reset(store: ProjectStore, project_id: str, state: TaskState, tier: Tier | None) -> None:
    store.save_task_state(
        project_id,
        state.model_copy(
            update={
                "status": TaskStatus.PENDING,
                "tier": tier,
                "error": None,
                "attempts": store.attempt_count(project_id, state.task_id),
            }
        ),
    )
