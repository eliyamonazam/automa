"""``TaskBrake``: a per-task spending brake on top of the ``BudgetGuard`` (D153).

The guard's caps (per call, daily, project) are hard limits on money. They do not
notice one task quietly burning ten times its estimate while the project cap still
has room (real case: a stock-scan task ran 18 ``run_command`` calls and cost $1.17
against an estimated $0.09-0.40). The brake is an *extra* stop for that case:

* every task gets a brake step of ``max(task_brake_factor x the high end of its
  estimated range, task_brake_min_usd)``; tasks without an estimate get the minimum;
* before each model call of a task, if the ledger shows that task has already cost
  *more* than ``step x (1 + continues)``, the call is refused with
  ``TaskBrakeStop`` (a ``BudgetError``, so every caller already stops on it) and a
  ``brake.stop`` event is recorded. Refusing happens before the inner guard sees
  the call, so nothing is reserved and no call runs after the brake;
* otherwise the call goes to the inner guard unchanged: the brake can only refuse
  more, never allow a call the ``BudgetGuard`` would refuse.

"Continue" (``automa brake continue``) appends one ``brake.continued`` event for
that task; it is only accepted while the task's latest brake stop has not been
continued yet, so the allowance can be raised one step at a time and never ahead
of time. Each continue lets the task spend one more step of the same size.

"Stop" in the web UI (D154) appends a ``brake.declined`` event: a record of the
user's decision only. It allows nothing, keeps the stop pending (a later continue is
still possible) and never changes what the brake or the guard refuse.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from types import MappingProxyType

from automa.budget.guard import BudgetError, cap_to_nusd
from automa.budget.ledger import CostLedger, nusd_to_usd
from automa.config import BudgetLimits
from automa.events import Event, EventSink
from automa.llm.gateway import CallEstimate, CallGuard
from automa.llm.types import LLMResult

BRAKE_STOP = "brake.stop"
BRAKE_CONTINUED = "brake.continued"
BRAKE_DECLINED = "brake.declined"


class TaskBrakeStop(BudgetError):
    """A task has cost more than its brake allows; the user decides whether to go on."""

    def __init__(
        self,
        task_id: str,
        *,
        spent_nusd: int,
        limit_nusd: int,
        step_nusd: int,
        steps: int,
    ) -> None:
        self.task_id = task_id
        self.spent_nusd = spent_nusd
        self.limit_nusd = limit_nusd
        self.step_nusd = step_nusd
        self.steps = steps
        super().__init__(
            f"task brake: task {task_id!r} has cost ${nusd_to_usd(spent_nusd):.4f}, more "
            f"than its brake ${nusd_to_usd(limit_nusd):.4f} ({steps} step(s) of "
            f"${nusd_to_usd(step_nusd):.4f}); far more than estimated"
        )


def brake_steps(
    task_ids: Sequence[str],
    limits: BudgetLimits,
    range_high_nusd: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Brake step (nano-USD) of every task in ``task_ids``.

    ``range_high_nusd`` is the high end of each task's realistic range from a fresh
    estimate; a task missing from it (or no estimate at all) gets the minimum.
    The product rounds *down*: the brake is never looser than asked.
    """
    minimum = cap_to_nusd(limits.task_brake_min_usd)
    highs = range_high_nusd or {}
    out: dict[str, int] = {}
    for tid in task_ids:
        high = max(0, int(highs.get(tid, 0)))
        out[tid] = max(minimum, math.floor(high * limits.task_brake_factor))
    return out


def brake_continues(events: Iterable[Event]) -> dict[str, int]:
    """How many times each task's brake was continued (``brake.continued`` events)."""
    out: dict[str, int] = {}
    for e in events:
        if e.type == BRAKE_CONTINUED:
            tid = e.data.get("task")
            if isinstance(tid, str):
                out[tid] = out.get(tid, 0) + 1
    return out


def pending_brake(events: Sequence[Event], task_id: str) -> Event | None:
    """The task's latest ``brake.stop`` if no ``brake.continued`` came after it."""
    for e in reversed(events):
        if e.data.get("task") != task_id:
            continue
        if e.type == BRAKE_CONTINUED:
            return None
        if e.type == BRAKE_STOP:
            return e
    return None


class BrakeError(Exception):
    """A brake continue was refused (nothing to continue)."""


def continue_task(
    events: Sequence[Event],
    sink: EventSink,
    project_id: str,
    task_id: str,
    *,
    via: str,
    now: datetime | None = None,
) -> Event:
    """Record one more brake step for ``task_id``; only after an un-continued stop."""
    stop = pending_brake(events, task_id)
    if stop is None:
        raise BrakeError(f"task {task_id!r} is not stopped by the spending brake")
    steps = brake_continues(events).get(task_id, 0) + 2
    event = Event(
        ts=now or datetime.now(UTC),
        type=BRAKE_CONTINUED,
        project_id=project_id,
        data={
            "task": task_id,
            "via": via,
            "steps": steps,
            "spent_nusd": stop.data.get("spent_nusd"),
            "step_nusd": stop.data.get("step_nusd"),
        },
    )
    sink.emit(event)
    return event


def declined_brake(events: Sequence[Event], task_id: str) -> bool:
    """Whether the user said "stop" after the task's pending brake stop (D154)."""
    for e in reversed(events):
        if e.data.get("task") != task_id:
            continue
        if e.type == BRAKE_DECLINED:
            return pending_brake(events, task_id) is not None
        if e.type in (BRAKE_STOP, BRAKE_CONTINUED):
            return False
    return False


def decline_task(
    events: Sequence[Event],
    sink: EventSink,
    project_id: str,
    task_id: str,
    *,
    via: str,
    now: datetime | None = None,
) -> Event:
    """Record that the user keeps ``task_id`` stopped (D154); allows nothing."""
    stop = pending_brake(events, task_id)
    if stop is None:
        raise BrakeError(f"task {task_id!r} is not stopped by the spending brake")
    event = Event(
        ts=now or datetime.now(UTC),
        type=BRAKE_DECLINED,
        project_id=project_id,
        data={"task": task_id, "via": via, "spent_nusd": stop.data.get("spent_nusd")},
    )
    sink.emit(event)
    return event


class TaskBrake:
    """``CallGuard`` that brakes one project's tasks, then defers to ``inner``."""

    def __init__(
        self,
        inner: CallGuard,
        ledger: CostLedger,
        project_id: str,
        steps: Mapping[str, int],
        *,
        default_step_nusd: int,
        continues: Mapping[str, int] | None = None,
        events: EventSink | None = None,
    ) -> None:
        if not project_id or not project_id.strip():
            raise ValueError("the task brake needs a project id")
        if default_step_nusd <= 0 or any(v <= 0 for v in steps.values()):
            raise ValueError("brake steps must be strictly positive")
        if any(v < 0 for v in (continues or {}).values()):
            raise ValueError("brake continues must not be negative")
        self._inner = inner
        self._ledger = ledger
        self._project_id = project_id
        self._steps = MappingProxyType(dict(steps))
        self._default = default_step_nusd
        self._continues = MappingProxyType(dict(continues or {}))
        self._sink = events
        self._lock = threading.Lock()
        self._announced: set[tuple[str, int]] = set()

    @property
    def inner(self) -> CallGuard:
        return self._inner

    def step_nusd(self, task_id: str) -> int:
        return self._steps.get(task_id, self._default)

    def allowed_steps(self, task_id: str) -> int:
        return 1 + self._continues.get(task_id, 0)

    def limit_nusd(self, task_id: str) -> int:
        return self.step_nusd(task_id) * self.allowed_steps(task_id)

    def authorize(self, estimate: CallEstimate) -> None:
        ctx = estimate.context
        if ctx.project_id == self._project_id and ctx.task_id is not None:
            self._check(ctx.task_id)
        self._inner.authorize(estimate)

    def record(self, estimate: CallEstimate, result: LLMResult) -> object:
        return self._inner.record(estimate, result)

    def release(self, estimate: CallEstimate) -> None:
        self._inner.release(estimate)

    def _check(self, task_id: str) -> None:
        spent = self._ledger.spent_nusd(project_id=self._project_id, task_id=task_id)
        limit = self.limit_nusd(task_id)
        if spent <= limit:
            return
        stop = TaskBrakeStop(
            task_id,
            spent_nusd=spent,
            limit_nusd=limit,
            step_nusd=self.step_nusd(task_id),
            steps=self.allowed_steps(task_id),
        )
        with self._lock:
            key = (task_id, stop.steps)
            fresh = key not in self._announced
            self._announced.add(key)
        if fresh and self._sink is not None:
            event = Event(
                ts=self._ledger.now(),
                type=BRAKE_STOP,
                project_id=self._project_id,
                data={
                    "task": task_id,
                    "spent_nusd": spent,
                    "limit_nusd": limit,
                    "step_nusd": stop.step_nusd,
                    "steps": stop.steps,
                    "spent_usd": nusd_to_usd(spent),
                    "limit_usd": nusd_to_usd(limit),
                },
            )
            try:
                self._sink.emit(event)
            except Exception as e:  # noqa: BLE001 - the call is refused either way
                raise stop from e
        raise stop
