"""``BudgetGuard``: hard spending caps enforced in code before every model call.

The guard is the only ``CallGuard`` meant for production. For each call it:

1. ``authorize``: refuses (raises ``BudgetExceeded``) unless the call's worst-case
   cost fits under the per-call cap *and*, together with everything already spent
   and every in-flight reservation, under the daily and project caps. An accepted
   estimate is reserved until the call is recorded or released.
2. ``record``: writes the actual cost to the ``CostLedger`` (always, even when the
   call turns out to have cost more than estimated) and frees the reservation.
3. ``release``: frees the reservation of a call that failed before any usage.

Reaching the daily or project cap is a *hard stop*: the scope is latched and every
later call in it is refused, even a tiny one, until the user resolves it (approval
flow arrives in M5). All arithmetic is in integer nano-USD; caps round *down* and
costs round *up*, so the guard can never under-count.

Warnings: when recorded spending first reaches 50% and 80% of the daily or project
cap, a ``budget.warning`` event is emitted (once per scope, key and threshold);
every new hard stop emits ``budget.stop``. Events go to the optional ``EventSink``
and are always kept in memory (``events()``). A failing sink never undoes the
ledger write or the latch; the failure is raised as ``BudgetError`` (fail closed).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from automa.budget.ledger import NANO_PER_USD, CostLedger, LedgerEntry, nusd_to_usd, usd_to_nusd
from automa.config import BudgetLimits
from automa.events import Event, EventSink
from automa.llm.gateway import CallEstimate
from automa.llm.types import LLMResult

# Percentages of the daily/project caps at which a warning event is emitted.
WARNING_THRESHOLDS_PCT: tuple[int, ...] = (50, 80)


class BudgetScope(StrEnum):
    PER_CALL = "per_call"
    DAILY = "daily"
    PROJECT = "project"


class BudgetError(Exception):
    """Base class for every refusal or misuse detected by the budget layer."""


class BudgetExceeded(BudgetError):
    """A call was refused because it would (or did) exceed a hard cap."""

    def __init__(
        self,
        scope: BudgetScope,
        *,
        limit_nusd: int,
        spent_nusd: int,
        requested_nusd: int,
        key: str | None = None,
        latched: bool = False,
    ):
        self.scope = scope
        self.limit_nusd = limit_nusd
        self.spent_nusd = spent_nusd
        self.requested_nusd = requested_nusd
        self.key = key
        self.latched = latched
        where = f" ({key})" if key else ""
        state = "hard stop already in effect; " if latched else ""
        super().__init__(
            f"budget cap '{scope.value}'{where} reached: {state}"
            f"limit ${self.limit_usd:.6f}, spent ${self.spent_usd:.6f}, "
            f"requested ${self.requested_usd:.6f}. User approval required."
        )

    @property
    def limit_usd(self) -> float:
        return nusd_to_usd(self.limit_nusd)

    @property
    def spent_usd(self) -> float:
        return nusd_to_usd(self.spent_nusd)

    @property
    def requested_usd(self) -> float:
        return nusd_to_usd(self.requested_nusd)


def cap_to_nusd(usd: float) -> int:
    """Convert a cap to nano-USD, rounding *down* (a cap is never loosened)."""
    if not math.isfinite(usd) or usd <= 0:
        raise BudgetError("budget caps must be finite and strictly positive")
    return math.floor(usd * NANO_PER_USD)


@dataclass(frozen=True)
class _Reservation:
    token: int
    estimate: CallEstimate
    nusd: int


class BudgetGuard:
    """Enforces ``BudgetLimits`` against a ``CostLedger``. Thread-safe."""

    def __init__(
        self,
        limits: BudgetLimits,
        ledger: CostLedger,
        *,
        events: EventSink | None = None,
    ):
        self._limits = limits
        self._sink = events
        self._events: list[Event] = []
        # (scope, key, pct) thresholds already announced.
        self._warned: set[tuple[BudgetScope, str, int]] = set()
        self._ledger = ledger
        self._per_call_cap = cap_to_nusd(limits.per_call_usd)
        self._daily_cap = cap_to_nusd(limits.daily_usd)
        self._project_cap = cap_to_nusd(limits.project_usd)
        self._lock = threading.Lock()
        self._pending: list[_Reservation] = []
        self._next_token = 0
        # (scope, key) -> the refusal that latched it.
        self._stopped: dict[tuple[BudgetScope, str], BudgetExceeded] = {}

    @property
    def limits(self) -> BudgetLimits:
        return self._limits

    @property
    def ledger(self) -> CostLedger:
        return self._ledger

    # --- inspection -----------------------------------------------------------

    def pending_nusd(self, *, project_id: str | None = None) -> int:
        with self._lock:
            return self._pending_locked(project_id)

    def is_stopped(self, scope: BudgetScope, key: str) -> bool:
        with self._lock:
            return (scope, key) in self._stopped

    def stops(self) -> list[BudgetExceeded]:
        with self._lock:
            return list(self._stopped.values())

    def events(self) -> list[Event]:
        """Every budget event emitted so far (also when no sink is configured)."""
        with self._lock:
            return list(self._events)

    # --- CallGuard protocol ---------------------------------------------------

    def authorize(self, estimate: CallEstimate) -> None:
        """Reserve ``estimate`` or raise ``BudgetError``. Never silently allows."""
        project_id = estimate.context.project_id
        if not project_id or not project_id.strip():
            # Every call must be attributable, otherwise the project cap is void.
            raise BudgetError("call has no project_id; refusing (project cap cannot apply)")
        requested = usd_to_nusd(estimate.max_cost_usd)
        with self._lock:
            day_key = self._ledger.today().isoformat()
            for scope, key in ((BudgetScope.DAILY, day_key), (BudgetScope.PROJECT, project_id)):
                latched = self._stopped.get((scope, key))
                if latched is not None:
                    raise BudgetExceeded(
                        scope,
                        limit_nusd=latched.limit_nusd,
                        spent_nusd=self._spent_locked(scope, key),
                        requested_nusd=requested,
                        key=key,
                        latched=True,
                    )
            if requested > self._per_call_cap:
                # Not latched: nothing was spent; the caller may retry smaller.
                raise BudgetExceeded(
                    BudgetScope.PER_CALL,
                    limit_nusd=self._per_call_cap,
                    spent_nusd=0,
                    requested_nusd=requested,
                )
            checks = (
                (BudgetScope.DAILY, day_key, self._daily_cap, self._pending_locked(None)),
                (
                    BudgetScope.PROJECT,
                    project_id,
                    self._project_cap,
                    self._pending_locked(project_id),
                ),
            )
            for scope, key, cap, pending in checks:
                committed = self._spent_locked(scope, key) + pending
                if committed + requested > cap:
                    err = BudgetExceeded(
                        scope,
                        limit_nusd=cap,
                        spent_nusd=committed,
                        requested_nusd=requested,
                        key=key,
                    )
                    self._stopped[(scope, key)] = err
                    stop_event = self._stop_event_locked(err, project_id)
                    break
            else:
                self._next_token += 1
                self._pending.append(_Reservation(self._next_token, estimate, requested))
                return
        # The call is refused whatever happens to the event; a sink failure is chained.
        try:
            self._deliver([stop_event])
        except BudgetError as sink_failure:
            raise err from sink_failure
        raise err

    def record(self, estimate: CallEstimate, result: LLMResult) -> LedgerEntry:
        """Persist the actual cost and free the reservation.

        The ledger write happens first and unconditionally: money that was spent is
        always recorded. Afterwards, an unreserved call (one that bypassed
        ``authorize``) raises ``BudgetError``, and any cap that the actual spending
        has now reached is latched.
        """
        with self._lock:
            entry = self._ledger.record(estimate, result)
            reserved = self._pop_reservation(estimate)
            day_key = entry.ts.date().isoformat()
            project_id = estimate.context.project_id
            scopes = [(BudgetScope.DAILY, day_key, self._daily_cap)]
            if project_id:
                scopes.append((BudgetScope.PROJECT, project_id, self._project_cap))
            new_events: list[Event] = []
            for scope, key, cap in scopes:
                spent = self._spent_locked(scope, key)
                new_events += self._warnings_locked(scope, key, cap, spent, project_id, entry)
                stop = self._latch_if_reached(scope, key, cap, spent, entry)
                if stop is not None:
                    new_events.append(self._stop_event_locked(stop, project_id))
        if not reserved:
            try:
                self._deliver(new_events)
            except BudgetError as sink_failure:
                raise BudgetError(
                    "recorded a call that was never authorized by the guard"
                ) from sink_failure
            raise BudgetError("recorded a call that was never authorized by the guard")
        self._deliver(new_events)
        return entry

    def release(self, estimate: CallEstimate) -> None:
        """Free the reservation of a call that failed without consuming tokens."""
        with self._lock:
            if not self._pop_reservation(estimate):
                raise BudgetError("released a call that was never authorized by the guard")

    # --- internals (call with the lock held) ----------------------------------

    def _pending_locked(self, project_id: str | None) -> int:
        return sum(
            r.nusd
            for r in self._pending
            if project_id is None or r.estimate.context.project_id == project_id
        )

    def _spent_locked(self, scope: BudgetScope, key: str) -> int:
        if scope is BudgetScope.DAILY:
            return self._ledger.spent_nusd(day=date.fromisoformat(key))
        return self._ledger.spent_nusd(project_id=key)

    def _pop_reservation(self, estimate: CallEstimate) -> bool:
        for i, r in enumerate(self._pending):
            if r.estimate is estimate or r.estimate == estimate:
                del self._pending[i]
                return True
        return False

    def _latch_if_reached(
        self, scope: BudgetScope, key: str, cap: int, spent: int, entry: LedgerEntry
    ) -> BudgetExceeded | None:
        if (scope, key) in self._stopped or spent < cap:
            return None
        err = BudgetExceeded(
            scope, limit_nusd=cap, spent_nusd=spent, requested_nusd=entry.cost_nusd, key=key
        )
        self._stopped[(scope, key)] = err
        return err

    def _warnings_locked(
        self,
        scope: BudgetScope,
        key: str,
        cap: int,
        spent: int,
        project_id: str | None,
        entry: LedgerEntry,
    ) -> list[Event]:
        out: list[Event] = []
        for pct in WARNING_THRESHOLDS_PCT:
            # Integer comparison: spent / cap >= pct / 100, with no float rounding.
            if spent * 100 < cap * pct or (scope, key, pct) in self._warned:
                continue
            self._warned.add((scope, key, pct))
            event = Event(
                ts=entry.ts,
                type="budget.warning",
                project_id=project_id,
                data={
                    "scope": scope.value,
                    "key": key,
                    "threshold_pct": pct,
                    "limit_nusd": cap,
                    "spent_nusd": spent,
                    "limit_usd": nusd_to_usd(cap),
                    "spent_usd": nusd_to_usd(spent),
                },
            )
            self._events.append(event)
            out.append(event)
        return out

    def _stop_event_locked(self, err: BudgetExceeded, project_id: str | None) -> Event:
        event = Event(
            ts=self._ledger.now(),
            type="budget.stop",
            project_id=project_id,
            data={
                "scope": err.scope.value,
                "key": err.key,
                "limit_nusd": err.limit_nusd,
                "spent_nusd": err.spent_nusd,
                "requested_nusd": err.requested_nusd,
                "limit_usd": err.limit_usd,
                "spent_usd": err.spent_usd,
            },
        )
        self._events.append(event)
        return event

    # --- event delivery (call WITHOUT the lock: sinks may do I/O) --------------

    def _deliver(self, events: list[Event]) -> None:
        if self._sink is None or not events:
            return
        failure: BaseException | None = None
        for event in events:
            try:
                self._sink.emit(event)
            except Exception as e:  # noqa: BLE001 - re-raised below, never swallowed
                failure = failure or e
        if failure is not None:
            raise BudgetError(f"budget event sink failed: {type(failure).__name__}") from failure
