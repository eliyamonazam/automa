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
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from automa.budget.ledger import NANO_PER_USD, CostLedger, LedgerEntry, nusd_to_usd, usd_to_nusd
from automa.config import BudgetLimits
from automa.llm.gateway import CallEstimate
from automa.llm.types import LLMResult


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

    def __init__(self, limits: BudgetLimits, ledger: CostLedger):
        self._limits = limits
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

    # --- CallGuard protocol ---------------------------------------------------

    def authorize(self, estimate: CallEstimate) -> None:
        """Reserve ``estimate`` or raise ``BudgetError``. Never silently allows."""
        project_id = estimate.context.project_id
        if not project_id:
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
                    raise err
            self._next_token += 1
            self._pending.append(_Reservation(self._next_token, estimate, requested))

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
            self._latch_if_reached(BudgetScope.DAILY, day_key, self._daily_cap, entry)
            if project_id:
                self._latch_if_reached(BudgetScope.PROJECT, project_id, self._project_cap, entry)
            if not reserved:
                raise BudgetError("recorded a call that was never authorized by the guard")
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

    def _latch_if_reached(self, scope: BudgetScope, key: str, cap: int, entry: LedgerEntry) -> None:
        if (scope, key) in self._stopped:
            return
        spent = self._spent_locked(scope, key)
        if spent >= cap:
            self._stopped[(scope, key)] = BudgetExceeded(
                scope, limit_nusd=cap, spent_nusd=spent, requested_nusd=entry.cost_nusd, key=key
            )
