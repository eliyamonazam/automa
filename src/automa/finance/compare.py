"""Estimate vs actual spending (Financial Controller, report comparison core; D67).

``automa run`` saves its advisory estimate as a ``finance.estimate`` event just
before the first model call (D64). ``compare_estimate`` takes a project's events
and the cost ledger entries and compares the **latest** such estimate with the
money actually recorded for that project at or after the estimate's timestamp:

* ``tasks``  — calls of the tasks the estimate covered (worker and verifier),
  compared with ``worst_nusd`` (the hard upper bound of those tasks);
* ``known``  — calls of the covered tasks whose expected cost was known,
  compared with ``expected_nusd`` (the sum of exactly those tasks' expectations);
* ``other``  — every other call of the project since then (planning, review,
  tasks the estimate did not cover).

Pure and read-only: no model call, no write, no authorization. The estimate has
no authority over money (the ``BudgetGuard`` alone has), and neither has this
comparison. Money stays in integer nano-USD.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from automa.budget import LedgerEntry
from automa.events import Event
from automa.finance.estimate import BudgetFit
from automa.finance.report import SpendGroup

ESTIMATE_EVENT = "finance.estimate"


class EstimateRecord(BaseModel):
    """The figures of one ``finance.estimate`` event (fields written by D64)."""

    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False, strict=True)

    ts: datetime
    plan_version: int = Field(ge=0)
    tasks: tuple[str, ...]
    excluded: tuple[str, ...] = ()
    router: bool = False
    expected_nusd: int = Field(ge=0)
    expected_complete: bool
    unknown: tuple[str, ...] = ()
    worst_nusd: int = Field(ge=0)
    project_remaining_nusd: int = Field(ge=0)
    assessment: BudgetFit

    @model_validator(mode="after")
    def _consistent(self) -> EstimateRecord:
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("duplicate task ids in the estimate")
        if not set(self.unknown) <= set(self.tasks):
            raise ValueError("unknown tasks must be estimated tasks")
        if self.expected_complete != (not self.unknown):
            raise ValueError("expected_complete contradicts the unknown tasks")
        if self.expected_nusd > self.worst_nusd:
            raise ValueError("the expected cost cannot exceed the worst case")
        return self

    @classmethod
    def from_event(cls, event: Event) -> EstimateRecord:
        """Parse a ``finance.estimate`` event; ``ValueError`` if it is not a valid one."""
        if not isinstance(event, Event) or event.type != ESTIMATE_EVENT:
            raise ValueError(f"not a {ESTIMATE_EVENT} event")
        data = dict(event.data)
        for key in ("tasks", "excluded", "unknown"):
            if isinstance(data.get(key), list):
                data[key] = tuple(data[key])
        if "assessment" in data and isinstance(data["assessment"], str):
            try:
                data["assessment"] = BudgetFit(data["assessment"])
            except ValueError as e:
                raise ValueError(f"invalid {ESTIMATE_EVENT} event: unknown assessment") from e
        try:
            return cls.model_validate({**data, "ts": event.ts})
        except ValidationError as e:
            raise ValueError(f"invalid {ESTIMATE_EVENT} event: {e.error_count()} error(s)") from e


class ComparisonOutcome(StrEnum):
    """How the actual spending since the latest estimate compares with it."""

    NO_ESTIMATE = "no_estimate"  # no finance.estimate event for the project
    INVALID_ESTIMATE = "invalid_estimate"  # the latest one cannot be read: no comparison
    NO_SPENDING = "no_spending"  # nothing recorded for the estimated tasks yet
    WITHIN_EXPECTED = "within_expected"  # known tasks <= expected (and tasks <= worst)
    ABOVE_EXPECTED = "above_expected"  # known tasks > expected, tasks still <= worst
    EXPECTED_UNKNOWN = "expected_unknown"  # only unknown tasks spent, tasks <= worst
    ABOVE_WORST = "above_worst"  # the estimated tasks spent more than the worst case


class EstimateComparison(BaseModel):
    """Latest estimate of one project next to the actual spending after it."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    project_id: str
    outcome: ComparisonOutcome
    estimate: EstimateRecord | None = None
    estimates: int = Field(default=0, ge=0)  # finance.estimate events seen
    invalid_reason: str | None = None
    tasks: SpendGroup = SpendGroup()  # calls of the estimated tasks
    known: SpendGroup = SpendGroup()  # calls of the estimated tasks with a known expectation
    other: SpendGroup = SpendGroup()  # other calls of the project since the estimate

    @property
    def total(self) -> SpendGroup:
        """Everything the project spent since the estimate."""
        return SpendGroup(
            calls=self.tasks.calls + self.other.calls,
            input_tokens=self.tasks.input_tokens + self.other.input_tokens,
            output_tokens=self.tasks.output_tokens + self.other.output_tokens,
            cost_nusd=self.tasks.cost_nusd + self.other.cost_nusd,
            reserved_nusd=self.tasks.reserved_nusd + self.other.reserved_nusd,
        )

    @property
    def expected_delta_nusd(self) -> int | None:
        """Known tasks' actual cost minus their expected cost (``None`` without estimate)."""
        if self.estimate is None:
            return None
        return self.known.cost_nusd - self.estimate.expected_nusd

    @model_validator(mode="after")
    def _consistent(self) -> EstimateComparison:
        has = self.estimate is not None
        needs = self.outcome not in (
            ComparisonOutcome.NO_ESTIMATE,
            ComparisonOutcome.INVALID_ESTIMATE,
        )
        if has != needs:
            raise ValueError("the outcome contradicts the presence of an estimate")
        if (self.outcome is ComparisonOutcome.INVALID_ESTIMATE) != (
            self.invalid_reason is not None
        ):
            raise ValueError("invalid_reason is set exactly for an invalid estimate")
        if (self.outcome is ComparisonOutcome.NO_ESTIMATE) != (self.estimates == 0):
            raise ValueError("no_estimate means no estimate event was seen")
        if self.known.cost_nusd > self.tasks.cost_nusd or self.known.calls > self.tasks.calls:
            raise ValueError("known tasks are a subset of the estimated tasks")
        return self


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def _outcome(record: EstimateRecord, tasks: SpendGroup, known: SpendGroup) -> ComparisonOutcome:
    if tasks.calls == 0:
        return ComparisonOutcome.NO_SPENDING
    if tasks.cost_nusd > record.worst_nusd:
        return ComparisonOutcome.ABOVE_WORST
    if known.calls == 0:
        return ComparisonOutcome.EXPECTED_UNKNOWN
    if known.cost_nusd > record.expected_nusd:
        return ComparisonOutcome.ABOVE_EXPECTED
    return ComparisonOutcome.WITHIN_EXPECTED


def compare_estimate(
    project_id: str,
    events: Iterable[Event],
    entries: Iterable[LedgerEntry],
) -> EstimateComparison:
    """Compare the latest ``finance.estimate`` of ``project_id`` with the ledger.

    ``events`` are the project's events (any types, any order); events of other
    projects are ignored. The latest estimate is the one with the greatest
    timestamp (the later one in input order on a tie). If it cannot be parsed the
    outcome is ``invalid_estimate``: an older estimate is never used instead, since
    the spending after it would include runs the latest estimate was made for.
    """
    if not isinstance(project_id, str) or not project_id:
        raise ValueError("project_id must be a non-empty string")
    latest: Event | None = None
    seen = 0
    for ev in events:
        if not isinstance(ev, Event):
            raise TypeError("events must be Event objects")
        if ev.type != ESTIMATE_EVENT or ev.project_id not in (None, project_id):
            continue
        seen += 1
        if latest is None or ev.ts >= latest.ts:
            latest = ev
    if latest is None:
        _check_entries(entries)
        return EstimateComparison(project_id=project_id, outcome=ComparisonOutcome.NO_ESTIMATE)
    try:
        record = EstimateRecord.from_event(latest)
    except ValueError as e:
        _check_entries(entries)
        return EstimateComparison(
            project_id=project_id,
            outcome=ComparisonOutcome.INVALID_ESTIMATE,
            estimates=seen,
            invalid_reason=str(e),
        )

    since = _utc(record.ts)
    covered = set(record.tasks)
    known_ids = covered - set(record.unknown)
    tasks, known, other = SpendGroup(), SpendGroup(), SpendGroup()
    for row in entries:
        if not isinstance(row, LedgerEntry):
            raise TypeError("entries must be LedgerEntry objects")
        if row.project_id != project_id or _utc(row.ts) < since:
            continue
        if row.task_id is not None and row.task_id in covered:
            tasks = tasks._add(row)
            if row.task_id in known_ids:
                known = known._add(row)
        else:
            other = other._add(row)
    return EstimateComparison(
        project_id=project_id,
        outcome=_outcome(record, tasks, known),
        estimate=record,
        estimates=seen,
        tasks=tasks,
        known=known,
        other=other,
    )


def _check_entries(entries: Iterable[LedgerEntry]) -> None:
    for e in entries:
        if not isinstance(e, LedgerEntry):
            raise TypeError("entries must be LedgerEntry objects")
