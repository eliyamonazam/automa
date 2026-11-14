"""Actual spending report from the cost ledger (Financial Controller, core; D65).

``spending_report`` groups recorded calls (``LedgerEntry``) by project, UTC day,
model and role and sums calls, tokens, cost and the per-call worst-case
reservations the guard checked before each call.

The ledger does not store a role. A call's role is the role of its task, looked
up in ``roles`` (``{(project_id, task_id): role}``, built by the caller from the
project plans). Verifier calls carry the task id, so they count towards the
task's role. Calls without a task (planning, review) go to ``no_task``; calls
whose task is not in ``roles`` go to ``unknown_role``. Calls without a project go
to ``no_project``. Every grouping therefore adds up exactly to the total.

Pure and read-only: no model call, no write, no authorization. Money stays in
integer nano-USD (the ledger's own unit), so sums are exact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from automa.budget import LedgerEntry, nusd_to_usd


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SpendGroup(_Frozen):
    """Sums over a set of recorded calls."""

    key: str | None = None
    calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_nusd: int = Field(default=0, ge=0)
    reserved_nusd: int = Field(default=0, ge=0)  # per-call worst cases checked by the guard

    @property
    def cost_usd(self) -> float:
        return nusd_to_usd(self.cost_nusd)

    def _add(self, e: LedgerEntry) -> SpendGroup:
        return SpendGroup(
            key=self.key,
            calls=self.calls + 1,
            input_tokens=self.input_tokens + e.input_tokens,
            output_tokens=self.output_tokens + e.output_tokens,
            cost_nusd=self.cost_nusd + e.cost_nusd,
            reserved_nusd=self.reserved_nusd + e.estimate_nusd,
        )


class SpendingReport(_Frozen):
    """Actual spending, grouped. Each grouping (with its extra buckets) sums to ``total``."""

    project_id: str | None = None  # filter, if any
    since: date | None = None  # inclusive UTC day filter
    until: date | None = None  # inclusive UTC day filter
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    total: SpendGroup
    by_project: tuple[SpendGroup, ...] = ()
    no_project: SpendGroup = SpendGroup()
    by_day: tuple[SpendGroup, ...] = ()  # chronological
    by_model: tuple[SpendGroup, ...] = ()
    by_role: tuple[SpendGroup, ...] = ()
    no_task: SpendGroup = SpendGroup()  # planning / review calls
    unknown_role: SpendGroup = SpendGroup()  # task not found in the role map

    @model_validator(mode="after")
    def _adds_up(self) -> SpendingReport:
        t = self.total.cost_nusd
        checks = (
            sum(g.cost_nusd for g in self.by_project) + self.no_project.cost_nusd,
            sum(g.cost_nusd for g in self.by_day),
            sum(g.cost_nusd for g in self.by_model),
            sum(g.cost_nusd for g in self.by_role)
            + self.no_task.cost_nusd
            + self.unknown_role.cost_nusd,
        )
        if any(c != t for c in checks):
            raise ValueError("report groups do not add up to the total")
        return self


def _day(e: LedgerEntry) -> date:
    ts = e.ts if e.ts.tzinfo is not None else e.ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).date()


def _by_cost(groups: Iterable[SpendGroup]) -> tuple[SpendGroup, ...]:
    """Most expensive first; ties by key so the order is deterministic."""
    return tuple(sorted(groups, key=lambda g: (-g.cost_nusd, g.key or "")))


def spending_report(
    entries: Iterable[LedgerEntry],
    *,
    roles: Mapping[tuple[str, str], str] | None = None,
    project_id: str | None = None,
    since: date | None = None,
    until: date | None = None,
) -> SpendingReport:
    """Group ``entries`` (optionally filtered by project and inclusive UTC days)."""
    if since is not None and until is not None and since > until:
        raise ValueError("since must not be after until")
    role_map: Mapping[tuple[str, str], str] = roles if roles is not None else {}
    for k, v in role_map.items():
        if not (
            isinstance(k, tuple)
            and len(k) == 2
            and all(isinstance(p, str) for p in k)
            and isinstance(v, str)
            and v
        ):
            raise TypeError("roles must map (project_id, task_id) to a role name")

    total = SpendGroup()
    no_project, no_task, unknown_role = SpendGroup(), SpendGroup(), SpendGroup()
    projects: dict[str, SpendGroup] = {}
    days: dict[str, SpendGroup] = {}
    models: dict[str, SpendGroup] = {}
    by_role: dict[str, SpendGroup] = {}
    first: datetime | None = None
    last: datetime | None = None

    def bump(bucket: dict[str, SpendGroup], key: str, e: LedgerEntry) -> None:
        bucket[key] = bucket.get(key, SpendGroup(key=key))._add(e)

    for e in entries:
        if not isinstance(e, LedgerEntry):
            raise TypeError("entries must be LedgerEntry objects")
        d = _day(e)
        if project_id is not None and e.project_id != project_id:
            continue
        if (since is not None and d < since) or (until is not None and d > until):
            continue
        total = total._add(e)
        first = e.ts if first is None or e.ts < first else first
        last = e.ts if last is None or e.ts > last else last
        if e.project_id is None:
            no_project = no_project._add(e)
        else:
            bump(projects, e.project_id, e)
        bump(days, d.isoformat(), e)
        bump(models, e.model, e)
        if e.task_id is None:
            no_task = no_task._add(e)
        else:
            role = role_map.get((e.project_id, e.task_id)) if e.project_id is not None else None
            if role is None:
                unknown_role = unknown_role._add(e)
            else:
                bump(by_role, role, e)

    return SpendingReport(
        project_id=project_id,
        since=since,
        until=until,
        first_ts=first,
        last_ts=last,
        total=total,
        by_project=_by_cost(projects.values()),
        no_project=no_project,
        by_day=tuple(sorted(days.values(), key=lambda g: g.key or "")),
        by_model=_by_cost(models.values()),
        by_role=_by_cost(by_role.values()),
        no_task=no_task,
        unknown_role=unknown_role,
    )
