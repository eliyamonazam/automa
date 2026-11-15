"""Unusual-cost detection (Financial Controller, core; D69).

A cost sample is one model call (from the cost ledger) or one judged attempt
(from the quality log). ``detect_anomalies`` compares each sample with the
history of *earlier* samples of the same kind, role and model: when there are at
least ``min_samples`` of them and the sample costs more than ``factor`` times
their median (and at least ``min_cost_nusd``), it is an anomaly.

Advisory only: nothing here calls a model, writes anything or stops a run. The
``BudgetGuard`` remains the only authority over money.

Every sample joins the history after it is checked, anomalies included, so a
lasting price change stops being flagged once it dominates the median. Money
stays in integer nano-USD; the threshold comparison is exact (``Fraction``).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from fractions import Fraction
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from automa.budget import LedgerEntry, nusd_to_usd
from automa.quality import QualityRecord

ANOMALY_EVENT = "finance.anomaly"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SampleKind(StrEnum):
    CALL = "call"  # one ledger row (worker or verifier call)
    ATTEMPT = "attempt"  # one judged attempt (worker cost from the quality log)


class AnomalySettings(_Frozen):
    factor: float = Field(default=3.0, gt=1.0, le=1_000.0)
    min_samples: int = Field(default=5, ge=2, le=10_000)
    min_cost_nusd: int = Field(default=1_000_000, ge=0)  # ignore anything below $0.001


class CostSample(_Frozen):
    kind: SampleKind
    ts: datetime
    project_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    model: str = Field(min_length=1)
    cost_nusd: int = Field(ge=0)
    attempt: int | None = Field(default=None, ge=1)  # attempts only
    call_id: int | None = None  # calls only (ledger row id)

    @model_validator(mode="after")
    def _kind_fields(self) -> CostSample:
        if self.kind is SampleKind.CALL and (self.call_id is None or self.attempt is not None):
            raise ValueError("a call sample needs call_id and no attempt")
        if self.kind is SampleKind.ATTEMPT and (self.attempt is None or self.call_id is not None):
            raise ValueError("an attempt sample needs attempt and no call_id")
        return self

    @property
    def key(self) -> tuple[SampleKind, str, str]:
        return (self.kind, self.role, self.model)


class Anomaly(_Frozen):
    sample: CostSample
    baseline_nusd: int = Field(ge=0)  # median of the earlier samples
    history: int = Field(ge=2)  # how many earlier samples
    threshold_nusd: int = Field(ge=0)  # smallest cost that would have been flagged

    @property
    def ratio(self) -> float | None:
        """cost / baseline; ``None`` when the baseline is zero."""
        if self.baseline_nusd == 0:
            return None
        return self.sample.cost_nusd / self.baseline_nusd

    def event_data(self) -> dict[str, Any]:
        """Payload for a ``finance.anomaly`` event (integer nUSD, JSON-safe)."""
        s = self.sample
        return {
            "kind": s.kind.value,
            "task_id": s.task_id,
            "attempt": s.attempt,
            "call_id": s.call_id,
            "role": s.role,
            "model": s.model,
            "cost_nusd": s.cost_nusd,
            "baseline_nusd": self.baseline_nusd,
            "threshold_nusd": self.threshold_nusd,
            "history": self.history,
            "sample_ts": _utc(s.ts).isoformat(),
        }

    @property
    def cost_usd(self) -> float:
        return nusd_to_usd(self.sample.cost_nusd)


class AnomalyReport(_Frozen):
    settings: AnomalySettings
    checked: int = Field(default=0, ge=0)  # samples with enough history
    without_history: int = Field(default=0, ge=0)  # samples with too little history
    anomalies: tuple[Anomaly, ...] = ()  # chronological

    @model_validator(mode="after")
    def _consistent(self) -> AnomalyReport:
        if len(self.anomalies) > self.checked:
            raise ValueError("more anomalies than checked samples")
        return self


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def call_samples(
    entries: Iterable[LedgerEntry], roles: Mapping[tuple[str, str], str]
) -> tuple[list[CostSample], int]:
    """Ledger rows as call samples, and how many rows were left out.

    A row needs a project, a task and a known role (``roles`` maps
    ``(project_id, task_id)`` to the task's role, as in ``spending_report``);
    planning/review calls and unknown tasks are left out and counted.
    """
    out: list[CostSample] = []
    left_out = 0
    for e in entries:
        if not isinstance(e, LedgerEntry):
            raise TypeError("entries must be LedgerEntry objects")
        role = (
            roles.get((e.project_id, e.task_id))
            if e.project_id is not None and e.task_id is not None
            else None
        )
        if role is None or e.project_id is None or e.task_id is None:
            left_out += 1
            continue
        out.append(
            CostSample(
                kind=SampleKind.CALL,
                ts=e.ts,
                project_id=e.project_id,
                task_id=e.task_id,
                role=role,
                model=e.model,
                cost_nusd=e.cost_nusd,
                call_id=e.id,
            )
        )
    return out, left_out


def attempt_samples(records: Iterable[QualityRecord]) -> list[CostSample]:
    """Judged attempts as attempt samples (role = task type, worker cost)."""
    out: list[CostSample] = []
    for r in records:
        if not isinstance(r, QualityRecord):
            raise TypeError("records must be QualityRecord objects")
        out.append(
            CostSample(
                kind=SampleKind.ATTEMPT,
                ts=r.judged_at,
                project_id=r.project_id,
                task_id=r.task_id,
                role=r.role,
                model=r.model,
                cost_nusd=r.cost_nusd,
                attempt=r.attempt,
            )
        )
    return out


def _threshold(baseline: int, factor: Fraction, floor: int) -> int:
    """Smallest integer cost that is > factor × baseline and ≥ floor."""
    above = math.floor(factor * baseline) + 1
    return max(above, floor)


def detect_anomalies(
    samples: Iterable[CostSample], settings: AnomalySettings | None = None
) -> AnomalyReport:
    """Flag samples that cost far more than the earlier samples of their (kind, role, model).

    Samples are processed in time order (UTC; ties keep the input order). Only
    strictly earlier input positions in that order count as history.
    """
    cfg = settings if settings is not None else AnomalySettings()
    factor = Fraction(cfg.factor)
    items = list(samples)
    for s in items:
        if not isinstance(s, CostSample):
            raise TypeError("samples must be CostSample objects")
    ordered = sorted(enumerate(items), key=lambda p: (_utc(p[1].ts), p[0]))

    history: dict[tuple[SampleKind, str, str], list[int]] = {}
    checked = without = 0
    found: list[Anomaly] = []
    for _, s in ordered:
        past = history.setdefault(s.key, [])
        if len(past) < cfg.min_samples:
            without += 1
        else:
            checked += 1
            baseline = int(statistics.median_low(past))
            threshold = _threshold(baseline, factor, cfg.min_cost_nusd)
            if s.cost_nusd >= threshold:
                found.append(
                    Anomaly(
                        sample=s,
                        baseline_nusd=baseline,
                        history=len(past),
                        threshold_nusd=threshold,
                    )
                )
        past.append(s.cost_nusd)

    return AnomalyReport(
        settings=cfg, checked=checked, without_history=without, anomalies=tuple(found)
    )
