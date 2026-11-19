"""Tests for unusual-cost detection (automa.finance.anomaly, D69).

Pure: samples, ledger entries and quality records are built by hand, plus round
trips through a real SQLite ledger and quality log in tmp_path. No provider,
no network, no model calls.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import automa.finance.anomaly as an_mod
from automa.budget import CostLedger, LedgerEntry
from automa.config import Tier
from automa.finance import (
    ANOMALY_EVENT,
    Anomaly,
    AnomalyReport,
    AnomalySettings,
    CostSample,
    SampleKind,
    attempt_samples,
    call_samples,
    detect_anomalies,
)
from automa.plan.models import ResultStatus, VerdictOutcome
from automa.quality import QualityLog, QualityRecord
from test_ledger import FixedClock, make_pair

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
# The original sensitivity (D69): most detector tests keep it; D132 raised the defaults.
NO_FLOOR = AnomalySettings(factor=3.0, min_cost_nusd=0)
LEGACY = AnomalySettings(factor=3.0, min_cost_nusd=1_000_000)
_ids = iter(range(1, 100_000))


def call(
    cost: int,
    i: int = 0,
    *,
    role: str = "writer",
    model: str = "m",
    task: str = "t1",
    project: str = "p1",
    ts: datetime | None = None,
) -> CostSample:
    return CostSample(
        kind=SampleKind.CALL,
        ts=ts if ts is not None else T0 + timedelta(minutes=i),
        project_id=project,
        task_id=task,
        role=role,
        model=model,
        cost_nusd=cost,
        call_id=next(_ids),
    )


def series(costs: list[int], **kw: Any) -> list[CostSample]:
    return [call(c, i, **kw) for i, c in enumerate(costs)]


def entry(
    cost: int, *, project: str | None = "p1", task: str | None = "t1", i: int = 0
) -> LedgerEntry:
    return LedgerEntry(
        id=next(_ids),
        ts=T0 + timedelta(minutes=i),
        project_id=project,
        task_id=task,
        tier=Tier.MEDIUM,
        model="m-model",
        response_model="m-model",
        input_tokens=1,
        output_tokens=1,
        cost_nusd=cost,
        estimate_nusd=cost,
    )


def rec(**kw: Any) -> QualityRecord:
    data: dict[str, Any] = {
        "project_id": "p1",
        "task_id": "a",
        "attempt": 1,
        "role": "writer",
        "tier": Tier.MEDIUM,
        "model": "m-medium",
        "result_status": ResultStatus.COMPLETED,
        "outcome": VerdictOutcome.PASS,
        "model_evaluated": False,
        "cost_nusd": 1000,
        "judged_at": T0,
    }
    data.update(kw)
    return QualityRecord(**data)


def costs(r: AnomalyReport) -> list[int]:
    return [a.sample.cost_nusd for a in r.anomalies]


# --- settings and models -------------------------------------------------------


def test_default_settings() -> None:
    s = AnomalySettings()
    assert s.factor == 5.0 and s.min_samples == 5 and s.min_cost_nusd == 50_000_000


@pytest.mark.parametrize(
    "kw",
    [
        {"factor": 1.0},
        {"factor": 0.5},
        {"factor": float("inf")},
        {"factor": float("nan")},
        {"factor": 1001},
        {"min_samples": 1},
        {"min_samples": 0},
        {"min_cost_nusd": -1},
        {"other": 1},
    ],
)
def test_invalid_settings(kw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AnomalySettings(**kw)


def test_settings_are_frozen() -> None:
    with pytest.raises(ValidationError):
        AnomalySettings().factor = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kw",
    [
        {"kind": SampleKind.CALL, "call_id": None},
        {"kind": SampleKind.CALL, "call_id": 1, "attempt": 1},
        {"kind": SampleKind.ATTEMPT, "attempt": None},
        {"kind": SampleKind.ATTEMPT, "attempt": 1, "call_id": 1},
        {"kind": SampleKind.CALL, "call_id": 1, "cost_nusd": -1},
        {"kind": SampleKind.CALL, "call_id": 1, "role": ""},
        {"kind": SampleKind.CALL, "call_id": 1, "model": ""},
        {"kind": SampleKind.ATTEMPT, "attempt": 0},
    ],
)
def test_invalid_sample(kw: dict[str, Any]) -> None:
    data: dict[str, Any] = {
        "ts": T0,
        "project_id": "p1",
        "task_id": "t1",
        "role": "writer",
        "model": "m",
        "cost_nusd": 1,
    }
    data.update(kw)
    with pytest.raises(ValidationError):
        CostSample(**data)


def test_report_rejects_more_anomalies_than_checked() -> None:
    a = Anomaly(sample=call(10), baseline_nusd=1, history=5, threshold_nusd=4)
    with pytest.raises(ValidationError):
        AnomalyReport(settings=AnomalySettings(), checked=0, anomalies=(a,))


# --- detection -------------------------------------------------------------------


def test_empty() -> None:
    r = detect_anomalies([])
    assert r == AnomalyReport(settings=AnomalySettings())


def test_default_settings_used_when_none() -> None:
    assert detect_anomalies([]).settings == AnomalySettings()


def test_steady_costs_no_anomaly() -> None:
    r = detect_anomalies(series([100] * 20), NO_FLOOR)
    assert r.anomalies == () and r.checked == 15 and r.without_history == 5


def test_spike_is_flagged() -> None:
    r = detect_anomalies(series([100] * 5 + [1000]), NO_FLOOR)
    assert costs(r) == [1000]
    a = r.anomalies[0]
    assert a.baseline_nusd == 100 and a.history == 5 and a.threshold_nusd == 301
    assert a.ratio == 10.0


def test_threshold_is_strictly_above_factor() -> None:
    assert detect_anomalies(series([100] * 5 + [300]), NO_FLOOR).anomalies == ()
    assert costs(detect_anomalies(series([100] * 5 + [301]), NO_FLOOR)) == [301]


def test_custom_factor() -> None:
    s = AnomalySettings(factor=1.5, min_cost_nusd=0)
    assert detect_anomalies(series([100] * 5 + [150]), s).anomalies == ()
    assert costs(detect_anomalies(series([100] * 5 + [151]), s)) == [151]


def test_fractional_factor_is_exact() -> None:
    # 0.1 is not exact in binary; threshold uses the float's exact value.
    s = AnomalySettings(factor=1.1, min_cost_nusd=0)
    r = detect_anomalies(series([10**9] * 5 + [1_100_000_001]), s)
    assert costs(r) == [1_100_000_001]
    assert r.anomalies[0].threshold_nusd == 1_100_000_001


def test_not_enough_history() -> None:
    r = detect_anomalies(series([100] * 4 + [10_000]), NO_FLOOR)
    assert r.anomalies == () and r.checked == 0 and r.without_history == 5


def test_min_samples_setting() -> None:
    s = AnomalySettings(min_samples=2, min_cost_nusd=0)
    assert costs(detect_anomalies(series([100, 100, 10_000]), s)) == [10_000]


def test_floor_suppresses_tiny_costs() -> None:
    r = detect_anomalies(series([10] * 5 + [500_000]), LEGACY)  # floor $0.001
    assert r.anomalies == () and r.checked == 1
    r = detect_anomalies(series([10] * 5 + [1_000_000]), LEGACY)
    assert costs(r) == [1_000_000] and r.anomalies[0].threshold_nusd == 1_000_000


def test_zero_baseline() -> None:
    r = detect_anomalies(series([0] * 5 + [1]), NO_FLOOR)
    assert costs(r) == [1]
    assert r.anomalies[0].ratio is None and r.anomalies[0].threshold_nusd == 1
    assert detect_anomalies(series([0] * 6), NO_FLOOR).anomalies == ()


def test_median_resists_one_outlier_in_history() -> None:
    # A single huge earlier call must not raise the baseline.
    r = detect_anomalies(series([100, 100, 10**9, 100, 100, 1000]), NO_FLOOR)
    assert costs(r) == [1000] and r.anomalies[0].baseline_nusd == 100


def test_median_low_for_even_history() -> None:
    r = detect_anomalies(series([100, 100, 200, 200, 100, 200, 700]), NO_FLOOR)
    # history at the last sample: six values, median_low = 100
    assert r.anomalies[-1].baseline_nusd == 100 and costs(r) == [700]


def test_anomalies_join_history_so_a_lasting_shift_stops_flagging() -> None:
    r = detect_anomalies(series([100] * 5 + [1000] * 10), NO_FLOOR)
    # flagged while the median is still 100; once 1000s dominate, no more.
    assert 1 <= len(r.anomalies) < 10
    assert all(a.baseline_nusd == 100 for a in r.anomalies)
    assert r.anomalies[-1].sample.ts < T0 + timedelta(minutes=14)


def test_separate_history_per_role_model_and_kind() -> None:
    s = (
        series([100] * 5)
        + [call(1000, 10, role="tester")]  # other role: no history
        + [call(1000, 11, model="m2")]  # other model: no history
    )
    att = CostSample(
        kind=SampleKind.ATTEMPT,
        ts=T0 + timedelta(minutes=12),
        project_id="p1",
        task_id="t1",
        role="writer",
        model="m",
        cost_nusd=1000,
        attempt=1,
    )
    r = detect_anomalies([*s, att], NO_FLOOR)
    assert r.anomalies == () and r.checked == 0


def test_history_spans_projects() -> None:
    s = [call(100, i, project=f"p{i}") for i in range(5)] + [call(1000, 9, project="new")]
    assert costs(detect_anomalies(s, NO_FLOOR)) == [1000]


def test_order_is_by_time_not_input() -> None:
    s = series([100] * 5 + [1000])
    r = detect_anomalies(list(reversed(s)), NO_FLOOR)
    assert costs(r) == [1000]


def test_later_samples_are_not_history() -> None:
    # The spike is first in time: nothing earlier, so not checked.
    s = [call(1000, 0)] + [call(100, i) for i in range(1, 7)]
    r = detect_anomalies(s, NO_FLOOR)
    assert r.anomalies == ()


def test_same_ts_keeps_input_order() -> None:
    s = [call(100, ts=T0) for _ in range(5)] + [call(1000, ts=T0)]
    assert costs(detect_anomalies(s, NO_FLOOR)) == [1000]
    s2 = [call(1000, ts=T0)] + [call(100, ts=T0) for _ in range(5)]
    assert detect_anomalies(s2, NO_FLOOR).anomalies == ()


def test_timezones_and_naive_are_utc() -> None:
    tz = timezone(timedelta(hours=3, minutes=30))
    early = [call(100, ts=(T0 + timedelta(minutes=i)).astimezone(tz)) for i in range(5)]
    spike = call(1000, ts=(T0 + timedelta(minutes=10)).replace(tzinfo=None))
    before = call(5000, ts=(T0 - timedelta(hours=1)).astimezone(tz))
    r = detect_anomalies([spike, *early, before], NO_FLOOR)
    # 'before' is first in time (no history); spike sees 6 earlier samples, median 100
    assert costs(r) == [1000] and r.anomalies[0].history == 6


def test_results_are_chronological() -> None:
    s = series([100] * 5 + [1000, 100, 2000])
    r = detect_anomalies(list(reversed(s)), NO_FLOOR)
    assert costs(r) == [1000, 2000]


def test_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        detect_anomalies([object()])  # type: ignore[list-item]


def test_accepts_generator_and_does_not_mutate() -> None:
    s = series([100] * 5 + [1000])
    before = [x.model_copy() for x in s]
    r = detect_anomalies(x for x in s)
    assert s == before
    assert r.checked == 1


# --- event payload ---------------------------------------------------------------


def test_event_data_is_json_safe() -> None:
    r = detect_anomalies(series([100] * 5 + [1000]), NO_FLOOR)
    data = r.anomalies[0].event_data()
    assert json.loads(json.dumps(data)) == data
    assert data["kind"] == "call" and data["cost_nusd"] == 1000
    assert data["baseline_nusd"] == 100 and data["threshold_nusd"] == 301
    assert data["attempt"] is None and isinstance(data["call_id"], int)
    assert data["sample_ts"].endswith("+00:00")
    assert ANOMALY_EVENT == "finance.anomaly"


def test_cost_usd() -> None:
    a = Anomaly(sample=call(2_000_000_000), baseline_nusd=1, history=5, threshold_nusd=4)
    assert a.cost_usd == 2.0


# --- sample builders --------------------------------------------------------------


def test_call_samples_role_from_map_and_left_out() -> None:
    roles = {("p1", "t1"): "writer", ("p2", "t1"): "tester"}
    entries = [
        entry(1, project="p1", task="t1"),
        entry(2, project="p2", task="t1"),
        entry(3, project="p1", task=None),  # planning/review
        entry(4, project=None, task="t1"),  # no project
        entry(5, project="p1", task="zz"),  # unknown task
    ]
    out, left_out = call_samples(entries, roles)
    assert [(s.role, s.cost_nusd) for s in out] == [("writer", 1), ("tester", 2)]
    assert left_out == 3
    assert all(s.kind is SampleKind.CALL and s.call_id is not None for s in out)
    assert out[0].call_id == entries[0].id and out[0].model == "m-model"


def test_call_samples_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        call_samples([object()], {})  # type: ignore[list-item]


def test_attempt_samples() -> None:
    out = attempt_samples([rec(attempt=2, cost_nusd=7, task_id="b", role="tester")])
    s = out[0]
    assert s.kind is SampleKind.ATTEMPT and s.attempt == 2 and s.call_id is None
    assert (s.task_id, s.role, s.model, s.cost_nusd, s.ts) == ("b", "tester", "m-medium", 7, T0)


def test_attempt_samples_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        attempt_samples([object()])  # type: ignore[list-item]


def test_attempt_spike_end_to_end() -> None:
    records = [
        rec(task_id=f"t{i}", cost_nusd=2_000_000, judged_at=T0 + timedelta(minutes=i))
        for i in range(5)
    ] + [rec(task_id="big", cost_nusd=50_000_000, judged_at=T0 + timedelta(hours=1))]
    r = detect_anomalies(attempt_samples(records))
    assert [a.sample.task_id for a in r.anomalies] == ["big"]
    assert r.anomalies[0].event_data()["attempt"] == 1


def test_round_trip_through_real_stores(tmp_path: Path) -> None:
    clock = FixedClock(datetime(2026, 11, 9, 12, 0, tzinfo=UTC))
    path = tmp_path / "ledger.sqlite3"
    with CostLedger(path, clock=clock) as led:
        for _ in range(5):
            led.record(*make_pair(0.01))
            clock.now += timedelta(minutes=1)
        led.record(*make_pair(0.2))
    with CostLedger(path, read_only=True) as ro:
        samples, left_out = call_samples(ro.entries(), {("p1", "t1"): "writer"})
    assert left_out == 0
    r = detect_anomalies(samples)
    assert len(r.anomalies) == 1 and r.anomalies[0].sample.cost_nusd == 200_000_000

    qpath = tmp_path / "quality.sqlite3"
    with QualityLog(qpath) as q:
        for i in range(6):
            q.record(
                rec(
                    task_id=f"t{i}",
                    cost_nusd=10**9 if i == 5 else 10**6,
                    judged_at=T0 + timedelta(minutes=i),
                )
            )
    with QualityLog(qpath, read_only=True) as q:
        r2 = detect_anomalies(attempt_samples(q.records()))
    assert [a.sample.task_id for a in r2.anomalies] == ["t5"]


def test_module_is_pure() -> None:
    """No model, provider, gateway, project store, guard or file I/O."""
    src = Path(an_mod.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any(
        m.startswith(("automa.llm", "automa.project", "automa.budget.", "automa.orchestrator"))
        for m in imported
    )
    for name in ("open(", "write_text", "sqlite3", "BudgetGuard(", "Gateway(", "Path("):
        assert name not in src
