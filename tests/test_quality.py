"""Tests for the quality log (automa.quality) and its wiring (D56).

Local only: FakeProvider behind the real gateway/guard/ledger, real agents, real
ProjectStore. No network, no real model calls.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import test_cli_run as cr
import test_orchestrator as to
from automa import cli
from automa.budget import CostLedger
from automa.config import Tier
from automa.orchestrator import Orchestrator
from automa.plan import Plan
from automa.plan.models import ResultStatus, VerdictOutcome
from automa.project import ProjectStore
from automa.quality import (
    QUALITY_FILE,
    QualityError,
    QualityLog,
    QualityRecord,
    QualityStats,
    sync_project,
)

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


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


@pytest.fixture
def log() -> QualityLog:
    return QualityLog(":memory:")


# --- the log ------------------------------------------------------------------


def test_record_is_idempotent_and_conflicts_fail(log: QualityLog) -> None:
    assert log.record(rec()) is True
    assert log.record(rec()) is False
    with pytest.raises(QualityError, match="different data"):
        log.record(rec(outcome=VerdictOutcome.FAIL))
    assert log.records() == [rec()]


def test_record_type_and_validation(log: QualityLog) -> None:
    with pytest.raises(TypeError):
        log.record({"project_id": "p1"})  # type: ignore[arg-type]
    for bad in (
        {"cost_nusd": -1},
        {"attempt": 0},
        {"role": "Bad Role"},
        {"project_id": "../x"},
        {"model": ""},
        {"judged_at": datetime(2026, 1, 1)},  # naive
    ):
        with pytest.raises(ValueError):
            rec(**bad)


def test_records_filters_and_order(log: QualityLog) -> None:
    log.record(rec(task_id="b", judged_at=T0 + timedelta(seconds=2)))
    log.record(rec(task_id="a", role="developer", model="m-large"))
    log.record(rec(project_id="p2", judged_at=T0 + timedelta(seconds=1)))
    assert [(r.project_id, r.task_id) for r in log.records()] == [
        ("p1", "a"),
        ("p2", "a"),
        ("p1", "b"),
    ]
    assert [r.task_id for r in log.records(role="developer")] == ["a"]
    assert [r.project_id for r in log.records(model="m-medium")] == ["p2", "p1"]
    assert len(log.records(project_id="p2")) == 1


def test_stats_per_role_and_model(log: QualityLog) -> None:
    log.record(rec(attempt=1, outcome=VerdictOutcome.FAIL, cost_nusd=10))
    log.record(rec(attempt=2, outcome=VerdictOutcome.PASS, cost_nusd=11))
    log.record(rec(attempt=3, tier=Tier.SMALL, outcome=VerdictOutcome.FAIL, cost_nusd=0))
    log.record(rec(task_id="b", model="m-large", tier=Tier.LARGE, cost_nusd=7))
    log.record(rec(task_id="c", role="developer", outcome=VerdictOutcome.FAIL, cost_nusd=5))
    stats = log.stats()
    assert [(s.role, s.model) for s in stats] == [
        ("developer", "m-medium"),
        ("writer", "m-large"),
        ("writer", "m-medium"),
    ]
    dev, large, medium = stats
    assert dev.passed == 0 and dev.cost_per_pass_nusd is None and dev.pass_rate == 0.0
    assert medium.attempts == 3 and medium.passed == 1 and medium.failed == 2
    assert medium.tiers == [Tier.SMALL, Tier.MEDIUM]
    assert medium.cost_nusd == 21
    assert medium.avg_cost_nusd == 7
    assert medium.cost_per_pass_nusd == 21
    assert large.pass_rate == 1.0 and large.cost_per_pass_nusd == 7
    assert [s.model for s in log.stats(role="writer")] == ["m-large", "m-medium"]


def test_average_cost_rounds_up() -> None:
    s = QualityStats(role="w", model="m", tiers=[Tier.SMALL], attempts=3, passed=2, cost_nusd=10)
    assert s.avg_cost_nusd == 4 and s.cost_per_pass_nusd == 5


def test_empty_log(log: QualityLog) -> None:
    assert log.records() == [] and log.stats() == []


def test_persists_across_reopen(tmp_path: Path) -> None:
    with QualityLog.for_workspace(tmp_path / "ws") as log:
        log.record(rec())
    assert (tmp_path / "ws" / QUALITY_FILE).is_file()
    with QualityLog.for_workspace(tmp_path / "ws") as log:
        assert log.records() == [rec()]


def test_foreign_database_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "q.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE other (x)")
    conn.commit()
    conn.close()
    with pytest.raises(QualityError, match="not an Automa quality log"):
        QualityLog(path)


def test_unknown_schema_version_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "q.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(QualityError, match="schema version 99"):
        QualityLog(path)


def test_garbage_file_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "q.sqlite3"
    path.write_bytes(b"this is not sqlite at all" * 100)
    with pytest.raises(QualityError):
        QualityLog(path)


def test_symlinked_file_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real.sqlite3"
    QualityLog(real).close()
    link = tmp_path / "q.sqlite3"
    link.symlink_to(real)
    with pytest.raises(QualityError, match="symlink"):
        QualityLog(link)


def test_corrupt_row_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "q.sqlite3"
    with QualityLog(path) as log:
        log.record(rec())
    conn = sqlite3.connect(path)
    conn.execute("UPDATE attempts SET tier = 'huge'")
    conn.commit()
    conn.close()
    with QualityLog(path) as log:
        with pytest.raises(QualityError, match="corrupt"):
            log.records()
        with pytest.raises(QualityError, match="corrupt"):
            log.stats()


def test_concurrent_records(log: QualityLog) -> None:
    def worker(n: int) -> None:
        for i in range(20):
            log.record(rec(task_id=f"t{n}", attempt=i + 1))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(log.records()) == 80


def test_quality_module_never_touches_budget_or_models() -> None:
    import automa.quality as q

    src = Path(q.__file__).read_text(encoding="utf-8")
    for forbidden in ("automa.budget", "automa.llm", "BudgetGuard", "PermissionPolicy"):
        assert forbidden not in src


# --- from the project store ------------------------------------------------------


def test_from_attempt_rejects_mismatch(tmp_path: Path) -> None:
    rig = to.Rig(
        tmp_path,
        [*to.write("a"), *to.write("b")],
        to.make_plan(to.task_dict("a"), to.task_dict("b")),
    )
    rig.run()
    plan = rig.store.load_plan(to.PID)
    (ra, va), (rb, _vb) = (
        rig.store.task_history(to.PID, "a")[0],
        rig.store.task_history(to.PID, "b")[0],
    )
    QualityRecord.from_attempt(to.PID, plan.task("a"), ra, va)
    with pytest.raises(QualityError):
        QualityRecord.from_attempt(to.PID, plan.task("b"), ra, va)
    with pytest.raises(QualityError):
        QualityRecord.from_attempt(to.PID, plan.task("a"), rb, va)


def test_sync_project_imports_judged_attempts_once(tmp_path: Path) -> None:
    rig = to.Rig(
        tmp_path, [*to.write("a", "bad"), *to.write("a", "ok")], to.make_plan(to.task_dict("a"))
    )
    rig.run()  # no quality log wired
    log = QualityLog(":memory:")
    assert sync_project(log, rig.store, to.PID) == 2
    assert sync_project(log, rig.store, to.PID) == 0
    rows = log.records()
    assert [(r.attempt, r.outcome) for r in rows] == [
        (1, VerdictOutcome.FAIL),
        (2, VerdictOutcome.PASS),
    ]
    history = rig.store.task_history(to.PID, "a")
    assert [r.cost_nusd for r in rows] == [res.cost_nusd for res, _ in history]
    assert all(r.cost_nusd > 0 for r in rows)
    assert rows[0].model == "fake-medium" and rows[0].role == "writer"


def test_sync_project_without_plan_is_zero(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path / "ws")
    store.create("goal", project_id="p1")
    assert sync_project(QualityLog(":memory:"), store, "p1") == 0


def test_sync_skips_unjudged_attempt(tmp_path: Path) -> None:
    # The verifier's model reply is invalid twice -> result stored without a verdict.
    plan = Plan.model_validate(
        {
            "goal": "g",
            "tasks": [
                {
                    "id": "a",
                    "title": "Write a",
                    "role": "writer",
                    "acceptance_criteria": [{"description": "is nice"}],
                }
            ],
        }
    )
    script = [*to.write("a"), "not json", "still not json", "nope"]
    rig = to.Rig(tmp_path, script, plan)
    rig.run()
    assert rig.store.task_history(to.PID, "a")[0][1] is None
    assert sync_project(QualityLog(":memory:"), rig.store, to.PID) == 0


# --- orchestrator wiring -----------------------------------------------------------


def orch_with(rig: to.Rig, log: Any) -> Orchestrator:
    return Orchestrator(
        rig.store,
        rig.worker,
        rig.verifier,
        rig.roles,
        rig.cfg,
        tools=rig.tools,
        clock=to.clock,
        quality=log,
    )


def test_orchestrator_rejects_wrong_quality_type(tmp_path: Path) -> None:
    rig = to.Rig(tmp_path, [])
    with pytest.raises(TypeError, match="quality"):
        orch_with(rig, object())


def test_orchestrator_records_every_verdict(tmp_path: Path) -> None:
    script = [*to.write("a", "bad"), *to.write("a", "bad"), *to.write("a", "ok")]
    rig = to.Rig(tmp_path, script, to.make_plan(to.task_dict("a")))
    log = QualityLog(":memory:")
    orch_with(rig, log).run(to.PID)
    rows = log.records()
    assert [(r.attempt, r.tier, r.model, r.outcome) for r in rows] == [
        (1, Tier.MEDIUM, "fake-medium", VerdictOutcome.FAIL),
        (2, Tier.MEDIUM, "fake-medium", VerdictOutcome.FAIL),
        (3, Tier.LARGE, "fake-large", VerdictOutcome.PASS),
    ]
    # same data as the store: a later sync adds nothing
    assert sync_project(log, rig.store, to.PID) == 0
    stats = {s.model: s for s in log.stats()}
    assert stats["fake-medium"].pass_rate == 0.0 and stats["fake-large"].pass_rate == 1.0


def test_orchestrator_without_log_behaves_as_before(tmp_path: Path) -> None:
    rig = to.Rig(tmp_path, to.write("a"), to.make_plan(to.task_dict("a")))
    report = orch_with(rig, None).run(to.PID)
    assert report.status.value == "done"
    assert "quality.record_failed" not in rig.event_types()


class BrokenLog(QualityLog):
    def record(self, rec: QualityRecord) -> bool:
        raise QualityError("disk full")


def test_failing_log_never_changes_the_run(tmp_path: Path) -> None:
    rig = to.Rig(tmp_path, to.write("a"), to.make_plan(to.task_dict("a")))
    report = orch_with(rig, BrokenLog(":memory:")).run(to.PID)
    assert report.status.value == "done"
    failed = [e for e in rig.store.events(to.PID).read() if e.type == "quality.record_failed"]
    assert len(failed) == 1 and failed[0].data["task"] == "a"
    assert "disk full" in failed[0].data["error"]
    # nothing is lost: the stored verdict can be imported later
    assert sync_project(QualityLog(":memory:"), rig.store, to.PID) == 1


def test_parallel_run_records_all(tmp_path: Path) -> None:
    script = [*to.write("a"), *to.write("b")]
    rig = to.Rig(tmp_path, script, to.make_plan(to.task_dict("a"), to.task_dict("b")))
    log = QualityLog(":memory:")
    Orchestrator(
        rig.store,
        rig.worker,
        rig.verifier,
        rig.roles,
        rig.cfg,
        tools=rig.tools,
        clock=to.clock,
        quality=log,
        settings=to.OrchestratorSettings(max_parallel=2),
    ).run(to.PID)
    assert sorted(r.task_id for r in log.records()) == ["a", "b"]


# --- CLI wiring ------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Env:
    return cr.Env(tmp_path, monkeypatch)


def test_cli_run_records_quality(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a", "bad"), *cr.write("a"))
    r = env.run(cr.PID, "--max-attempts", "2")
    assert r.exit_code == cli.EXIT_OK, r.output
    with QualityLog.for_workspace(env.ws) as log:
        rows = log.records(project_id=cr.PID)
    assert [r.outcome for r in rows] == [VerdictOutcome.FAIL, VerdictOutcome.PASS]
    with CostLedger.for_workspace(env.ws) as ledger:
        assert sum(r.cost_nusd for r in rows) <= ledger.spent_nusd(project_id=cr.PID)


def test_cli_run_with_broken_log_is_config_error_without_calls(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    env.ws.mkdir(parents=True, exist_ok=True)
    (env.ws / QUALITY_FILE).write_bytes(b"garbage" * 200)
    r = env.run(cr.PID)
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert len(env.provider.requests) == 0
    assert env.store.task_history(cr.PID, "a") == []
    assert env.status().value == "new"


def test_bench_run_records_quality(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    import test_bench as tb

    env = tb.Env(tmp_path, monkeypatch)
    env.provider.push(
        json.dumps({"questions": [tb.QUESTION]}), json.dumps(tb.PLAN), *tb.build_attempt()
    )
    r = env.bench(tb.BID)
    assert r.exit_code == cli.EXIT_OK, r.output
    pid = env.record().project_id
    with QualityLog.for_workspace(env.ws) as log:
        rows = log.records(project_id=pid)
    assert [(r.task_id, r.outcome) for r in rows] == [("build", VerdictOutcome.PASS)]
