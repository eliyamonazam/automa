"""Tests for the ``unusual spending:`` lines and ``finance.anomaly`` events of
``automa run`` (D71).

Local only: FakeProvider behind the real gateway/guard/ledger, real Orchestrator,
ProjectStore, QualityLog and anomaly check. History is written directly through
the real ``CostLedger`` and ``QualityLog``. No network, no real model calls.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import test_cli_finance_report as fr
import test_cli_run as cr
import test_cli_run_router as rr
from automa import cli
from automa.budget.ledger import CostLedger, usd_to_nusd
from automa.config import Tier
from automa.events import JsonlEventLog
from automa.llm import ProviderError
from automa.quality import QUALITY_FILE, QualityError, QualityLog
from test_ledger import FixedClock, make_pair

# Twenty times the router test prices: a fake call costs a few $0.001 (above the
# default $0.001 floor) while the pre-call cap still fits per_call_usd.
TOML = """
workspace = "{ws}"
[tiers]
small = "fake-small"
medium = "fake-medium"
large = "fake-large"
[prices."fake-small"]
input_per_mtok = 10.0
output_per_mtok = 20.0
[prices."fake-medium"]
input_per_mtok = 20.0
output_per_mtok = 40.0
[prices."fake-large"]
input_per_mtok = 40.0
output_per_mtok = 80.0
[limits]
project_usd = 5.0
daily_usd = 2.0
per_call_usd = 0.25
"""

PAST = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
CHEAP = 0.0005  # history cost per call: a fake call of this run is > 3x


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Env:
    e = cr.Env(tmp_path, monkeypatch)
    e.config.write_text(TOML.format(ws=e.ws.as_posix()), encoding="utf-8")
    return e


def history(
    env: cr.Env, n: int = 5, cost: float = CHEAP, project: str = "old", model: str = "fake-medium"
) -> None:
    """``n`` earlier writer calls of ``model`` in project ``old`` (task ``h``)."""
    if not env.store.exists(project):
        fr.add_project(env, project, ("h", "writer"))
    clock = FixedClock(PAST)
    with CostLedger(fr.ledger_path(env), clock=clock) as led:
        for i in range(n):
            clock.now = PAST + timedelta(minutes=i)
            est, res = make_pair(cost, project=project, task="h", tier=Tier.MEDIUM)
            led.record(est.model_copy(update={"model": model}), res)


def anomaly_events(env: cr.Env) -> list[dict[str, Any]]:
    return [e.data for e in env.store.events(cr.PID).read() if e.type == "finance.anomaly"]


def lines(output: str) -> list[str]:
    return [x for x in output.splitlines() if x.startswith("unusual spending: ")]


def run_one(env: cr.Env, *args: str) -> Any:
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    return env.run(cr.PID, "--yes", *args)


def run_ledger_ids(env: cr.Env) -> list[int]:
    with CostLedger(fr.ledger_path(env), read_only=True) as led:
        return [e.id for e in led.entries(project_id=cr.PID)]


# --- nothing unusual -----------------------------------------------------------------


def test_fresh_workspace_none_this_run(env: cr.Env) -> None:
    r = run_one(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert lines(r.output) == ["unusual spending: none this run"]
    assert anomaly_events(env) == []


def test_line_comes_after_cost(env: cr.Env) -> None:
    out = run_one(env).output.splitlines()
    cost = next(i for i, x in enumerate(out) if x.startswith("cost: "))
    unusual = next(i for i, x in enumerate(out) if x.startswith("unusual spending: "))
    status = next(i for i, x in enumerate(out) if x.startswith("status: "))
    assert cost < unusual < status


def test_too_little_history_is_not_flagged(env: cr.Env) -> None:
    history(env, n=3)  # the verifier call then has 4 earlier ones (worker included)
    r = run_one(env)
    assert lines(r.output) == ["unusual spending: none this run"]
    assert anomaly_events(env) == []


def test_history_of_another_model_is_not_a_baseline(env: cr.Env) -> None:
    history(env, model="fake-small")
    assert lines(run_one(env).output) == ["unusual spending: none this run"]


def test_similar_cost_is_not_flagged(env: cr.Env) -> None:
    history(env, cost=0.005)
    assert lines(run_one(env).output) == ["unusual spending: none this run"]


# --- calls -----------------------------------------------------------------------------


def test_spike_against_ledger_history(env: cr.Env) -> None:
    history(env)
    r = run_one(env)
    assert r.exit_code == cli.EXIT_OK, r.output  # advisory: the run is unchanged
    ids = run_ledger_ids(env)
    assert len(ids) == 2  # worker + verifier
    (head,) = lines(r.output)
    assert head.startswith("unusual spending: 2 this run (advisory: nothing was stopped")
    assert f"automa finance report --project {cr.PID}" in head
    detail = [x for x in r.output.splitlines() if x.startswith("  call #")]
    assert len(detail) == 2
    assert all("a  writer  fake-medium" in x for x in detail)
    # The worker call of this run is history for its verifier call.
    assert "5 earlier" in detail[0] and "6 earlier" in detail[1]
    assert f"vs median {cli._usd(usd_to_nusd(CHEAP))}" in detail[0]
    events = anomaly_events(env)
    assert [e["call_id"] for e in events] == ids
    assert [e["history"] for e in events] == [5, 6]
    for e in events:
        assert e["kind"] == "call" and e["task_id"] == "a" and e["role"] == "writer"
        assert e["model"] == "fake-medium" and e["attempt"] is None
        assert e["baseline_nusd"] == usd_to_nusd(CHEAP)
        assert e["cost_nusd"] >= e["threshold_nusd"] > 3 * e["baseline_nusd"]
    json.dumps(events)


def test_events_carry_the_project_id(env: cr.Env) -> None:
    history(env)
    run_one(env)
    raw = [e for e in env.store.events(cr.PID).read() if e.type == "finance.anomaly"]
    assert raw and all(e.project_id == cr.PID for e in raw)


def test_history_spike_of_another_project_is_not_reported(env: cr.Env) -> None:
    history(env, n=6)
    history(env, n=1, cost=0.05)  # an old anomaly in project "old"
    history(env, n=7, cost=0.05)  # the new price becomes the median
    r = run_one(env)
    assert lines(r.output) == ["unusual spending: none this run"]
    assert anomaly_events(env) == []


def test_calls_of_earlier_runs_are_not_reported_again(env: cr.Env) -> None:
    history(env)
    env.project(cr.task_dict("a"), cr.task_dict("b", "a"))
    env.script(*cr.write("a"), ProviderError("down"))
    first = env.run(cr.PID, "--yes")
    assert first.exit_code == cli.EXIT_PROVIDER, first.output
    first_ids = [e["call_id"] for e in anomaly_events(env)]
    assert first_ids and lines(first.output)[0].startswith(f"unusual spending: {len(first_ids)} ")
    env.script(*cr.write("b"))
    second = env.run(cr.PID, "--yes")
    assert second.exit_code == cli.EXIT_OK, second.output
    events = anomaly_events(env)
    later = [e["call_id"] for e in events[len(first_ids) :]]
    assert set(later).isdisjoint(first_ids)
    assert all(e["task_id"] == "b" for e in events[len(first_ids) :])
    assert len(set(e["call_id"] for e in events)) == len(events)  # no duplicates


def test_runs_after_an_interrupt(env: cr.Env) -> None:
    history(env)
    env.project(cr.task_dict("a"))
    env.script(cr.calls(("write_file", {"path": "a.txt", "content": "ok"})), KeyboardInterrupt())
    r = env.run(cr.PID, "--yes")
    assert r.exit_code == cli.EXIT_INTERRUPTED
    assert lines(r.output)[0].startswith("unusual spending: 1 this run")
    assert [e["call_id"] for e in anomaly_events(env)] == run_ledger_ids(env)


# --- judged attempts -------------------------------------------------------------------


def test_attempt_spike_against_quality_history(env: cr.Env) -> None:
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=5)  # 1000 nUSD per attempt
    r = run_one(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    (event,) = anomaly_events(env)
    assert event["kind"] == "attempt" and event["attempt"] == 1 and event["call_id"] is None
    assert event["task_id"] == "a" and event["baseline_nusd"] == 1000
    assert lines(r.output)[0].startswith("unusual spending: 1 this run")
    assert any(x.startswith("  attempt 1  a  writer  fake-medium") for x in r.output.splitlines())


def test_attempts_judged_before_the_run_are_not_reported(env: cr.Env) -> None:
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=5)
    env.project(cr.task_dict("a"), cr.task_dict("b", "a"))
    env.script(*cr.write("a"), ProviderError("down"))
    env.run(cr.PID, "--yes")
    assert [e["task_id"] for e in anomaly_events(env)] == ["a"]
    env.script(*cr.write("b"))
    env.run(cr.PID, "--yes")
    assert [e["task_id"] for e in anomaly_events(env)] == ["a", "b"]


# --- the same check as automa finance report -------------------------------------------


def test_matches_finance_report(env: cr.Env) -> None:
    history(env)
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=5)
    assert run_one(env).exit_code == cli.EXIT_OK
    data = fr.report_json(env, "--project", cr.PID)
    shown = {(a["kind"], a["call_id"], a["attempt"]) for a in data["anomalies"]["anomalies"]}
    events = {(e["kind"], e["call_id"], e["attempt"]) for e in anomaly_events(env)}
    assert events == shown and len(events) == 3
    for a in data["anomalies"]["anomalies"]:
        match = next(
            e
            for e in anomaly_events(env)
            if e["call_id"] == a["call_id"] and e["attempt"] == a["attempt"]
        )
        for key, value in match.items():
            assert a[key] == value


# --- advisory: failures never change the run -------------------------------------------


def test_unreadable_quality_records_check_calls_only(
    env: cr.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    history(env)

    def broken(self: QualityLog, **kw: Any) -> list[Any]:
        raise QualityError("cannot read quality log: boom")

    monkeypatch.setattr(QualityLog, "records", broken)
    r = run_one(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    (head,) = lines(r.output)
    assert "judged attempts not checked" in head
    assert {e["kind"] for e in anomaly_events(env)} == {"call"}


def test_check_failure_is_unavailable(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    history(env)

    def boom(*a: Any, **kw: Any) -> Any:
        raise ValueError("bad\nsample")

    monkeypatch.setattr(cli, "detect_anomalies", boom)
    r = run_one(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert lines(r.output) == ["unusual spending: unavailable (bad sample)"]
    assert anomaly_events(env) == []


def test_unwritable_event_is_ignored(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    history(env)
    real = JsonlEventLog.emit

    def emit(self: JsonlEventLog, event: Any) -> None:
        if event.type == "finance.anomaly":
            raise OSError("disk full")
        real(self, event)

    monkeypatch.setattr(JsonlEventLog, "emit", emit)
    r = run_one(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert lines(r.output)[0].startswith("unusual spending: 2 this run")


def test_unreadable_plan_of_another_project_leaves_its_calls_out(env: cr.Env) -> None:
    history(env)
    plan = next((env.ws / "projects" / "old").glob("plan*"))
    plan.write_text("{not json", encoding="utf-8")
    r = run_one(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert lines(r.output) == ["unusual spending: none this run"]  # no baseline left


def test_no_extra_model_calls(env: cr.Env) -> None:
    history(env)
    run_one(env)
    assert len(env.provider.requests) == 2  # worker + verifier only


def test_finance_report_still_writes_nothing(env: cr.Env) -> None:
    history(env)
    run_one(env)
    before = {p: p.read_bytes() for p in env.ws.rglob("*") if p.is_file()}
    CliRunner().invoke(cli.app, ["finance", "report", "--config", str(env.config)])
    after = {p: p.read_bytes() for p in env.ws.rglob("*") if p.is_file()}
    assert before == after
    assert (env.ws / QUALITY_FILE).exists()
