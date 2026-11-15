"""Tests for the "estimate vs actual" part of ``automa finance report`` (D68).

Estimate events are written through the project's real event log and ledger rows
through the real ``CostLedger`` (no provider); one end-to-end test runs
``automa run`` with the ``FakeProvider``. No network, no real model calls.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import test_cli_run as cr
from automa import cli
from automa.budget.ledger import CostLedger, usd_to_nusd
from automa.events import Event
from test_ledger import FixedClock, make_pair

T0 = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Env:
    return cr.Env(tmp_path, monkeypatch)


def report(env: cr.Env, *args: str) -> Any:
    return CliRunner().invoke(cli.app, ["finance", "report", *args, "--config", str(env.config)])


def report_json(env: cr.Env, *args: str) -> dict[str, Any]:
    r = report(env, *args, "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    data: dict[str, Any] = json.loads(r.output)
    return data


def comparisons(env: cr.Env, *args: str) -> dict[str, dict[str, Any]]:
    return {c["project"]: c for c in report_json(env, *args)["estimate_vs_actual"]}


def record(env: cr.Env, *calls: tuple[float, str | None, str | None, int]) -> None:
    """Each call: (cost_usd, project, task, minute offset from T0)."""
    env.ws.mkdir(parents=True, exist_ok=True)
    clock = FixedClock(T0)
    with CostLedger(env.ws / cli.LEDGER_FILENAME, clock=clock) as led:
        for cost, project, task, offset in calls:
            clock.now = T0 + timedelta(minutes=offset)
            led.record(*make_pair(cost, project=project, task=task))


def est_data(**over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "plan_version": 1,
        "tasks": ["t1", "t2"],
        "excluded": [],
        "router": False,
        "expected_nusd": usd_to_nusd(0.5),
        "expected_complete": True,
        "unknown": [],
        "worst_nusd": usd_to_nusd(2.0),
        "project_remaining_nusd": usd_to_nusd(10.0),
        "assessment": "worst_fits",
    }
    d.update(over)
    return d


def add_project(env: cr.Env, pid: str = "p1") -> None:
    env.store.create("goal", project_id=pid)


def estimate(env: cr.Env, pid: str = "p1", minute: int = 0, **over: Any) -> None:
    env.store.events(pid).emit(
        Event(
            ts=T0 + timedelta(minutes=minute),
            type="finance.estimate",
            project_id=pid,
            data=est_data(**over),
        )
    )


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


# --- JSON -----------------------------------------------------------------------


def test_no_projects_is_an_empty_list(env: cr.Env) -> None:
    assert report_json(env)["estimate_vs_actual"] == []


def test_project_without_estimate(env: cr.Env) -> None:
    add_project(env)
    c = comparisons(env)["p1"]
    assert c["outcome"] == "no_estimate"
    assert c["estimate"] is None and c["expected_delta_nusd"] is None
    assert c["estimates"] == 0


def test_within_expected(env: cr.Env) -> None:
    add_project(env)
    estimate(env)
    record(env, (0.1, "p1", "t1", 1), (0.2, "p1", "t2", 2))
    c = comparisons(env)["p1"]
    assert c["outcome"] == "within_expected"
    assert c["tasks"]["cost_nusd"] == usd_to_nusd(0.3)
    assert c["tasks"]["calls"] == 2
    assert c["known"]["cost_nusd"] == usd_to_nusd(0.3)
    assert c["expected_delta_nusd"] == usd_to_nusd(0.3) - usd_to_nusd(0.5)
    assert c["estimate"]["worst_nusd"] == usd_to_nusd(2.0)
    assert c["estimate"]["worst_usd"] == pytest.approx(2.0)
    assert c["estimate"]["assessment"] == "worst_fits"
    assert c["estimate"]["tasks"] == ["t1", "t2"]


def test_above_expected(env: cr.Env) -> None:
    add_project(env)
    estimate(env)
    record(env, (0.6, "p1", "t1", 1))
    c = comparisons(env)["p1"]
    assert c["outcome"] == "above_expected"
    assert c["expected_delta_nusd"] == usd_to_nusd(0.6) - usd_to_nusd(0.5)
    assert c["expected_delta_usd"] == pytest.approx(0.1)


def test_above_worst(env: cr.Env) -> None:
    add_project(env)
    estimate(env)
    record(env, (1.5, "p1", "t1", 1), (1.0, "p1", "t2", 2))
    assert comparisons(env)["p1"]["outcome"] == "above_worst"


def test_expected_unknown(env: cr.Env) -> None:
    add_project(env)
    estimate(env, expected_nusd=0, expected_complete=False, unknown=["t1", "t2"])
    record(env, (0.1, "p1", "t1", 1))
    c = comparisons(env)["p1"]
    assert c["outcome"] == "expected_unknown"
    assert c["known"]["calls"] == 0
    assert c["estimate"]["expected_complete"] is False
    assert c["estimate"]["unknown"] == ["t1", "t2"]


def test_no_spending_and_spending_before_estimate_is_ignored(env: cr.Env) -> None:
    add_project(env)
    estimate(env, minute=10)
    record(env, (0.4, "p1", "t1", 5))
    c = comparisons(env)["p1"]
    assert c["outcome"] == "no_spending"
    assert c["total"]["calls"] == 0


def test_other_calls_are_separate_and_counted_in_total(env: cr.Env) -> None:
    add_project(env)
    estimate(env)
    record(env, (0.1, "p1", "t1", 1), (0.05, "p1", None, 2), (0.07, "p1", "t9", 3))
    c = comparisons(env)["p1"]
    assert c["other"]["calls"] == 2
    assert c["other"]["cost_nusd"] == usd_to_nusd(0.05) + usd_to_nusd(0.07)
    assert c["total"]["cost_nusd"] == usd_to_nusd(0.1) + c["other"]["cost_nusd"]
    assert c["outcome"] == "within_expected"


def test_latest_estimate_is_used(env: cr.Env) -> None:
    add_project(env)
    estimate(env, minute=0, worst_nusd=usd_to_nusd(0.1), expected_nusd=0)
    estimate(env, minute=10)
    record(env, (0.3, "p1", "t1", 11))
    c = comparisons(env)["p1"]
    assert c["estimates"] == 2
    assert c["outcome"] == "within_expected"
    assert c["estimate"]["ts"] == (T0 + timedelta(minutes=10)).isoformat()


def test_invalid_latest_estimate_does_not_fall_back(env: cr.Env) -> None:
    add_project(env)
    estimate(env, minute=0)
    estimate(env, minute=10, expected_nusd=usd_to_nusd(5.0))  # expected > worst
    record(env, (0.1, "p1", "t1", 11))
    c = comparisons(env)["p1"]
    assert c["outcome"] == "invalid_estimate"
    assert c["estimate"] is None
    assert c["invalid_reason"]
    assert c["estimates"] == 2


def test_other_projects_do_not_mix(env: cr.Env) -> None:
    add_project(env, "p1")
    add_project(env, "p2")
    estimate(env, "p1")
    record(env, (0.1, "p1", "t1", 1), (9.0, "p2", "t1", 1))
    cs = comparisons(env)
    assert cs["p1"]["tasks"]["cost_nusd"] == usd_to_nusd(0.1)
    assert cs["p1"]["outcome"] == "within_expected"
    assert cs["p2"]["outcome"] == "no_estimate"


def test_project_option_limits_comparisons(env: cr.Env) -> None:
    add_project(env, "p1")
    add_project(env, "p2")
    estimate(env, "p2")
    assert list(comparisons(env, "--project", "p2")) == ["p2"]


def test_days_do_not_limit_the_comparison(env: cr.Env) -> None:
    add_project(env)
    estimate(env)
    record(env, (0.1, "p1", "t1", 1))
    data = report_json(env, "--since", "2026-11-20")
    assert data["total"]["calls"] == 0
    assert data["estimate_vs_actual"][0]["tasks"]["calls"] == 1


def test_missing_ledger_still_compares_with_no_spending(env: cr.Env) -> None:
    add_project(env)
    estimate(env)
    assert comparisons(env)["p1"]["outcome"] == "no_spending"
    assert not (env.ws / cli.LEDGER_FILENAME).exists()


# --- unreadable event logs are warnings -------------------------------------------


def test_corrupt_event_log_is_a_warning(env: cr.Env) -> None:
    add_project(env, "p1")
    add_project(env, "p2")
    estimate(env, "p2")
    path = env.store.events("p1").path
    path.write_text("{not json\n", encoding="utf-8")
    data = report_json(env)
    assert [c["project"] for c in data["estimate_vs_actual"]] == ["p2"]
    assert any("event log of p1" in w for w in data["warnings"])


def test_corrupt_event_log_text_warning_and_exit_ok(env: cr.Env) -> None:
    add_project(env)
    env.store.events("p1").path.write_text("garbage\n", encoding="utf-8")
    r = report(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "warning: event log of p1 is unreadable" in r.output


def test_symlinked_event_log_is_not_read(env: cr.Env, tmp_path: Path) -> None:
    add_project(env)
    outside = tmp_path / "outside.jsonl"
    other = Event(ts=T0, type="finance.estimate", project_id="p1", data=est_data())
    outside.write_text(other.model_dump_json() + "\n", encoding="utf-8")
    path = env.store.events("p1").path
    if path.exists():
        path.unlink()
    path.symlink_to(outside)
    data = report_json(env)
    assert data["estimate_vs_actual"] == []
    assert any("symlink" in w for w in data["warnings"])


def test_non_utf8_event_log_is_a_warning(env: cr.Env) -> None:
    add_project(env)
    env.store.events("p1").path.write_bytes(b"\xff\xfe\n")
    data = report_json(env)
    assert data["estimate_vs_actual"] == []
    assert data["warnings"]


def test_broken_ledger_is_still_exit_2(env: cr.Env) -> None:
    add_project(env)
    estimate(env)
    env.ws.mkdir(parents=True, exist_ok=True)
    (env.ws / cli.LEDGER_FILENAME).write_bytes(b"not a database" * 100)
    assert report(env).exit_code == cli.EXIT_CONFIG


# --- text -----------------------------------------------------------------------


def test_text_section_within_expected(env: cr.Env) -> None:
    add_project(env)
    estimate(env)
    record(env, (0.1, "p1", "t1", 1), (0.05, "p1", None, 2))
    r = report(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    out = r.output
    assert "estimate vs actual" in out
    assert "p1  within_expected" in out
    assert "expected $0.5000, worst case $2.0000" in out
    assert "estimated tasks spent $0.1000 in 1 call(s)" in out
    assert "vs expected $0.5000 (-$0.4000)" in out
    assert "other calls since" in out
    assert "verifier calls" in out
    assert out.rstrip().endswith("the budget guard alone decides spending")


def test_text_above_worst_is_loud(env: cr.Env) -> None:
    add_project(env)
    estimate(env)
    record(env, (3.0, "p1", "t1", 1))
    assert "ABOVE THE WORST CASE" in report(env).output


def test_text_partial_expectation(env: cr.Env) -> None:
    add_project(env)
    estimate(env, expected_complete=False, unknown=["t2"])
    record(env, (0.1, "p1", "t1", 1))
    assert "expected $0.5000 for 1 task(s), 1 unknown" in report(env).output


def test_text_invalid_estimate(env: cr.Env) -> None:
    add_project(env)
    estimate(env, assessment="bogus")
    out = report(env).output
    assert "p1  invalid_estimate" in out
    assert "no comparison" in out


def test_text_without_estimates(env: cr.Env) -> None:
    add_project(env, "p1")
    add_project(env, "p2")
    assert "2 project(s) without an estimate" in report(env).output
    assert "p1  no estimate yet" in report(env, "--project", "p1").output


def test_text_no_projects(env: cr.Env) -> None:
    assert "  no projects" in report(env).output


def test_text_escapes_control_characters_in_reason(env: cr.Env) -> None:
    add_project(env)
    estimate(env, tasks=["t1", "t1"])  # duplicate ids -> invalid
    out = report(env).output
    assert "\x1b" not in out
    assert "invalid_estimate" in out


# --- read-only and no provider ----------------------------------------------------


def test_report_writes_nothing_and_builds_no_provider(env: cr.Env) -> None:
    add_project(env)
    estimate(env)
    record(env, (0.1, "p1", "t1", 1))
    before = tree(env.ws)
    assert report(env).exit_code == cli.EXIT_OK
    assert report(env, "--json").exit_code == cli.EXIT_OK
    assert tree(env.ws) == before
    assert env.factory_calls == 0


def test_end_to_end_with_a_fake_provider_run(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    assert env.run(cr.PID).exit_code == cli.EXIT_OK
    c = comparisons(env, "--project", cr.PID)[cr.PID]
    assert c["estimate"] is not None and c["estimate"]["tasks"] == ["a"]
    assert c["tasks"]["calls"] >= 1
    assert c["outcome"] != "above_worst"
    assert "estimate vs actual" in report(env, "--project", cr.PID).output
