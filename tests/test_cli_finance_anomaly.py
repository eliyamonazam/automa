"""Tests for the unusual-spending section of ``automa finance report`` (D70).

Ledger rows and quality records are written directly through the real
``CostLedger`` and ``QualityLog`` (no provider); the command never builds a
provider. No network, no real model calls.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

import test_cli_finance_report as fr
import test_cli_run as cr
from automa import cli
from automa.budget.ledger import CostLedger, usd_to_nusd
from automa.quality import QualityLog
from test_finance_anomaly import rec
from test_ledger import FixedClock, make_pair

T0 = fr.T0
env = fr.env  # fixture
report = fr.report
report_json = fr.report_json


def calls(env: cr.Env, *items: tuple[float, str, str, int]) -> None:
    """Each call: (cost_usd, project, task, minute offset from T0), in order."""
    clock = FixedClock(T0)
    with CostLedger(fr.ledger_path(env), clock=clock) as led:
        for cost, project, task, minute in items:
            clock.now = T0 + timedelta(minutes=minute)
            led.record(*make_pair(cost, project=project, task=task))


def steady(project: str = "p1", task: str = "a", n: int = 5, start: int = 0) -> list[Any]:
    return [(0.01, project, task, start + i) for i in range(n)]


def anomalies(data: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = data["anomalies"]["anomalies"]
    return out


def quality_path(env: cr.Env) -> Path:
    env.ws.mkdir(parents=True, exist_ok=True)
    return env.ws / cli.QUALITY_FILE


# --- empty -----------------------------------------------------------------------


def test_no_ledger_no_quality_log(env: cr.Env) -> None:
    data = report_json(env)
    a = data["anomalies"]
    assert a["anomalies"] == [] and a["checked"] == 0 and a["total"] == 0
    assert a["attempts_checked"] is True
    assert a["settings"] == {
        "factor": 3.0,
        "min_samples": 5,
        "min_cost_nusd": 1_000_000,
        "min_cost_usd": 0.001,
    }
    assert not (env.ws / cli.QUALITY_FILE).exists()


def test_text_section_when_nothing_unusual(env: cr.Env) -> None:
    fr.add_project(env, "p1", ("a", "writer"))
    calls(env, *steady())
    r = report(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "unusual spending" in r.output
    assert "  none\n" in r.output
    assert "0 sample(s) checked, 5 with too little history yet" in r.output
    assert "advisory only" in r.output


# --- ledger calls ------------------------------------------------------------------


def test_spike_after_history_is_flagged(env: cr.Env) -> None:
    fr.add_project(env, "p1", ("a", "writer"))
    calls(env, *steady(), (0.05, "p1", "a", 10))
    data = report_json(env)
    found = anomalies(data)
    assert len(found) == 1
    a = found[0]
    assert a["kind"] == "call" and a["project"] == "p1" and a["task_id"] == "a"
    assert a["role"] == "writer" and a["model"] == "m-small"
    assert a["cost_nusd"] == usd_to_nusd(0.05)
    assert a["baseline_nusd"] == usd_to_nusd(0.01)
    assert a["history"] == 5 and a["ratio"] == pytest.approx(5.0)
    assert a["call_id"] is not None and a["attempt"] is None
    assert data["anomalies"]["checked"] == 1 and data["anomalies"]["without_history"] == 5
    json.dumps(data)  # JSON-safe


def test_spike_in_text(env: cr.Env) -> None:
    fr.add_project(env, "p1", ("a", "writer"))
    calls(env, *steady(), (0.05, "p1", "a", 10))
    r = report(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    line = next(ln for ln in r.output.splitlines() if "vs median" in ln)
    assert "call call #" in line and "p1/a  writer  m-small" in line
    assert "$0.0500 vs median $0.0100 (5.0x, 5 earlier)" in line


def test_too_little_history_is_not_flagged(env: cr.Env) -> None:
    fr.add_project(env, "p1", ("a", "writer"))
    calls(env, *steady(n=4), (0.5, "p1", "a", 10))
    assert anomalies(report_json(env)) == []


def test_calls_without_role_are_left_out_and_counted(env: cr.Env) -> None:
    fr.add_project(env, "p1", ("a", "writer"))
    calls(env, *steady(), (0.01, "p1", "zzz", 7))
    clock = FixedClock(T0)
    with CostLedger(fr.ledger_path(env), clock=clock) as led:
        led.record(*make_pair(0.9, project=None, task=None))  # planning/review
    data = report_json(env)
    assert data["anomalies"]["left_out_calls"] == 2
    assert anomalies(data) == []
    assert "2 call(s) without a task role not checked" in report(env).output


def test_history_is_shared_across_projects_and_project_filter_narrows(env: cr.Env) -> None:
    fr.add_project(env, "p1", ("a", "writer"))
    fr.add_project(env, "p2", ("b", "writer"))
    # p1 builds the history; p2's first call is already unusual against it
    calls(env, *steady("p1"), (0.05, "p2", "b", 10), (0.06, "p1", "a", 11))
    every = anomalies(report_json(env))
    assert [a["project"] for a in every] == ["p2", "p1"]
    only_p2 = report_json(env, "--project", "p2")
    assert [a["project"] for a in anomalies(only_p2)] == ["p2"]
    assert only_p2["anomalies"]["total"] == 2  # detection itself is not filtered
    only_p1 = report_json(env, "--project", "p1")
    assert [a["cost_nusd"] for a in anomalies(only_p1)] == [usd_to_nusd(0.06)]


def test_project_filter_still_uses_other_projects_roles(env: cr.Env) -> None:
    fr.add_project(env, "p1", ("a", "writer"))
    fr.add_project(env, "p2", ("b", "writer"))
    calls(env, *steady("p1"), (0.05, "p2", "b", 10))
    # without p1's roles the p2 call would have no history at all
    assert len(anomalies(report_json(env, "--project", "p2"))) == 1


def test_day_filter_narrows_shown_but_not_history(env: cr.Env) -> None:
    fr.add_project(env, "p1", ("a", "writer"))
    day = 24 * 60
    calls(env, *steady(), (0.05, "p1", "a", day), (0.07, "p1", "a", 2 * day))
    data = report_json(env, "--since", "2026-11-11")
    assert [a["cost_nusd"] for a in anomalies(data)] == [usd_to_nusd(0.07)]
    data = report_json(env, "--until", "2026-11-10")
    assert [a["cost_nusd"] for a in anomalies(data)] == [usd_to_nusd(0.05)]
    data = report_json(env, "--until", "2026-11-09")
    assert anomalies(data) == [] and data["anomalies"]["total"] == 2
    r = report(env, "--until", "2026-11-09")
    assert "none in the selected project/days" in r.output


def test_different_roles_have_separate_history(env: cr.Env) -> None:
    fr.add_project(env, "p1", ("a", "writer"), ("b", "coder"))
    calls(env, *steady(task="a"), (0.05, "p1", "b", 10))
    assert anomalies(report_json(env)) == []


def test_unreadable_other_plan_warns_as_history_only(env: cr.Env) -> None:
    fr.add_project(env, "p1", ("a", "writer"))
    fr.add_project(env, "p2", ("b", "writer"))
    env.store.plan_file("p2").write_text("{broken", encoding="utf-8")
    calls(env, *steady())
    data = report_json(env, "--project", "p1")
    assert len(data["warnings"]) == 1
    assert "p2" in data["warnings"][0] and "anomaly history only" in data["warnings"][0]


# --- judged attempts ----------------------------------------------------------------


def write_attempts(env: cr.Env, costs: list[int]) -> None:
    with QualityLog(quality_path(env)) as log:
        for i, c in enumerate(costs):
            log.record(
                rec(
                    task_id=f"t{i}",
                    cost_nusd=c,
                    judged_at=T0 + timedelta(minutes=i),
                )
            )


def test_attempt_spike_is_flagged(env: cr.Env) -> None:
    write_attempts(env, [2_000_000] * 5 + [9_000_000])
    data = report_json(env)
    found = anomalies(data)
    assert len(found) == 1
    assert found[0]["kind"] == "attempt" and found[0]["attempt"] == 1
    assert found[0]["call_id"] is None and found[0]["task_id"] == "t5"
    r = report(env)
    assert "attempt attempt 1" in r.output


def test_quality_log_is_not_changed(env: cr.Env) -> None:
    write_attempts(env, [2_000_000] * 6)
    before = fr.tree(env.ws)
    report_json(env)
    report(env, "--project", "p1")
    assert fr.tree(env.ws) == before
    assert env.factory_calls == 0


def test_broken_quality_log_is_a_warning(env: cr.Env) -> None:
    fr.add_project(env, "p1", ("a", "writer"))
    calls(env, *steady(), (0.05, "p1", "a", 10))
    quality_path(env).write_bytes(b"not a sqlite database at all" * 10)
    data = report_json(env)
    assert data["anomalies"]["attempts_checked"] is False
    assert any("quality log" in w for w in data["warnings"])
    assert len(anomalies(data)) == 1  # ledger calls still checked
    r = report(env)
    assert r.exit_code == cli.EXIT_OK
    assert "judged attempts not checked" in r.output


def test_symlinked_quality_log_is_a_warning(env: cr.Env, tmp_path: Path) -> None:
    real = tmp_path / "elsewhere.sqlite3"
    with QualityLog(real) as log:
        log.record(rec())
    os.symlink(real, quality_path(env))
    data = report_json(env)
    assert data["anomalies"]["attempts_checked"] is False
    assert any("symlink" in w for w in data["warnings"])


# --- safety ----------------------------------------------------------------------------


def test_control_characters_in_model_are_escaped(env: cr.Env) -> None:
    fr.add_project(env, "p1", ("a", "writer"))
    clock = FixedClock(T0)
    with CostLedger(fr.ledger_path(env), clock=clock) as led:
        for i, cost in enumerate([0.01] * 5 + [0.05]):
            clock.now = T0 + timedelta(minutes=i)
            est, res = make_pair(cost, project="p1", task="a")
            led.record(est.model_copy(update={"model": "evil\x1b[2J\nmodel"}), res)
    r = report(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "\x1b" not in r.output
    assert any("vs median" in ln and "evil" in ln for ln in r.output.splitlines())


def test_works_without_api_key(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    fr.add_project(env, "p1", ("a", "writer"))
    calls(env, *steady(), (0.05, "p1", "a", 10))
    assert len(anomalies(report_json(env))) == 1
