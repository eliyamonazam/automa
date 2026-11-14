"""Tests for ``automa finance report`` — the read-only spending report CLI (D66).

Ledger rows are written directly through the real ``CostLedger`` (no provider);
the command itself never builds a provider. No network, no real model calls.
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
from automa.plan import Plan
from test_ledger import FixedClock, make_pair

T0 = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)


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


def ledger_path(env: cr.Env) -> Path:
    env.ws.mkdir(parents=True, exist_ok=True)
    return env.ws / cli.LEDGER_FILENAME


def record(env: cr.Env, *calls: tuple[float, str | None, str | None, int]) -> None:
    """Each call: (cost_usd, project, task, day offset from T0)."""
    clock = FixedClock(T0)
    with CostLedger(ledger_path(env), clock=clock) as led:
        for cost, project, task, offset in calls:
            clock.now = T0 + timedelta(days=offset)
            led.record(*make_pair(cost, project=project, task=task))


def add_project(env: cr.Env, pid: str, *tasks: tuple[str, str]) -> None:
    env.store.create("goal", project_id=pid)
    if tasks:
        env.store.save_plan(
            pid,
            Plan.model_validate(
                {"goal": "goal", "tasks": [cr.task_dict(t, role=role) for t, role in tasks]}
            ),
        )


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def keyed(groups: list[dict[str, Any]]) -> dict[str, int]:
    return {g["key"]: g["cost_nusd"] for g in groups}


# --- empty / missing ledger ---------------------------------------------------------


def test_missing_ledger_is_empty_and_not_created(env: cr.Env) -> None:
    before = tree(env.ws)
    data = report_json(env)
    assert data["total"]["calls"] == 0 and data["total"]["cost_usd"] == 0
    assert data["by_project"] == [] and data["warnings"] == []
    assert not ledger_path(env).exists()
    assert tree(env.ws) == before
    assert env.factory_calls == 0


def test_missing_ledger_text(env: cr.Env) -> None:
    r = report(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "spending report: all projects" in r.output
    assert "no recorded calls" in r.output


# --- groupings ----------------------------------------------------------------------


def test_groups_by_project_day_model_and_role(env: cr.Env) -> None:
    add_project(env, "p1", ("a", "writer"), ("b", "coder"))
    add_project(env, "p2", ("a", "researcher"))
    record(
        env,
        (0.01, "p1", "a", 0),
        (0.02, "p1", "b", 1),
        (0.04, "p2", "a", 1),
        (0.08, "p1", None, 0),  # planning
        (0.16, None, None, 2),  # no project
        (0.32, "p1", "ghost", 2),  # task not in the plan
    )
    data = report_json(env)
    n = usd_to_nusd
    with CostLedger(ledger_path(env), read_only=True) as ro:
        assert data["total"]["cost_nusd"] == ro.spent_nusd()
    assert data["total"]["calls"] == 6
    assert keyed(data["by_project"]) == {
        "p1": n(0.01) + n(0.02) + n(0.08) + n(0.32),
        "p2": n(0.04),
    }
    assert [g["key"] for g in data["by_project"]] == ["p1", "p2"]
    assert data["no_project"]["cost_nusd"] == n(0.16)
    assert [g["key"] for g in data["by_day"]] == ["2026-11-09", "2026-11-10", "2026-11-11"]
    assert keyed(data["by_model"]) == {"m-small": data["total"]["cost_nusd"]}
    # Role keys include the project: task "a" is a writer in p1, a researcher in p2.
    assert keyed(data["by_role"]) == {"writer": n(0.01), "coder": n(0.02), "researcher": n(0.04)}
    assert data["no_task"]["cost_nusd"] == n(0.08) + n(0.16)
    assert data["unknown_role"]["cost_nusd"] == n(0.32)
    assert data["first_ts"].startswith("2026-11-09") and data["last_ts"].startswith("2026-11-11")
    for name in ("by_project", "by_day", "by_model"):
        extra = data["no_project"]["cost_nusd"] if name == "by_project" else 0
        assert sum(g["cost_nusd"] for g in data[name]) + extra == data["total"]["cost_nusd"]


def test_reserved_is_the_guards_worst_case(env: cr.Env) -> None:
    add_project(env, "p1", ("a", "writer"))
    record(env, (0.01, "p1", "a", 0))  # make_pair reserves twice the cost
    data = report_json(env)
    assert data["total"]["reserved_nusd"] == 2 * usd_to_nusd(0.01)
    assert data["total"]["input_tokens"] == 40 and data["total"]["output_tokens"] == 10


def test_text_output(env: cr.Env) -> None:
    add_project(env, "p1", ("a", "writer"))
    record(env, (0.01, "p1", "a", 0), (0.02, None, None, 0))
    r = report(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    out = r.output
    assert "total: $0.0300 in 2 call(s)" in out
    assert "by project:" in out and "  p1  $0.0100" in out and "(no project)  $0.0200" in out
    assert "by day (UTC):" in out and "2026-11-09" in out
    assert "by model:" in out and "m-small" in out
    assert "  writer  $0.0100" in out and "(no task: planning/review)  $0.0200" in out
    assert "(unknown role)" not in out  # empty buckets are not shown
    assert "read-only" in out


# --- filters ------------------------------------------------------------------------


def test_project_filter(env: cr.Env) -> None:
    add_project(env, "p1", ("a", "writer"))
    add_project(env, "p2", ("a", "coder"))
    record(env, (0.01, "p1", "a", 0), (0.02, "p2", "a", 0), (0.04, None, None, 0))
    data = report_json(env, "--project", "p2")
    assert data["project"] == "p2"
    assert data["total"]["cost_nusd"] == usd_to_nusd(0.02)
    assert keyed(data["by_role"]) == {"coder": usd_to_nusd(0.02)}
    assert data["no_project"]["calls"] == 0


def test_day_filters_are_inclusive(env: cr.Env) -> None:
    add_project(env, "p1", ("a", "writer"))
    record(env, *[(0.01 * (i + 1), "p1", "a", i) for i in range(4)])
    data = report_json(env, "--since", "2026-11-10", "--until", "2026-11-11")
    assert [g["key"] for g in data["by_day"]] == ["2026-11-10", "2026-11-11"]
    assert data["total"]["cost_nusd"] == usd_to_nusd(0.02) + usd_to_nusd(0.03)
    assert (data["since"], data["until"]) == ("2026-11-10", "2026-11-11")
    only_since = report_json(env, "--since", "2026-11-12")
    assert [g["key"] for g in only_since["by_day"]] == ["2026-11-12"]
    only_until = report_json(env, "--until", "2026-11-09")
    assert [g["key"] for g in only_until["by_day"]] == ["2026-11-09"]


def test_day_filter_in_text(env: cr.Env) -> None:
    r = report(env, "--since", "2026-11-10")
    assert r.exit_code == cli.EXIT_OK
    assert "days 2026-11-10 to ... (UTC)" in r.output


# --- errors -------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["yesterday", "2026-13-01", "2026/11/10", ""])
@pytest.mark.parametrize("option", ["--since", "--until"])
def test_invalid_day_is_user_error(env: cr.Env, option: str, value: str) -> None:
    r = report(env, option, value)
    assert r.exit_code == cli.EXIT_USER
    assert option in r.output


def test_since_after_until_is_user_error(env: cr.Env) -> None:
    r = report(env, "--since", "2026-11-12", "--until", "2026-11-10")
    assert r.exit_code == cli.EXIT_USER
    assert "--since" in r.output


@pytest.mark.parametrize("pid", ["nope", "../etc", "a b"])
def test_unknown_or_invalid_project_is_user_error(env: cr.Env, pid: str) -> None:
    r = report(env, "--project", pid)
    assert r.exit_code == cli.EXIT_USER
    assert not ledger_path(env).exists()


def test_broken_ledger_is_config_error(env: cr.Env) -> None:
    ledger_path(env).write_bytes(b"not a database" * 100)
    r = report(env)
    assert r.exit_code == cli.EXIT_CONFIG
    assert "ledger" in r.output


def test_symlinked_ledger_is_config_error(env: cr.Env, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.sqlite3"
    with CostLedger(target) as led:
        led.record(*make_pair(0.01))
    ledger_path(env).symlink_to(target)
    r = report(env)
    assert r.exit_code == cli.EXIT_CONFIG


def test_broken_config_is_config_error(env: cr.Env) -> None:
    env.config.write_text("not = [valid", encoding="utf-8")
    r = report(env)
    assert r.exit_code == cli.EXIT_CONFIG


# --- read-only ----------------------------------------------------------------------


def test_nothing_is_written_and_no_provider(env: cr.Env) -> None:
    add_project(env, "p1", ("a", "writer"))
    record(env, (0.01, "p1", "a", 0))
    before = tree(env.ws)
    report_json(env)
    report(env)
    report(env, "--project", "p1", "--since", "2026-11-01")
    assert tree(env.ws) == before
    assert env.factory_calls == 0


def test_works_without_api_key(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    record(env, (0.01, None, None, 0))
    assert report_json(env)["total"]["calls"] == 1


def test_unreadable_plan_is_a_warning(env: cr.Env) -> None:
    add_project(env, "p1", ("a", "writer"))
    add_project(env, "p2", ("a", "coder"))
    env.store.plan_file("p2").write_text("{broken", encoding="utf-8")
    record(env, (0.01, "p1", "a", 0), (0.02, "p2", "a", 0))
    data = report_json(env)
    assert keyed(data["by_role"]) == {"writer": usd_to_nusd(0.01)}
    assert data["unknown_role"]["cost_nusd"] == usd_to_nusd(0.02)
    assert len(data["warnings"]) == 1 and "p2" in data["warnings"][0]
    r = report(env)
    assert r.exit_code == cli.EXIT_OK
    assert r.output.startswith("warning: plan of p2")


def test_project_without_plan_counts_as_unknown_role(env: cr.Env) -> None:
    add_project(env, "p1")
    record(env, (0.01, "p1", "a", 0))
    data = report_json(env)
    assert data["unknown_role"]["calls"] == 1 and data["warnings"] == []


def test_control_characters_in_model_are_escaped(env: cr.Env) -> None:
    clock = FixedClock(T0)
    est, res = make_pair(0.01, project=None, task=None)
    est = est.model_copy(update={"model": "evil\x1b[2Jmodel"})
    with CostLedger(ledger_path(env), clock=clock) as led:
        led.record(est, res)
    r = report(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "\x1b" not in r.output


def test_help_lists_finance_report() -> None:
    r = CliRunner().invoke(cli.app, ["finance", "--help"])
    assert r.exit_code == cli.EXIT_OK
    assert "report" in r.output
