"""Tests for the data router in ``automa run`` (D60).

Local only: FakeProvider behind the real gateway/guard/ledger, real Orchestrator,
ProjectStore, QualityLog and Router. No network, no real model calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import test_cli_run as cr
from automa import cli
from automa.config import Tier
from automa.plan.models import ResultStatus, VerdictOutcome
from automa.quality import QUALITY_FILE, QualityLog, QualityRecord

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)

# small is cheaper than medium (the writer role's tier), large is more expensive.
TOML = """
workspace = "{ws}"
[tiers]
small = "fake-small"
medium = "fake-medium"
large = "fake-large"
[prices."fake-small"]
input_per_mtok = 0.5
output_per_mtok = 1.0
[prices."fake-medium"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[prices."fake-large"]
input_per_mtok = 2.0
output_per_mtok = 4.0
[limits]
project_usd = 5.0
daily_usd = 2.0
per_call_usd = 0.25
{router}
"""


def configure(env: cr.Env, router: str = "") -> None:
    env.config.write_text(TOML.format(ws=env.ws.as_posix(), router=router), encoding="utf-8")


def rec(
    attempt: int, *, model: str, tier: Tier, outcome: VerdictOutcome, **kw: Any
) -> QualityRecord:
    data: dict[str, Any] = {
        "project_id": "old",
        "task_id": f"{model}-t",
        "attempt": attempt,
        "role": "writer",
        "tier": tier,
        "model": model,
        "result_status": ResultStatus.COMPLETED,
        "outcome": outcome,
        "model_evaluated": False,
        "cost_nusd": 1000,
        "judged_at": T0,
    }
    data.update(kw)
    return QualityRecord(**data)


def seed(env: cr.Env, model: str, tier: Tier, passed: int, failed: int = 0, **kw: Any) -> Path:
    env.ws.mkdir(parents=True, exist_ok=True)
    path = env.ws / QUALITY_FILE
    with QualityLog(path) as log:
        n = 0
        for outcome, count in ((VerdictOutcome.PASS, passed), (VerdictOutcome.FAIL, failed)):
            for _ in range(count):
                n += 1
                log.record(rec(n, model=model, tier=tier, outcome=outcome, **kw))
    return path


def seed_small_good(env: cr.Env) -> None:
    seed(env, "fake-small", Tier.SMALL, passed=5)


def router_events(env: cr.Env) -> list[dict[str, Any]]:
    return [e.data for e in env.store.events(cr.PID).read() if e.type == "router.decision"]


def models_used(env: cr.Env) -> list[str]:
    return [r.model for r in env.provider.requests]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Env:
    e = cr.Env(tmp_path, monkeypatch)
    configure(e)
    return e


def run_one(env: cr.Env, *args: str) -> Any:
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    return env.run(cr.PID, *args)


# --- off by default ----------------------------------------------------------------


def test_router_off_by_default_keeps_requested_tier(env: cr.Env) -> None:
    seed_small_good(env)
    r = run_one(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "router: off" in r.output
    assert set(models_used(env)) == {"fake-medium"}
    assert router_events(env) == []
    assert env.store.task_states(cr.PID)["a"].tier is Tier.MEDIUM


def test_no_router_flag_overrides_enabled_config(env: cr.Env) -> None:
    configure(env, "[router]\nenabled = true")
    seed_small_good(env)
    r = run_one(env, "--no-router")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "router: off" in r.output
    assert set(models_used(env)) == {"fake-medium"}
    assert router_events(env) == []


# --- on ------------------------------------------------------------------------------


def test_enabled_in_config_routes_to_cheaper_qualified_tier(env: cr.Env) -> None:
    configure(env, "[router]\nenabled = true")
    seed_small_good(env)
    r = run_one(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "router: on" in r.output
    assert "at least 5 judged attempt(s)" in r.output
    assert "80.0%" in r.output
    assert set(models_used(env)) == {"fake-small"}
    assert router_events(env) == [
        {
            "task": "a",
            "role": "writer",
            "requested": "medium",
            "tier": "small",
            "model": "fake-small",
            "reason": "cheaper",
            "attempts": 5,
            "passed": 5,
        }
    ]
    assert env.store.task_states(cr.PID)["a"].tier is Tier.SMALL


def test_router_flag_overrides_disabled_config(env: cr.Env) -> None:
    configure(env, "[router]\nenabled = false")
    seed_small_good(env)
    r = run_one(env, "--router")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "router: on" in r.output
    assert set(models_used(env)) == {"fake-small"}


def test_router_settings_from_config_are_used(env: cr.Env) -> None:
    configure(env, "[router]\nenabled = true\nmin_attempts = 6\nmin_pass_rate = 1")
    seed_small_good(env)  # only 5 attempts: not enough with min_attempts = 6
    r = run_one(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "at least 6 judged attempt(s)" in r.output
    assert "100.0%" in r.output
    assert set(models_used(env)) == {"fake-medium"}
    assert [e["reason"] for e in router_events(env)] == ["requested"]


def test_router_without_quality_data_keeps_requested_tier(env: cr.Env) -> None:
    r = run_one(env, "--router")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert set(models_used(env)) == {"fake-medium"}
    ev = router_events(env)
    assert [(e["tier"], e["reason"], e["attempts"]) for e in ev] == [("medium", "requested", 0)]


def test_router_never_moves_to_more_expensive_tier_without_proof(env: cr.Env) -> None:
    seed(env, "fake-large", Tier.LARGE, passed=10)
    r = run_one(env, "--router")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert set(models_used(env)) == {"fake-medium"}


def test_router_replaces_proven_weak_tier(env: cr.Env) -> None:
    seed(env, "fake-medium", Tier.MEDIUM, passed=1, failed=4)
    with QualityLog(env.ws / QUALITY_FILE) as log:
        for n in range(1, 6):
            log.record(rec(n, model="fake-large", tier=Tier.LARGE, outcome=VerdictOutcome.PASS))
    r = run_one(env, "--router")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert set(models_used(env)) == {"fake-large"}
    assert [e["reason"] for e in router_events(env)] == ["replaced"]


def test_stats_of_other_role_do_not_route(env: cr.Env) -> None:
    seed(env, "fake-small", Tier.SMALL, passed=5, role="developer")
    r = run_one(env, "--router")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert set(models_used(env)) == {"fake-medium"}


def test_escalation_continues_from_routed_tier(env: cr.Env) -> None:
    seed_small_good(env)
    env.project(cr.task_dict("a"))
    # escalate_after defaults to 2 rejections: small, small, then medium.
    env.script(*cr.write("a", "bad"), *cr.write("a", "bad"), *cr.write("a"))
    r = env.run(cr.PID, "--router", "--max-attempts", "3")
    assert r.exit_code == cli.EXIT_OK, r.output
    used = models_used(env)
    assert used[0] == "fake-small"
    assert used[-1] == "fake-medium"
    assert len(router_events(env)) == 1


def test_resume_does_not_route_again(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    env.script(cr.ProviderError("down"))
    r = env.run(cr.PID)  # no router: tier medium saved with the first attempt
    assert r.exit_code != cli.EXIT_OK
    seed_small_good(env)
    env.script(*cr.write("a"))
    r = env.run(cr.PID, "--router")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert set(models_used(env)) == {"fake-medium"}
    assert router_events(env) == []


def test_routed_attempts_are_recorded_with_routed_model(env: cr.Env) -> None:
    seed_small_good(env)
    r = run_one(env, "--router")
    assert r.exit_code == cli.EXIT_OK, r.output
    with QualityLog(env.ws / QUALITY_FILE, read_only=True) as log:
        mine = log.records(project_id=cr.PID)
    assert [(x.model, x.tier) for x in mine] == [("fake-small", Tier.SMALL)]


def test_router_snapshot_ignores_attempts_of_same_run(env: cr.Env) -> None:
    # Task a passes on medium; even with min_attempts = 1 that new row must not
    # route task b of the same run (the router reads a snapshot taken at start).
    configure(env, "[router]\nenabled = true\nmin_attempts = 1")
    env.project(cr.task_dict("a"), cr.task_dict("b", "a"))
    env.script(*cr.write("a"), *cr.write("b"))
    r = env.run(cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert set(models_used(env)) == {"fake-medium"}
    assert [e["reason"] for e in router_events(env)] == ["requested", "requested"]


# --- failures ------------------------------------------------------------------------


def test_broken_quality_log_with_router_is_config_error_before_model_call(env: cr.Env) -> None:
    env.ws.mkdir(parents=True)
    (env.ws / QUALITY_FILE).write_bytes(b"not a database at all" * 10)
    env.project(cr.task_dict("a"))
    r = env.run(cr.PID, "--router")
    assert r.exit_code == cli.EXIT_CONFIG
    assert env.provider.requests == []


def test_corrupt_quality_row_with_router_is_config_error_before_model_call(
    env: cr.Env,
) -> None:
    import sqlite3

    path = seed(env, "fake-small", Tier.SMALL, passed=5)
    con = sqlite3.connect(path)
    with con:
        con.execute("UPDATE attempts SET role = 'Not A Role'")
    con.close()
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    r = env.run(cr.PID, "--router")
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert env.provider.requests == []
    assert env.status() is not cr.ProjectStatus.RUNNING
    # Without the router the same log rows are never read for routing.
    r = env.run(cr.PID, "--no-router")
    assert r.exit_code == cli.EXIT_OK, r.output


def test_router_line_printed_before_run(env: cr.Env) -> None:
    r = run_one(env)
    lines = r.output.splitlines()
    assert lines.index(next(x for x in lines if x.startswith("memory:"))) + 1 == lines.index(
        next(x for x in lines if x.startswith("router:"))
    )


def test_run_help_lists_router_options() -> None:
    from typer.testing import CliRunner

    r = CliRunner().invoke(cli.app, ["run", "--help"])
    assert r.exit_code == 0
    assert "--router" in r.output and "--no-router" in r.output


def test_bench_run_has_no_router_option() -> None:
    from typer.testing import CliRunner

    r = CliRunner().invoke(cli.app, ["bench", "run", "--help"])
    assert r.exit_code == 0
    assert "--router" not in r.output
