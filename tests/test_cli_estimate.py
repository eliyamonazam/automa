"""Tests for ``automa estimate <id>`` — the read-only cost estimate (D63).

Local only: FakeProvider behind the real gateway/guard/ledger for the runs that
create history; the estimate itself never builds a provider. No network, no real
model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import test_cli_run as cr
import test_cli_run_router as rr
from automa import cli
from automa.budget import cap_to_nusd
from automa.budget.ledger import CostLedger, nusd_to_usd
from automa.config import Tier
from automa.finance import EstimateSettings
from automa.llm import ProviderError
from automa.quality import QUALITY_FILE
from automa.roles.registry import RoleRegistry

S = EstimateSettings()
WRITER_STEPS = min(RoleRegistry.builtin().get("writer").max_steps, S.worker_max_steps)
PER_ATTEMPT = WRITER_STEPS + S.verifier_calls
CAP = cap_to_nusd(0.25)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Env:
    e = cr.Env(tmp_path, monkeypatch)
    rr.configure(e)
    return e


def estimate(env: cr.Env, *args: str) -> Any:
    return CliRunner().invoke(cli.app, ["estimate", *args, "--config", str(env.config)])


def estimate_json(env: cr.Env, *args: str) -> dict[str, Any]:
    r = estimate(env, cr.PID, *args, "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    data: dict[str, Any] = json.loads(r.output)
    return data


def by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {t["task_id"]: t for t in data["tasks"]}


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def worst_usd(attempts: int) -> float:
    return nusd_to_usd(attempts * PER_ATTEMPT * CAP)


# --- errors -------------------------------------------------------------------------


def test_unknown_project_is_user_error(env: cr.Env) -> None:
    r = estimate(env, "nope")
    assert r.exit_code == cli.EXIT_USER
    assert "nope" in r.output


def test_project_without_plan_is_user_error(env: cr.Env) -> None:
    env.project(plan=False)
    r = estimate(env, cr.PID)
    assert r.exit_code == cli.EXIT_USER
    assert "no plan" in r.output


@pytest.mark.parametrize("value", ["0", "51"])
def test_invalid_max_attempts(env: cr.Env, value: str) -> None:
    env.project(cr.task_dict("a"))
    r = estimate(env, cr.PID, "--max-attempts", value)
    assert r.exit_code == cli.EXIT_USER
    assert "--max-attempts" in r.output


def test_unconfigured_tier_is_config_error(env: cr.Env) -> None:
    env.project(cr.task_dict("a", tier="small"))
    env.write_config(tiers='medium = "fake-medium"\nlarge = "fake-large"')
    r = estimate(env, cr.PID)
    assert r.exit_code == cli.EXIT_CONFIG
    assert "small" in r.output


def test_broken_quality_log_is_config_error(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    (env.ws / QUALITY_FILE).write_bytes(b"not a database" * 100)
    r = estimate(env, cr.PID)
    assert r.exit_code == cli.EXIT_CONFIG


def test_broken_ledger_is_config_error(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    (env.ws / cli.LEDGER_FILENAME).write_bytes(b"not a database" * 100)
    r = estimate(env, cr.PID)
    assert r.exit_code == cli.EXIT_CONFIG
    assert "ledger" in r.output


# --- fresh project ------------------------------------------------------------------


def test_fresh_project_without_history(env: cr.Env) -> None:
    env.project(cr.task_dict("a"), cr.task_dict("b", "a"))
    before = tree(env.ws)
    data = estimate_json(env)
    assert tree(env.ws) == before  # nothing written: no ledger, no quality log
    assert not (env.ws / cli.LEDGER_FILENAME).exists()
    assert not (env.ws / QUALITY_FILE).exists()
    assert env.factory_calls == 0
    tasks = by_id(data)
    assert list(tasks) == ["a", "b"]
    a = tasks["a"]
    assert (a["requested"], a["tier"], a["model"], a["routed"]) == (
        "medium",
        "medium",
        "fake-medium",
        False,
    )
    assert a["basis"] == "no_data" and a["expected_usd"] is None
    assert a["max_calls"] == S.max_attempts * PER_ATTEMPT
    assert a["worst_usd"] == pytest.approx(worst_usd(S.max_attempts))
    assert data["worst_usd"] == pytest.approx(2 * worst_usd(S.max_attempts))
    assert data["expected_usd"] == 0 and data["expected_complete"] is False
    assert data["unknown"] == ["a", "b"]
    assert data["excluded"] == []
    assert data["router"] is False
    assert data["budget"]["project_cap_usd"] == pytest.approx(5.0)
    assert data["budget"]["project_remaining_usd"] == pytest.approx(5.0)
    assert data["budget"]["daily_cap_usd"] == pytest.approx(2.0)
    assert data["assessment"] == "unknown"
    assert data["settings"]["max_attempts"] == S.max_attempts


def test_no_api_key_needed(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(config: Any) -> Any:
        raise AssertionError("estimate must not build a provider")

    monkeypatch.setattr(cli, "make_provider", boom)
    env.project(cr.task_dict("a"))
    r = estimate(env, cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output


def test_expected_cost_from_history(env: cr.Env) -> None:
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=5)  # 1000 nUSD per attempt
    env.project(cr.task_dict("a"))
    data = estimate_json(env)
    a = by_id(data)["a"]
    assert a["basis"] == "history"
    assert a["samples"] == 5 and a["pass_rate"] == 1.0
    assert a["expected_attempts"] == 1.0
    assert a["expected_usd"] == pytest.approx(nusd_to_usd(1000))
    assert data["expected_complete"] is True
    assert data["assessment"] == "expected_fits"  # worst case ($32) > $5 cap


def test_max_attempts_option_changes_worst_case(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    data = estimate_json(env, "--max-attempts", "1")
    assert data["settings"]["max_attempts"] == 1
    assert by_id(data)["a"]["max_calls"] == PER_ATTEMPT
    assert data["worst_usd"] == pytest.approx(worst_usd(1))


def test_worst_fits_with_small_worst_case(env: cr.Env) -> None:
    env.config.write_text(
        env.config.read_text(encoding="utf-8").replace("project_usd = 5.0", "project_usd = 50.0"),
        encoding="utf-8",
    )
    env.project(cr.task_dict("a"))
    data = estimate_json(env)
    assert data["assessment"] == "worst_fits"


def test_expected_exceeds_budget_left(env: cr.Env) -> None:
    # 10 USD per attempt in the history; the project cap is 5 USD.
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=5, cost_nusd=10 * 10**9)
    env.project(cr.task_dict("a"))
    data = estimate_json(env)
    assert data["assessment"] == "expected_exceeds"
    r = estimate(env, cr.PID)
    assert "exceeds the project budget left" in r.output


def test_router_option(env: cr.Env) -> None:
    rr.seed_small_good(env)
    env.project(cr.task_dict("a"))
    routed = by_id(estimate_json(env, "--router"))["a"]
    assert (routed["tier"], routed["model"], routed["routed"]) == ("small", "fake-small", True)
    assert routed["requested"] == "medium"
    plain = by_id(estimate_json(env))["a"]  # [router] enabled is off by default
    assert plain["tier"] == "medium" and plain["routed"] is False


def test_router_enabled_in_config(env: cr.Env) -> None:
    rr.configure(env, router="[router]\nenabled = true")
    rr.seed_small_good(env)
    env.project(cr.task_dict("a"))
    data = estimate_json(env)
    assert data["router"] is True
    assert by_id(data)["a"]["tier"] == "small"
    assert by_id(estimate_json(env, "--no-router"))["a"]["tier"] == "medium"


# --- after runs ---------------------------------------------------------------------


def test_passed_tasks_are_excluded_and_spending_is_from_ledger(env: cr.Env) -> None:
    env.project(cr.task_dict("a"), cr.task_dict("b"))
    env.script(*cr.write("a"), ProviderError("down"))
    r = env.run(cr.PID)
    assert r.exit_code == cli.EXIT_PROVIDER, r.output
    with CostLedger(env.ws / cli.LEDGER_FILENAME) as ledger:
        spent = ledger.spent_nusd(project_id=cr.PID)
    assert spent > 0
    data = estimate_json(env)
    assert data["excluded"] == [{"task_id": "a", "reason": "passed"}]
    assert list(by_id(data)) == ["b"]
    assert data["budget"]["project_spent_usd"] == pytest.approx(nusd_to_usd(spent))
    assert data["budget"]["daily_spent_usd"] == pytest.approx(nusd_to_usd(spent))
    assert data["budget"]["project_remaining_usd"] == pytest.approx(
        nusd_to_usd(cap_to_nusd(5.0) - spent)
    )


def test_estimate_does_not_change_ledger_or_project(env: cr.Env) -> None:
    env.project(cr.task_dict("a"), cr.task_dict("b"))
    env.script(*cr.write("a"), ProviderError("down"))
    env.run(cr.PID)
    before = tree(env.ws)
    estimate_json(env)
    r = estimate(env, cr.PID)
    assert r.exit_code == cli.EXIT_OK
    assert tree(env.ws) == before


def test_resumed_task_counts_remaining_attempts(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a", "bad"), ProviderError("down"))
    r = env.run(cr.PID)
    assert r.exit_code == cli.EXIT_PROVIDER, r.output
    a = by_id(estimate_json(env))["a"]
    assert a["done_attempts"] == 1
    assert a["reverify"] is False
    assert a["tier"] == "medium"
    assert a["max_calls"] == (S.max_attempts - 1) * PER_ATTEMPT


def test_resumed_task_is_never_routed_again(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a", "bad"), ProviderError("down"))
    env.run(cr.PID)
    rr.seed_small_good(env)
    a = by_id(estimate_json(env, "--router"))["a"]
    assert a["tier"] == "medium" and a["routed"] is False


def test_failed_and_blocked_tasks_are_excluded(env: cr.Env) -> None:
    env.project(cr.task_dict("a"), cr.task_dict("b", "a"), cr.task_dict("c"))
    env.script(*cr.write("a", "bad"), *cr.write("c"))
    r = env.run(cr.PID, "--max-attempts", "1")
    assert r.exit_code == cli.EXIT_TASKS_FAILED, r.output
    data = estimate_json(env)
    reasons = {e["task_id"]: e["reason"] for e in data["excluded"]}
    assert reasons == {"a": "failed", "b": "blocked", "c": "passed"}
    assert data["tasks"] == []
    assert data["worst_usd"] == 0
    assert data["assessment"] == "worst_fits"


def test_dependent_of_failed_task_will_be_blocked(env: cr.Env) -> None:
    env.project(cr.task_dict("a"), cr.task_dict("b", "a"))
    env.script(*cr.write("a", "bad"))
    env.run(cr.PID, "--max-attempts", "1")
    # Simulate a crash before the orchestrator blocked b: b is still pending.
    store = env.store
    assert store.task_states(cr.PID)["b"].status.value == "blocked"
    (store.project_dir(cr.PID) / "tasks" / "b" / "state.json").unlink()
    assert store.task_states(cr.PID)["b"].status.value == "pending"
    data = estimate_json(env)
    reasons = {e["task_id"]: e["reason"] for e in data["excluded"]}
    assert reasons == {"a": "failed", "b": "will be blocked (dependency a will not pass)"}
    assert data["tasks"] == []


# --- text output --------------------------------------------------------------------


def test_text_output(env: cr.Env) -> None:
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=5)
    env.project(cr.task_dict("a"), cr.task_dict("b", role="developer"))
    r = estimate(env, cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    out = r.output
    assert out.startswith(f"project: {cr.PID} (")
    assert "plan v1: 2 task(s), 2 to run" in out
    assert "router: off" in out
    assert "  a [writer] medium (fake-medium): about $" in out
    assert "(history); expected $0.0000 (~1.00 attempt(s)); guard cap $" in out
    assert "  b [developer] medium (fake-medium): about $" in out
    assert "(defaults); expected unknown" in out
    assert "total: about $" in out
    assert "; expected $0.0000 + 1 task(s) unknown; guard cap (worst case) $" in out
    assert (
        "range basis: partly from your quality history, partly from default token profiles" in out
    )
    assert "toman" not in out
    assert "budget: project $0.0000 spent of $5.0000, $5.0000 left" in out
    assert "assessment: the known part fits" in out
    assert "advisory only" in out


def test_text_output_lists_excluded_and_progress(env: cr.Env) -> None:
    env.project(cr.task_dict("a"), cr.task_dict("b"))
    env.script(*cr.write("a"), *cr.write("b", "bad"), ProviderError("down"))
    env.run(cr.PID)
    r = estimate(env, cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "  a: passed, not counted" in r.output
    assert "1 attempt(s) done" in r.output


def test_help_lists_estimate() -> None:
    r = CliRunner().invoke(cli.app, ["--help"])
    assert "estimate" in r.output
    r = CliRunner().invoke(cli.app, ["estimate", "--help"])
    assert r.exit_code == 0
    for option in ("--json", "--router", "--max-attempts", "--config"):
        assert option in r.output


def test_symlinked_ledger_is_config_error(env: cr.Env, tmp_path: Path) -> None:
    env.project(cr.task_dict("a"))
    real = tmp_path / "elsewhere.sqlite3"
    with CostLedger(real):
        pass
    (env.ws / cli.LEDGER_FILENAME).symlink_to(real)
    r = estimate(env, cr.PID)
    assert r.exit_code == cli.EXIT_CONFIG
    assert "symlink" in r.output
