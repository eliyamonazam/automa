"""Tests for the ``estimate:`` line of ``automa run`` (D64).

Local only: FakeProvider behind the real gateway/guard/ledger, real Orchestrator,
ProjectStore, QualityLog, Router and estimate. No network, no real model calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import test_cli_estimate as ce
import test_cli_run as cr
import test_cli_run_router as rr
from automa import cli
from automa.budget import cap_to_nusd
from automa.budget.ledger import CostLedger, nusd_to_usd
from automa.config import Tier
from automa.llm import ProviderError
from automa.quality import QUALITY_FILE


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Env:
    e = cr.Env(tmp_path, monkeypatch)
    rr.configure(e)
    return e


def line(output: str) -> str:
    found = [x for x in output.splitlines() if x.startswith("estimate: ")]
    assert len(found) == 1, output
    return found[0]


def estimate_events(env: cr.Env) -> list[dict[str, Any]]:
    return [e.data for e in env.store.events(cr.PID).read() if e.type == "finance.estimate"]


# --- the line ----------------------------------------------------------------------


def test_fresh_project_prints_unknown_expected_and_worst_case(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    r = env.run(cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    text = line(r.output)
    assert "1 task(s) to run" in text
    assert "+ 1 task(s) unknown" in text
    assert f"worst {cli._usd(ce.PER_ATTEMPT * ce.CAP * ce.S.max_attempts)}" in text
    assert f"project budget left {cli._usd(cap_to_nusd(5.0))}" in text
    assert "the known part fits, the expected cost is incomplete" in text
    assert f"details: automa estimate {cr.PID}" in text


def test_line_comes_after_router_and_before_cost(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    lines = env.run(cr.PID).output.splitlines()
    router = next(i for i, x in enumerate(lines) if x.startswith("router: "))
    est = next(i for i, x in enumerate(lines) if x.startswith("estimate: "))
    cost = next(i for i, x in enumerate(lines) if x.startswith("cost: "))
    assert router < est < cost


def test_computed_before_any_model_call(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    real = cli._run_estimate

    def spy(*args: Any, **kw: Any) -> str:
        seen.append(len(env.provider.requests))
        return real(*args, **kw)

    monkeypatch.setattr(cli, "_run_estimate", spy)
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    assert env.run(cr.PID).exit_code == cli.EXIT_OK
    assert seen == [0]
    assert env.provider.requests  # the run itself did call the model


def test_expected_cost_from_history(env: cr.Env) -> None:
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=5)  # 1000 nUSD per attempt
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    text = line(env.run(cr.PID).output)
    assert f"expected {cli._usd(1000)};" in text
    assert "unknown" not in text
    assert "the expected cost fits, the worst case does not" in text


def test_worst_fits(env: cr.Env) -> None:
    env.config.write_text(
        env.config.read_text(encoding="utf-8").replace("project_usd = 5.0", "project_usd = 50.0"),
        encoding="utf-8",
    )
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    r = env.run(cr.PID, "--max-attempts", "1")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "even the worst case fits" in line(r.output)


def test_expected_exceeds_is_only_advisory(env: cr.Env) -> None:
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=5, cost_nusd=10 * 10**9)
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    r = env.run(cr.PID)
    assert "will likely stop at the project cap" in line(r.output)
    # Advisory: the run still goes ahead and finishes normally.
    assert r.exit_code == cli.EXIT_OK, r.output
    assert env.provider.requests


def test_max_attempts_of_the_run_is_used(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    text = line(env.run(cr.PID, "--max-attempts", "2").output)
    assert f"worst {cli._usd(2 * ce.PER_ATTEMPT * ce.CAP)};" in text


# --- same inputs as automa estimate ------------------------------------------------


@pytest.mark.parametrize("router_args", [(), ("--router",)])
def test_matches_automa_estimate(env: cr.Env, router_args: tuple[str, ...]) -> None:
    rr.seed_small_good(env)
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=3, failed=2, task_id="m-t")
    env.project(cr.task_dict("a"), cr.task_dict("b", "a"))
    before = ce.estimate_json(env, *router_args, "--max-attempts", "3")
    env.script(*cr.write("a"), *cr.write("b"))
    r = env.run(cr.PID, *router_args, "--max-attempts", "3")
    assert r.exit_code == cli.EXIT_OK, r.output
    (event,) = estimate_events(env)
    assert nusd_to_usd(event["expected_nusd"]) == pytest.approx(before["expected_usd"])
    assert nusd_to_usd(event["worst_nusd"]) == pytest.approx(before["worst_usd"])
    assert event["assessment"] == before["assessment"]
    assert event["tasks"] == [t["task_id"] for t in before["tasks"]]
    assert event["router"] is bool(router_args)
    assert event["unknown"] == before["unknown"]


def test_router_of_the_run_changes_the_estimate(env: cr.Env) -> None:
    # small: 1000 nUSD and always passes; medium has no history.
    rr.seed_small_good(env)
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    off = line(env.run(cr.PID).output)
    assert "unknown" in off
    env.store.create("again", project_id="proj-2")
    env.store.save_plan(
        "proj-2",
        cr.Plan.model_validate({"goal": "again", "tasks": [cr.task_dict("a")]}),
    )
    env.script(*cr.write("a"))
    r = env.run("proj-2", "--router")
    assert r.exit_code == cli.EXIT_OK, r.output
    on = line(r.output)
    assert f"expected {cli._usd(1000)};" in on and "unknown" not in on


def test_resumed_run_excludes_passed_and_uses_ledger_spending(env: cr.Env) -> None:
    env.project(cr.task_dict("a"), cr.task_dict("b"))
    env.script(*cr.write("a"), ProviderError("down"))
    assert env.run(cr.PID).exit_code == cli.EXIT_PROVIDER
    with CostLedger(env.ws / cli.LEDGER_FILENAME) as ledger:
        spent = ledger.spent_nusd(project_id=cr.PID)
    env.script(*cr.write("b"))
    r = env.run(cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    text = line(r.output)
    assert "1 task(s) to run" in text
    assert f"project budget left {cli._usd(cap_to_nusd(5.0) - spent)}" in text
    second = estimate_events(env)[-1]
    assert second["tasks"] == ["b"] and second["excluded"] == ["a"]
    assert second["project_remaining_nusd"] == cap_to_nusd(5.0) - spent


def test_done_project_has_nothing_to_run(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    env.run(cr.PID)
    r = env.run(cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    text = line(r.output)
    assert "0 task(s) to run" in text and "even the worst case fits" in text


# --- never stops or changes the run -------------------------------------------------


def test_unconfigured_tier_is_unavailable_and_the_orchestrator_reports_it(env: cr.Env) -> None:
    env.project(cr.task_dict("a", tier="small"))
    env.write_config(tiers='medium = "fake-medium"\nlarge = "fake-large"')
    r = env.run(cr.PID)
    assert line(r.output).startswith("estimate: unavailable (")
    assert r.exit_code == cli.EXIT_CONFIG, r.output  # same as without the estimate
    assert env.provider.requests == []
    assert estimate_events(env) == []


def test_unreadable_stats_with_router_off_is_unavailable_but_runs(
    env: cr.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automa.quality import QualityError, QualityLog

    def broken(self: QualityLog, *a: Any, **kw: Any) -> Any:
        raise QualityError("broken row")

    monkeypatch.setattr(QualityLog, "stats", broken)
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    r = env.run(cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert line(r.output) == ("estimate: unavailable (the quality log statistics cannot be read)")


def test_unreadable_stats_with_router_on_is_still_a_config_error(
    env: cr.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automa.quality import QualityError, QualityLog

    def broken(self: QualityLog, *a: Any, **kw: Any) -> Any:
        raise QualityError("broken row")

    monkeypatch.setattr(QualityLog, "stats", broken)
    env.project(cr.task_dict("a"))
    r = env.run(cr.PID, "--router")
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert env.provider.requests == []
    assert "estimate: " not in r.output


def test_estimate_error_never_stops_the_run(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **kw: Any) -> Any:
        raise ValueError("bad\nestimate")

    monkeypatch.setattr(cli, "estimate_plan", boom)
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    r = env.run(cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert line(r.output) == "estimate: unavailable (bad estimate)"


def test_unwritable_event_never_stops_the_run(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    from automa.events import Event, JsonlEventLog

    real = JsonlEventLog.emit

    def emit(self: JsonlEventLog, event: Event) -> None:
        if event.type == "finance.estimate":
            raise OSError("disk full")
        real(self, event)

    monkeypatch.setattr(JsonlEventLog, "emit", emit)
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    r = env.run(cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "1 task(s) to run" in line(r.output)


def test_budget_guard_still_stops_the_run(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    env.write_config(per_call=0.000001)
    env.script(*cr.write("a"))
    r = env.run(cr.PID)
    assert "estimate: " in r.output
    assert r.exit_code == cli.EXIT_BUDGET, r.output
    assert env.provider.requests == []


def test_estimate_writes_nothing_but_the_event(env: cr.Env) -> None:
    """With the guard stopping before the provider, the quality log stays empty and
    the only new event of the estimate is finance.estimate."""
    env.project(cr.task_dict("a"))
    env.write_config(per_call=0.000001)
    env.run(cr.PID)
    assert (env.ws / QUALITY_FILE).exists()
    assert len(estimate_events(env)) == 1


def test_saved_estimate_compares_with_the_run(env: cr.Env) -> None:
    """The finance.estimate event of a real (fake-provider) run parses and its
    tasks' ledger spending since then is within the worst case (D67)."""
    from automa.finance import ComparisonOutcome, compare_estimate

    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    assert env.run(cr.PID).exit_code == cli.EXIT_OK
    with CostLedger(env.ws / cli.LEDGER_FILENAME, read_only=True) as ledger:
        entries = ledger.entries()
    c = compare_estimate(cr.PID, env.store.events(cr.PID).read(), entries)
    assert c.estimate is not None and c.estimate.tasks == ("a",)
    assert c.tasks.calls >= 1
    assert c.tasks.cost_nusd == sum(e.cost_nusd for e in entries if e.task_id == "a")
    assert c.outcome is not ComparisonOutcome.ABOVE_WORST
