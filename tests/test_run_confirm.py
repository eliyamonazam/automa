"""Tests for asking before spending (D121): the ``cost range:`` line of ``automa new``
and the "continue?" question of ``automa run`` above ``[limits] confirm_above_usd``.

Local only: FakeProvider behind the real gateway/guard/ledger, real Orchestrator and
ProjectStore. No network, no real model calls.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import test_cli as tc
import test_cli_run as cr
from automa import cli
from automa.budget import guard as guard_module
from automa.config import AutomaConfig, BudgetLimits
from automa.events import JsonlEventLog
from automa.project import ProjectStatus
from automa.quality import QUALITY_FILE

ALWAYS = "confirm_above_usd = 0\n"  # appended under [limits]: always ask


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Env:
    return cr.Env(tmp_path, monkeypatch)


def limits(env: cr.Env, extra: str, tail: str = "") -> None:
    env.write_config()
    with env.config.open("a", encoding="utf-8") as f:
        f.write(extra + tail)


def terminal(monkeypatch: pytest.MonkeyPatch, is_tty: bool = True) -> None:
    monkeypatch.setattr(cli, "_stdin_is_terminal", lambda: is_tty)


def run(env: cr.Env, *args: str, input: str | None = None) -> Any:
    return CliRunner().invoke(
        cli.app, ["run", cr.PID, *args, "--config", str(env.config)], input=input
    )


def confirms(env: cr.Env) -> list[dict[str, Any]]:
    return [e.data for e in env.store.events(cr.PID).read() if e.type == cli.CONFIRM_EVENT]


def ready(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))


def assert_nothing_ran(env: cr.Env) -> None:
    assert env.provider.requests == []
    assert env.status() is ProjectStatus.NEW
    assert not (env.ws / "projects" / cr.PID / "files" / "a.txt").exists()


# --- config ------------------------------------------------------------------------


def test_default_threshold_is_thirty_cents() -> None:
    assert BudgetLimits().confirm_above_usd == 0.3
    assert AutomaConfig().limits.confirm_above_usd == 0.3


@pytest.mark.parametrize("value", [0, 0.0, 0.3, 2, 1_000_000])
def test_threshold_accepts_zero_and_positive(value: float) -> None:
    assert BudgetLimits(confirm_above_usd=value).confirm_above_usd == value


@pytest.mark.parametrize("value", [-0.01, float("inf"), float("nan"), 1_000_001])
def test_threshold_rejects_negative_and_non_finite(value: float) -> None:
    with pytest.raises(ValidationError):
        BudgetLimits(confirm_above_usd=value)


def test_threshold_does_not_change_the_caps() -> None:
    a = BudgetLimits()
    b = BudgetLimits(confirm_above_usd=1000)
    assert (a.project_usd, a.daily_usd, a.per_call_usd) == (
        b.project_usd,
        b.daily_usd,
        b.per_call_usd,
    )


def test_budget_guard_never_reads_the_threshold() -> None:
    # A question before starting, never a cap: the guard must not depend on it.
    assert "confirm_above" not in inspect.getsource(guard_module)


def test_bad_threshold_in_toml_is_config_error(env: cr.Env) -> None:
    limits(env, "confirm_above_usd = -1\n")
    ready(env)
    r = run(env)
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert env.provider.requests == []


def test_threshold_nusd() -> None:
    assert cli._confirm_threshold_nusd(0) == 0
    assert cli._confirm_threshold_nusd(0.3) == 300_000_000


# --- automa run: below the threshold ------------------------------------------------


def test_below_threshold_runs_without_asking(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch)
    ready(env)
    r = run(env)  # cheap fake prices: the range stays below the default $0.30
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "confirm:" not in r.output and "Continue?" not in r.output
    assert confirms(env) == []
    assert env.status() is ProjectStatus.DONE


def test_high_threshold_never_asks(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch, False)
    limits(env, "confirm_above_usd = 1000\n")
    ready(env)
    assert run(env).exit_code == cli.EXIT_OK


# --- automa run: above the threshold ------------------------------------------------


def test_no_terminal_without_yes_stops_before_anything(
    env: cr.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal(monkeypatch, False)
    limits(env, ALWAYS)
    ready(env)
    r = run(env)
    assert r.exit_code == cli.EXIT_NOT_CONFIRMED == 11, r.output
    assert "not started" in r.stderr and "no terminal to ask" in r.stderr
    assert f"automa run {cr.PID} --yes" in r.stderr
    assert "Nothing was spent" in r.stderr
    assert_nothing_ran(env)
    assert confirms(env)[-1]["decision"] == "no_terminal"
    assert "cost: " not in r.output  # the run never started


def test_yes_starts_without_asking(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch, False)
    limits(env, ALWAYS)
    ready(env)
    r = run(env, "--yes")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "starting (--yes)" in r.output and "Continue?" not in r.output
    assert env.status() is ProjectStatus.DONE
    assert confirms(env)[-1]["decision"] == "yes_flag"


def test_short_yes_flag(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch, False)
    limits(env, ALWAYS)
    ready(env)
    assert run(env, "-y").exit_code == cli.EXIT_OK


def test_terminal_answer_yes_runs(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch)
    limits(env, ALWAYS)
    ready(env)
    r = run(env, input="y\n")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "Continue?" in r.output
    assert "asking above $0.0000 ([limits] confirm_above_usd)" in r.output
    assert env.status() is ProjectStatus.DONE
    assert confirms(env)[-1]["decision"] == "confirmed"


@pytest.mark.parametrize("answer", ["n\n", "no\n", "\n"])
def test_terminal_answer_no_or_enter_declines(
    env: cr.Env, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    terminal(monkeypatch)
    limits(env, ALWAYS)
    ready(env)
    r = run(env, input=answer)
    assert r.exit_code == cli.EXIT_NOT_CONFIRMED, r.output
    assert f"not started: nothing was spent and project {cr.PID} is unchanged" in r.output
    assert_nothing_ran(env)
    assert confirms(env)[-1]["decision"] == "declined"


def test_end_of_input_declines(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch)
    limits(env, ALWAYS)
    ready(env)
    r = run(env, input="")  # Ctrl+D / closed input: never taken as yes
    assert r.exit_code == cli.EXIT_NOT_CONFIRMED, r.output
    assert_nothing_ran(env)
    assert confirms(env)[-1]["decision"] == "declined"


def test_question_comes_after_estimate_and_before_any_call(
    env: cr.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []
    real = cli._confirm_spend

    def spy(*args: Any, **kw: Any) -> None:
        seen.append(len(env.provider.requests))
        real(*args, **kw)

    monkeypatch.setattr(cli, "_confirm_spend", spy)
    terminal(monkeypatch)
    limits(env, ALWAYS)
    ready(env)
    r = run(env, input="y\n")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert seen == [0]
    lines = r.output.splitlines()
    est = next(i for i, x in enumerate(lines) if x.startswith("estimate: "))
    ask = next(i for i, x in enumerate(lines) if "Continue?" in x)
    cost = next(i for i, x in enumerate(lines) if x.startswith("cost: "))
    assert est < ask < cost


def test_declined_resume_keeps_paused_project_paused(
    env: cr.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal(monkeypatch)
    limits(env, ALWAYS)
    ready(env)
    env.store.set_status(cr.PID, ProjectStatus.PAUSED)
    r = run(env, input="n\n")
    assert r.exit_code == cli.EXIT_NOT_CONFIRMED
    assert env.status() is ProjectStatus.PAUSED
    assert env.provider.requests == []


def test_question_shows_toman_when_configured(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch)
    limits(env, ALWAYS, "[display]\ntoman_per_usd = 323000\n")
    ready(env)
    r = run(env, input="n\n")
    assert r.exit_code == cli.EXIT_NOT_CONFIRMED
    ask = next(x for x in r.output.splitlines() if "Continue?" in x)
    assert re.search(r"up to \$\d+\.\d{4} \(≈ [\d,]+ toman\)", ask), ask


def test_question_shows_the_upper_end_of_the_range(
    env: cr.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal(monkeypatch)
    limits(env, ALWAYS)
    ready(env)
    r = run(env, input="n\n")
    est = next(x for x in r.output.splitlines() if x.startswith("estimate: "))
    high = re.search(r"about \$[\d.]+ to (\$[\d.]+)", est)
    assert high is not None, est
    assert f"up to {high.group(1)};" in r.output
    data = confirms(env)[-1]
    assert data["threshold_nusd"] == 0 and data["range_high_nusd"] > 0


def test_unavailable_estimate_is_not_asked_and_says_so(
    env: cr.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal(monkeypatch, False)
    limits(env, ALWAYS)
    env.write_config(tiers='medium = "fake-medium"\nlarge = "fake-large"')
    with env.config.open("a", encoding="utf-8") as f:
        f.write(ALWAYS)
    env.project(cr.task_dict("a", tier="small"))
    r = run(env)
    # The real problem (unconfigured tier) is still reported by the run itself.
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert "confirm: not asked, no cost range" in r.output
    assert env.provider.requests == []
    assert confirms(env) == []


# --- _confirm_spend directly ---------------------------------------------------------


def _cfg(threshold: float) -> AutomaConfig:
    return AutomaConfig(limits=BudgetLimits(confirm_above_usd=threshold))


def test_exactly_at_threshold_is_not_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch, False)
    log = JsonlEventLog(tmp_path / "e.jsonl")
    cli._confirm_spend(_cfg(0.3), log, "p", 300_000_000, yes=False)  # no exception
    assert not (tmp_path / "e.jsonl").exists() or log.read() == []


def test_one_nano_above_threshold_is_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    terminal(monkeypatch, False)
    log = JsonlEventLog(tmp_path / "e.jsonl")
    with pytest.raises(typer.Exit) as info:
        cli._confirm_spend(_cfg(0.3), log, "p", 300_000_001, yes=False)
    assert info.value.exit_code == cli.EXIT_NOT_CONFIRMED
    assert [e.data["decision"] for e in log.read()] == ["no_terminal"]


def test_unwritable_event_log_does_not_change_the_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import typer

    terminal(monkeypatch, False)
    log = JsonlEventLog(tmp_path / "missing-dir" / "sub" / "e.jsonl")

    def broken(*_: Any, **__: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(log, "emit", broken)
    with pytest.raises(typer.Exit) as info:
        cli._confirm_spend(_cfg(0), log, "p", 1, yes=False)
    assert info.value.exit_code == cli.EXIT_NOT_CONFIRMED
    cli._confirm_spend(_cfg(0), log, "p", 1, yes=True)  # --yes still starts


def test_stdin_is_terminal_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class Closed:
        def isatty(self) -> bool:
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(cli.sys, "stdin", Closed())
    assert cli._stdin_is_terminal() is False
    monkeypatch.setattr(cli.sys, "stdin", None)
    assert cli._stdin_is_terminal() is False


# --- automa new: the cost range line -------------------------------------------------


@pytest.fixture
def new_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tc.Env:
    return tc.Env(tmp_path, monkeypatch)


def cost_range(output: str) -> str:
    found = [x for x in output.splitlines() if x.startswith("cost range: ")]
    assert len(found) == 1, output
    return found[0]


def test_new_prints_cost_range_after_the_plan(new_env: tc.Env) -> None:
    new_env.provider.push(tc.NO_QUESTIONS, tc.GOOD_PLAN)
    r = new_env.run("new", "Build a site", "--id", "site")
    assert r.exit_code == 0, r.output
    line = cost_range(r.output)
    assert re.search(r"about \$\d+\.\d{4} to \$\d+\.\d{4}", line), line
    assert "from default token profiles and catalog prices (no history yet)" in line
    assert "guard cap (worst case) $" in line
    assert "start with: automa run site" in line
    assert "will ask before starting" not in line  # cheap fake prices
    lines = r.output.splitlines()
    assert lines.index(line) > next(i for i, x in enumerate(lines) if x.startswith("status: "))


def test_new_cost_range_does_not_call_or_create_anything(new_env: tc.Env) -> None:
    new_env.provider.push(tc.NO_QUESTIONS, tc.GOOD_PLAN)
    r = new_env.run("new", "Build a site", "--id", "site")
    assert r.exit_code == 0, r.output
    assert len(new_env.provider.requests) == 2  # questions + plan only
    assert not (new_env.ws / QUALITY_FILE).exists()
    assert [e.type for e in new_env.store.events("site").read()].count("finance.estimate") == 0


def test_new_says_run_will_ask_above_threshold(new_env: tc.Env) -> None:
    new_env.write_config(extra="confirm_above_usd = 0\n")
    new_env.provider.push(tc.NO_QUESTIONS, tc.GOOD_PLAN)
    r = new_env.run("new", "Build a site", "--id", "site")
    assert r.exit_code == 0, r.output
    assert "automa run will ask before starting (above $0.0000)" in cost_range(r.output)


def test_new_cost_range_with_toman(new_env: tc.Env) -> None:
    new_env.write_config(extra="[display]\ntoman_per_usd = 323000\n")
    new_env.provider.push(tc.NO_QUESTIONS, tc.GOOD_PLAN)
    r = new_env.run("new", "Build a site", "--id", "site")
    assert r.exit_code == 0, r.output
    assert re.search(r"\(≈ [\d,]+ to [\d,]+ toman\)", cost_range(r.output))


def test_new_cost_range_unavailable_never_fails_new(
    new_env: tc.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: Any, **__: Any) -> Any:
        raise ValueError("bad\nestimate")

    monkeypatch.setattr(cli, "estimate_plan", boom)
    new_env.provider.push(tc.NO_QUESTIONS, tc.GOOD_PLAN)
    r = new_env.run("new", "Build a site", "--id", "site")
    assert r.exit_code == 0, r.output
    line = cost_range(r.output)
    assert "unavailable (bad estimate)" in line and "automa estimate site" in line
    assert new_env.store.has_plan("site")


def test_new_cost_range_matches_automa_estimate(new_env: tc.Env) -> None:
    new_env.provider.push(tc.NO_QUESTIONS, tc.GOOD_PLAN)
    r = new_env.run("new", "Build a site", "--id", "site")
    rng = re.search(r"about \$[\d.]+ to \$[\d.]+", cost_range(r.output))
    assert rng is not None
    e = new_env.run("estimate", "site")
    assert e.exit_code == 0, e.output
    total = next(x for x in e.output.splitlines() if x.startswith("total: "))
    assert rng.group(0) in total
