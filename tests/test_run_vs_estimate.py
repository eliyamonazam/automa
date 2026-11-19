"""Tests for the actual cost next to the estimated range at the end of ``automa run``
(D122): ``range_verdict``/``check_range`` and the ``vs estimate:`` line with its
``finance.actual`` event.

Local only: FakeProvider behind the real gateway/guard/ledger, real Orchestrator and
ProjectStore. No network, no real model calls.
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import test_cli_run as cr
import test_finance_compare as fc
from automa import cli
from automa.budget import guard as guard_module
from automa.finance import (
    ACTUAL_EVENT,
    RangeCheck,
    RangeVerdict,
    check_range,
    compare_estimate,
    range_verdict,
)
from automa.finance import compare as compare_module
from automa.project import ProjectStatus

V = RangeVerdict
T0 = fc.T0
RANGE = {"range_low_nusd": 100, "range_high_nusd": 1_000, "range_basis": "defaults"}


# --- range_verdict -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("actual", "verdict"),
    [(0, V.BELOW), (99, V.BELOW), (100, V.WITHIN), (500, V.WITHIN), (1_000, V.WITHIN)]
    + [(1_001, V.ABOVE), (10**12, V.ABOVE)],
)
def test_range_verdict_is_inclusive(actual: int, verdict: RangeVerdict) -> None:
    assert range_verdict(actual, 100, 1_000) is verdict


def test_range_verdict_zero_range() -> None:
    assert range_verdict(0, 0, 0) is V.WITHIN
    assert range_verdict(1, 0, 0) is V.ABOVE


# --- RangeCheck --------------------------------------------------------------------


def check(**over: Any) -> RangeCheck:
    d: dict[str, Any] = {
        "actual_nusd": 500,
        "other_nusd": 0,
        "low_nusd": 100,
        "high_nusd": 1_000,
        "worst_nusd": 2_000,
        "verdict": V.WITHIN,
    }
    d.update(over)
    return RangeCheck(**d)


def test_range_check_valid() -> None:
    c = check()
    assert c.verdict is V.WITHIN and not c.above_worst


def test_range_check_above_worst() -> None:
    assert check(actual_nusd=3_000, verdict=V.ABOVE).above_worst


@pytest.mark.parametrize(
    "over",
    [
        {"verdict": V.ABOVE},  # contradicts the figures
        {"actual_nusd": 50},  # below, but says within
        {"low_nusd": 2_000},  # low > high
        {"high_nusd": 3_000},  # high > worst
        {"actual_nusd": -1},
        {"other_nusd": -1},
        {"unexpected": 1},
        {"actual_nusd": 1.5},  # strict: no floats for money
    ],
)
def test_range_check_rejects_inconsistent(over: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        check(**over)


def test_range_check_is_frozen() -> None:
    c = check()
    with pytest.raises(ValidationError):
        c.actual_nusd = 1  # type: ignore[misc]


# --- check_range -------------------------------------------------------------------


def test_check_range_within() -> None:
    comp = compare_estimate("p1", [fc.est(**RANGE)], [fc.entry(300), fc.entry(200)])
    c = check_range(comp)
    assert c is not None
    assert (c.actual_nusd, c.low_nusd, c.high_nusd, c.worst_nusd) == (500, 100, 1_000, 2_000)
    assert c.verdict is V.WITHIN and c.other_nusd == 0


def test_check_range_below_without_spending() -> None:
    c = check_range(compare_estimate("p1", [fc.est(**RANGE)], []))
    assert c is not None and c.actual_nusd == 0 and c.verdict is V.BELOW


def test_check_range_above_and_above_worst() -> None:
    c = check_range(compare_estimate("p1", [fc.est(**RANGE)], [fc.entry(1_500)]))
    assert c is not None and c.verdict is V.ABOVE and not c.above_worst
    c = check_range(compare_estimate("p1", [fc.est(**RANGE)], [fc.entry(2_500)]))
    assert c is not None and c.verdict is V.ABOVE and c.above_worst


def test_check_range_other_calls_are_separate() -> None:
    comp = compare_estimate(
        "p1", [fc.est(**RANGE)], [fc.entry(200), fc.entry(5_000, task=None), fc.entry(7, task="x")]
    )
    c = check_range(comp)
    assert c is not None
    assert c.actual_nusd == 200 and c.other_nusd == 5_007 and c.verdict is V.WITHIN


def test_check_range_ignores_spending_before_the_estimate() -> None:
    old = fc.entry(5_000, ts=T0 - timedelta(minutes=1))
    c = check_range(compare_estimate("p1", [fc.est(**RANGE)], [old, fc.entry(300)]))
    assert c is not None and c.actual_nusd == 300


def test_check_range_none_without_estimate() -> None:
    assert check_range(compare_estimate("p1", [], [fc.entry()])) is None


def test_check_range_none_for_invalid_estimate() -> None:
    bad = fc.est(**{**RANGE, "range_high_nusd": 5_000})  # above the worst case
    comp = compare_estimate("p1", [bad], [])
    assert comp.estimate is None
    assert check_range(comp) is None


def test_check_range_none_for_estimate_before_ranges() -> None:
    assert check_range(compare_estimate("p1", [fc.est()], [fc.entry()])) is None


def test_check_range_since_rejects_an_earlier_runs_estimate() -> None:
    comp = compare_estimate("p1", [fc.est(**RANGE)], [fc.entry()])
    assert check_range(comp, since=T0 + timedelta(seconds=1)) is None
    assert check_range(comp, since=T0) is not None
    assert check_range(comp, since=T0 - timedelta(hours=1)) is not None


def test_check_range_since_naive_is_utc() -> None:
    comp = compare_estimate("p1", [fc.est(**RANGE)], [fc.entry()])
    naive = datetime(2026, 11, 10, 12, 0, 1)  # one second after T0, no tzinfo
    assert check_range(comp, since=naive) is None


def test_check_range_rejects_non_comparison() -> None:
    with pytest.raises(TypeError):
        check_range("nope")  # type: ignore[arg-type]


def test_compare_module_stays_pure() -> None:
    tree = ast.parse(inspect.getsource(compare_module))
    names = {
        n.module if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in getattr(n, "names", [])
    }
    for forbidden in ("automa.llm", "automa.budget.guard", "httpx", "subprocess", "os"):
        assert forbidden not in names


def test_guard_never_reads_the_actual_event() -> None:
    src = inspect.getsource(guard_module)
    assert ACTUAL_EVENT not in src and "range_verdict" not in src and "RangeCheck" not in src


# --- automa run ----------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Env:
    monkeypatch.setattr(cli, "_stdin_is_terminal", lambda: False)
    return cr.Env(tmp_path, monkeypatch)


def run(env: cr.Env) -> Any:
    return CliRunner().invoke(cli.app, ["run", cr.PID, "--yes", "--config", str(env.config)])


def actuals(env: cr.Env) -> list[dict[str, Any]]:
    return [e.data for e in env.store.events(cr.PID).read() if e.type == ACTUAL_EVENT]


def ready(env: cr.Env, *tasks: str) -> None:
    ids = tasks or ("a",)
    env.project(*(cr.task_dict(t) for t in ids))
    env.script(*[x for t in ids for x in cr.write(t)])


def vs_line(output: str) -> str:
    lines = [ln for ln in output.splitlines() if ln.startswith("vs estimate:")]
    assert len(lines) == 1, output
    return lines[0]


def set_range(monkeypatch: pytest.MonkeyPatch, low: int, high: int) -> None:
    """Make every task's estimated range ``low``..``high`` (split over the tasks)."""
    real = cli.estimate_plan

    def fake(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        n = len(result.tasks)
        tasks = [
            t.model_copy(update={"range_low_nusd": low // n, "range_high_nusd": high // n})
            for t in result.tasks
        ]
        return result.model_copy(update={"tasks": tasks})

    monkeypatch.setattr(cli, "estimate_plan", fake)


def test_run_prints_actual_vs_range_and_saves_event(env: cr.Env) -> None:
    ready(env)
    r = run(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    line = vs_line(r.output)
    assert line.startswith("vs estimate: actual $")
    assert "for the estimated tasks vs range about $" in line
    out = r.output.splitlines()
    assert out.index(line) == next(i for i, x in enumerate(out) if x.startswith("cost:")) + 1
    (ev,) = actuals(env)
    assert set(ev) == {
        "actual_nusd",
        "other_nusd",
        "range_low_nusd",
        "range_high_nusd",
        "worst_nusd",
        "verdict",
    }
    est = [e.data for e in env.store.events(cr.PID).read() if e.type == "finance.estimate"][-1]
    assert ev["range_low_nusd"] == est["range_low_nusd"]
    assert ev["range_high_nusd"] == est["range_high_nusd"]
    assert ev["worst_nusd"] == est["worst_nusd"]
    assert ev["actual_nusd"] > 0
    assert (
        ev["verdict"]
        == range_verdict(ev["actual_nusd"], ev["range_low_nusd"], ev["range_high_nusd"]).value
    )
    assert line.endswith(ev["verdict"])


def test_run_actual_matches_the_ledger(env: cr.Env) -> None:
    ready(env, "a", "b")
    r = run(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    from automa.budget import CostLedger

    with CostLedger.for_workspace(env.ws) as ledger:
        spent = ledger.spent_nusd(project_id=cr.PID)
    (ev,) = actuals(env)
    assert ev["actual_nusd"] + ev["other_nusd"] == spent


def test_run_below_default_range(env: cr.Env) -> None:
    ready(env)
    r = run(env)
    assert vs_line(r.output).endswith(": below")
    assert actuals(env)[0]["verdict"] == "below"


def test_run_within(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    set_range(monkeypatch, 0, 10_000_000)
    ready(env)
    r = run(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert vs_line(r.output).endswith(": within")
    assert actuals(env)[0]["verdict"] == "within"


def test_run_above(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    set_range(monkeypatch, 0, 1)
    ready(env)
    r = run(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    line = vs_line(r.output)
    assert ": above" in line and "guard cap" not in line
    assert actuals(env)[0]["verdict"] == "above"


def test_run_shows_toman_with_display_rate(env: cr.Env) -> None:
    env.write_config()
    with env.config.open("a", encoding="utf-8") as f:
        f.write("\n[display]\ntoman_per_usd = 323000\n")
    ready(env)
    r = run(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    line = vs_line(r.output)
    assert line.count("toman)") == 2  # the actual cost and the range


def test_run_without_estimate_uses_no_older_one(
    env: cr.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready(env, "a", "b")
    # An estimate of an earlier run is still in the log.
    old = fc.est(ts=datetime.now(UTC) - timedelta(hours=1), project=cr.PID, **RANGE)
    env.store.events(cr.PID).emit(old)

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("no estimate today")

    monkeypatch.setattr(cli, "estimate_plan", broken)
    r = run(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert vs_line(r.output) == "vs estimate: no cost range was estimated for this run"
    assert actuals(env) == []


def test_run_unreadable_comparison_never_changes_the_run(
    env: cr.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk\nwent away")

    monkeypatch.setattr(cli, "compare_estimate", broken)
    ready(env)
    r = run(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert vs_line(r.output) == "vs estimate: unavailable (disk went away)"
    assert env.status() is ProjectStatus.DONE
    assert actuals(env) == []


def test_run_unwritable_actual_event_is_tolerated(
    env: cr.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_emit = cli.JsonlEventLog.emit

    def emit(self: Any, event: Any) -> Any:
        if event.type == ACTUAL_EVENT:
            raise OSError("read-only")
        return real_emit(self, event)

    monkeypatch.setattr(cli.JsonlEventLog, "emit", emit)
    ready(env)
    r = run(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert vs_line(r.output).startswith("vs estimate: actual $")
    assert actuals(env) == []


def test_run_stopped_by_provider_error_still_compares(env: cr.Env) -> None:
    from automa.llm.provider import ProviderError

    env.project(cr.task_dict("a"))
    env.script(ProviderError("gateway down"))
    r = run(env)
    assert r.exit_code != cli.EXIT_OK
    assert vs_line(r.output).startswith("vs estimate: actual $0.0000")
    assert len(actuals(env)) == 1


def test_declined_run_prints_no_comparison(env: cr.Env) -> None:
    env.write_config()
    with env.config.open("a", encoding="utf-8") as f:
        f.write("confirm_above_usd = 0\n")
    ready(env)
    r = CliRunner().invoke(cli.app, ["run", cr.PID, "--config", str(env.config)])
    assert r.exit_code == cli.EXIT_NOT_CONFIRMED, r.output
    assert "vs estimate:" not in r.output
    assert actuals(env) == []


def test_each_run_compares_only_its_own_spending(env: cr.Env) -> None:
    ready(env)
    assert run(env).exit_code == cli.EXIT_OK
    first = actuals(env)[0]["actual_nusd"]
    # A second run on a finished project spends nothing new.
    r = run(env)
    assert first > 0
    assert vs_line(r.output).endswith("vs range about $0.0000 to $0.0000: within")
    second = actuals(env)[1]
    assert second["actual_nusd"] == 0 and second["verdict"] == "within"
