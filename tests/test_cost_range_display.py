"""Tests for showing the realistic cost range (D120): ``[display] toman_per_usd``,
``automa estimate`` (text and JSON), the ``estimate:`` line of ``automa run`` and the
range fields of the ``finance.estimate`` event.

Local only: FakeProvider behind the real gateway/guard/ledger; no network, no real
model calls. Display settings never change a price, a cap or the budget guard.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import test_cli_estimate as ce
import test_cli_run as cr
import test_cli_run_router as rr
from automa import cli
from automa.config import AutomaConfig, ConfigError, DisplayConfig, Tier, load_config
from automa.events import Event
from automa.finance import EstimateRecord, RangeBasis

RATE = 323_000


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Env:
    e = cr.Env(tmp_path, monkeypatch)
    rr.configure(e)
    return e


def with_rate(env: cr.Env, rate: str = str(RATE)) -> None:
    text = env.config.read_text(encoding="utf-8")
    env.config.write_text(text + f"\n[display]\ntoman_per_usd = {rate}\n", encoding="utf-8")


def run_line(output: str) -> str:
    (found,) = [x for x in output.splitlines() if x.startswith("estimate: ")]
    return found


def events(env: cr.Env, pid: str = cr.PID) -> list[dict[str, Any]]:
    return [e.data for e in env.store.events(pid).read() if e.type == "finance.estimate"]


# --- config ------------------------------------------------------------------------


def test_display_defaults_to_dollars_only() -> None:
    assert AutomaConfig().display.toman_per_usd is None
    assert DisplayConfig().toman_per_usd is None


@pytest.mark.parametrize("value", [323000, 323000.5, 1, 100_000_000])
def test_valid_rates(value: float) -> None:
    assert DisplayConfig(toman_per_usd=value).toman_per_usd == value


@pytest.mark.parametrize("value", [0, -1, 100_000_001, float("inf"), float("nan"), "323000", True])
def test_invalid_rates(value: Any) -> None:
    with pytest.raises(ValueError):
        DisplayConfig(toman_per_usd=value)


def test_unknown_display_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        DisplayConfig.model_validate({"toman_per_usd": 1, "currency": "irr"})


def test_display_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "automa.toml"
    path.write_text("[display]\ntoman_per_usd = 323000\n", encoding="utf-8")
    config = load_config(path, env_file=None, environ={})
    assert config.display.toman_per_usd == 323000


def test_bad_display_from_toml_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "automa.toml"
    path.write_text('[display]\ntoman_per_usd = "lots"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="display"):
        load_config(path, env_file=None, environ={})


def test_display_never_changes_limits_or_prices(tmp_path: Path) -> None:
    base = tmp_path / "a.toml"
    base.write_text("", encoding="utf-8")
    rated = tmp_path / "b.toml"
    rated.write_text("[display]\ntoman_per_usd = 323000\n", encoding="utf-8")
    a = load_config(base, env_file=None, environ={})
    b = load_config(rated, env_file=None, environ={})
    assert a.limits == b.limits and a.prices == b.prices and a.models == b.models


def test_example_config_documents_display() -> None:
    text = (Path(__file__).parents[1] / "automa.toml.example").read_text(encoding="utf-8")
    assert "# [display]" in text and "toman_per_usd = 323000" in text


# --- formatting --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nusd", "rate", "want"),
    [
        (0, RATE, "0"),
        (10**9, RATE, "323,000"),
        (60_000_000, RATE, "19,400"),  # 19,380 -> nearest 100
        (340_000_000, RATE, "109,800"),  # 109,820
        (150, RATE, "0"),
        (10**9, 1.5, "0"),  # 1.5 toman -> nearest 100
        (10**12, 323000.5, "323,000,500"),
    ],
)
def test_toman(nusd: int, rate: float, want: str) -> None:
    assert cli._toman(nusd, rate) == want


def test_toman_int() -> None:
    assert cli._toman_int(10**9, RATE) == 323_000
    assert cli._toman_int(10**9, None) is None


def test_range_text() -> None:
    assert cli._range_text(60_000_000, 340_000_000, None) == "about $0.0600 to $0.3400"
    assert cli._range_text(60_000_000, 340_000_000, RATE) == (
        "about $0.0600 to $0.3400 (≈ 19,400 to 109,800 toman)"
    )


def test_every_range_basis_has_a_text() -> None:
    assert set(cli._RANGE_BASIS_TEXT) == set(RangeBasis)


# --- automa estimate ---------------------------------------------------------------


def test_estimate_json_range_without_rate(env: cr.Env) -> None:
    env.project(cr.task_dict("a"), cr.task_dict("b", "a"))
    data = ce.estimate_json(env)
    rng = data["range"]
    assert rng["basis"] == "defaults"
    assert 0 < rng["low_usd"] <= rng["high_usd"] <= data["worst_usd"]
    assert rng["toman_per_usd"] is None
    assert rng["low_toman"] is None and rng["high_toman"] is None
    assert rng["low_usd"] == pytest.approx(sum(t["range_low_usd"] for t in data["tasks"]))
    assert rng["high_usd"] == pytest.approx(sum(t["range_high_usd"] for t in data["tasks"]))
    for t in data["tasks"]:
        assert t["range_basis"] == "defaults"
        assert t["range_low_usd"] <= t["range_high_usd"] <= t["worst_usd"]


def test_estimate_json_range_with_rate(env: cr.Env) -> None:
    with_rate(env)
    env.project(cr.task_dict("a"))
    rng = ce.estimate_json(env)["range"]
    assert rng["toman_per_usd"] == RATE
    assert rng["low_toman"] == round(rng["low_usd"] * RATE / 100) * 100
    assert rng["high_toman"] == round(rng["high_usd"] * RATE / 100) * 100


def test_estimate_json_history_basis(env: cr.Env) -> None:
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=5)
    env.project(cr.task_dict("a"))
    data = ce.estimate_json(env)
    assert data["range"]["basis"] == "history"
    assert data["tasks"][0]["range_basis"] == "history"


def test_estimate_uses_history_costs_not_only_statistics(env: cr.Env) -> None:
    # Same pass rate, very different cost per attempt: only the range moves.
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=5, cost_nusd=1_000)
    env.project(cr.task_dict("a"))
    cheap = ce.estimate_json(env)["range"]
    (env.ws / cli.QUALITY_FILE).unlink()
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=5, cost_nusd=10**8)
    dear = ce.estimate_json(env)["range"]
    assert cheap["high_usd"] < dear["high_usd"]


def test_estimate_text_with_rate(env: cr.Env) -> None:
    with_rate(env)
    env.project(cr.task_dict("a"))
    r = ce.estimate(env, cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    total = next(x for x in r.output.splitlines() if x.startswith("total: "))
    assert total.startswith("total: about $")
    assert "toman); expected" in total
    assert "guard cap (worst case) $" in total
    task = next(x for x in r.output.splitlines() if x.startswith("  a [writer]"))
    assert "toman) (defaults); expected unknown" in task
    assert "range basis: from default token profiles and catalog prices (no history yet)" in (
        r.output
    )


def test_estimate_writes_nothing(env: cr.Env) -> None:
    with_rate(env)
    env.project(cr.task_dict("a"))
    before = sorted(p.relative_to(env.ws) for p in env.ws.rglob("*"))
    assert ce.estimate(env, cr.PID).exit_code == cli.EXIT_OK
    assert sorted(p.relative_to(env.ws) for p in env.ws.rglob("*")) == before


# --- automa run --------------------------------------------------------------------


def test_run_line_with_rate(env: cr.Env) -> None:
    with_rate(env)
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    r = env.run(cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    text = run_line(r.output)
    (event,) = events(env)
    low = cli._toman(event["range_low_nusd"], RATE)
    high = cli._toman(event["range_high_nusd"], RATE)
    assert f"(≈ {low} to {high} toman), from default token profiles" in text
    assert "guard cap (worst case) $" in text


def test_run_event_records_range(env: cr.Env) -> None:
    env.project(cr.task_dict("a"), cr.task_dict("b", "a"))
    env.script(*cr.write("a"), *cr.write("b"))
    assert env.run(cr.PID).exit_code == cli.EXIT_OK
    (event,) = events(env)
    assert event["range_basis"] == "defaults"
    assert 0 < event["range_low_nusd"] <= event["range_high_nusd"] <= event["worst_nusd"]
    assert "toman" not in str(event)  # display only, never stored


def test_run_range_uses_quality_history(env: cr.Env) -> None:
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=5, cost_nusd=1_000)
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    r = env.run(cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    (event,) = events(env)
    assert event["range_basis"] == "history"
    assert "from your quality history" in run_line(r.output)


def test_unreadable_costs_fall_back_to_defaults(
    env: cr.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=5)

    def broken(*_: Any, **__: Any) -> Any:
        raise cli.QualityError("broken")

    # Statistics stay readable; only the per-attempt costs fail.
    monkeypatch.setattr(cli, "attempt_costs", broken)
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    r = env.run(cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output  # advisory: the run goes ahead
    (event,) = events(env)
    assert event["range_basis"] == "defaults"


def test_rate_never_changes_the_guard(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    plain = ce.estimate_json(env)
    with_rate(env)
    rated = ce.estimate_json(env)
    assert plain["worst_usd"] == rated["worst_usd"]
    assert plain["per_call_cap_usd"] == rated["per_call_cap_usd"]
    assert plain["budget"] == rated["budget"]
    assert plain["range"]["low_usd"] == rated["range"]["low_usd"]


# --- finance.estimate record -------------------------------------------------------


def _event(**over: Any) -> Event:
    data: dict[str, Any] = {
        "plan_version": 1,
        "tasks": ["t1"],
        "expected_nusd": 0,
        "expected_complete": False,
        "unknown": ["t1"],
        "worst_nusd": 2_000,
        "project_remaining_nusd": 10_000,
        "assessment": "unknown",
    }
    data.update(over)
    return Event(
        ts=datetime(2026, 11, 19, tzinfo=UTC), type="finance.estimate", project_id="p", data=data
    )


def test_record_without_range_still_parses() -> None:
    rec = EstimateRecord.from_event(_event())
    assert rec.range_low_nusd is None and rec.range_high_nusd is None and rec.range_basis is None


def test_record_with_range() -> None:
    rec = EstimateRecord.from_event(
        _event(range_low_nusd=100, range_high_nusd=900, range_basis="mixed")
    )
    assert (rec.range_low_nusd, rec.range_high_nusd) == (100, 900)
    assert rec.range_basis is RangeBasis.MIXED


@pytest.mark.parametrize(
    "over",
    [
        {"range_low_nusd": 100},  # one end only
        {"range_high_nusd": 100, "range_basis": "defaults"},
        {"range_low_nusd": 900, "range_high_nusd": 100, "range_basis": "defaults"},
        {"range_low_nusd": 100, "range_high_nusd": 2_001, "range_basis": "defaults"},
        {"range_low_nusd": -1, "range_high_nusd": 100, "range_basis": "defaults"},
        {"range_low_nusd": 1, "range_high_nusd": 100},  # no basis
        {"range_basis": "defaults"},  # basis without a range
        {"range_low_nusd": 1, "range_high_nusd": 100, "range_basis": "guess"},
        {"range_low_nusd": 1.5, "range_high_nusd": 100, "range_basis": "defaults"},
    ],
)
def test_invalid_record_range(over: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        EstimateRecord.from_event(_event(**over))
