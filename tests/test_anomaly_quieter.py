"""Tests for the quieter unusual-spending warning (D132).

* defaults: flagged only above 5x the usual cost *and* at least $0.05;
* ``[finance] anomaly_factor`` / ``anomaly_min_usd`` in automa.toml;
* a real project's sample is compared with real projects only once there is
  enough of that history (benchmarks are small, cheap jobs);
* the web UI shows at most one summary line, never the raw events.

Local only: hand-made samples and events, FakeProvider behind the real
gateway/guard/ledger for the ``automa run`` cases. No network, no model calls.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import test_cli_run as cr
import test_cli_run_anomaly as ra
from automa import bench, cli
from automa.budget.ledger import usd_to_nusd
from automa.config import AutomaConfig, FinanceConfig, load_config
from automa.events import Event
from automa.finance import (
    ANOMALY_EVENT,
    AnomalySettings,
    BaselineSource,
    CostSample,
    SampleKind,
    anomaly_settings,
    detect_anomalies,
    is_benchmark_project,
)
from automa.finance.compare import ESTIMATE_EVENT
from automa.webui import progress as pg
from automa.webui import render as rd

T0 = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
BENCH = "bench-20261118-100000-html-page"
REAL = "p-20261119-120848-8faf2e"
CENT = 10_000_000  # $0.01 in nUSD
_ids = iter(range(1, 1_000_000))


def call(cost: int, minute: int, project: str = REAL, model: str = "m") -> CostSample:
    return CostSample(
        kind=SampleKind.CALL,
        ts=T0 + timedelta(minutes=minute),
        project_id=project,
        task_id="t1",
        role="writer",
        model=model,
        cost_nusd=cost,
        call_id=next(_ids),
    )


def flagged_at(r: Any, minute: int) -> list[Any]:
    """The anomalies of the samples at or after ``minute`` (the "run" of a test)."""
    at = T0 + timedelta(minutes=minute)
    return [a for a in r.anomalies if a.sample.ts >= at]


def bench_history(n: int = 10, cost: int = CENT // 10, start: int = 0) -> list[CostSample]:
    return [call(cost, start + i, project=f"{BENCH}{i}") for i in range(n)]


def real_history(n: int, cost: int, start: int = 100) -> list[CostSample]:
    return [call(cost, start + i, project=f"p-old-{i}") for i in range(n)]


# --- defaults and settings -------------------------------------------------------


def test_defaults_are_five_times_and_five_cents() -> None:
    s = AnomalySettings()
    assert s.factor == 5.0 and s.min_cost_nusd == 50_000_000 and s.min_samples == 5
    f = FinanceConfig()
    assert f.anomaly_factor == 5.0 and f.anomaly_min_usd == 0.05
    assert anomaly_settings(AutomaConfig()) == AnomalySettings()


def test_config_section_is_read(tmp_path: Path) -> None:
    path = tmp_path / "automa.toml"
    path.write_text("[finance]\nanomaly_factor = 8\nanomaly_min_usd = 0.2\n", encoding="utf-8")
    config = load_config(path, env_file=None, environ={})
    s = anomaly_settings(config)
    assert s.factor == 8.0 and s.min_cost_nusd == usd_to_nusd(0.2)


def test_config_zero_floor_allowed() -> None:
    s = anomaly_settings(AutomaConfig(finance=FinanceConfig(anomaly_min_usd=0)))
    assert s.min_cost_nusd == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"anomaly_factor": 1},
        {"anomaly_factor": 0.5},
        {"anomaly_factor": 1001},
        {"anomaly_factor": "5"},
        {"anomaly_factor": True},
        {"anomaly_min_usd": -0.01},
        {"anomaly_min_usd": 1001},
        {"anomaly_min_usd": float("nan")},
        {"unknown": 1},
    ],
)
def test_config_rejects_bad_values(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FinanceConfig.model_validate(bad)


def test_config_file_with_bad_section_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "automa.toml"
    path.write_text("[finance]\nanomaly_factor = 0.5\n", encoding="utf-8")
    with pytest.raises(cli.ConfigError):
        load_config(path, env_file=None, environ={})


# --- both conditions --------------------------------------------------------------


def test_high_ratio_but_small_cost_is_not_flagged() -> None:
    hist = real_history(5, CENT // 100)  # $0.0001 each
    r = detect_anomalies([*hist, call(CENT, 200)])  # $0.01: 100x, below $0.05
    assert r.anomalies == () and r.checked == 1


def test_large_cost_but_low_ratio_is_not_flagged() -> None:
    hist = real_history(5, 10 * CENT)  # $0.10
    r = detect_anomalies([*hist, call(50 * CENT, 200)])  # $0.50 = exactly 5x
    assert r.anomalies == ()


def test_both_conditions_flag() -> None:
    hist = real_history(5, 10 * CENT)
    r = detect_anomalies([*hist, call(50 * CENT + 1, 200)])
    assert [a.sample.cost_nusd for a in r.anomalies] == [50 * CENT + 1]
    assert r.anomalies[0].baseline_source is BaselineSource.PROJECTS


def test_floor_alone_decides_with_zero_baseline() -> None:
    hist = real_history(5, 0)
    assert detect_anomalies([*hist, call(4 * CENT, 200)]).anomalies == ()
    assert len(detect_anomalies([*hist, call(5 * CENT, 201)]).anomalies) == 1


# --- benchmarks out of the baseline ------------------------------------------------


def test_benchmark_prefix_matches_bench_module() -> None:
    assert bench.benchmark_project_id("x", T0).startswith("bench-")
    assert is_benchmark_project(bench.benchmark_project_id("html-page", T0))
    assert not is_benchmark_project(REAL)
    assert not is_benchmark_project("benchmark")  # no dash: an ordinary id
    assert not is_benchmark_project("my-bench-1")


def test_owner_case_real_calls_against_cheap_benchmarks_are_quiet() -> None:
    """The screenshot case: three ordinary calls were 3-5x the small benchmarks."""
    s = AnomalySettings(factor=3.0, min_cost_nusd=0)  # even with the old sensitivity…
    cheap = bench_history(10, CENT // 10)
    real = real_history(5, CENT // 2)  # earlier real calls cost about the same
    run = [call(CENT // 2, 300), call(CENT * 4 // 10, 301), call(CENT * 6 // 10, 302)]
    r = detect_anomalies([*cheap, *real, *run], s)
    assert flagged_at(r, 300) == []  # …compared with real projects, they are normal


def test_owner_case_with_defaults_and_no_real_history() -> None:
    """No real history yet: the benchmarks are the baseline, but 3-5x and cents stay quiet."""
    cheap = bench_history(10, CENT // 10)
    run = [call(3 * CENT // 10, 300), call(4 * CENT // 10, 301), call(5 * CENT // 10, 302)]
    assert detect_anomalies([*cheap, *run]).anomalies == ()


def test_falls_back_to_all_projects_until_enough_real_history() -> None:
    s = AnomalySettings(factor=3.0, min_cost_nusd=0)
    cheap = bench_history(10, 100)
    real = real_history(4, 10_000)  # one short of min_samples
    r = detect_anomalies([*cheap, *real, call(10_000, 300)], s)
    flagged = r.anomalies[-1]
    assert flagged.baseline_source is BaselineSource.ALL
    assert flagged.history == 14 and flagged.baseline_nusd == 100


def test_real_history_used_once_it_is_enough() -> None:
    s = AnomalySettings(factor=3.0, min_cost_nusd=0)
    r = detect_anomalies([*bench_history(10, 100), *real_history(5, 10_000), call(10_000, 300)], s)
    assert flagged_at(r, 300) == []  # the run call matches the real median
    r = detect_anomalies([*bench_history(10, 100), *real_history(5, 10_000), call(30_001, 300)], s)
    (a,) = flagged_at(r, 300)
    assert a.baseline_source is BaselineSource.PROJECTS
    assert a.history == 5 and a.baseline_nusd == 10_000
    assert a.event_data()["baseline_source"] == "projects"


def test_benchmark_samples_still_use_all_history() -> None:
    s = AnomalySettings(factor=3.0, min_cost_nusd=0)
    hist = [*real_history(5, 10_000, start=0), *bench_history(5, 100, start=50)]
    spike = call(40_000, 300, project=f"{BENCH}-new")
    (a,) = detect_anomalies([*hist, spike], s).anomalies
    assert a.baseline_source is BaselineSource.ALL and a.history == 10


def test_benchmark_samples_never_join_the_real_history() -> None:
    s = AnomalySettings(factor=3.0, min_cost_nusd=0)
    # Five *expensive* benchmark calls must not make a real spike look normal.
    hist = [*real_history(5, 100, start=0), *bench_history(5, 1_000_000, start=50)]
    (a,) = flagged_at(detect_anomalies([*hist, call(1_000, 300)], s), 300)
    assert a.baseline_nusd == 100 and a.history == 5


def test_keys_stay_separate_per_role_and_model() -> None:
    s = AnomalySettings(factor=3.0, min_cost_nusd=0)
    hist = real_history(5, 100)
    other_model = call(1_000, 300, model="other")
    assert detect_anomalies([*hist, other_model], s).anomalies == ()  # no history for it


def test_event_data_records_the_source() -> None:
    s = AnomalySettings(factor=3.0, min_cost_nusd=0)
    (a,) = detect_anomalies([*bench_history(5, 100), call(1_000, 300)], s).anomalies
    data = a.event_data()
    assert data["baseline_source"] == "all"
    assert json.loads(json.dumps(data)) == data


# --- web UI: one line at most --------------------------------------------------------


def _ev(type_: str, minute: int, **data: object) -> Event:
    return Event(ts=T0 + timedelta(minutes=minute), type=type_, project_id=REAL, data=data)


def _anomaly(minute: int, cost: int, baseline: int) -> Event:
    return _ev(ANOMALY_EVENT, minute, kind="call", cost_nusd=cost, baseline_nusd=baseline)


def test_no_anomaly_events_no_line() -> None:
    assert pg.unusual_spending([]) is None
    assert pg.unusual_spending([_ev("task.started", 1)]) is None


def test_old_noisy_warnings_are_not_shown() -> None:
    """Events recorded with the old 3x / $0.001 thresholds (the screenshot)."""
    events = [
        _ev(ESTIMATE_EVENT, 0),
        _anomaly(1, 3 * CENT // 10, CENT // 10),  # 3x, $0.003
        _anomaly(2, 5 * CENT // 10, CENT // 10),  # 5x, $0.005
        _anomaly(3, 4 * CENT // 10, CENT // 10),
    ]
    assert pg.unusual_spending(events) is None


def test_real_warning_is_one_summary() -> None:
    events = [
        _ev(ESTIMATE_EVENT, 0),
        _anomaly(1, 60 * CENT, 10 * CENT),  # $0.60, 6x
        _anomaly(2, 2 * CENT, CENT // 10),  # too small: dropped
        _anomaly(3, 80 * CENT, 10 * CENT),  # $0.80, 8x
    ]
    u = pg.unusual_spending(events)
    assert u is not None
    assert u.count == 2 and u.largest_usd == pytest.approx(0.8) and u.ratio == pytest.approx(8.0)


def test_only_the_latest_run_counts() -> None:
    events = [
        _ev(ESTIMATE_EVENT, 0),
        _anomaly(1, 80 * CENT, 10 * CENT),  # an earlier run
        _ev(ESTIMATE_EVENT, 10),
        _anomaly(11, 60 * CENT, 10 * CENT),
    ]
    u = pg.unusual_spending(events)
    assert u is not None and u.count == 1 and u.largest_usd == pytest.approx(0.6)
    assert pg.unusual_spending([*events[:2], _ev(ESTIMATE_EVENT, 10)]) is None


def test_without_estimate_all_events_count() -> None:
    u = pg.unusual_spending([_anomaly(1, 80 * CENT, 10 * CENT)])
    assert u is not None and u.count == 1


def test_zero_baseline_has_no_ratio() -> None:
    u = pg.unusual_spending([_anomaly(1, 10 * CENT, 0)])
    assert u is not None and u.ratio is None


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"cost_nusd": 80 * CENT},
        {"cost_nusd": "800000000", "baseline_nusd": 1},
        {"cost_nusd": 80 * CENT, "baseline_nusd": None},
        {"cost_nusd": True, "baseline_nusd": 0},
        {"cost_nusd": 80 * CENT, "baseline_nusd": -1},
        {"cost_nusd": 0.8, "baseline_nusd": 0},
        {"cost_nusd": [1], "baseline_nusd": 0},
    ],
)
def test_malformed_events_are_skipped(data: dict[str, Any]) -> None:
    assert pg.unusual_spending([_ev(ANOMALY_EVENT, 1, **data)]) is None


def test_custom_settings_apply_to_stored_events() -> None:
    events = [_anomaly(1, 3 * CENT // 10, CENT // 10)]
    loose = AnomalySettings(factor=2.0, min_cost_nusd=0)
    u = pg.unusual_spending(events, loose)
    assert u is not None and u.count == 1


def test_recent_events_leave_out_anomalies() -> None:
    events = [_ev("task.started", 1, task="a"), _anomaly(2, 80 * CENT, 10 * CENT)]
    lines = pg.recent_events(events)
    assert [e.type for e in lines] == ["task.started"]


def _view(**kw: Any) -> pg.ProgressView:
    return pg.ProgressView(passed=1, total=1, **kw)


def test_render_one_line_when_unusual() -> None:
    u = pg.UnusualSpending(count=2, largest_usd=0.8, ratio=8.0)
    html = rd._progress(_view(unusual=u))
    assert html.count("Unusual spending") == 1
    assert '<p class="warn">Unusual spending: 2 steps cost much more than usual' in html
    assert "$0.8000" in html and "8.0× the usual cost" in html
    assert "nothing was stopped" in html
    assert ANOMALY_EVENT not in html


def test_render_singular_and_no_ratio() -> None:
    html = rd._progress(_view(unusual=pg.UnusualSpending(count=1, largest_usd=0.1, ratio=None)))
    assert "1 step cost much more" in html and "the usual cost" not in html


def test_render_nothing_when_quiet() -> None:
    assert "Unusual spending" not in rd._progress(_view())


def test_build_progress_uses_given_settings(tmp_path: Path) -> None:
    from automa.project import ProjectStore

    store = ProjectStore.for_workspace(tmp_path)
    store.create("goal", project_id=REAL)
    store.events(REAL).emit(_anomaly(1, 3 * CENT // 10, CENT // 10))
    quiet = pg.build_progress(store, REAL, [], [], running=False)
    assert quiet.unusual is None and quiet.events and quiet.events[0].type != ANOMALY_EVENT
    loose = AnomalySettings(factor=2.0, min_cost_nusd=0)
    noisy = pg.build_progress(store, REAL, [], [], running=False, anomaly=loose)
    assert noisy.unusual is not None and noisy.unusual.count == 1


# --- automa run with the defaults ------------------------------------------------------

DEFAULT_TOML = ra.TOML.split("[finance]")[0]


@pytest.fixture
def default_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Env:
    e = cr.Env(tmp_path, monkeypatch)
    e.config.write_text(DEFAULT_TOML.format(ws=e.ws.as_posix()), encoding="utf-8")
    return e


def test_run_with_defaults_is_quiet_for_small_calls(default_env: cr.Env) -> None:
    """The same history that the 3x/$0.001 tests flag stays quiet by default."""
    ra.history(default_env)
    r = ra.run_one(default_env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert ra.lines(r.output) == ["unusual spending: none this run"]
    assert ra.anomaly_events(default_env) == []


def test_run_against_benchmark_history_uses_real_projects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old sensitivity, cheap benchmarks, and real history at the run's own cost."""
    env = cr.Env(tmp_path, monkeypatch)
    env.config.write_text(ra.TOML.format(ws=env.ws.as_posix()), encoding="utf-8")
    ra.history(env, n=5, cost=ra.CHEAP, project="bench-20261101-000000-x")
    flagged = ra.run_one(env)
    assert ra.lines(flagged.output)[0].startswith("unusual spending: 2 this run")
    assert {e["baseline_source"] for e in ra.anomaly_events(env)} == {"all"}


def test_finance_report_json_shows_default_settings(default_env: cr.Env) -> None:
    from typer.testing import CliRunner

    r = CliRunner().invoke(
        cli.app, ["finance", "report", "--json", "--config", str(default_env.config)]
    )
    assert r.exit_code == cli.EXIT_OK, r.output
    settings = json.loads(r.output)["anomalies"]["settings"]
    assert settings["factor"] == 5.0 and settings["min_cost_usd"] == 0.05


def test_run_compared_with_real_projects_not_benchmarks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same cheap benchmarks, plus five real calls costing more than this run's: quiet."""
    env = cr.Env(tmp_path, monkeypatch)
    env.config.write_text(ra.TOML.format(ws=env.ws.as_posix()), encoding="utf-8")
    ra.history(env, n=5, cost=ra.CHEAP, project="bench-20261101-000000-x")
    ra.history(env, n=5, cost=0.02, project="old")
    r = ra.run_one(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert ra.lines(r.output) == ["unusual spending: none this run"]
    assert ra.anomaly_events(env) == []
