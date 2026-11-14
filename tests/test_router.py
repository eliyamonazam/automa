"""Tests for the data router (automa.router, D58).

Pure: statistics are built by hand or recorded into a real QualityLog in a temp
dir. No provider, no network, no model calls.
"""

from __future__ import annotations

import sqlite3
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from automa.config import AutomaConfig, ConfigError, RouterConfig, Tier, load_config
from automa.plan.models import ResultStatus, VerdictOutcome
from automa.quality import QualityError, QualityLog, QualityRecord, QualityStats
from automa.router import RouteDecision, Router, RouteReason

PRICES = {
    "s-model": {"input_per_mtok": 0.1, "output_per_mtok": 0.4},
    "m-model": {"input_per_mtok": 1.0, "output_per_mtok": 4.0},
    "l-model": {"input_per_mtok": 5.0, "output_per_mtok": 20.0},
}
TIERS = {"small": "s-model", "medium": "m-model", "large": "l-model"}


def cfg(
    tiers: dict[str, str] | None = None,
    prices: dict[str, dict[str, float]] | None = None,
    **router: object,
) -> AutomaConfig:
    return AutomaConfig.model_validate(
        {
            "tiers": TIERS if tiers is None else tiers,
            "prices": PRICES if prices is None else prices,
            "router": router,
        }
    )


def st(model: str, attempts: int, passed: int, role: str = "developer") -> QualityStats:
    return QualityStats(
        role=role, model=model, tiers=[Tier.SMALL], attempts=attempts, passed=passed, cost_nusd=0
    )


def route(stats: list[QualityStats], requested: Tier, role: str = "developer", **kw: object):
    return Router(cfg(**kw), stats).route(role, requested)


# --- config ---------------------------------------------------------------------------


def test_router_config_defaults() -> None:
    c = AutomaConfig()
    assert c.router == RouterConfig(enabled=False, min_attempts=5, min_pass_rate=0.8)


def test_router_config_from_toml(tmp_path: Path) -> None:
    p = tmp_path / "automa.toml"
    p.write_text("[router]\nenabled = true\nmin_attempts = 3\nmin_pass_rate = 1\n")
    c = load_config(p, env_file=None, environ={})
    assert c.router.enabled is True
    assert c.router.min_attempts == 3
    assert c.router.min_pass_rate == 1


@pytest.mark.parametrize(
    "body",
    [
        "enabled = 1",
        'enabled = "yes"',
        "min_attempts = 0",
        "min_attempts = -1",
        "min_attempts = 2.0",
        "min_attempts = true",
        "min_attempts = 100001",
        'min_attempts = "5"',
        "min_pass_rate = 0",
        "min_pass_rate = -0.1",
        "min_pass_rate = 1.01",
        "min_pass_rate = true",
        'min_pass_rate = "0.5"',
        "min_pass_rate = nan",
        "min_pass_rate = inf",
        "unknown = 1",
        "model = 'x'",
    ],
)
def test_router_config_rejects(tmp_path: Path, body: str) -> None:
    p = tmp_path / "automa.toml"
    p.write_text(f"[router]\n{body}\n")
    tomllib.loads(p.read_text())  # the TOML itself is valid
    with pytest.raises(ConfigError, match="router"):
        load_config(p, env_file=None, environ={})


def test_router_config_frozen() -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic frozen error
        AutomaConfig().router.enabled = True  # type: ignore[misc]


# --- no data / not enough data --------------------------------------------------------


@pytest.mark.parametrize("tier", list(Tier))
def test_no_data_keeps_requested(tier: Tier) -> None:
    d = route([], tier)
    assert d.tier is tier
    assert d.reason is RouteReason.REQUESTED
    assert not d.changed
    assert d.model == TIERS[tier.value]
    assert [e.tier for e in d.evidence] == [Tier.SMALL, Tier.MEDIUM, Tier.LARGE]
    assert all(e.attempts == 0 and e.pass_rate is None for e in d.evidence)
    assert not any(e.qualified or e.proven_weak for e in d.evidence)


def test_too_few_attempts_does_not_qualify() -> None:
    d = route([st("s-model", 4, 4)], Tier.MEDIUM)
    assert d.tier is Tier.MEDIUM
    assert d.evidence[0].qualified is False
    assert d.evidence[0].proven_weak is False


def test_too_few_failures_does_not_prove_weak() -> None:
    d = route([st("m-model", 4, 0), st("l-model", 10, 10)], Tier.MEDIUM)
    assert d.tier is Tier.MEDIUM
    assert d.reason is RouteReason.REQUESTED


# --- cheaper ---------------------------------------------------------------------------


def test_cheaper_qualified_tier_is_chosen() -> None:
    d = route([st("s-model", 5, 4)], Tier.MEDIUM)
    assert d.tier is Tier.SMALL
    assert d.model == "s-model"
    assert d.reason is RouteReason.CHEAPER
    assert d.changed
    assert d.evidence[0].qualified
    assert d.evidence[0].pass_rate == pytest.approx(0.8)


def test_cheapest_of_several_qualified() -> None:
    d = route([st("s-model", 10, 9), st("m-model", 10, 10)], Tier.LARGE)
    assert d.tier is Tier.SMALL


def test_below_threshold_cheaper_not_chosen() -> None:
    d = route([st("s-model", 10, 7)], Tier.MEDIUM)
    assert d.tier is Tier.MEDIUM
    assert d.evidence[0].proven_weak


def test_requested_qualified_and_cheapest_stays() -> None:
    d = route([st("s-model", 10, 10), st("m-model", 10, 10)], Tier.SMALL)
    assert d.tier is Tier.SMALL
    assert d.reason is RouteReason.REQUESTED


def test_requested_qualified_but_cheaper_also_qualified() -> None:
    d = route([st("s-model", 10, 8), st("m-model", 10, 10)], Tier.MEDIUM)
    assert d.tier is Tier.SMALL
    assert d.reason is RouteReason.CHEAPER


def test_never_more_expensive_on_a_hunch() -> None:
    # Requested tier has no data: a qualified but pricier tier is not chosen.
    d = route([st("l-model", 100, 100)], Tier.MEDIUM)
    assert d.tier is Tier.MEDIUM
    assert d.reason is RouteReason.REQUESTED


def test_never_more_expensive_when_requested_qualified() -> None:
    d = route([st("m-model", 10, 8), st("l-model", 10, 10)], Tier.MEDIUM)
    assert d.tier is Tier.MEDIUM


def test_min_pass_rate_edge_is_inclusive() -> None:
    # 7/10 with min 0.7: 0.7 * 10 is 7.000000000000001 in floating point.
    d = route([st("s-model", 10, 7)], Tier.MEDIUM, min_pass_rate=0.7)
    assert d.tier is Tier.SMALL


def test_just_below_threshold() -> None:
    d = route([st("s-model", 1000, 799)], Tier.MEDIUM)
    assert d.tier is Tier.MEDIUM
    assert d.evidence[0].proven_weak


def test_min_pass_rate_one_needs_all_passed() -> None:
    assert route([st("s-model", 9, 8)], Tier.MEDIUM, min_pass_rate=1).tier is Tier.MEDIUM
    assert route([st("s-model", 9, 9)], Tier.MEDIUM, min_pass_rate=1).tier is Tier.SMALL


def test_min_attempts_setting() -> None:
    assert route([st("s-model", 2, 2)], Tier.MEDIUM, min_attempts=2).tier is Tier.SMALL
    assert route([st("s-model", 2, 2)], Tier.MEDIUM, min_attempts=3).tier is Tier.MEDIUM


def test_settings_argument_overrides_config() -> None:
    r = Router(cfg(), [st("s-model", 2, 2)], RouterConfig(min_attempts=2))
    assert r.settings.min_attempts == 2
    assert r.route("developer", Tier.MEDIUM).tier is Tier.SMALL


# --- replaced (requested proven weak) --------------------------------------------------


def test_weak_requested_replaced_by_pricier_qualified() -> None:
    d = route([st("s-model", 10, 2), st("m-model", 10, 9)], Tier.SMALL)
    assert d.tier is Tier.MEDIUM
    assert d.reason is RouteReason.REPLACED
    assert d.evidence[0].proven_weak


def test_weak_requested_replaced_by_cheapest_qualified() -> None:
    d = route([st("s-model", 10, 2), st("m-model", 10, 9), st("l-model", 10, 10)], Tier.SMALL)
    assert d.tier is Tier.MEDIUM


def test_weak_requested_replaced_by_cheaper_qualified() -> None:
    d = route([st("s-model", 10, 9), st("m-model", 10, 1)], Tier.MEDIUM)
    assert d.tier is Tier.SMALL
    assert d.reason is RouteReason.REPLACED


def test_weak_requested_nothing_qualified_keeps_requested() -> None:
    d = route([st("s-model", 10, 2), st("m-model", 10, 3), st("l-model", 3, 3)], Tier.SMALL)
    assert d.tier is Tier.SMALL
    assert d.reason is RouteReason.REQUESTED


# --- roles and models -----------------------------------------------------------------


def test_stats_of_other_roles_ignored() -> None:
    d = route([st("s-model", 10, 10, role="writer")], Tier.MEDIUM)
    assert d.tier is Tier.MEDIUM
    d = route([st("s-model", 10, 10, role="writer")], Tier.MEDIUM, role="writer")
    assert d.tier is Tier.SMALL


def test_stats_of_unconfigured_models_ignored() -> None:
    d = route([st("old-model", 100, 100)], Tier.MEDIUM)
    assert d.tier is Tier.MEDIUM
    assert all(e.attempts == 0 for e in d.evidence)


def test_remapped_tier_uses_new_model_evidence() -> None:
    # Evidence follows the model, not the tier name.
    stats = [st("s-model", 10, 10)]
    tiers = {"small": "other", "medium": "m-model"}
    prices = {**PRICES, "other": {"input_per_mtok": 0.1, "output_per_mtok": 0.1}}
    d = route(stats, Tier.MEDIUM, tiers=tiers, prices=prices)
    assert d.tier is Tier.MEDIUM


def test_two_tiers_same_model_requested_wins_tie() -> None:
    tiers = {"small": "m-model", "medium": "m-model"}
    d = route([st("m-model", 10, 10)], Tier.MEDIUM, tiers=tiers)
    assert d.tier is Tier.MEDIUM
    assert d.reason is RouteReason.REQUESTED
    assert d.evidence[0].qualified and d.evidence[1].qualified


def test_equal_price_tie_lower_tier_wins_if_not_requested() -> None:
    prices = {
        "a": {"input_per_mtok": 1.0, "output_per_mtok": 1.0},
        "b": {"input_per_mtok": 1.0, "output_per_mtok": 1.0},
        "c": {"input_per_mtok": 2.0, "output_per_mtok": 1.0},
    }
    tiers = {"small": "a", "medium": "b", "large": "c"}
    d = route([st("a", 5, 5), st("b", 5, 5)], Tier.LARGE, tiers=tiers, prices=prices)
    assert d.tier is Tier.SMALL


def test_price_not_tier_order_decides() -> None:
    # A misconfigured "large" that is cheaper than "medium" is still the cheaper one.
    prices = {
        "m-model": {"input_per_mtok": 1.0, "output_per_mtok": 4.0},
        "cheap-large": {"input_per_mtok": 0.1, "output_per_mtok": 0.1},
    }
    tiers = {"medium": "m-model", "large": "cheap-large"}
    d = route([st("cheap-large", 5, 5)], Tier.MEDIUM, tiers=tiers, prices=prices)
    assert d.tier is Tier.LARGE
    assert d.reason is RouteReason.CHEAPER


def test_only_configured_tiers_considered() -> None:
    tiers = {"medium": "m-model", "large": "l-model"}
    d = route([st("s-model", 100, 100)], Tier.LARGE, tiers=tiers)
    assert d.tier is Tier.LARGE
    assert [e.tier for e in d.evidence] == [Tier.MEDIUM, Tier.LARGE]


def test_free_model_price() -> None:
    prices = {**PRICES, "free": {"input_per_mtok": 0.0, "output_per_mtok": 0.0}}
    tiers = {"small": "free", "medium": "m-model"}
    d = route([st("free", 5, 5)], Tier.MEDIUM, tiers=tiers, prices=prices)
    assert d.tier is Tier.SMALL
    assert d.evidence[0].price_per_mtok == 0


# --- errors -------------------------------------------------------------------------


def test_unconfigured_requested_tier_is_config_error() -> None:
    with pytest.raises(ConfigError, match="tier 'small'"):
        route([], Tier.SMALL, tiers={"medium": "m-model"})


def test_no_tiers_at_all() -> None:
    with pytest.raises(ConfigError):
        Router(AutomaConfig(), []).route("developer", Tier.MEDIUM)


@pytest.mark.parametrize("role", ["", "Developer", "a b", "x" * 33, "-x", "dev\n"])
def test_invalid_role(role: str) -> None:
    with pytest.raises(ValueError, match="invalid role"):
        route([], Tier.MEDIUM, role=role)


def test_duplicate_stats_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Router(cfg(), [st("s-model", 5, 5), st("s-model", 6, 6)])


def test_decision_is_frozen_and_serializable() -> None:
    d = route([st("s-model", 5, 5)], Tier.MEDIUM)
    assert isinstance(d, RouteDecision)
    with pytest.raises(Exception):  # noqa: B017
        d.tier = Tier.LARGE  # type: ignore[misc]
    again = RouteDecision.model_validate_json(d.model_dump_json())
    assert again == d


def test_router_is_deterministic_and_pure() -> None:
    stats = [st("s-model", 10, 9), st("m-model", 10, 1)]
    r = Router(cfg(), stats)
    first = r.route("developer", Tier.MEDIUM)
    assert all(r.route("developer", Tier.MEDIUM) == first for _ in range(5))
    assert stats == [st("s-model", 10, 9), st("m-model", 10, 1)]


# --- from a real quality log ---------------------------------------------------------


def _rec(i: int, model: str, tier: Tier, passed: bool, role: str = "developer") -> QualityRecord:
    return QualityRecord(
        project_id=f"p{i:04d}",
        task_id="t1",
        attempt=1,
        role=role,
        tier=tier,
        model=model,
        result_status=ResultStatus.COMPLETED,
        outcome=VerdictOutcome.PASS if passed else VerdictOutcome.FAIL,
        model_evaluated=True,
        cost_nusd=1000,
        judged_at=datetime(2026, 11, 1, tzinfo=UTC) + timedelta(minutes=i),
    )


def test_from_log(tmp_path: Path) -> None:
    with QualityLog(tmp_path / "q.sqlite3") as log:
        for i in range(6):
            log.record(_rec(i, "s-model", Tier.SMALL, passed=i != 0))  # 5/6
        for i in range(6, 8):
            log.record(_rec(i, "s-model", Tier.SMALL, passed=True, role="writer"))
        r = Router.from_log(cfg(), log)
    d = r.route("developer", Tier.MEDIUM)
    assert d.tier is Tier.SMALL
    assert d.evidence[0].attempts == 6 and d.evidence[0].passed == 5
    assert r.route("writer", Tier.MEDIUM).tier is Tier.MEDIUM  # 2 < min_attempts


def test_from_log_counts_attempts_of_any_tier(tmp_path: Path) -> None:
    # A model used under another tier name still counts for that model.
    with QualityLog(tmp_path / "q.sqlite3") as log:
        for i in range(5):
            log.record(_rec(i, "s-model", Tier.MEDIUM, passed=True))
        r = Router.from_log(cfg(), log)
    assert r.route("developer", Tier.LARGE).tier is Tier.SMALL


def test_from_empty_log(tmp_path: Path) -> None:
    with QualityLog(tmp_path / "q.sqlite3") as log:
        r = Router.from_log(cfg(), log)
    assert r.route("developer", Tier.MEDIUM).tier is Tier.MEDIUM


def test_from_broken_log_fails_closed(tmp_path: Path) -> None:
    p = tmp_path / "q.sqlite3"
    with QualityLog(p) as log:
        log.record(_rec(0, "s-model", Tier.SMALL, passed=True))
    conn = sqlite3.connect(p)
    conn.execute("UPDATE attempts SET tier = 'huge'")
    conn.commit()
    conn.close()
    with QualityLog(p) as log, pytest.raises(QualityError):
        Router.from_log(cfg(), log)


def test_example_config_router_section() -> None:
    example = Path(__file__).resolve().parent.parent / "automa.toml.example"
    data = tomllib.loads(example.read_text(encoding="utf-8"))
    assert RouterConfig.model_validate(data["router"]) == RouterConfig()
