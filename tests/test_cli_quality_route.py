"""Tests for ``automa quality route`` — the read-only data router preview (D61).

Local only: real QualityLog, Router and config; no provider is ever built, no
network, no model calls, nothing written.
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
from automa.config import Tier
from automa.quality import QUALITY_FILE


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Env:
    e = cr.Env(tmp_path, monkeypatch)
    rr.configure(e)
    return e


def route(env: cr.Env, *args: str) -> Any:
    return CliRunner().invoke(cli.app, ["quality", "route", *args, "--config", str(env.config)])


def route_json(env: cr.Env, *args: str) -> dict[str, Any]:
    r = route(env, *args, "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    data: dict[str, Any] = json.loads(r.output)
    return data


def by_role(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {e["role"]: e for e in data["roles"]}


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


def snapshot(path: Path) -> tuple[bytes, int]:
    return path.read_bytes(), path.stat().st_mtime_ns


# --- no data ------------------------------------------------------------------------


def test_no_log_keeps_requested_tiers_and_creates_nothing(env: cr.Env) -> None:
    data = route_json(env)
    assert not env.ws.exists()
    assert env.factory_calls == 0
    assert data["enabled"] is False
    assert (data["min_attempts"], data["min_pass_rate"]) == (5, 0.8)
    roles = by_role(data)
    assert set(roles) == {"architect", "developer", "researcher", "writer"}
    w = roles["writer"]
    assert (w["requested"], w["tier"], w["model"], w["reason"]) == (
        "medium",
        "medium",
        "fake-medium",
        "requested",
    )
    assert w["changed"] is False and w["error"] is None
    assert [e["tier"] for e in w["evidence"]] == ["small", "medium", "large"]
    assert all(e["attempts"] == 0 and e["pass_rate"] is None for e in w["evidence"])
    assert roles["architect"]["tier"] == "large"


def test_text_output_when_off(env: cr.Env) -> None:
    r = route(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    lines = r.output.splitlines()
    assert lines[0].startswith("router: off (preview only")
    assert "at least 5 judged attempt(s)" in lines[1] and "80.0%" in lines[1]
    assert "writer  requested medium -> medium (fake-medium): kept the requested tier" in lines
    assert any("small" in ln and "0/0 passed (-)" in ln and "not enough data" in ln for ln in lines)
    assert not env.ws.exists()


def test_text_output_when_enabled(env: cr.Env) -> None:
    rr.configure(env, "[router]\nenabled = true\nmin_attempts = 2")
    r = route(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert r.output.splitlines()[0] == "router: on"
    assert "at least 2 judged attempt(s)" in r.output


# --- decisions match the Router -----------------------------------------------------


def test_cheaper_qualified_tier_is_shown(env: cr.Env) -> None:
    rr.seed_small_good(env)
    w = by_role(route_json(env, "--role", "writer"))["writer"]
    assert (w["tier"], w["model"], w["reason"], w["changed"]) == (
        "small",
        "fake-small",
        "cheaper",
        True,
    )
    small = w["evidence"][0]
    assert small == {
        "tier": "small",
        "model": "fake-small",
        "price_per_mtok": 1.5,
        "attempts": 5,
        "passed": 5,
        "pass_rate": 1.0,
        "qualified": True,
        "proven_weak": False,
    }
    r = route(env, "--role", "writer")
    assert "writer  requested medium -> small (fake-small): a cheaper qualified tier" in r.output
    assert "5/5 passed (100.0%)  qualified" in r.output


def test_never_more_expensive_without_proven_weakness(env: cr.Env) -> None:
    rr.seed(env, "fake-large", Tier.LARGE, passed=5)
    w = by_role(route_json(env, "--role", "writer"))["writer"]
    assert (w["tier"], w["reason"]) == ("medium", "requested")


def test_proven_weak_requested_tier_is_replaced(env: cr.Env) -> None:
    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=1, failed=4)
    rr.seed(env, "fake-large", Tier.LARGE, passed=5, project_id="old2")
    w = by_role(route_json(env, "--role", "writer"))["writer"]
    assert (w["tier"], w["reason"]) == ("large", "replaced")
    assert w["evidence"][1]["proven_weak"] is True
    r = route(env, "--role", "writer")
    assert "proven weak; cheapest qualified tier" in r.output
    assert "1/5 passed (20.0%)  proven weak" in r.output


def test_thresholds_from_config_are_used_even_when_off(env: cr.Env) -> None:
    rr.configure(env, "[router]\nenabled = false\nmin_attempts = 2\nmin_pass_rate = 0.5")
    rr.seed(env, "fake-small", Tier.SMALL, passed=1, failed=1)
    data = route_json(env, "--role", "writer")
    assert data["enabled"] is False
    assert by_role(data)["writer"]["tier"] == "small"


def test_stats_of_another_role_do_not_count(env: cr.Env) -> None:
    rr.seed(env, "fake-small", Tier.SMALL, passed=5, role="developer")
    roles = by_role(route_json(env))
    assert roles["writer"]["tier"] == "medium"
    assert roles["developer"]["tier"] == "small"


def test_matches_the_router_directly(env: cr.Env) -> None:
    from automa.config import load_config
    from automa.quality import QualityLog
    from automa.router import Router

    rr.seed(env, "fake-medium", Tier.MEDIUM, passed=2, failed=3)
    rr.seed(env, "fake-small", Tier.SMALL, passed=5, project_id="old2")
    config = load_config(env.config)
    with QualityLog(env.ws / QUALITY_FILE, read_only=True) as log:
        router = Router.from_log(config, log)
    for role, entry in by_role(route_json(env)).items():
        d = router.route(role, Tier(entry["requested"]))
        assert (entry["tier"], entry["reason"]) == (d.tier.value, d.reason.value)


# --- --tier and --role ----------------------------------------------------------------


def test_tier_option_overrides_requested_tier(env: cr.Env) -> None:
    rr.seed_small_good(env)
    roles = by_role(route_json(env, "--tier", "large"))
    assert all(e["requested"] == "large" for e in roles.values())
    assert roles["writer"]["tier"] == "small"
    assert roles["architect"]["tier"] == "large"  # no data for architect


def test_invalid_tier_is_user_error(env: cr.Env) -> None:
    r = route(env, "--tier", "huge")
    assert r.exit_code == cli.EXIT_USER
    assert "invalid tier" in r.output


def test_unconfigured_tier_option_is_config_error(env: cr.Env) -> None:
    env.config.write_text(
        env.config.read_text(encoding="utf-8").replace('large = "fake-large"\n', ""),
        encoding="utf-8",
    )
    r = route(env, "--tier", "large")
    assert r.exit_code == cli.EXIT_CONFIG
    assert "no model configured for tier 'large'" in r.output


def test_unconfigured_default_tier_is_reported_per_role(env: cr.Env) -> None:
    env.config.write_text(
        env.config.read_text(encoding="utf-8").replace('large = "fake-large"\n', ""),
        encoding="utf-8",
    )
    roles = by_role(route_json(env))
    a = roles["architect"]
    assert a["tier"] is None and "no model configured for tier 'large'" in a["error"]
    assert [e["tier"] for e in a["evidence"]] == ["small", "medium"]
    assert roles["writer"]["error"] is None
    r = route(env, "--role", "architect")
    assert r.exit_code == cli.EXIT_OK
    assert "architect  requested large: no model configured" in r.output


def test_role_filter(env: cr.Env) -> None:
    assert list(by_role(route_json(env, "--role", "developer"))) == ["developer"]


def test_invalid_role_is_user_error(env: cr.Env) -> None:
    r = route(env, "--role", "Bad Role!")
    assert r.exit_code == cli.EXIT_USER
    assert "invalid role" in r.output


def test_unknown_role_needs_tier(env: cr.Env) -> None:
    rr.seed(env, "fake-small", Tier.SMALL, passed=5, role="translator")
    roles = by_role(route_json(env))
    t = roles["translator"]
    assert t["requested"] is None and t["tier"] is None and "--tier" in t["error"]
    assert t["evidence"][0]["attempts"] == 5
    t2 = by_role(route_json(env, "--role", "translator", "--tier", "medium"))["translator"]
    assert (t2["tier"], t2["reason"]) == ("small", "cheaper")


# --- read-only and failures -------------------------------------------------------------


def test_log_is_not_modified(env: cr.Env) -> None:
    path = rr.seed(env, "fake-small", Tier.SMALL, passed=5)
    before, files = snapshot(path), tree(env.ws)
    route(env)
    route_json(env)
    assert snapshot(path) == before
    assert tree(env.ws) == files
    assert env.factory_calls == 0


def test_no_api_key_needed(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOMA_API_KEY", raising=False)
    assert route(env).exit_code == cli.EXIT_OK


def test_broken_log_is_config_error(env: cr.Env) -> None:
    env.ws.mkdir(parents=True)
    path = env.ws / QUALITY_FILE
    path.write_bytes(b"garbage" * 200)
    before = snapshot(path)
    r = route(env)
    assert r.exit_code == cli.EXIT_CONFIG
    assert snapshot(path) == before


def test_symlinked_log_is_refused(env: cr.Env, tmp_path: Path) -> None:
    real = rr.seed(env, "fake-small", Tier.SMALL, passed=5)
    moved = tmp_path / "elsewhere.sqlite3"
    real.rename(moved)
    (env.ws / QUALITY_FILE).symlink_to(moved)
    assert route(env).exit_code == cli.EXIT_CONFIG


def test_help_lists_route(env: cr.Env) -> None:
    r = CliRunner().invoke(cli.app, ["quality", "--help"])
    assert r.exit_code == 0 and "route" in r.output
