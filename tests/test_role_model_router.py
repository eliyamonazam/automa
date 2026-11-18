"""Data-driven model choice (D98): the router over a role's own model list.

For a role with ``[roles.<name>] models`` the router's evidence is the quality
statistics of the role's model at each tier, so it starts the task on the cheapest
role model that has proven good enough. Local only: FakeProvider behind the real
gateway/guard/ledger. No network, no real model calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import test_cli_quality_route as qr
import test_cli_run as cr
import test_cli_run_router as rr
import test_orchestrator as to
import test_role_models as rm
from automa import cli
from automa.budget import BudgetGuard
from automa.config import AutomaConfig, ConfigError, Tier
from automa.finance.estimate import estimate_plan
from automa.orchestrator import Orchestrator
from automa.project import ProjectStatus
from automa.quality import QualityStats
from automa.roles import RoleRegistry
from automa.router import Router, RouteReason
from automa.tools import default_tools

TIERS = {"small": "an-small", "medium": "an-medium", "large": "an-large"}


def priced(family: str, total: float) -> dict[str, Any]:
    """A catalog entry whose input + output price is ``total`` USD per 1M tokens."""
    return {
        "family": family,
        "capabilities": ["text", "code"],
        "input_per_mtok": total / 3,
        "output_per_mtok": total * 2 / 3,
    }


def data(roles: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "tiers": TIERS,
        "models": {
            # tier models: cheap, so a role ladder is never "replaced" by them
            "an-small": priced("anthropic", 0.3),
            "an-medium": priced("anthropic", 0.6),
            "an-large": priced("anthropic", 0.9),
            "ds-cheap": priced("deepseek", 3.0),
            "ds-mid": priced("deepseek", 6.0),
            "ds-exp": priced("deepseek", 12.0),
            "ds-other": priced("deepseek", 0.1),  # cheapest of all, never in a ladder
        },
    }
    out["roles"] = (
        {"writer": {"models": ["ds-cheap", "ds-mid", "ds-exp"]}} if roles is None else roles
    )
    out.update(extra)
    return out


def config(roles: dict[str, Any] | None = None, **extra: Any) -> AutomaConfig:
    return AutomaConfig.model_validate(data(roles, **extra))


def stat(
    model: str, passed: int, attempts: int, role: str = "writer", cost: int = 0
) -> QualityStats:
    return QualityStats(
        role=role,
        model=model,
        tiers=[Tier.MEDIUM],
        attempts=attempts,
        passed=passed,
        cost_nusd=cost,
    )


# --- Router: evidence ---------------------------------------------------------------


def test_evidence_uses_role_models_and_catalog_prices() -> None:
    ev = Router(config(), []).evidence("writer")
    assert [(e.tier, e.model) for e in ev] == [
        (Tier.SMALL, "ds-cheap"),
        (Tier.MEDIUM, "ds-mid"),
        (Tier.LARGE, "ds-exp"),
    ]
    assert [round(e.price_per_mtok, 9) for e in ev] == [3.0, 6.0, 12.0]
    assert all(e.attempts == 0 and e.avg_cost_nusd is None for e in ev)


def test_evidence_of_role_without_models_is_unchanged() -> None:
    ev = Router(config(), []).evidence("developer")
    assert [e.model for e in ev] == ["an-small", "an-medium", "an-large"]


def test_evidence_short_ladder_repeats_last_model() -> None:
    c = config({"writer": {"models": ["ds-cheap", "ds-mid"]}})
    ev = Router(c, [stat("ds-mid", 5, 5)]).evidence("writer")
    assert [e.model for e in ev] == ["ds-cheap", "ds-mid", "ds-mid"]
    assert [e.attempts for e in ev] == [0, 5, 5]


def test_evidence_reports_average_cost() -> None:
    ev = Router(config(), [stat("ds-cheap", 4, 5, cost=11)]).evidence("writer")
    assert ev[0].avg_cost_nusd == 3  # ceil(11 / 5), never under-counted
    assert ev[1].avg_cost_nusd is None


# --- Router: decisions --------------------------------------------------------------


def test_cheapest_qualified_role_model_is_chosen() -> None:
    r = Router(config(), [stat("ds-cheap", 5, 5), stat("ds-mid", 5, 5)])
    d = r.route("writer", Tier.MEDIUM)
    assert (d.tier, d.model, d.reason, d.source) == (
        Tier.SMALL,
        "ds-cheap",
        RouteReason.CHEAPER,
        "role",
    )
    assert d.changed


def test_role_model_kept_without_data() -> None:
    d = Router(config(), []).route("writer", Tier.MEDIUM)
    assert (d.tier, d.model, d.reason) == (Tier.MEDIUM, "ds-mid", RouteReason.REQUESTED)


def test_not_enough_attempts_does_not_qualify() -> None:
    d = Router(config(), [stat("ds-cheap", 4, 4)]).route("writer", Tier.MEDIUM)
    assert d.model == "ds-mid"


def test_low_pass_rate_does_not_qualify() -> None:
    d = Router(config(), [stat("ds-cheap", 3, 5)]).route("writer", Tier.MEDIUM)
    assert d.model == "ds-mid"


def test_never_more_expensive_role_model_without_proven_weakness() -> None:
    d = Router(config(), [stat("ds-exp", 5, 5)]).route("writer", Tier.MEDIUM)
    assert (d.model, d.reason) == ("ds-mid", RouteReason.REQUESTED)


def test_proven_weak_role_model_is_replaced() -> None:
    r = Router(config(), [stat("ds-mid", 1, 5), stat("ds-exp", 5, 5)])
    d = r.route("writer", Tier.MEDIUM)
    assert (d.tier, d.model, d.reason) == (Tier.LARGE, "ds-exp", RouteReason.REPLACED)


def test_proven_weak_without_qualified_alternative_keeps_default() -> None:
    d = Router(config(), [stat("ds-mid", 1, 5)]).route("writer", Tier.MEDIUM)
    assert (d.model, d.reason) == ("ds-mid", RouteReason.REQUESTED)


def test_tier_model_statistics_do_not_route_a_role_ladder() -> None:
    # an-small is cheap and good, but not in the writer's ladder
    d = Router(config(), [stat("an-small", 5, 5)]).route("writer", Tier.MEDIUM)
    assert d.model == "ds-mid"


def test_model_outside_ladder_is_never_chosen() -> None:
    r = Router(config(), [stat("ds-other", 50, 50), stat("ds-mid", 0, 5)])
    d = r.route("writer", Tier.MEDIUM)
    assert d.model in {"ds-cheap", "ds-mid", "ds-exp"}
    assert d.model == "ds-mid"


def test_statistics_of_another_role_do_not_count() -> None:
    d = Router(config(), [stat("ds-cheap", 5, 5, role="developer")]).route("writer", Tier.MEDIUM)
    assert d.model == "ds-mid"


def test_role_without_models_still_routes_on_tier_models() -> None:
    r = Router(config(), [stat("an-small", 5, 5, role="developer")])
    d = r.route("developer", Tier.MEDIUM)
    assert (d.tier, d.model, d.source) == (Tier.SMALL, "an-small", "tier")


def test_short_ladder_tie_keeps_requested_tier() -> None:
    c = config({"writer": {"models": ["ds-mid", "ds-cheap"]}})  # same model at medium/large
    d = Router(c, [stat("ds-cheap", 5, 5)]).route("writer", Tier.LARGE)
    assert (d.tier, d.model, d.reason) == (Tier.LARGE, "ds-cheap", RouteReason.REQUESTED)


def test_short_ladder_picks_lowest_tier_of_cheaper_model() -> None:
    c = config({"writer": {"models": ["ds-cheap", "ds-mid"]}})
    d = Router(c, [stat("ds-cheap", 5, 5)]).route("writer", Tier.LARGE)
    assert (d.tier, d.model, d.reason) == (Tier.SMALL, "ds-cheap", RouteReason.CHEAPER)


def test_unconfigured_requested_tier_is_still_config_error() -> None:
    c = config(tiers={"small": "an-small", "medium": "an-medium"})
    with pytest.raises(ConfigError):
        Router(c, []).route("writer", Tier.LARGE)


def test_exact_pass_rate_edge_qualifies() -> None:
    c = config(router={"min_attempts": 5, "min_pass_rate": 0.8})
    d = Router(c, [stat("ds-cheap", 4, 5)]).route("writer", Tier.MEDIUM)
    assert d.model == "ds-cheap"


# --- orchestrator -------------------------------------------------------------------


class PRig(rm.RRig):
    """``RRig`` over the priced catalog of this module."""

    def rebuild(self, **limits: float) -> None:
        d = data(self.roles_cfg)
        if limits:
            d["limits"] = limits
        self.cfg = AutomaConfig.model_validate(d)
        events = self.store.events(to.PID)
        self.guard = BudgetGuard(self.cfg.limits, self.ledger, events=events)
        from automa.agents import VerifierAgent, WorkerAgent
        from automa.llm import LLMGateway

        self.gateway = LLMGateway(self.cfg, self.provider, guard=self.guard)
        self.worker = WorkerAgent(self.gateway, events=events, clock=to.clock)
        self.verifier = VerifierAgent(self.gateway, events=events, clock=to.clock)


LADDER = {"writer": {"models": ["ds-cheap", "ds-mid", "ds-exp"]}}


def run_with(rig: PRig, router: Router) -> Any:
    return Orchestrator(
        rig.store,
        rig.worker,
        rig.verifier,
        rig.roles,
        rig.cfg,
        tools=rig.tools,
        clock=to.clock,
        router=router,
    ).run(to.PID)


def test_orchestrator_starts_on_cheapest_qualified_role_model(tmp_path: Path) -> None:
    rig = PRig(tmp_path, to.write("a"), to.make_plan(to.task_dict("a")), LADDER)
    report = run_with(rig, Router(rig.cfg, [stat("ds-cheap", 5, 5)]))
    assert report.status is ProjectStatus.DONE
    assert set(rig.models()) == {"ds-cheap"}
    assert rig.state("a").model == "ds-cheap"
    assert rig.events("router.skipped") == []
    (decision,) = rig.events("router.decision")
    assert decision == {
        "task": "a",
        "role": "writer",
        "requested": "medium",
        "tier": "small",
        "model": "ds-cheap",
        "reason": "cheaper",
        "attempts": 5,
        "passed": 5,
        "source": "role",
    }
    (attempt,) = rig.events("orchestrator.attempt")
    assert (attempt["tier"], attempt["model"], attempt["model_source"]) == (
        "small",
        "ds-cheap",
        "role",
    )


def test_orchestrator_routed_calls_go_through_ledger(tmp_path: Path) -> None:
    rig = PRig(tmp_path, to.write("a"), to.make_plan(to.task_dict("a")), LADDER)
    run_with(rig, Router(rig.cfg, [stat("ds-cheap", 5, 5)]))
    entries = rig.ledger.entries()
    assert entries and {(e.model, e.tier) for e in entries} == {("ds-cheap", "small")}


def test_orchestrator_keeps_role_default_without_data(tmp_path: Path) -> None:
    rig = PRig(tmp_path, to.write("a"), to.make_plan(to.task_dict("a")), LADDER)
    run_with(rig, Router(rig.cfg, []))
    assert set(rig.models()) == {"ds-mid"}
    (decision,) = rig.events("router.decision")
    assert (decision["reason"], decision["source"]) == ("requested", "role")


def test_orchestrator_budget_still_bounds_replaced_model(tmp_path: Path) -> None:
    rig = PRig(
        tmp_path,
        to.write("a"),
        to.make_plan(to.task_dict("a")),
        LADDER,
    )
    rig.rebuild(per_call_usd=0.0001)
    router = Router(rig.cfg, [stat("ds-mid", 0, 5), stat("ds-exp", 5, 5)])
    report = run_with(rig, router)
    assert report.status is not ProjectStatus.DONE
    assert rig.models() == []  # refused by the BudgetGuard before any call
    (decision,) = rig.events("router.decision")
    assert decision["model"] == "ds-exp"


def test_orchestrator_does_not_route_role_without_models_on_role_data(tmp_path: Path) -> None:
    rig = PRig(tmp_path, to.write("a"), to.make_plan(to.task_dict("a")), {})
    run_with(rig, Router(rig.cfg, [stat("ds-cheap", 5, 5)]))
    (decision,) = rig.events("router.decision")
    assert decision["model"] == "an-medium" and "source" not in decision


# --- estimate -----------------------------------------------------------------------


def test_estimate_follows_routed_role_model() -> None:
    c = config()
    roles = RoleRegistry.builtin(tools=default_tools())
    plan = to.make_plan(to.task_dict("a"))
    est = estimate_plan(plan, c, roles, [], router=Router(c, [stat("ds-cheap", 5, 5)]))
    (t,) = est.tasks
    assert (t.tier, t.model, t.routed) == (Tier.SMALL, "ds-cheap", True)


def test_estimate_without_data_keeps_role_default() -> None:
    c = config()
    roles = RoleRegistry.builtin(tools=default_tools())
    est = estimate_plan(to.make_plan(to.task_dict("a")), c, roles, [], router=Router(c, []))
    (t,) = est.tasks
    assert (t.tier, t.model, t.routed) == (Tier.MEDIUM, "ds-mid", False)


# --- automa quality route -----------------------------------------------------------

CATALOG = """
[models."ds-cheap"]
family = "deepseek"
capabilities = ["text", "code"]
input_per_mtok = 0.1
output_per_mtok = 0.2
[models."ds-exp"]
family = "deepseek"
capabilities = ["text", "code"]
input_per_mtok = 3.0
output_per_mtok = 6.0
[roles.writer]
models = ["ds-cheap", "fake-medium", "ds-exp"]
[models."fake-medium"]
family = "anthropic"
capabilities = ["text"]
input_per_mtok = 1.0
output_per_mtok = 2.0
"""


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Env:
    e = cr.Env(tmp_path, monkeypatch)
    # fake-medium moves from [prices] to the catalog (a model may not be priced twice)
    toml = rr.TOML.replace(
        '[prices."fake-medium"]\ninput_per_mtok = 1.0\noutput_per_mtok = 2.0\n', ""
    )
    e.config.write_text(toml.format(ws=e.ws.as_posix(), router=CATALOG), encoding="utf-8")
    return e


def test_quality_route_shows_role_model_choice(env: cr.Env) -> None:
    rr.seed(env, "ds-cheap", Tier.SMALL, passed=5)
    w = qr.by_role(qr.route_json(env, "--role", "writer"))["writer"]
    assert (w["tier"], w["model"], w["reason"], w["source"]) == (
        "small",
        "ds-cheap",
        "cheaper",
        "role",
    )
    assert [e["model"] for e in w["evidence"]] == ["ds-cheap", "fake-medium", "ds-exp"]
    assert w["evidence"][0]["avg_cost_usd"] == pytest.approx(1e-6)
    r = qr.route(env, "--role", "writer")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "writer  requested medium -> small (ds-cheap): " in r.output
    assert "[role models]" in r.output
    assert "avg $0.000001/attempt" in r.output


def test_quality_route_other_roles_use_tier_models(env: cr.Env) -> None:
    d = qr.by_role(qr.route_json(env))["developer"]
    assert d["source"] == "tier"
    assert [e["model"] for e in d["evidence"]] == ["fake-small", "fake-medium", "fake-large"]


def test_quality_route_preview_matches_run(env: cr.Env) -> None:
    rr.seed(env, "ds-cheap", Tier.SMALL, passed=5)
    preview = qr.by_role(qr.route_json(env, "--role", "writer"))["writer"]
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a"))
    r = env.run(cr.PID, "--router")
    assert r.exit_code == cli.EXIT_OK, r.output
    (decision,) = rr.router_events(env)
    assert (decision["tier"], decision["model"]) == (preview["tier"], preview["model"])
    assert set(rr.models_used(env)) == {"ds-cheap"}


def test_quality_route_help_still_works() -> None:
    r = CliRunner().invoke(cli.app, ["quality", "route", "--help"])
    assert r.exit_code == 0
