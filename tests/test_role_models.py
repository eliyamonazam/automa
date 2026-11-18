"""Per-role model selection (D97): ``[roles.<name>] models``/``capabilities``,
``AutomaConfig.model_for_role``, the worker's model override, the orchestrator's
choice (recorded in the task state), escalation, the router and the estimate.

Local only: FakeProvider behind a real LLMGateway + BudgetGuard + CostLedger. No
network, no real model calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import test_orchestrator as to
import test_parallel as tp
from automa.agents import VerifierAgent, WorkerAgent
from automa.budget import BudgetGuard
from automa.config import AutomaConfig, ConfigError, Tier, load_config
from automa.finance.estimate import estimate_plan
from automa.orchestrator import Orchestrator, StopReason
from automa.plan.models import TaskState, TaskStatus
from automa.project import ProjectStatus
from automa.roles import RoleRegistry
from automa.router import Router
from automa.tools import default_tools

PRICE = {"input_per_mtok": 1.0, "output_per_mtok": 2.0}
TIERS = {"small": "an-small", "medium": "an-medium", "large": "an-large"}


def model(family: str, caps: tuple[str, ...] = ("text", "code"), **price: float) -> dict:
    return {"family": family, "capabilities": list(caps), **{**PRICE, **price}}


def cfg_data(roles: dict[str, Any] | None = None, tiers: dict[str, str] | None = None) -> dict:
    data: dict[str, Any] = {
        "tiers": TIERS if tiers is None else tiers,
        "models": {
            "an-small": model("anthropic"),
            "an-medium": model("anthropic"),
            "an-large": model("anthropic"),
            "ds-a": model("deepseek"),
            "ds-b": model("deepseek"),
            "ds-c": model("deepseek"),
            "txt-only": model("openai", ("text",)),
            "img-only": model("openai", ("image_out",)),
            "vision": model("google", ("text", "vision")),
        },
        "prices": {"priced-only": PRICE},
    }
    if roles is not None:
        data["roles"] = roles
    return data


def cfg(roles: dict[str, Any] | None = None, tiers: dict[str, str] | None = None) -> AutomaConfig:
    return AutomaConfig.model_validate(cfg_data(roles, tiers))


# --- config ------------------------------------------------------------------------


def test_defaults_no_role_models() -> None:
    c = cfg()
    assert c.role_models("writer") == ()
    assert c.model_for_role("writer", Tier.MEDIUM) == "an-medium"


def test_role_models_with_capabilities_valid() -> None:
    c = cfg({"developer": {"models": ["ds-a", "ds-b"], "capabilities": ["code"]}})
    assert c.role_models("developer") == ("ds-a", "ds-b")
    assert c.roles["developer"].capabilities[0].value == "code"
    assert c.role_models("writer") == ()


def test_skills_and_models_together() -> None:
    c = cfg({"writer": {"skills": ["x"], "models": ["txt-only"]}})
    assert c.role_skills() == {"writer": ("x",)}
    assert c.role_models("writer") == ("txt-only",)


def test_models_only_role_is_not_in_role_skills() -> None:
    assert cfg({"writer": {"models": ["txt-only"]}}).role_skills() == {}


@pytest.mark.parametrize(
    ("role", "fragment"),
    [
        ({"models": ["nope"]}, "not in the [models] catalog"),
        ({"models": ["priced-only"]}, "not in the [models] catalog"),
        ({"models": ["img-only"]}, "'text' capability"),
        ({"models": ["txt-only"], "capabilities": ["code"]}, "'code' capability"),
        ({"models": ["ds-a", "vision"], "capabilities": ["vision"]}, "ds-a"),
        ({"capabilities": ["code"]}, "need a non-empty models list"),
        ({"capabilities": ["code"], "models": []}, "need a non-empty models list"),
        ({"models": "ds-a"}, "not a string"),
        ({"models": ["ds-a", "ds-a"]}, "must not repeat"),
        ({"models": ["bad id"]}, "invalid model ids"),
        ({"models": [3]}, "invalid model ids"),
        ({"models": [f"m{i}" for i in range(11)]}, "at most 10"),
        ({"models": ["ds-a"], "capabilities": "code"}, "not a string"),
        ({"models": ["ds-a"], "capabilities": ["code", "code"]}, "must not repeat"),
        ({"models": ["ds-a"], "capabilities": ["telepathy"]}, "capabilities"),
        ({"models": ["ds-a"], "tier": "large"}, "tier"),
        ({"models": ["ds-a"], "tools": ["run_command"]}, "tools"),
        ({"models": ["ds-a"], "risk": "high"}, "risk"),
    ],
)
def test_role_models_rejected(role: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ValueError) as e:
        AutomaConfig.model_validate(cfg_data({"developer": role}))
    assert fragment in str(e.value)


def test_capability_error_names_the_role() -> None:
    with pytest.raises(ValueError, match=r"\[roles.developer\]"):
        cfg({"developer": {"models": ["nope"]}})


def test_role_models_from_toml(tmp_path: Path) -> None:
    toml = tmp_path / "automa.toml"
    toml.write_text(
        """
[tiers]
medium = "an-medium"
[models."an-medium"]
family = "anthropic"
capabilities = ["text"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[models."ds-a"]
family = "deepseek"
capabilities = ["text", "code"]
input_per_mtok = 0.1
output_per_mtok = 0.2
[roles.developer]
models = ["ds-a"]
capabilities = ["code"]
""",
        encoding="utf-8",
    )
    c = load_config(toml, env_file=None, environ={})
    assert c.model_for_role("developer", Tier.MEDIUM) == "ds-a"
    toml.write_text(
        toml.read_text(encoding="utf-8").replace('["text", "code"]', '["text"]'),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="'code' capability"):
        load_config(toml, env_file=None, environ={})


def test_example_config_mentions_role_models() -> None:
    text = (Path(__file__).parent.parent / "automa.toml.example").read_text(encoding="utf-8")
    assert "capabilities = [" in text and "[roles.developer]" in text


# --- model_for_role ----------------------------------------------------------------


def test_ladder_follows_configured_tier_positions() -> None:
    c = cfg({"developer": {"models": ["ds-a", "ds-b", "ds-c"]}})
    assert [c.model_for_role("developer", t) for t in Tier] == ["ds-a", "ds-b", "ds-c"]


def test_short_ladder_uses_last_entry_for_higher_tiers() -> None:
    c = cfg({"developer": {"models": ["ds-a", "ds-b"]}})
    assert [c.model_for_role("developer", t) for t in Tier] == ["ds-a", "ds-b", "ds-b"]
    one = cfg({"developer": {"models": ["ds-a"]}})
    assert {one.model_for_role("developer", t) for t in Tier} == {"ds-a"}


def test_positions_skip_unconfigured_tiers() -> None:
    tiers = {"small": "an-small", "large": "an-large"}
    c = cfg({"developer": {"models": ["ds-a", "ds-b", "ds-c"]}}, tiers)
    assert c.model_for_role("developer", Tier.SMALL) == "ds-a"
    assert c.model_for_role("developer", Tier.LARGE) == "ds-b"


def test_unconfigured_tier_is_still_a_config_error() -> None:
    c = cfg({"developer": {"models": ["ds-a"]}}, {"medium": "an-medium"})
    with pytest.raises(ConfigError, match="tier 'large'"):
        c.model_for_role("developer", Tier.LARGE)
    with pytest.raises(ConfigError, match="tier 'large'"):
        c.model_for_role("writer", Tier.LARGE)


def test_ladder_never_gets_weaker_when_escalating() -> None:
    c = cfg({"developer": {"models": ["ds-a", "ds-b"]}})
    picks = [c.model_for_role("developer", t) for t in Tier]
    ladder = list(c.role_models("developer"))
    assert [ladder.index(p) for p in picks] == sorted(ladder.index(p) for p in picks)


# --- task state ----------------------------------------------------------------------


def test_old_task_state_without_model_still_loads() -> None:
    s = TaskState.model_validate(
        {"task_id": "a", "status": "pending", "updated_at": "2026-11-10T12:00:00Z"}
    )
    assert s.model is None


def test_task_state_model_round_trips() -> None:
    s = TaskState.model_validate(
        {"task_id": "a", "model": "ds-a", "updated_at": "2026-11-10T12:00:00Z"}
    )
    assert TaskState.model_validate_json(s.model_dump_json()).model == "ds-a"
    with pytest.raises(ValueError):
        TaskState.model_validate({"task_id": "a", "model": "", "updated_at": s.updated_at})


# --- orchestrator ------------------------------------------------------------------


class RRig(to.Rig):
    """``to.Rig`` with a role-model config (writer: medium tier by default)."""

    roles_cfg: dict[str, Any] = {}

    def __init__(self, tmp_path: Path, script: list[Any], plan: Any, roles_cfg: dict, **kw: Any):
        self.roles_cfg = roles_cfg
        super().__init__(tmp_path, script, plan, **kw)

    def rebuild(self, **limits: float) -> None:
        data = cfg_data(self.roles_cfg, self.tiers)
        if limits:
            data["limits"] = limits
        self.cfg = AutomaConfig.model_validate(data)
        events = self.store.events(to.PID)
        self.guard = BudgetGuard(self.cfg.limits, self.ledger, events=events)
        from automa.llm import LLMGateway

        self.gateway = LLMGateway(self.cfg, self.provider, guard=self.guard)
        self.worker = WorkerAgent(self.gateway, events=events, clock=to.clock)
        self.verifier = VerifierAgent(self.gateway, events=events, clock=to.clock)

    def events(self, type_: str) -> list[dict[str, Any]]:
        return [e.data for e in self.store.events(to.PID).read() if e.type == type_]


WRITER = {"writer": {"models": ["ds-a", "ds-b"]}}


def test_task_runs_on_role_model_and_state_records_it(tmp_path: Path) -> None:
    rig = RRig(tmp_path, to.write("a"), to.make_plan(to.task_dict("a")), WRITER)
    report = rig.run()
    assert report.status is ProjectStatus.DONE
    # writer starts at medium = position 1 of 3 configured tiers → ds-b
    assert rig.models() == ["ds-b", "ds-b"]
    assert rig.state("a").model == "ds-b"
    assert rig.state("a").tier is None or rig.state("a").tier is Tier.MEDIUM
    (result, _), *_ = rig.store.task_history(to.PID, "a")
    assert (result.model, result.tier) == ("ds-b", Tier.MEDIUM)
    (attempt,) = rig.events("orchestrator.attempt")
    assert attempt["model"] == "ds-b" and attempt["model_source"] == "role"
    (started,) = rig.events("worker.started")
    assert started["model"] == "ds-b"


def test_every_call_goes_through_ledger_with_role_model(tmp_path: Path) -> None:
    rig = RRig(tmp_path, to.write("a"), to.make_plan(to.task_dict("a")), WRITER)
    rig.run()
    entries = rig.ledger.entries()
    assert len(entries) == 2
    assert {(e.model, e.tier) for e in entries} == {("ds-b", "medium")}


def test_other_roles_keep_tier_models(tmp_path: Path) -> None:
    rig = RRig(
        tmp_path,
        to.write("a"),
        to.make_plan(to.task_dict("a")),
        {"developer": {"models": ["ds-a"]}},
    )
    rig.run()
    assert rig.models() == ["an-medium", "an-medium"]
    assert rig.state("a").model == "an-medium"
    (attempt,) = rig.events("orchestrator.attempt")
    assert attempt["model"] == "an-medium" and "model_source" not in attempt


def test_escalation_climbs_the_role_ladder(tmp_path: Path) -> None:
    script = [*to.write("a", "bad"), *to.write("a", "ok")]
    plan = to.make_plan(to.task_dict("a", tier="small"))
    rig = RRig(tmp_path, script, plan, WRITER)
    assert rig.run(escalate_after=1).status is ProjectStatus.DONE
    assert rig.models() == ["ds-a", "ds-a", "ds-b", "ds-b"]
    tiers = [(r.tier, r.model) for r, _ in rig.store.task_history(to.PID, "a")]
    assert tiers == [(Tier.SMALL, "ds-a"), (Tier.MEDIUM, "ds-b")]
    assert rig.state("a").model == "ds-b"
    esc = rig.events("orchestrator.escalated")
    assert [(e["from"], e["to"]) for e in esc] == [("small", "medium")]


def test_escalation_past_ladder_end_stays_on_last_model(tmp_path: Path) -> None:
    script = [*to.write("a", "bad"), *to.write("a", "ok")]
    plan = to.make_plan(to.task_dict("a"))  # medium
    rig = RRig(tmp_path, script, plan, WRITER)
    assert rig.run(escalate_after=1).status is ProjectStatus.DONE
    assert rig.models() == ["ds-b"] * 4
    tiers = [r.tier for r, _ in rig.store.task_history(to.PID, "a")]
    assert tiers == [Tier.MEDIUM, Tier.LARGE]  # the tier still escalates


def test_state_keeps_model_while_pending_after_rejection(tmp_path: Path) -> None:
    script = [*to.write("a", "bad"), to.ProviderError("down")]
    plan = to.make_plan(to.task_dict("a", tier="small"))
    rig = RRig(tmp_path, script, plan, WRITER)
    assert rig.run(escalate_after=1).reason is StopReason.PROVIDER
    st = rig.state("a")
    assert st.status is TaskStatus.PENDING and st.tier is Tier.MEDIUM
    assert st.model == "ds-b"  # the attempt that hit the provider error


def test_resume_recomputes_model_from_saved_tier(tmp_path: Path) -> None:
    script = [*to.write("a", "bad"), to.ProviderError("down"), *to.write("a", "ok")]
    plan = to.make_plan(to.task_dict("a", tier="small"))
    rig = RRig(tmp_path, script, plan, WRITER)
    rig.run(escalate_after=1)
    rig.rebuild()
    assert rig.run(escalate_after=1).status is ProjectStatus.DONE
    assert rig.models()[-2:] == ["ds-b", "ds-b"]


def test_budget_cap_applies_to_role_model_price(tmp_path: Path) -> None:
    roles = {"writer": {"models": ["pricey"]}}
    rig = RRig(tmp_path, to.write("a"), to.make_plan(to.task_dict("a")), {})
    data = cfg_data(roles)
    data["models"]["pricey"] = model("openai", input_per_mtok=1e6, output_per_mtok=1e6)
    rig.cfg = AutomaConfig.model_validate(data)
    events = rig.store.events(to.PID)
    rig.guard = BudgetGuard(rig.cfg.limits, rig.ledger, events=events)
    from automa.llm import LLMGateway

    rig.gateway = LLMGateway(rig.cfg, rig.provider, guard=rig.guard)
    rig.worker = WorkerAgent(rig.gateway, events=events, clock=to.clock)
    rig.verifier = VerifierAgent(rig.gateway, events=events, clock=to.clock)
    report = rig.run()
    assert report.status is ProjectStatus.PAUSED and report.reason is StopReason.BUDGET
    assert rig.provider.requests == [] and rig.ledger.entries() == []
    assert rig.store.task_history(to.PID, "a") == []


def test_unknown_role_with_models_is_rejected(tmp_path: Path) -> None:
    rig = RRig(tmp_path, [], to.make_plan(to.task_dict("a")), {"ghost": {"models": ["ds-a"]}})
    with pytest.raises(ConfigError, match="unknown role"):
        rig.orch()


def test_unknown_role_with_only_skills_is_not_checked_here(tmp_path: Path) -> None:
    # Skills of unknown roles are rejected when the roles are loaded (D47), not here.
    rig = RRig(tmp_path, [], to.make_plan(to.task_dict("a")), {"ghost": {"skills": ["s"]}})
    rig.orch()


def test_unconfigured_tier_with_role_models_pauses_config(tmp_path: Path) -> None:
    plan = to.make_plan(to.task_dict("a", tier="large"))
    rig = RRig(tmp_path, [], plan, WRITER, tiers={"medium": "an-medium"})
    report = rig.run()
    assert report.reason is StopReason.CONFIG
    assert rig.provider.requests == []


def test_parallel_run_uses_role_models(tmp_path: Path) -> None:
    plan = to.make_plan(to.task_dict("a"), to.task_dict("b"))
    # Each request is answered from its own task's script (threads interleave).
    router = tp.Router({"a": to.write("a"), "b": to.write("b")})
    rig = RRig(tmp_path, [router] * router.total(), plan, WRITER)
    assert rig.run(max_parallel=2).status is ProjectStatus.DONE
    assert set(rig.models()) == {"ds-b"}
    assert rig.state("a").model == rig.state("b").model == "ds-b"


# --- worker override ---------------------------------------------------------------


def test_worker_rejects_unpriced_model_before_any_call(tmp_path: Path) -> None:
    rig = RRig(tmp_path, to.write("a"), to.make_plan(to.task_dict("a")), {})
    plan = rig.store.load_plan(to.PID)
    role = rig.roles.get("writer")
    orch = rig.orch()
    executor = orch._executor(to.PID, role.policy(), task_id="a")
    for bad in ("unpriced", "img-only", ""):
        with pytest.raises(ConfigError):
            rig.worker.run(plan.tasks[0], role, executor, project_id=to.PID, model=bad)
    assert rig.provider.requests == []


# --- router and estimate ----------------------------------------------------------


def test_router_is_skipped_for_role_with_models(tmp_path: Path) -> None:
    rig = RRig(tmp_path, to.write("a"), to.make_plan(to.task_dict("a")), WRITER)
    router = Router(rig.cfg, [])
    Orchestrator(
        rig.store,
        rig.worker,
        rig.verifier,
        rig.roles,
        rig.cfg,
        tools=rig.tools,
        clock=to.clock,
        router=router,
    ).run(to.PID)
    assert rig.events("router.skipped") == [
        {"task": "a", "role": "writer", "reason": "role_models"}
    ]
    assert rig.events("router.decision") == []
    assert rig.models() == ["ds-b", "ds-b"]


def test_router_still_routes_roles_without_models(tmp_path: Path) -> None:
    rig = RRig(tmp_path, to.write("a"), to.make_plan(to.task_dict("a")), {})
    Orchestrator(
        rig.store,
        rig.worker,
        rig.verifier,
        rig.roles,
        rig.cfg,
        tools=rig.tools,
        clock=to.clock,
        router=Router(rig.cfg, []),
    ).run(to.PID)
    assert rig.events("router.skipped") == []
    assert len(rig.events("router.decision")) == 1


def test_estimate_uses_role_model() -> None:
    c = cfg(WRITER)
    roles = RoleRegistry.builtin(tools=default_tools())
    plan = to.make_plan(to.task_dict("a"), to.task_dict("b", role="developer"))
    est = estimate_plan(plan, c, roles, [], router=Router(c, []))
    by_id = {t.task_id: t for t in est.tasks}
    assert by_id["a"].model == "ds-b" and by_id["a"].routed is False
    assert by_id["b"].model == c.model_for(by_id["b"].tier)
