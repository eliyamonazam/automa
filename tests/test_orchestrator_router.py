"""The data router wired into the Orchestrator (D59).

FakeProvider behind a real LLMGateway + BudgetGuard; real Worker/Verifier, store and
quality log. No network, no real model calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import test_orchestrator as to
from automa.config import AutomaConfig, Tier
from automa.llm import ProviderError
from automa.orchestrator import Orchestrator, OrchestratorSettings, StopReason
from automa.plan import TaskStatus
from automa.project import ProjectStatus
from automa.quality import QualityLog, QualityStats
from automa.router import Router

PRICES = {
    "fake-small": {"input_per_mtok": 0.1, "output_per_mtok": 0.2},
    "fake-medium": {"input_per_mtok": 1.0, "output_per_mtok": 2.0},
    "fake-large": {"input_per_mtok": 10.0, "output_per_mtok": 20.0},
}


def make_cfg(tiers: dict[str, str] | None = None, **limits: float) -> AutomaConfig:
    tiers = to.ALL_TIERS if tiers is None else tiers
    data: dict[str, Any] = {
        "tiers": tiers,
        "prices": {m: PRICES[m] for m in tiers.values()},
        "router": {"min_attempts": 3, "min_pass_rate": 0.8},
    }
    if limits:
        data["limits"] = limits
    return AutomaConfig.model_validate(data)


class Rig(to.Rig):
    def rebuild(self, **limits: float) -> None:
        super().rebuild(**limits)
        # same wiring as the base rig, priced so that small < medium < large
        self.cfg = make_cfg(self.tiers, **limits)
        events = self.store.events(to.PID)
        self.guard = to.BudgetGuard(self.cfg.limits, self.ledger, events=events)
        self.gateway = to.LLMGateway(self.cfg, self.provider, guard=self.guard)
        self.worker = to.WorkerAgent(self.gateway, events=events, clock=to.clock)
        self.verifier = to.VerifierAgent(self.gateway, events=events, clock=to.clock)

    def router(self, *stats: QualityStats) -> Router:
        return Router(self.cfg, stats)

    def orch_r(
        self, router: Router | None, quality: QualityLog | None = None, **settings: Any
    ) -> Orchestrator:
        return Orchestrator(
            self.store,
            self.worker,
            self.verifier,
            self.roles,
            self.cfg,
            settings=OrchestratorSettings(**settings) if settings else None,
            tools=self.tools,
            clock=to.clock,
            quality=quality,
            router=router,
        )

    def decisions(self) -> list[dict[str, Any]]:
        return [e.data for e in self.store.events(to.PID).read() if e.type == "router.decision"]


def st(model: str, attempts: int, passed: int, role: str = "writer") -> QualityStats:
    return QualityStats(
        role=role, model=model, tiers=[Tier.SMALL], attempts=attempts, passed=passed, cost_nusd=0
    )


GOOD_SMALL = st("fake-small", 5, 5)
WEAK_MEDIUM = st("fake-medium", 5, 1)
GOOD_LARGE = st("fake-large", 5, 5)


def one_task(tmp_path: Path, script: list[Any], **task: Any) -> Rig:
    return Rig(tmp_path, script, to.make_plan(to.task_dict("a", **task)))


# --- construction -------------------------------------------------------------------


def test_rejects_wrong_router_type(tmp_path: Path) -> None:
    rig = one_task(tmp_path, [])
    with pytest.raises(TypeError, match="router"):
        rig.orch_r(object())  # type: ignore[arg-type]


def test_rejects_router_from_other_config(tmp_path: Path) -> None:
    rig = one_task(tmp_path, [])
    other = Router(make_cfg({"small": "fake-small", "medium": "fake-medium"}), [])
    with pytest.raises(ValueError, match="different configuration"):
        rig.orch_r(other)
    # an equal (not identical) configuration is fine
    rig.orch_r(Router(make_cfg(), []))


# --- routing ---------------------------------------------------------------------------


def test_without_router_nothing_changes(tmp_path: Path) -> None:
    rig = one_task(tmp_path, to.write("a"))
    report = rig.orch_r(None).run(to.PID)
    assert report.status is ProjectStatus.DONE
    assert rig.models()[0] == "fake-medium"
    assert rig.decisions() == []


def test_cheaper_qualified_tier_starts_the_task(tmp_path: Path) -> None:
    rig = one_task(tmp_path, to.write("a"))
    report = rig.orch_r(rig.router(GOOD_SMALL)).run(to.PID)
    assert report.status is ProjectStatus.DONE
    assert set(rig.models()) == {"fake-small"}  # worker and verifier calls use the tier
    assert rig.state("a").tier is Tier.SMALL
    assert rig.decisions() == [
        {
            "task": "a",
            "role": "writer",
            "requested": "medium",
            "tier": "small",
            "model": "fake-small",
            "reason": "cheaper",
            "attempts": 5,
            "passed": 5,
        }
    ]


def test_no_data_keeps_requested_and_is_logged(tmp_path: Path) -> None:
    rig = one_task(tmp_path, to.write("a"))
    rig.orch_r(rig.router()).run(to.PID)
    assert rig.models()[0] == "fake-medium"
    (d,) = rig.decisions()
    assert (d["tier"], d["reason"], d["attempts"], d["passed"]) == ("medium", "requested", 0, 0)


def test_stats_of_other_roles_do_not_route(tmp_path: Path) -> None:
    rig = one_task(tmp_path, to.write("a"))
    rig.orch_r(rig.router(st("fake-small", 9, 9, role="developer"))).run(to.PID)
    assert rig.models()[0] == "fake-medium"


def test_weak_requested_replaced_by_pricier_qualified(tmp_path: Path) -> None:
    rig = one_task(tmp_path, to.write("a"))
    rig.orch_r(rig.router(WEAK_MEDIUM, GOOD_LARGE)).run(to.PID)
    assert rig.models()[0] == "fake-large"
    assert rig.decisions()[0]["reason"] == "replaced"


def test_plan_tier_is_the_requested_tier(tmp_path: Path) -> None:
    rig = one_task(tmp_path, to.write("a"), tier="large")
    rig.orch_r(rig.router(GOOD_SMALL)).run(to.PID)
    (d,) = rig.decisions()
    assert (d["requested"], d["tier"]) == ("large", "small")
    assert rig.models()[0] == "fake-small"


def test_router_ignores_enabled_flag(tmp_path: Path) -> None:
    # Deciding whether to route is the caller's job (CLI); a given router routes.
    rig = one_task(tmp_path, to.write("a"))
    assert rig.cfg.router.enabled is False
    rig.orch_r(rig.router(GOOD_SMALL)).run(to.PID)
    assert rig.models()[0] == "fake-small"


def test_routed_tier_still_escalates(tmp_path: Path) -> None:
    script = [*to.write("a", "bad"), *to.write("a", "bad"), *to.write("a", "ok")]
    rig = one_task(tmp_path, script)
    log = QualityLog(":memory:")
    report = rig.orch_r(rig.router(GOOD_SMALL), quality=log).run(to.PID)
    assert report.status is ProjectStatus.DONE
    assert [(r.attempt, r.tier) for r in log.records()] == [
        (1, Tier.SMALL),
        (2, Tier.SMALL),
        (3, Tier.MEDIUM),
    ]
    assert len(rig.decisions()) == 1  # routed once, at the first attempt
    assert "orchestrator.escalated" in rig.event_types()


def test_resume_never_routes_again(tmp_path: Path) -> None:
    rig = one_task(tmp_path, [ProviderError("boom"), *to.write("a")])
    first = rig.orch_r(rig.router(GOOD_SMALL)).run(to.PID)
    assert first.status is ProjectStatus.PAUSED and first.reason is StopReason.PROVIDER
    assert rig.state("a").tier is Tier.SMALL
    # new data would now keep medium, but the saved choice is kept
    second = rig.orch_r(rig.router(WEAK_MEDIUM, GOOD_LARGE)).run(to.PID)
    assert second.status is ProjectStatus.DONE
    assert set(rig.models()) == {"fake-small"}
    assert len(rig.decisions()) == 1


def test_task_with_history_is_not_routed(tmp_path: Path) -> None:
    # attempt 1 without a router, then the run stops; the next run has a router
    rig = one_task(tmp_path, [*to.write("a", "bad"), ProviderError("x"), *to.write("a")])
    assert rig.orch_r(None).run(to.PID).reason is StopReason.PROVIDER
    rig.orch_r(rig.router(GOOD_SMALL)).run(to.PID)
    assert rig.decisions() == []
    assert set(rig.models()) == {"fake-medium"}


def test_unconfigured_requested_tier_stops_before_any_call(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        to.write("a"),
        to.make_plan(to.task_dict("a", tier="large")),
        tiers={"small": "fake-small", "medium": "fake-medium"},
    )
    report = rig.orch_r(rig.router(st("fake-small", 5, 5))).run(to.PID)
    assert report.status is ProjectStatus.PAUSED and report.reason is StopReason.CONFIG
    assert report.task_id == "a" and "large" in (report.detail or "")
    assert rig.provider.requests == []
    assert rig.decisions() == []
    assert rig.state("a").status is TaskStatus.PENDING


def test_budget_guard_still_bounds_a_pricier_route(tmp_path: Path) -> None:
    rig = one_task(tmp_path, to.write("a"))
    rig.rebuild(project_usd=1.0, daily_usd=1.0, per_call_usd=0.02)
    report = rig.orch_r(rig.router(WEAK_MEDIUM, GOOD_LARGE)).run(to.PID)
    assert rig.decisions()[0]["tier"] == "large"
    assert report.reason is StopReason.BUDGET
    assert rig.provider.requests == []  # refused before the provider was called


def test_quality_records_the_routed_model(tmp_path: Path) -> None:
    rig = one_task(tmp_path, to.write("a"))
    log = QualityLog(":memory:")
    rig.orch_r(rig.router(GOOD_SMALL), quality=log).run(to.PID)
    (rec,) = log.records()
    assert (rec.tier, rec.model) == (Tier.SMALL, "fake-small")


def test_each_task_routed_once_in_sequence(tmp_path: Path) -> None:
    plan = to.make_plan(to.task_dict("a"), to.task_dict("b", deps=["a"]))
    rig = Rig(tmp_path, [*to.write("a"), *to.write("b")], plan)
    rig.orch_r(rig.router(GOOD_SMALL)).run(to.PID)
    assert [d["task"] for d in rig.decisions()] == ["a", "b"]


def test_parallel_tasks_are_routed(tmp_path: Path) -> None:
    plan = to.make_plan(to.task_dict("a"), to.task_dict("b"))
    rig = Rig(tmp_path, [], plan)
    rig.provider = to.FakeProvider(default="done")
    rig.rebuild()
    # every worker step just finishes; acceptance checks then fail → retries until failed
    report = rig.orch_r(rig.router(GOOD_SMALL), max_parallel=2, max_attempts=1).run(to.PID)
    assert report.status is ProjectStatus.FAILED
    assert sorted(d["task"] for d in rig.decisions()) == ["a", "b"]
    assert {rig.state(t).tier for t in "ab"} == {Tier.SMALL}
