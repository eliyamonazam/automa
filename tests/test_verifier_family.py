"""Cross-family verifier (D96): ``[verifier]`` config, gateway model override,
evaluator choice, verdict fields and the quality log column.

Local only: FakeProvider behind a real LLMGateway + BudgetGuard + CostLedger. No
network, no real model calls.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

import test_orchestrator as to
import test_verifier as tv
from automa.agents import VerifierAgent
from automa.agents.verifier import EvaluatorReason, choose_evaluator
from automa.budget import BudgetExceeded, BudgetGuard, CostLedger
from automa.config import AutomaConfig, ConfigError, Tier, load_config
from automa.llm import FakeProvider, LLMGateway, Message
from automa.llm.gateway import CallContext
from automa.orchestrator import Orchestrator
from automa.plan.models import Verdict, VerdictOutcome
from automa.quality import QualityLog, QualityRecord, sync_project

CTX = CallContext(project_id="p1")
PRICE = {"input_per_mtok": 1.0, "output_per_mtok": 2.0}


def model(family: str, caps: tuple[str, ...] = ("text", "code"), **price: float) -> dict:
    return {"family": family, "capabilities": list(caps), **{**PRICE, **price}}


def cfg_data(**verifier: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "tiers": {"small": "an-small", "medium": "an-medium", "large": "an-large"},
        "models": {
            "an-small": model("anthropic"),
            "an-medium": model("anthropic"),
            "an-large": model("anthropic"),
            "oa-judge": model("openai"),
            "gg-judge": model("google"),
            "an-judge": model("anthropic"),
            "img-only": model("openai", ("image_out",)),
        },
        "prices": {"priced-only": PRICE},
    }
    if verifier:
        data["verifier"] = verifier
    return data


def cfg(**verifier: Any) -> AutomaConfig:
    return AutomaConfig.model_validate(cfg_data(**verifier))


ON = {"prefer_other_family": True, "models": ["oa-judge", "gg-judge"]}


# --- config ----------------------------------------------------------------------


def test_verifier_config_defaults_off() -> None:
    c = cfg()
    assert c.verifier.prefer_other_family is False and c.verifier.models == ()


def test_verifier_config_valid() -> None:
    c = cfg(**ON)
    assert c.verifier.models == ("oa-judge", "gg-judge")


def test_models_without_prefer_is_allowed() -> None:
    assert cfg(models=["oa-judge"]).verifier.prefer_other_family is False


@pytest.mark.parametrize(
    ("verifier", "fragment"),
    [
        ({"prefer_other_family": True}, "non-empty models"),
        ({"prefer_other_family": True, "models": []}, "non-empty models"),
        ({"models": ["nope"]}, "not in the [models] catalog"),
        ({"models": ["priced-only"]}, "not in the [models] catalog"),
        ({"models": ["img-only"]}, "'text' capability"),
        ({"models": "oa-judge"}, "not a string"),
        ({"models": ["oa-judge", "oa-judge"]}, "must not repeat"),
        ({"models": ["bad id"]}, "invalid model ids"),
        ({"models": [1]}, "invalid model ids"),
        ({"models": [f"m{i}" for i in range(21)]}, "at most 20"),
        ({"prefer_other_family": "yes", "models": ["oa-judge"]}, "prefer_other_family"),
        ({"prefer_other_family": 1, "models": ["oa-judge"]}, "prefer_other_family"),
        ({"unknown": True}, "unknown"),
    ],
)
def test_verifier_config_rejects(verifier: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ValueError, match="verifier") as e:
        AutomaConfig.model_validate(cfg_data(**verifier))
    assert fragment in str(e.value)


def test_verifier_config_from_toml(tmp_path: Path) -> None:
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
[models."oa-judge"]
family = "openai"
capabilities = ["text"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[verifier]
prefer_other_family = true
models = ["oa-judge"]
""",
        encoding="utf-8",
    )
    c = load_config(toml, env_file=None, environ={})
    assert c.verifier.prefer_other_family and c.verifier.models == ("oa-judge",)
    toml.write_text(
        toml.read_text(encoding="utf-8").replace('["oa-judge"]', '["missing"]'),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="not in the"):
        load_config(toml, env_file=None, environ={})


def test_example_config_mentions_verifier_section() -> None:
    text = (Path(__file__).parent.parent / "automa.toml.example").read_text(encoding="utf-8")
    assert "[verifier]" in text and "prefer_other_family" in text


# --- gateway model override --------------------------------------------------------


class Gw:
    def __init__(self, tmp_path: Path, c: AutomaConfig, script: list[Any], **limits: float):
        if limits:
            c = c.model_copy(update={"limits": c.limits.model_copy(update=limits)})
        self.cfg = c
        self.provider = FakeProvider(script)
        self.ledger = CostLedger(tmp_path / "ledger.sqlite3")
        self.guard = BudgetGuard(c.limits, self.ledger)
        self.gateway = LLMGateway(c, self.provider, guard=self.guard)


def test_gateway_default_uses_tier_model(tmp_path: Path) -> None:
    g = Gw(tmp_path, cfg(), ["hi"])
    g.gateway.complete([Message.user("x")], tier=Tier.MEDIUM, max_tokens=10, context=CTX)
    assert g.provider.requests[0].model == "an-medium"


def test_gateway_model_override_is_guarded_and_recorded(tmp_path: Path) -> None:
    g = Gw(tmp_path, cfg(), ["hi"])
    r = g.gateway.complete(
        [Message.user("x")], tier=Tier.MEDIUM, max_tokens=10, model="oa-judge", context=CTX
    )
    assert g.provider.requests[0].model == "oa-judge"
    (entry,) = g.ledger.entries()
    assert entry.model == "oa-judge" and entry.tier == "medium"
    assert r.cost_usd > 0


def test_gateway_override_uses_override_price(tmp_path: Path) -> None:
    c = AutomaConfig.model_validate(
        {
            **cfg_data(),
            "models": {
                **cfg_data()["models"],
                "pricey": model("openai", input_per_mtok=1_000_000.0, output_per_mtok=1e6),
            },
        }
    )
    g = Gw(tmp_path, c, ["hi"])
    # The per-call cap applies to the override's (much higher) price: refused before sending.
    with pytest.raises(BudgetExceeded):
        g.gateway.complete(
            [Message.user("x")], tier=Tier.SMALL, max_tokens=1000, model="pricey", context=CTX
        )
    assert g.provider.requests == []
    assert g.ledger.entries() == []


def test_gateway_override_priced_only_model_allowed(tmp_path: Path) -> None:
    g = Gw(tmp_path, cfg(), ["hi"])
    g.gateway.complete(
        [Message.user("x")], tier=Tier.SMALL, max_tokens=10, model="priced-only", context=CTX
    )
    assert g.provider.requests[0].model == "priced-only"


@pytest.mark.parametrize("bad", ["unpriced-model", "img-only", ""])
def test_gateway_override_rejects_before_any_call(tmp_path: Path, bad: str) -> None:
    g = Gw(tmp_path, cfg(), ["hi"])
    with pytest.raises(ConfigError):
        g.gateway.complete(
            [Message.user("x")], tier=Tier.SMALL, max_tokens=10, model=bad, context=CTX
        )
    assert g.provider.requests == []
    assert g.ledger.entries() == []


def test_gateway_override_non_string_rejected(tmp_path: Path) -> None:
    g = Gw(tmp_path, cfg(), ["hi"])
    with pytest.raises(ConfigError):
        g.gateway.complete(
            [Message.user("x")],
            tier=Tier.SMALL,
            max_tokens=10,
            model=123,  # type: ignore[arg-type]
            context=CTX,
        )
    assert g.provider.requests == []


# --- choose_evaluator ----------------------------------------------------------------


def test_choice_disabled() -> None:
    ch = choose_evaluator(cfg(models=["oa-judge"]), "an-medium")
    assert ch.model is None and ch.reason is EvaluatorReason.DISABLED and not ch.fallback
    assert ch.worker_family == "anthropic"


def test_choice_first_other_family_in_order() -> None:
    ch = choose_evaluator(cfg(**ON), "an-medium")
    assert ch.model == "oa-judge" and ch.reason is EvaluatorReason.OTHER_FAMILY
    ch = choose_evaluator(cfg(prefer_other_family=True, models=["gg-judge", "oa-judge"]), "an-x")
    # an-x is not catalogued: family unknown, so no cross-family claim is made
    assert ch.model is None and ch.reason is EvaluatorReason.WORKER_FAMILY_UNKNOWN


def test_choice_skips_same_family() -> None:
    c = cfg(prefer_other_family=True, models=["an-judge", "gg-judge"])
    ch = choose_evaluator(c, "an-medium")
    assert ch.model == "gg-judge"


def test_choice_worker_is_other_family() -> None:
    # work done by openai: the first listed model is openai too, so google is chosen
    ch = choose_evaluator(cfg(**ON), "oa-judge")
    assert ch.model == "gg-judge" and ch.worker_family == "openai"


def test_choice_no_other_family() -> None:
    ch = choose_evaluator(cfg(prefer_other_family=True, models=["an-judge"]), "an-medium")
    assert ch.model is None and ch.reason is EvaluatorReason.NO_OTHER_FAMILY and ch.fallback


def test_choice_worker_priced_only_is_unknown() -> None:
    ch = choose_evaluator(cfg(**ON), "priced-only")
    assert ch.reason is EvaluatorReason.WORKER_FAMILY_UNKNOWN and ch.fallback
    assert ch.worker_family is None


# --- VerifierAgent -----------------------------------------------------------------


class VRig(tv.Rig):
    def __init__(self, tmp_path: Path, script: list[Any], c: AutomaConfig, **limits: float):
        super().__init__(tmp_path, script)
        if limits:
            c = c.model_copy(update={"limits": c.limits.model_copy(update=limits)})
        self.cfg = c
        self.guard = BudgetGuard(c.limits, self.ledger)
        self.gateway = LLMGateway(c, self.provider, guard=self.guard)


def result(**kw: Any) -> Any:
    return tv.make_result(**{"model": "an-medium", **kw})


def ok_rig(tmp_path: Path, c: AutomaConfig, *script: Any) -> VRig:
    rig = VRig(tmp_path, list(script) or [tv.judgment((1, True, ""))], c)
    rig.write("hello.txt", "hi")
    return rig


def evaluator_events(rig: VRig) -> list[dict[str, Any]]:
    return [e.data for e in rig.events.events if e.type == "verifier.evaluator"]


def test_agent_uses_other_family_model(tmp_path: Path) -> None:
    rig = ok_rig(tmp_path, cfg(**ON))
    v = rig.verify(result=result())
    assert v.passed and v.model_evaluated
    assert rig.provider.requests[0].model == "oa-judge"
    assert (v.evaluator_model, v.evaluator_family) == ("oa-judge", "openai")
    (ev,) = evaluator_events(rig)
    assert ev == {
        "task": "t1",
        "attempt": 1,
        "model": "oa-judge",
        "family": "openai",
        "worker_model": "an-medium",
        "worker_family": "anthropic",
        "reason": "other_family",
        "fallback": False,
    }
    (entry,) = rig.ledger.entries()
    assert entry.model == "oa-judge" and entry.task_id == "t1" and entry.tier == "medium"


def test_agent_disabled_behaves_as_before_but_records_evaluator(tmp_path: Path) -> None:
    rig = ok_rig(tmp_path, cfg())
    v = rig.verify(result=result())
    assert rig.provider.requests[0].model == "an-medium"
    assert (v.evaluator_model, v.evaluator_family) == ("an-medium", "anthropic")
    assert evaluator_events(rig) == []


def test_agent_fallback_no_other_family(tmp_path: Path) -> None:
    rig = ok_rig(tmp_path, cfg(prefer_other_family=True, models=["an-judge"]))
    v = rig.verify(result=result())
    assert rig.provider.requests[0].model == "an-medium"  # tier model, as before
    assert v.evaluator_model == "an-medium"
    (ev,) = evaluator_events(rig)
    assert ev["reason"] == "no_other_family" and ev["fallback"] is True


def test_agent_fallback_unknown_worker_family(tmp_path: Path) -> None:
    rig = ok_rig(tmp_path, cfg(**ON))
    v = rig.verify(result=result(model="priced-only"))
    assert rig.provider.requests[0].model == "an-medium"
    assert v.evaluator_family == "anthropic"
    (ev,) = evaluator_events(rig)
    assert ev["reason"] == "worker_family_unknown" and ev["worker_family"] is None


def test_agent_uncatalogued_tier_model_has_no_family(tmp_path: Path) -> None:
    rig = VRig(tmp_path, [tv.judgment((1, True, ""))], tv.config())
    rig.write("hello.txt", "hi")
    v = rig.verify()
    assert (v.evaluator_model, v.evaluator_family) == ("fake-medium", None)


def test_agent_tier_argument_still_used_for_fallback(tmp_path: Path) -> None:
    rig = ok_rig(tmp_path, cfg(prefer_other_family=True, models=["an-judge"]))
    v = rig.verify(result=result(), tier=Tier.LARGE)
    assert rig.provider.requests[0].model == "an-large" and v.evaluator_model == "an-large"


def test_agent_retries_stay_on_the_chosen_model(tmp_path: Path) -> None:
    rig = ok_rig(tmp_path, cfg(**ON), "not json", tv.judgment((1, True, "")))
    v = rig.verify(result=result())
    assert [r.model for r in rig.provider.requests] == ["oa-judge", "oa-judge"]
    assert v.evaluator_model == "oa-judge"
    assert len(evaluator_events(rig)) == 1


def test_agent_no_model_evaluation_records_no_evaluator(tmp_path: Path) -> None:
    rig = VRig(tmp_path, [], cfg(**ON))
    v = rig.verify(result=result())  # hello.txt missing: the check fails
    assert not v.passed and not v.model_evaluated
    assert v.evaluator_model is None and v.evaluator_family is None
    assert rig.provider.requests == [] and evaluator_events(rig) == []


def test_agent_failed_result_records_no_evaluator(tmp_path: Path) -> None:
    rig = VRig(tmp_path, [], cfg(**ON))
    v = rig.verify(result=result(status="failed", error="boom"))
    assert v.evaluator_model is None and rig.provider.requests == []


def test_agent_budget_refusal_on_other_family_propagates(tmp_path: Path) -> None:
    c = AutomaConfig.model_validate(
        {
            **cfg_data(prefer_other_family=True, models=["pricey"]),
            "models": {
                **cfg_data()["models"],
                "pricey": model("openai", input_per_mtok=1e6, output_per_mtok=1e6),
            },
        }
    )
    rig = ok_rig(tmp_path, c)
    with pytest.raises(BudgetExceeded):
        rig.verify(result=result())
    # never silently switched to a cheaper model, never sent
    assert rig.provider.requests == [] and rig.ledger.entries() == []
    assert "verifier.verdict" not in rig.types()


def test_agent_project_cap_counts_cross_family_calls(tmp_path: Path) -> None:
    rig = VRig(tmp_path, [tv.judgment((1, True, ""))] * 3, cfg(**ON), project_usd=0.01)
    rig.write("hello.txt", "hi")
    rig.verify(result=result())
    spent = sum(e.cost_usd for e in rig.ledger.entries())
    assert spent > 0
    assert all(e.model == "oa-judge" for e in rig.ledger.entries())


# --- Verdict fields ------------------------------------------------------------------


def verdict(**kw: Any) -> Verdict:
    data: dict[str, Any] = {
        "task_id": "t1",
        "attempt": 1,
        "outcome": "pass",
        "model_evaluated": True,
        "created_at": tv.T0,
    }
    data.update(kw)
    return Verdict.model_validate(data)


def test_old_verdict_without_evaluator_still_loads() -> None:
    v = verdict()
    assert v.evaluator_model is None and v.evaluator_family is None
    assert Verdict.model_validate_json(v.model_dump_json()) == v


@pytest.mark.parametrize(
    "kw",
    [
        {"model_evaluated": False, "evaluator_model": "m"},
        {"model_evaluated": False, "evaluator_family": "openai"},
        {"evaluator_family": "openai"},
        {"evaluator_model": "m", "evaluator_family": "OpenAI"},
        {"evaluator_model": ""},
        {"evaluator_model": "x" * 201},
    ],
)
def test_verdict_evaluator_rules(kw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        verdict(**kw)


def test_verdict_with_evaluator_round_trips() -> None:
    v = verdict(evaluator_model="oa-judge", evaluator_family="openai")
    assert Verdict.model_validate_json(v.model_dump_json()) == v


# --- quality log ---------------------------------------------------------------------


def qrec(**kw: Any) -> QualityRecord:
    data: dict[str, Any] = {
        "project_id": "p1",
        "task_id": "a",
        "attempt": 1,
        "role": "writer",
        "tier": Tier.MEDIUM,
        "model": "an-medium",
        "result_status": "completed",
        "outcome": VerdictOutcome.PASS,
        "model_evaluated": True,
        "cost_nusd": 1000,
        "judged_at": tv.T0,
    }
    data.update(kw)
    return QualityRecord(**data)


def test_quality_log_stores_verifier_family() -> None:
    log = QualityLog(":memory:")
    r = qrec(verifier_model="oa-judge", verifier_family="openai")
    assert log.record(r) is True
    assert log.records() == [r]
    assert log.record(r) is False
    with pytest.raises(Exception, match="different data"):
        log.record(qrec(verifier_model="gg-judge", verifier_family="google"))


def test_quality_record_rejects_bad_family() -> None:
    with pytest.raises(ValueError):
        qrec(verifier_family="Not A Family")


def _v1_log(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
CREATE TABLE attempts (
    project_id      TEXT    NOT NULL,
    task_id         TEXT    NOT NULL,
    attempt         INTEGER NOT NULL CHECK (attempt >= 1),
    role            TEXT    NOT NULL,
    tier            TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    result_status   TEXT    NOT NULL,
    outcome         TEXT    NOT NULL CHECK (outcome IN ('pass', 'fail')),
    model_evaluated INTEGER NOT NULL CHECK (model_evaluated IN (0, 1)),
    cost_nusd       INTEGER NOT NULL CHECK (cost_nusd >= 0),
    judged_at       TEXT    NOT NULL,
    PRIMARY KEY (project_id, task_id, attempt)
);
INSERT INTO attempts VALUES
  ('p1', 'a', 1, 'writer', 'medium', 'an-medium', 'completed', 'pass', 1, 1000,
   '2026-11-10T12:00:00+00:00');
PRAGMA user_version = 1;
"""
    )
    conn.commit()
    conn.close()


def test_v1_log_read_only_is_readable_and_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "quality.sqlite3"
    _v1_log(path)
    before = path.read_bytes()
    with QualityLog(path, read_only=True) as log:
        (r,) = log.records()
        assert r.verifier_model is None and r.verifier_family is None
        assert log.stats()[0].attempts == 1
    assert path.read_bytes() == before


def test_v1_log_is_migrated_on_write_open(tmp_path: Path) -> None:
    path = tmp_path / "quality.sqlite3"
    _v1_log(path)
    with QualityLog(path) as log:
        (old,) = log.records()
        assert old == qrec()
        assert log.record(qrec()) is False  # the old row is still the same row
        new = qrec(attempt=2, verifier_model="oa-judge", verifier_family="openai")
        assert log.record(new) is True
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone() == (2,)
    conn.close()
    with QualityLog(path, read_only=True) as log:
        assert [r.verifier_family for r in log.records()] == [None, "openai"]


def test_log_with_unexpected_columns_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "quality.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE attempts (x TEXT); PRAGMA user_version = 1;")
    conn.commit()
    conn.close()
    for ro in (True, False):
        with pytest.raises(Exception, match="unexpected columns"):
            QualityLog(path, read_only=ro)


# --- orchestrator end to end -----------------------------------------------------------


def test_orchestrator_records_verifier_family_in_quality_log(tmp_path: Path) -> None:
    criteria = [
        {"description": "a.txt says ok", "check": {"type": "file_contains", "path": "a.txt"}},
        {"description": "reads well"},
    ]
    criteria[0]["check"]["text"] = "ok"
    plan = to.make_plan(to.task_dict("a", acceptance_criteria=criteria))
    script = [*to.write("a"), tv.judgment((1, True, ""))]
    rig = to.Rig(tmp_path, script, plan)
    c = cfg(**ON)
    events = rig.store.events(to.PID)
    guard = BudgetGuard(c.limits, rig.ledger, events=events)
    gateway = LLMGateway(c, rig.provider, guard=guard)
    from automa.agents import WorkerAgent

    log = QualityLog(":memory:")
    report = Orchestrator(
        rig.store,
        WorkerAgent(gateway, events=events, clock=to.clock),
        VerifierAgent(gateway, events=events, clock=to.clock),
        rig.roles,
        c,
        tools=rig.tools,
        clock=to.clock,
        quality=log,
    ).run(to.PID)
    assert report.status.value == "done"
    models = [r.model for r in rig.provider.requests]
    assert models[-1] == "oa-judge" and all(m.startswith("an-") for m in models[:-1])
    (row,) = log.records()
    assert (row.model, row.verifier_model, row.verifier_family) == (
        models[0],
        "oa-judge",
        "openai",
    )
    assert sync_project(log, rig.store, to.PID) == 0  # same data as the stored verdict
    assert "verifier.evaluator" in rig.event_types()
