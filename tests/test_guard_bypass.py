"""Exhaustive tests that no path lets a model call escape the budget guard.

Every test runs with ``FakeProvider`` (no network). The invariants checked here:

1. A refused call never reaches the provider and leaves no reservation behind.
2. Every call that reached the provider is recorded in the ledger, at a cost that is
   never below what the tokens are worth (worst case when usage is missing or bad).
3. Misuse of the guard API (record/release without authorize, double record, ...)
   raises, and never loses money already spent.
4. Structurally, ``LLMGateway.complete`` is the only place a provider is called, and
   it always authorizes first; no other ``CallGuard`` exists in production code.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

import automa
from automa.budget import (
    BudgetError,
    BudgetExceeded,
    BudgetGuard,
    BudgetScope,
    CostLedger,
    usd_to_nusd,
)
from automa.config import AutomaConfig, BudgetLimits, Tier
from automa.llm import (
    CallContext,
    CallEstimate,
    ChatResponse,
    FakeProvider,
    LLMGateway,
    LLMResult,
    Message,
    ProviderError,
    Usage,
    compute_cost,
)

SRC = Path(automa.__file__).resolve().parent
CTX = CallContext(project_id="p1", task_id="t1")
# 1 token = $0.001 at these prices; "hi" with max_tokens=10 reserves 19 tokens.
CONFIG = AutomaConfig.model_validate(
    {
        "tiers": {"small": "m-small"},
        "prices": {"m-small": {"input_per_mtok": 1000.0, "output_per_mtok": 1000.0}},
        "limits": {"project_usd": 1.0, "daily_usd": 2.0, "per_call_usd": 0.5},
    }
)


class FixedClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def ledger() -> Iterator[CostLedger]:
    with CostLedger(":memory:", clock=FixedClock()) as led:
        yield led


@pytest.fixture
def guard(ledger: CostLedger) -> BudgetGuard:
    return BudgetGuard(CONFIG.limits, ledger)


def _gw(guard: object, *script: object) -> tuple[LLMGateway, FakeProvider]:
    fake = FakeProvider(script)  # type: ignore[arg-type]
    return LLMGateway(CONFIG, fake, guard=guard), fake  # type: ignore[arg-type]


def _resp(usage: Usage) -> ChatResponse:
    return ChatResponse(model="m-small", message=Message.assistant("x"), usage=usage)


def _call(gw: LLMGateway, *, max_tokens: int = 10, ctx: CallContext | None = CTX) -> LLMResult:
    return gw.complete([Message.user("hi")], tier=Tier.SMALL, max_tokens=max_tokens, context=ctx)


def _assert_untouched(fake: FakeProvider, guard: BudgetGuard) -> None:
    assert fake.requests == []
    assert guard.pending_nusd() == 0
    assert guard.ledger.entries() == []


# --- 1. construction: there is no way to build an unguarded gateway -------------


@pytest.mark.parametrize(
    "bogus",
    [
        None,
        object(),
        "guard",
        pytest.param("ledger", id="cost-ledger-is-not-a-guard"),
        pytest.param("no-release", id="missing-release"),
        pytest.param("attr-not-callable", id="authorize-not-callable"),
    ],
)
def test_gateway_rejects_anything_that_is_not_a_guard(bogus: object, ledger: CostLedger) -> None:
    class NoRelease:
        def authorize(self, estimate: CallEstimate) -> None: ...

        def record(self, estimate: CallEstimate, result: LLMResult) -> None: ...

    class NotCallable:
        authorize = None
        record = release = staticmethod(lambda *a: None)

    candidate = (
        {
            "ledger": ledger,
            "no-release": NoRelease(),
            "attr-not-callable": NotCallable(),
        }.get(bogus, bogus)
        if isinstance(bogus, str)
        else bogus
    )
    fake = FakeProvider(["x"])
    with pytest.raises(TypeError):
        LLMGateway(CONFIG, fake, guard=candidate)  # type: ignore[arg-type]
    assert fake.requests == []


def test_from_config_cannot_skip_guard() -> None:
    config = CONFIG.model_copy(update={"api_key": SecretStr("sk-test-not-real")})
    with pytest.raises(TypeError):
        LLMGateway.from_config(config, guard=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        LLMGateway.from_config(config)  # type: ignore[call-arg]


# --- 2. every refusal stops the call before the provider ------------------------


@pytest.mark.parametrize(
    "ctx",
    [None, CallContext(), CallContext(project_id=""), CallContext(project_id="   \t")],
    ids=["no-context", "no-project", "empty-project", "blank-project"],
)
def test_unattributed_calls_never_reach_provider(
    guard: BudgetGuard, ctx: CallContext | None
) -> None:
    gw, fake = _gw(guard, "x")
    with pytest.raises(BudgetError):
        _call(gw, ctx=ctx)
    _assert_untouched(fake, guard)


def test_per_call_cap_refusal_never_reaches_provider(guard: BudgetGuard) -> None:
    gw, fake = _gw(guard, "x")
    with pytest.raises(BudgetExceeded) as exc:
        _call(gw, max_tokens=600)  # 0.609 USD worst case > 0.5 per call
    assert exc.value.scope is BudgetScope.PER_CALL
    _assert_untouched(fake, guard)


@pytest.mark.parametrize("scope", [BudgetScope.PROJECT, BudgetScope.DAILY])
def test_latched_scope_refuses_even_tiny_calls(guard: BudgetGuard, scope: BudgetScope) -> None:
    # Fill the scope with real (recorded) spending through the gateway itself.
    big = _resp(Usage(input_tokens=1, output_tokens=499))  # 0.5 USD each
    projects = ["p1", "p1"] if scope is BudgetScope.PROJECT else ["a", "b", "c", "d"]
    gw, fake = _gw(guard, *[big] * len(projects))
    for pid in projects:
        _call(gw, max_tokens=480, ctx=CallContext(project_id=pid))
    assert guard.is_stopped(scope, "p1" if scope is BudgetScope.PROJECT else "2026-11-09")
    sent = len(fake.requests)
    fake.push("x")
    target = "p1" if scope is BudgetScope.PROJECT else "fresh-project"
    with pytest.raises(BudgetExceeded) as exc:
        _call(gw, max_tokens=1, ctx=CallContext(project_id=target))
    assert exc.value.latched and exc.value.scope is scope
    assert len(fake.requests) == sent and fake.remaining == 1
    assert guard.pending_nusd() == 0


def test_foreign_guard_exception_blocks_provider(ledger: CostLedger) -> None:
    class Exploding(BudgetGuard):
        def authorize(self, estimate: CallEstimate) -> None:
            raise RuntimeError("guard broke")

    guard = Exploding(CONFIG.limits, ledger)
    gw, fake = _gw(guard, "x")
    with pytest.raises(RuntimeError, match="guard broke"):
        _call(gw)
    _assert_untouched(fake, guard)


# --- 3. provider failures release, successes always hit the ledger --------------


@pytest.mark.parametrize(
    "failure",
    [ProviderError("down", retryable=True), RuntimeError("bug"), KeyboardInterrupt()],
    ids=["provider-error", "unexpected-error", "keyboard-interrupt"],
)
def test_provider_failure_releases_reservation(guard: BudgetGuard, failure: BaseException) -> None:
    gw, fake = _gw(guard, failure, "ok")
    with pytest.raises(type(failure)):
        _call(gw)
    assert guard.pending_nusd() == 0 and guard.ledger.entries() == []
    # The guard is not left wedged: the next call goes through and is recorded.
    _call(gw)
    assert len(guard.ledger.entries()) == 1 and guard.pending_nusd() == 0


def test_postprocessing_failure_still_records_worst_case(
    guard: BudgetGuard, monkeypatch: pytest.MonkeyPatch
) -> None:
    gw, fake = _gw(guard, "x")

    def broken(*args: object, **kwargs: object) -> float:
        raise ArithmeticError("pricing bug")

    monkeypatch.setattr(gw, "_actual_cost", broken)
    with pytest.raises(ArithmeticError):
        _call(gw)
    (entry,) = guard.ledger.entries()
    assert entry.cost_nusd == usd_to_nusd(0.019)  # the reserved worst case
    assert guard.pending_nusd() == 0


def test_output_beyond_max_tokens_is_fully_charged_and_latches(guard: BudgetGuard) -> None:
    # The provider ignores max_tokens and bills 1.2 USD for a call reserved at 0.019.
    gw, fake = _gw(guard, _resp(Usage(input_tokens=200, output_tokens=1000)), "x")
    result = _call(gw)
    assert result.cost_usd == pytest.approx(1.2)
    assert guard.ledger.spent_nusd(project_id="p1") == usd_to_nusd(1.2)
    assert guard.is_stopped(BudgetScope.PROJECT, "p1")
    with pytest.raises(BudgetExceeded):
        _call(gw)
    assert len(fake.requests) == 1


@pytest.mark.parametrize("reported", [None, 0.0, 0.000001])
def test_zero_usage_is_charged_worst_case_whatever_is_reported(
    guard: BudgetGuard, reported: float | None
) -> None:
    gw, _ = _gw(guard, _resp(Usage(reported_cost_usd=reported)))
    result = _call(gw)
    assert result.cost_usd == pytest.approx(0.019)
    assert guard.ledger.spent_nusd() == usd_to_nusd(0.019)


def test_missing_input_tokens_charged_at_estimated_maximum(guard: BudgetGuard) -> None:
    gw, _ = _gw(guard, _resp(Usage(input_tokens=0, output_tokens=5)))
    result = _call(gw)
    price = CONFIG.price_for("m-small")
    assert result.cost_usd == pytest.approx(
        compute_cost(Usage(input_tokens=9, output_tokens=5), price)
    )


def test_reported_cost_above_tokens_is_charged(guard: BudgetGuard) -> None:
    gw, _ = _gw(guard, _resp(Usage(input_tokens=1, output_tokens=1, reported_cost_usd=0.3)))
    assert _call(gw).cost_usd == pytest.approx(0.3)
    assert guard.ledger.spent_nusd() == usd_to_nusd(0.3)


def test_every_provider_call_has_exactly_one_ledger_entry(guard: BudgetGuard) -> None:
    script: list[object] = ["a", ProviderError("x"), "b", RuntimeError("y"), "c"]
    gw, fake = _gw(guard, *script)
    ok = 0
    for _ in script:
        try:
            _call(gw)
            ok += 1
        except (ProviderError, RuntimeError):
            pass
    assert len(fake.requests) == len(script)
    assert len(guard.ledger.entries()) == ok == 3
    assert guard.pending_nusd() == 0


# --- 4. direct misuse of the guard API ------------------------------------------


def _estimate(cost: float = 0.01, project: str = "p1") -> CallEstimate:
    return CallEstimate(
        tier=Tier.SMALL,
        model="m-small",
        max_input_tokens=5,
        max_output_tokens=5,
        max_cost_usd=cost,
        context=CallContext(project_id=project),
    )


def _result(cost: float = 0.01) -> LLMResult:
    return LLMResult(
        tier=Tier.SMALL, response=_resp(Usage(input_tokens=5, output_tokens=5)), cost_usd=cost
    )


def test_record_without_authorize_is_written_then_rejected(guard: BudgetGuard) -> None:
    with pytest.raises(BudgetError, match="never authorized"):
        guard.record(_estimate(), _result(0.7))
    # The money is still counted against the caps.
    assert guard.ledger.spent_nusd(project_id="p1") == usd_to_nusd(0.7)
    with pytest.raises(BudgetExceeded):
        guard.authorize(_estimate(0.4))


def test_double_record_is_rejected_but_counted(guard: BudgetGuard) -> None:
    est = _estimate()
    guard.authorize(est)
    guard.record(est, _result())
    with pytest.raises(BudgetError):
        guard.record(est, _result())
    assert len(guard.ledger.entries()) == 2


@pytest.mark.parametrize("order", ["release-only", "release-after-record", "double-release"])
def test_release_misuse_is_rejected(guard: BudgetGuard, order: str) -> None:
    est = _estimate()
    if order == "release-after-record":
        guard.authorize(est)
        guard.record(est, _result())
    elif order == "double-release":
        guard.authorize(est)
        guard.release(est)
    with pytest.raises(BudgetError, match="never authorized"):
        guard.release(est)


def test_record_after_release_is_rejected(guard: BudgetGuard) -> None:
    est = _estimate()
    guard.authorize(est)
    guard.release(est)
    with pytest.raises(BudgetError):
        guard.record(est, _result())
    assert len(guard.ledger.entries()) == 1


def test_record_with_other_estimate_keeps_original_reservation(guard: BudgetGuard) -> None:
    authorized = _estimate(0.2)
    guard.authorize(authorized)
    with pytest.raises(BudgetError):
        guard.record(_estimate(0.01), _result(0.01))
    # The 0.2 reservation is still held: the budget stays conservative.
    assert guard.pending_nusd() == usd_to_nusd(0.2)


def test_ledger_failure_during_record_keeps_reservation(guard: BudgetGuard) -> None:
    est = _estimate(0.3)
    guard.authorize(est)
    guard.ledger.close()
    with pytest.raises(Exception):  # noqa: B017 - any ledger failure must propagate
        guard.record(est, _result(0.3))
    assert guard.pending_nusd() == usd_to_nusd(0.3)


@pytest.mark.parametrize("cost", [-0.01, float("inf"), float("nan")])
def test_estimates_cannot_be_negative_or_infinite(cost: float) -> None:
    with pytest.raises(ValidationError):
        _estimate(cost)


def test_estimates_and_limits_are_immutable(guard: BudgetGuard) -> None:
    est = _estimate()
    with pytest.raises(ValidationError):
        est.max_cost_usd = 0.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        guard.limits.project_usd = 1e9  # type: ignore[misc]


def test_concurrent_reservations_count_toward_caps(guard: BudgetGuard) -> None:
    # Two in-flight 0.45 USD calls fit; a third would exceed the 1 USD project cap.
    first, second = _estimate(0.45), _estimate(0.45)
    guard.authorize(first)
    guard.authorize(second)
    with pytest.raises(BudgetExceeded) as exc:
        guard.authorize(_estimate(0.45))
    assert exc.value.scope is BudgetScope.PROJECT


def test_daily_cap_spans_projects(guard: BudgetGuard) -> None:
    for pid in ("a", "b", "c", "d"):
        est = _estimate(0.5, pid)
        guard.authorize(est)
        guard.record(est, _result(0.5))
    with pytest.raises(BudgetExceeded) as exc:
        guard.authorize(_estimate(0.01, "e"))
    assert exc.value.scope is BudgetScope.DAILY


# --- 5. multiple guards / restarts share the same ledger ------------------------


def test_second_guard_on_same_ledger_sees_spending(ledger: CostLedger) -> None:
    first = BudgetGuard(CONFIG.limits, ledger)
    est = _estimate(0.5)
    first.authorize(est)
    first.record(est, _result(0.95))
    second = BudgetGuard(CONFIG.limits, ledger)
    with pytest.raises(BudgetExceeded):
        second.authorize(_estimate(0.1))


def test_restart_keeps_caps_from_persisted_ledger(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    clock = FixedClock()
    limits = BudgetLimits(project_usd=1.0, daily_usd=2.0, per_call_usd=0.5)
    with CostLedger(path, clock=clock) as led:
        g = BudgetGuard(limits, led)
        gw, _ = _gw(g, _resp(Usage(input_tokens=1, output_tokens=999)))
        _call(gw)  # 1.0 USD: project cap reached
    with CostLedger(path, clock=clock) as led:
        g = BudgetGuard(limits, led)
        gw, fake = _gw(g, "x")
        with pytest.raises(BudgetExceeded):
            _call(gw, max_tokens=1)
        assert fake.requests == []


# --- 6. structural guarantees over the source tree ------------------------------


def _modules() -> list[tuple[Path, ast.Module]]:
    return [(p, ast.parse(p.read_text(encoding="utf-8"))) for p in sorted(SRC.rglob("*.py"))]


def test_only_the_gateway_calls_a_provider() -> None:
    callers: list[str] = []
    for path, tree in _modules():
        rel = path.relative_to(SRC).as_posix()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "complete"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "_provider"
            ):
                callers.append(rel)
    assert callers == ["llm/gateway.py"]


def test_providers_are_not_imported_outside_llm_package() -> None:
    offenders = []
    for path, tree in _modules():
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("llm/"):
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom):
                names = [a.name for a in node.names]
                mod = node.module or ""
                if mod.startswith(("automa.llm.openai_compat", "automa.llm.fake")):
                    offenders.append(f"{rel}: {mod}")
            if {"OpenAICompatProvider", "FakeProvider"} & set(names):
                offenders.append(f"{rel}: {names}")
    assert offenders == []


def test_gateway_complete_authorizes_before_calling_provider() -> None:
    tree = ast.parse((SRC / "llm" / "gateway.py").read_text(encoding="utf-8"))
    (fn,) = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "complete" and n.args.args[0].arg == "self"
    ]
    lines: dict[str, list[int]] = {"authorize": [], "provider": []}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Attribute) and owner.attr == "_guard":
                if node.func.attr == "authorize":
                    lines["authorize"].append(node.lineno)
            if isinstance(owner, ast.Attribute) and owner.attr == "_provider":
                lines["provider"].append(node.lineno)
    assert len(lines["authorize"]) == 1 and len(lines["provider"]) == 1
    assert lines["authorize"][0] < lines["provider"][0]


def test_budget_guard_is_the_only_production_guard() -> None:
    # No NullGuard / AllowAll style class may exist in production code.
    guards = []
    for path, tree in _modules():
        for node in ast.walk(tree):
            is_protocol = isinstance(node, ast.ClassDef) and any(
                isinstance(b, ast.Name) and b.id == "Protocol" for b in node.bases
            )
            if isinstance(node, ast.ClassDef) and not is_protocol:
                methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
                if "authorize" in methods:
                    guards.append(f"{path.relative_to(SRC).as_posix()}:{node.name}")
    assert guards == ["budget/guard.py:BudgetGuard"]
