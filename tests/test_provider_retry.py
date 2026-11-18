"""Retrying transient provider failures without ever loosening the budget (D91).

Found by the first real run: one gateway hiccup paused a whole project, and the
pause did not say what the gateway answered (it was HTTP 402, "insufficient
balance"). These tests pin down:

- only retryable failures are retried, at most ``max_retries`` times;
- every retry is a new call that the BudgetGuard must authorize first;
- a request that may have been billed (sent, then timed out) is charged at the
  worst case instead of being released;
- the gateway's own error message reaches the user, with secrets redacted.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import ValidationError

from automa.budget import BudgetExceeded, BudgetGuard, CostLedger, usd_to_nusd
from automa.config import AutomaConfig, ProviderConfig, Tier
from automa.llm import CallContext, FakeProvider, LLMGateway, Message, ProviderError
from automa.llm import gateway as gateway_mod
from automa.llm.openai_compat import OpenAICompatProvider
from automa.llm.types import ChatRequest

CTX = CallContext(project_id="p1", task_id="t1")
# 1 token = $0.001; "hi" with max_tokens=10 reserves 19 tokens = $0.019 worst case.
WORST = usd_to_nusd(0.019)


def config(**provider: object) -> AutomaConfig:
    return AutomaConfig.model_validate(
        {
            "tiers": {"small": "m-small"},
            "prices": {"m-small": {"input_per_mtok": 1000.0, "output_per_mtok": 1000.0}},
            "limits": {"project_usd": 1.0, "daily_usd": 2.0, "per_call_usd": 0.5},
            "provider": provider,
        }
    )


class FixedClock:
    def __call__(self) -> datetime:
        return datetime(2026, 11, 18, 12, 0, tzinfo=UTC)


@pytest.fixture
def ledger() -> Iterator[CostLedger]:
    with CostLedger(":memory:", clock=FixedClock()) as led:
        yield led


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waited: list[float] = []
    monkeypatch.setattr(gateway_mod, "_sleep", waited.append)
    return waited


def gw(ledger: CostLedger, *script: object, **provider: object):
    cfg = config(**provider)
    guard = BudgetGuard(cfg.limits, ledger)
    fake = FakeProvider(script)  # type: ignore[arg-type]
    return LLMGateway(cfg, fake, guard=guard), fake, guard


def call(g: LLMGateway):
    return g.complete([Message.user("hi")], tier=Tier.SMALL, max_tokens=10, context=CTX)


def transient(**kw: object) -> ProviderError:
    return ProviderError("provider returned HTTP 503", status=503, retryable=True, **kw)


# --- retry policy ---------------------------------------------------------------


def test_transient_failure_is_retried_then_succeeds(ledger: CostLedger, sleeps: list[float]):
    g, fake, guard = gw(ledger, transient(), transient(), "ok")
    assert call(g).text == "ok"
    assert len(fake.requests) == 3
    assert sleeps == [3.0, 10.0]  # default delays
    assert len(ledger.entries()) == 1 and guard.pending_nusd() == 0


def test_retries_are_bounded(ledger: CostLedger, sleeps: list[float]):
    g, fake, guard = gw(ledger, *[transient()] * 10)
    with pytest.raises(ProviderError, match="HTTP 503"):
        call(g)
    assert len(fake.requests) == 4  # 1 + max_retries (3)
    assert sleeps == [3.0, 10.0, 30.0]
    assert guard.pending_nusd() == 0 and ledger.entries() == []


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_non_retryable_failures_stop_immediately(
    ledger: CostLedger, sleeps: list[float], status: int
):
    # 402 (insufficient balance) must not be hammered: retrying cannot fix it.
    err = ProviderError(f"provider returned HTTP {status}", status=status)
    g, fake, guard = gw(ledger, err, "ok")
    with pytest.raises(ProviderError):
        call(g)
    assert len(fake.requests) == 1 and sleeps == []
    assert guard.pending_nusd() == 0


def test_retries_can_be_turned_off(ledger: CostLedger, sleeps: list[float]):
    g, fake, _ = gw(ledger, transient(), "ok", max_retries=0)
    with pytest.raises(ProviderError):
        call(g)
    assert len(fake.requests) == 1 and sleeps == []


def test_retry_after_hint_is_honoured_and_capped(ledger: CostLedger, sleeps: list[float]):
    g, _, _ = gw(ledger, transient(retry_after_s=20.0), transient(retry_after_s=9999.0), "ok")
    call(g)
    assert sleeps == [20.0, 120.0]


def test_last_delay_repeats(ledger: CostLedger, sleeps: list[float]):
    g, _, _ = gw(ledger, *[transient()] * 3, "ok", retry_delays_s=[1.0])
    call(g)
    assert sleeps == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "bad",
    [
        {"max_retries": -1},
        {"max_retries": 7},
        {"retry_delays_s": []},
        {"retry_delays_s": [-1.0]},
        {"retry_delays_s": [121.0]},
        {"timeout_s": 0},
        {"unknown": 1},
    ],
)
def test_invalid_provider_config_is_rejected(bad: dict[str, object]):
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(bad)


# --- budget: every retry is a fresh, guarded call -------------------------------


def test_each_retry_is_authorized_by_the_guard(ledger: CostLedger, sleeps: list[float]):
    g, fake, guard = gw(ledger, transient(), "ok")
    calls: list[str] = []
    real = guard.authorize

    def spy(estimate):  # type: ignore[no-untyped-def]
        calls.append("authorize")
        return real(estimate)

    guard.authorize = spy  # type: ignore[method-assign]
    call(g)
    assert calls == ["authorize", "authorize"]


def test_retry_is_blocked_when_the_cap_is_reached(ledger: CostLedger, sleeps: list[float]):
    # Timeouts may have been billed, so each one is charged at the worst case.
    # With a cap of two worst cases, the third attempt must be refused by the guard
    # before it reaches the provider.
    cfg = AutomaConfig.model_validate(
        {
            "tiers": {"small": "m-small"},
            "prices": {"m-small": {"input_per_mtok": 1000.0, "output_per_mtok": 1000.0}},
            "limits": {"project_usd": 0.04, "daily_usd": 2.0, "per_call_usd": 0.02},
        }
    )
    guard = BudgetGuard(cfg.limits, ledger)
    timeout = ProviderError("request timed out", retryable=True, maybe_billed=True)
    fake = FakeProvider([timeout, timeout, "ok", "ok"])
    g = LLMGateway(cfg, fake, guard=guard)
    with pytest.raises(BudgetExceeded):
        call(g)
    assert len(fake.requests) == 2
    assert ledger.spent_nusd(project_id="p1") == 2 * WORST
    assert guard.pending_nusd() == 0


def test_maybe_billed_failure_is_charged_at_worst_case(ledger: CostLedger, sleeps: list[float]):
    timeout = ProviderError("request timed out", retryable=True, maybe_billed=True)
    g, _, guard = gw(ledger, timeout, "ok")
    call(g)
    entries = ledger.entries()
    assert len(entries) == 2
    assert entries[0].cost_nusd == WORST  # the unanswered attempt
    assert guard.pending_nusd() == 0


def test_unbilled_failure_is_released(ledger: CostLedger, sleeps: list[float]):
    g, _, guard = gw(ledger, transient(), "ok")
    call(g)
    assert len(ledger.entries()) == 1  # only the successful call
    assert guard.pending_nusd() == 0


def test_maybe_billed_non_retryable_is_still_charged(ledger: CostLedger, sleeps: list[float]):
    err = ProviderError("cut off", maybe_billed=True)
    g, _, guard = gw(ledger, err)
    with pytest.raises(ProviderError):
        call(g)
    (entry,) = ledger.entries()
    assert entry.cost_nusd == WORST and guard.pending_nusd() == 0


# --- HTTP provider: classification and error messages ---------------------------

KEY = "1xai-SECRETsecretSECRET123456"
REQ = ChatRequest(model="m", messages=(Message.user("hi"),), max_tokens=5)


def provider(handler) -> OpenAICompatProvider:  # type: ignore[no-untyped-def]
    return OpenAICompatProvider("https://gw.test/v1", KEY, transport=httpx.MockTransport(handler))


def raised(handler) -> ProviderError:  # type: ignore[no-untyped-def]
    with pytest.raises(ProviderError) as exc:
        provider(handler).complete(REQ)
    return exc.value


def test_402_message_reaches_the_user():
    e = raised(
        lambda r: httpx.Response(
            402, json={"error": {"message": "موجودی کافی نیست", "code": "insufficient_balance"}}
        )
    )
    assert e.status == 402 and not e.retryable and not e.maybe_billed
    assert str(e) == "provider returned HTTP 402: موجودی کافی نیست"


def test_plain_text_and_string_errors():
    assert "overloaded" in str(raised(lambda r: httpx.Response(503, text="overloaded")))
    assert "bad model" in str(raised(lambda r: httpx.Response(400, json={"error": "bad model"})))
    assert "nope" in str(raised(lambda r: httpx.Response(422, json={"detail": "nope"})))
    assert str(raised(lambda r: httpx.Response(500))) == "provider returned HTTP 500"


def test_error_message_never_contains_the_key():
    e = raised(
        lambda r: httpx.Response(
            401,
            json={"error": {"message": f"invalid key {KEY} (Bearer {KEY}) sk-abcdefgh12345"}},
        )
    )
    text = str(e)
    assert KEY not in text and "SECRET" not in text and "sk-abcdefgh12345" not in text
    assert "[redacted]" in text


def test_error_message_is_single_line_and_bounded():
    e = raised(
        lambda r: httpx.Response(500, json={"error": {"message": "a\nb\r\x1b[31m" + "x" * 5000}})
    )
    text = str(e)
    assert "\n" not in text and "\x1b" not in text
    assert len(text) < 400


def test_retry_after_header_is_parsed():
    e = raised(lambda r: httpx.Response(429, headers={"Retry-After": "7"}))
    assert e.retryable and e.retry_after_s == 7.0
    e = raised(lambda r: httpx.Response(429, headers={"Retry-After": "soon"}))
    assert e.retry_after_s is None


def test_connect_failure_is_retryable_and_not_billed():
    def handler(r):  # type: ignore[no-untyped-def]
        raise httpx.ConnectError("refused", request=r)

    e = raised(handler)
    assert e.retryable and not e.maybe_billed


def test_read_timeout_is_retryable_and_maybe_billed():
    def handler(r):  # type: ignore[no-untyped-def]
        raise httpx.ReadTimeout("slow", request=r)

    e = raised(handler)
    assert e.retryable and e.maybe_billed


def test_error_object_with_200_includes_message():
    e = raised(lambda r: httpx.Response(200, json={"error": {"message": "model overloaded"}}))
    assert "model overloaded" in str(e)


def test_timeout_comes_from_config():
    cfg = config(timeout_s=42)
    assert cfg.provider.timeout_s == 42


# --- models that reject `temperature` (D111) --------------------------------------


def temperature_rejected() -> ProviderError:
    # Verbatim from a live gpt-5.5 call on 1xAi.
    return ProviderError(
        "provider returned HTTP 400: `temperature` is deprecated for this model.", status=400
    )


def test_rejected_temperature_is_dropped_and_remembered(ledger: CostLedger, sleeps: list[float]):
    g, fake, guard = gw(ledger, temperature_rejected(), "ok", "again")
    assert call(g).text == "ok"
    assert fake.requests[0].temperature is not None
    assert fake.requests[1].temperature is None
    assert sleeps == []  # not a transient failure: no wait
    assert call(g).text == "again"
    assert fake.requests[2].temperature is None  # remembered for this model
    assert len(ledger.entries()) == 2 and guard.pending_nusd() == 0


def test_temperature_retry_happens_once(ledger: CostLedger, sleeps: list[float]):
    g, fake, _ = gw(ledger, temperature_rejected(), temperature_rejected(), "ok")
    with pytest.raises(ProviderError, match="temperature"):
        call(g)
    assert len(fake.requests) == 2


def test_temperature_retry_is_guarded(ledger: CostLedger, sleeps: list[float]):
    g, fake, guard = gw(ledger, temperature_rejected(), "ok")
    calls: list[str] = []
    real = guard.authorize

    def spy(estimate):  # type: ignore[no-untyped-def]
        calls.append("authorize")
        return real(estimate)

    guard.authorize = spy  # type: ignore[method-assign]
    call(g)
    assert calls == ["authorize", "authorize"]


def test_other_400s_are_not_retried(ledger: CostLedger, sleeps: list[float]):
    err = ProviderError("provider returned HTTP 400: bad tools", status=400)
    g, fake, _ = gw(ledger, err, "ok")
    with pytest.raises(ProviderError):
        call(g)
    assert len(fake.requests) == 1


def test_wire_omits_temperature_when_none():
    from automa.llm.openai_compat import request_to_wire

    req = ChatRequest(model="m", messages=(Message.user("hi"),), max_tokens=5, temperature=None)
    assert "temperature" not in request_to_wire(req)
    assert request_to_wire(req.model_copy(update={"temperature": 0.2}))["temperature"] == 0.2


# --- reply did not fit in max_tokens (D117) ----------------------------------------


def out_of_room() -> ProviderError:
    # Verbatim from a live Claude call on 1xAi (the provider marks it maybe_billed).
    return ProviderError(
        "provider returned HTTP 400: Could not finish the message because max_tokens or "
        "model output limit was reached. Please try again with higher max_tokens.",
        status=400,
        maybe_billed=True,
    )


def test_output_limit_retries_with_a_larger_cap(ledger: CostLedger, sleeps: list[float]):
    g, fake, guard = gw(ledger, out_of_room(), "ok")
    assert call(g).text == "ok"
    assert [r.max_tokens for r in fake.requests] == [10, 20]
    assert sleeps == []
    # The failed attempt may have been billed: charged at its worst case.
    entries = ledger.entries()
    assert len(entries) == 2 and entries[0].cost_nusd == WORST
    assert guard.pending_nusd() == 0


def test_output_limit_grows_at_most_twice(ledger: CostLedger, sleeps: list[float]):
    g, fake, _ = gw(ledger, out_of_room(), out_of_room(), out_of_room(), "ok")
    with pytest.raises(ProviderError, match="max_tokens"):
        call(g)
    assert [r.max_tokens for r in fake.requests] == [10, 20, 40]


def test_grown_request_is_authorized_at_its_new_worst_case(ledger: CostLedger, sleeps: list[float]):
    g, _, guard = gw(ledger, out_of_room(), "ok")
    seen: list[int] = []
    real = guard.authorize

    def spy(estimate):  # type: ignore[no-untyped-def]
        seen.append(estimate.max_output_tokens)
        return real(estimate)

    guard.authorize = spy  # type: ignore[method-assign]
    call(g)
    assert seen == [10, 20]


def test_provider_marks_output_limit_as_maybe_billed():
    msg = (
        "Could not finish the message because max_tokens or model output limit was "
        "reached. Please try again with higher max_tokens."
    )
    e = raised(lambda r: httpx.Response(400, json={"error": {"message": msg}}))
    assert e.maybe_billed and not e.retryable
