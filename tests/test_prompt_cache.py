"""Prompt caching (D167): markers on the wire, cached-token usage, conservative cost.

FakeProvider and httpx.MockTransport only; no network, no real calls.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime

import httpx
import pytest

from automa.budget import BudgetGuard, CostLedger
from automa.config import AutomaConfig, ModelPrice, ProviderConfig, Tier
from automa.llm import CallContext, FakeProvider, LLMGateway, Message, ProviderError
from automa.llm.gateway import CACHE_WRITE_FACTOR, compute_cost, estimate_max_cost
from automa.llm.openai_compat import OpenAICompatProvider, request_to_wire, response_from_wire
from automa.llm.types import ChatRequest, ChatResponse, Usage

CTX = CallContext(project_id="p1", task_id="t1")


def config(model: str = "claude-sonnet-5-5", **provider: object) -> AutomaConfig:
    return AutomaConfig.model_validate(
        {
            "tiers": {"small": model},
            "prices": {
                model: {
                    "input_per_mtok": 3.0,
                    "output_per_mtok": 15.0,
                    "cache_read_per_mtok": 0.3,
                }
            },
            "limits": {"project_usd": 5.0, "daily_usd": 5.0, "per_call_usd": 1.0},
            "provider": provider,
        }
    )


@pytest.fixture
def ledger() -> Iterator[CostLedger]:
    clock = lambda: datetime(2026, 11, 22, 12, 0, tzinfo=UTC)  # noqa: E731
    with CostLedger(":memory:", clock=clock) as led:
        yield led


def reply(text: str = "ok", **usage: int) -> ChatResponse:
    return ChatResponse(
        model="x",
        message=Message.assistant(text),
        usage=Usage(**usage),  # type: ignore[arg-type]
    )


def msgs() -> list[Message]:
    return [Message.system("rules " * 50), Message.user("task"), Message.assistant("step")]


def test_provider_config_caches() -> None:
    assert not ProviderConfig().caches("claude-sonnet-5-5")
    on = ProviderConfig(prompt_cache=True)
    assert on.caches("Claude-Opus-5-5") and not on.caches("gpt-5.5")
    assert ProviderConfig(prompt_cache=True, prompt_cache_models=("gpt",)).caches("gpt-5.5")


def test_wire_markers_on_system_and_last_user_or_tool() -> None:
    req = ChatRequest(
        model="m",
        messages=(
            Message.system("sys"),
            Message.user("task"),
            Message.assistant("a"),
            Message.tool("c1", "result"),
        ),
        max_tokens=10,
        cache=True,
    )
    body = request_to_wire(req)
    m = body["messages"]
    assert m[0]["content"] == [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
    ]
    assert m[1]["content"] == "task"  # only the last user/tool message is marked
    assert m[3]["content"][0]["cache_control"] == {"type": "ephemeral"}
    plain = request_to_wire(req.model_copy(update={"cache": False}))
    assert all(isinstance(x["content"], str) for x in plain["messages"])


def test_cached_tokens_parsed_from_both_styles() -> None:
    base = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    a = response_from_wire(
        base
        | {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 800},
            }
        },
        fallback_model="m",
    )
    assert a.usage.cached_input_tokens == 800
    b = response_from_wire(
        base
        | {"usage": {"prompt_tokens": 100, "completion_tokens": 5, "cache_read_input_tokens": 500}},
        fallback_model="m",
    )
    assert b.usage.cached_input_tokens == 100  # never more than the input
    c = response_from_wire(base | {"usage": {"prompt_tokens": 10}}, fallback_model="m")
    assert c.usage.cached_input_tokens == 0


def test_cost_is_conservative_with_cache() -> None:
    price = ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0, cache_read_per_mtok=0.3)
    usage = Usage(input_tokens=10_000, output_tokens=100, cached_input_tokens=8_000)
    plain = compute_cost(usage, price)
    cached = compute_cost(usage, price, cache=True)
    assert cached == pytest.approx((2_000 * 3 * CACHE_WRITE_FACTOR + 8_000 * 0.3 + 100 * 15) / 1e6)
    assert cached < plain
    # nothing read from the cache: every input token may have been written (1.25x)
    cold = compute_cost(Usage(input_tokens=10_000, output_tokens=100), price, cache=True)
    assert cold == pytest.approx((10_000 * 3 * CACHE_WRITE_FACTOR + 100 * 15) / 1e6)
    assert cold > plain
    # no cache read price: cached tokens at the full price (never under-counted)
    no_read = ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0)
    assert compute_cost(usage, no_read, cache=True) >= compute_cost(usage, no_read) - 1e-12
    # the worst-case estimate assumes every input token is written
    req = ChatRequest(model="m", messages=tuple(msgs()), max_tokens=100, cache=True)
    n, worst = estimate_max_cost(req, price)
    assert worst == pytest.approx((n * 3 * CACHE_WRITE_FACTOR + 100 * 15) / 1e6)


def test_gateway_sends_cache_only_when_on_and_records_cheaper_cost(ledger: CostLedger) -> None:
    for on, expected in ((False, False), (True, True)):
        cfg = config(prompt_cache=on)
        fake = FakeProvider(
            [reply(input_tokens=10_000, output_tokens=100, cached_input_tokens=9_000)]
        )
        g = LLMGateway(cfg, fake, guard=BudgetGuard(cfg.limits, ledger))
        r = g.complete(msgs(), tier=Tier.SMALL, max_tokens=100, context=CTX)
        assert fake.requests[0].cache is expected
        if on:
            assert r.cost_usd == pytest.approx(
                (1_000 * 3 * CACHE_WRITE_FACTOR + 9_000 * 0.3 + 100 * 15) / 1e6
            )
        else:
            assert r.cost_usd == pytest.approx((10_000 * 3 + 100 * 15) / 1e6)


def test_gpt_models_are_not_marked(ledger: CostLedger) -> None:
    cfg = config("gpt-5.5", prompt_cache=True)
    fake = FakeProvider([reply(input_tokens=10, output_tokens=1)])
    LLMGateway(cfg, fake, guard=BudgetGuard(cfg.limits, ledger)).complete(
        msgs(), tier=Tier.SMALL, max_tokens=10, context=CTX
    )
    assert fake.requests[0].cache is False


def test_gateway_drops_markers_once_refused(ledger: CostLedger) -> None:
    cfg = config(prompt_cache=True)
    refused = ProviderError("provider returned HTTP 400: unknown field cache_control", status=400)
    fake = FakeProvider(
        [refused, reply(input_tokens=10, output_tokens=1), reply(input_tokens=10, output_tokens=1)]
    )
    g = LLMGateway(cfg, fake, guard=BudgetGuard(cfg.limits, ledger))
    g.complete(msgs(), tier=Tier.SMALL, max_tokens=10, context=CTX)
    assert [r.cache for r in fake.requests] == [True, False]
    g.complete(msgs(), tier=Tier.SMALL, max_tokens=10, context=CTX)
    assert fake.requests[-1].cache is False  # remembered for this model


def test_auth_or_budget_errors_do_not_drop_markers(ledger: CostLedger) -> None:
    cfg = config(prompt_cache=True)
    fake = FakeProvider([ProviderError("provider returned HTTP 401: bad key", status=401)])
    g = LLMGateway(cfg, fake, guard=BudgetGuard(cfg.limits, ledger))
    with pytest.raises(ProviderError):
        g.complete(msgs(), tier=Tier.SMALL, max_tokens=10, context=CTX)
    assert len(fake.requests) == 1


def test_markers_reach_the_http_body() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 40},
                },
            },
        )

    with OpenAICompatProvider(
        "https://gw.example/v1", "k", transport=httpx.MockTransport(handler)
    ) as p:
        resp = p.complete(
            ChatRequest(model="claude-x", messages=tuple(msgs()[:2]), max_tokens=5, cache=True)
        )
    assert seen[0]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert resp.usage.cached_input_tokens == 40
