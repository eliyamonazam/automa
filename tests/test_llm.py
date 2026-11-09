"""Tests for the model gateway. No network: FakeProvider and httpx.MockTransport only."""

from __future__ import annotations

import json

import httpx
import pytest

from automa.config import AutomaConfig, ConfigError, ModelPrice, Tier
from automa.llm import (
    CallContext,
    CallEstimate,
    ChatRequest,
    ChatResponse,
    FakeProvider,
    FinishReason,
    LLMGateway,
    LLMResult,
    Message,
    OpenAICompatProvider,
    Provider,
    ProviderError,
    Role,
    ToolCall,
    ToolSpec,
    Usage,
    compute_cost,
    estimate_max_cost,
)
from automa.llm.openai_compat import request_to_wire, response_from_wire

FAKE_KEY = "fake-test-key-xyz"


@pytest.fixture
def config() -> AutomaConfig:
    return AutomaConfig.model_validate(
        {
            "base_url": "https://gateway.example/v1",
            "tiers": {"small": "m-small", "large": "m-large"},
            "prices": {
                "m-small": {"input_per_mtok": 1.0, "output_per_mtok": 2.0},
                "m-large": {"input_per_mtok": 10.0, "output_per_mtok": 30.0},
            },
        }
    )


class RecordingGuard:
    def __init__(self, *, refuse: bool = False):
        self.refuse = refuse
        self.authorized: list[CallEstimate] = []
        self.recorded: list[tuple[CallEstimate, LLMResult]] = []

    def authorize(self, estimate: CallEstimate) -> None:
        self.authorized.append(estimate)
        if self.refuse:
            raise RuntimeError("budget exceeded")

    def record(self, estimate: CallEstimate, result: LLMResult) -> None:
        self.recorded.append((estimate, result))


# --- types -----------------------------------------------------------------


def test_message_validation() -> None:
    with pytest.raises(ValueError):
        Message(role=Role.TOOL, content="x")  # missing tool_call_id
    with pytest.raises(ValueError):
        Message(role=Role.USER, tool_calls=(ToolCall(id="1", name="f"),))
    with pytest.raises(ValueError):
        Message(role=Role.USER)
    assert Message.tool("c1", "ok").tool_call_id == "c1"


def test_request_requires_bounded_output() -> None:
    msgs = (Message.user("hi"),)
    with pytest.raises(ValueError):
        ChatRequest(model="m", messages=msgs, max_tokens=0)
    with pytest.raises(ValueError):
        ChatRequest(model="m", messages=(), max_tokens=10)
    with pytest.raises(ValueError):
        ChatRequest(model="m", messages=msgs, max_tokens=10, extra=1)  # type: ignore[call-arg]


# --- cost ------------------------------------------------------------------


def test_compute_cost() -> None:
    price = ModelPrice(input_per_mtok=1.0, output_per_mtok=2.0)
    usage = Usage(input_tokens=1_000_000, output_tokens=500_000)
    assert compute_cost(usage, price) == pytest.approx(2.0)


def test_estimate_is_pessimistic() -> None:
    price = ModelPrice(input_per_mtok=1.0, output_per_mtok=2.0)
    text = "word " * 400  # ~2000 chars ≈ 500 real tokens
    req = ChatRequest(model="m", messages=(Message.user(text),), max_tokens=100)
    tokens, cost = estimate_max_cost(req, price)
    assert tokens >= len(text) // 2
    assert cost >= compute_cost(Usage(input_tokens=len(text) // 4, output_tokens=100), price)


def test_estimate_counts_tools() -> None:
    price = ModelPrice(input_per_mtok=1.0, output_per_mtok=1.0)
    base = ChatRequest(model="m", messages=(Message.user("x"),), max_tokens=1)
    spec = ToolSpec(name="read_file", description="d" * 200, parameters={"type": "object"})
    with_tool = base.model_copy(update={"tools": (spec,)})
    assert estimate_max_cost(with_tool, price)[0] > estimate_max_cost(base, price)[0]


# --- fake provider -----------------------------------------------------------


def test_fake_provider_script_and_recording() -> None:
    boom = ProviderError("down", status=503, retryable=True)
    fake = FakeProvider(["one", boom, lambda r: f"echo {r.messages[-1].content}"])
    assert isinstance(fake, Provider)
    req = ChatRequest(model="m", messages=(Message.user("ping"),), max_tokens=5)
    first = fake.complete(req)
    assert first.message.content == "one" and first.usage.output_tokens > 0
    with pytest.raises(ProviderError) as info:
        fake.complete(req)
    assert info.value.retryable
    assert fake.complete(req).message.content == "echo ping"
    with pytest.raises(ProviderError, match="exhausted"):
        fake.complete(req)
    assert len(fake.requests) == 4


def test_fake_provider_default() -> None:
    fake = FakeProvider(default="same")
    req = ChatRequest(model="m", messages=(Message.user("x"),), max_tokens=5)
    assert [fake.complete(req).message.content for _ in range(3)] == ["same"] * 3


# --- gateway -----------------------------------------------------------------


def test_gateway_resolves_tier_and_prices(config: AutomaConfig) -> None:
    fake = FakeProvider(["hello"])
    gw = LLMGateway(config, fake)
    result = gw.complete([Message.user("hi")], tier=Tier.LARGE, max_tokens=50)
    assert fake.requests[0].model == "m-large"
    assert result.text == "hello" and result.tier is Tier.LARGE
    expected = compute_cost(result.response.usage, config.price_for("m-large"))
    assert result.cost_usd == pytest.approx(expected) and result.cost_usd > 0


def test_gateway_unknown_tier(config: AutomaConfig) -> None:
    fake = FakeProvider(["x"])
    with pytest.raises(ConfigError):
        LLMGateway(config, fake).complete([Message.user("hi")], tier=Tier.MEDIUM, max_tokens=5)
    assert fake.requests == []


def test_gateway_guard_authorizes_before_and_records_after(config: AutomaConfig) -> None:
    guard = RecordingGuard()
    gw = LLMGateway(config, FakeProvider(["ok"]), guard=guard)
    ctx = CallContext(project_id="p1", task_id="t1")
    result = gw.complete([Message.user("hi")], tier=Tier.SMALL, max_tokens=10, context=ctx)
    assert len(guard.authorized) == 1 and guard.authorized[0].context == ctx
    est = guard.authorized[0]
    assert est.model == "m-small" and est.max_output_tokens == 10
    assert est.max_cost_usd >= result.cost_usd
    assert guard.recorded == [(est, result)]


def test_gateway_guard_refusal_blocks_provider(config: AutomaConfig) -> None:
    fake = FakeProvider(["never"])
    gw = LLMGateway(config, fake, guard=RecordingGuard(refuse=True))
    with pytest.raises(RuntimeError, match="budget"):
        gw.complete([Message.user("hi")], tier=Tier.SMALL, max_tokens=10)
    assert fake.requests == [] and fake.remaining == 1


def test_gateway_provider_failure_not_recorded(config: AutomaConfig) -> None:
    guard = RecordingGuard()
    gw = LLMGateway(config, FakeProvider([ProviderError("x")]), guard=guard)
    with pytest.raises(ProviderError):
        gw.complete([Message.user("hi")], tier=Tier.SMALL, max_tokens=10)
    assert len(guard.authorized) == 1 and guard.recorded == []


def _resp(usage: Usage, model: str = "m-small") -> ChatResponse:
    return ChatResponse(model=model, message=Message.assistant("x"), usage=usage)


def test_gateway_missing_usage_charges_worst_case(config: AutomaConfig) -> None:
    guard = RecordingGuard()
    gw = LLMGateway(config, FakeProvider([_resp(Usage())]), guard=guard)
    result = gw.complete([Message.user("hi")], tier=Tier.SMALL, max_tokens=1000)
    assert result.cost_usd > 0
    assert result.cost_usd == pytest.approx(guard.authorized[0].max_cost_usd)


def test_gateway_reported_cost_never_lowers_cost(config: AutomaConfig) -> None:
    usage_low = Usage(input_tokens=1000, output_tokens=1000, reported_cost_usd=0.0)
    usage_high = Usage(input_tokens=1000, output_tokens=1000, reported_cost_usd=5.0)
    gw = LLMGateway(config, FakeProvider([_resp(usage_low), _resp(usage_high)]))
    computed = compute_cost(usage_low, config.price_for("m-small"))
    low = gw.complete([Message.user("a")], tier=Tier.SMALL, max_tokens=10)
    high = gw.complete([Message.user("a")], tier=Tier.SMALL, max_tokens=10)
    assert low.cost_usd == pytest.approx(computed)
    assert high.cost_usd == pytest.approx(5.0)


def test_gateway_prices_requested_model_not_reported(config: AutomaConfig) -> None:
    # Provider claims a different (unpriced) model; we still price the requested one.
    usage = Usage(input_tokens=1000, output_tokens=0)
    gw = LLMGateway(config, FakeProvider([_resp(usage, model="other/unknown")]))
    result = gw.complete([Message.user("a")], tier=Tier.LARGE, max_tokens=10)
    assert result.cost_usd == pytest.approx(compute_cost(usage, config.price_for("m-large")))


def test_from_config_requires_key(config: AutomaConfig) -> None:
    with pytest.raises(ConfigError):
        LLMGateway.from_config(config)


# --- OpenAI-compatible provider (MockTransport, no network) -----------------


def _completion_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "model": "m-small",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
    body.update(overrides)
    return body


def _provider(handler) -> OpenAICompatProvider:  # type: ignore[no-untyped-def]
    return OpenAICompatProvider(
        "https://gateway.example/v1/", FAKE_KEY, transport=httpx.MockTransport(handler)
    )


def test_openai_provider_request_and_response() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion_body())

    tool = ToolSpec(name="list_dir", parameters={"type": "object", "properties": {}})
    req = ChatRequest(
        model="m-small",
        messages=(Message.system("s"), Message.user("u")),
        max_tokens=64,
        tools=(tool,),
    )
    with _provider(handler) as provider:
        resp = provider.complete(req)
    assert str(seen[0].url) == "https://gateway.example/v1/chat/completions"
    assert seen[0].headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(seen[0].content)
    assert body["model"] == "m-small" and body["max_tokens"] == 64 and body["stream"] is False
    assert body["tools"][0]["function"]["name"] == "list_dir"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert resp.message.content == "hi"
    assert resp.usage == Usage(input_tokens=12, output_tokens=3)


def test_openai_provider_parses_tool_calls_and_reported_cost() -> None:
    body = _completion_body(
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"a"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        usage={"prompt_tokens": 5, "completion_tokens": 7, "cost": 0.0012},
    )
    resp = response_from_wire(body, fallback_model="m")
    assert resp.finish_reason is FinishReason.TOOL_CALLS
    assert resp.message.tool_calls == (
        ToolCall(id="c1", name="read_file", arguments='{"path":"a"}'),
    )
    assert resp.usage.reported_cost_usd == pytest.approx(0.0012)


def test_tool_messages_round_trip_to_wire() -> None:
    call = ToolCall(id="c1", name="f", arguments="{}")
    req = ChatRequest(
        model="m",
        messages=(
            Message.user("u"),
            Message(role=Role.ASSISTANT, tool_calls=(call,)),
            Message.tool("c1", "result"),
        ),
        max_tokens=5,
    )
    wire = request_to_wire(req)["messages"]
    assert wire[1]["tool_calls"][0]["function"]["name"] == "f"
    assert wire[2] == {"role": "tool", "content": "result", "tool_call_id": "c1"}


@pytest.mark.parametrize(
    ("status", "retryable"), [(400, False), (401, False), (429, True), (500, True), (503, True)]
)
def test_openai_provider_http_errors_hide_key(status: int, retryable: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": f"bad key {FAKE_KEY}"}})

    req = ChatRequest(model="m", messages=(Message.user("u"),), max_tokens=5)
    with pytest.raises(ProviderError) as info:
        _provider(handler).complete(req)
    assert info.value.status == status and info.value.retryable is retryable
    assert FAKE_KEY not in str(info.value)


@pytest.mark.parametrize(
    "handler_result",
    [
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"error": {"message": "x"}}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"nomessage": 1}]}),
    ],
)
def test_openai_provider_bad_bodies(handler_result: httpx.Response) -> None:
    req = ChatRequest(model="m", messages=(Message.user("u"),), max_tokens=5)
    with pytest.raises(ProviderError):
        _provider(lambda r: handler_result).complete(req)


def test_openai_provider_network_error_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot connect {FAKE_KEY}")

    req = ChatRequest(model="m", messages=(Message.user("u"),), max_tokens=5)
    with pytest.raises(ProviderError) as info:
        _provider(handler).complete(req)
    assert info.value.retryable and FAKE_KEY not in str(info.value)


def test_openai_provider_requires_key() -> None:
    with pytest.raises(ValueError):
        OpenAICompatProvider("https://x/v1", "")
