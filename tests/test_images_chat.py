"""Chat-based image models (D103): ``image_api = "chat"`` in the catalog, token pricing
with a ``max_tokens`` worst case, the ``/chat/completions`` transport and the same
format/dimension checks in ``generate_image``.

Local only: ``FakeImageProvider`` or ``httpx.MockTransport`` behind a real
BudgetGuard + CostLedger. No network, no real model calls.
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from tests.test_images import jpeg, png

from automa.budget import BudgetExceeded, BudgetGuard, CostLedger
from automa.config import (
    IMAGE_CHAT_MAX_INPUT_TOKENS,
    MAX_IMAGE_CHAT_TOKENS,
    AutomaConfig,
    ConfigError,
    ImageApi,
    ModelInfo,
    Tier,
)
from automa.llm.gateway import CallContext, _estimate_input_tokens
from automa.llm.images import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PROMPT_CHARS,
    FakeImageProvider,
    ImageGateway,
    ImageRequest,
    ImageSize,
    OpenAICompatImageProvider,
    chat_image_prompt,
    chat_image_request_to_wire,
    chat_image_response_from_wire,
)
from automa.llm.provider import ProviderError
from automa.llm.types import ChatRequest, Message, Usage
from automa.tools import GenerateImage, GenerateImageArgs, ToolError, Workspace

CTX = CallContext(project_id="p1", task_id="t1")
TEXT = {"family": "anthropic", "capabilities": ["text"], "input_per_mtok": 1, "output_per_mtok": 2}
# $2 in / $30 out per 1M tokens, max 4000 output tokens:
# worst = (1200 * 2 + 4000 * 30) / 1e6 = 0.1224
CHAT_IMG: dict[str, Any] = {
    "family": "google",
    "capabilities": ["text", "image_out"],
    "input_per_mtok": 2,
    "output_per_mtok": 30,
    "image_api": "chat",
    "image_max_tokens": 4000,
}
WORST = (IMAGE_CHAT_MAX_INPUT_TOKENS * 2 + 4000 * 30) / 1_000_000


def data_uri(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def cfg_data(images: dict[str, Any] | None = None, **chat: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "tiers": {"small": "txt", "medium": "txt"},
        "models": {"txt": TEXT, "chat-img": {**CHAT_IMG, **chat}},
        "limits": {"project_usd": 1.0, "daily_usd": 1.0, "per_call_usd": 0.25},
    }
    if images is not None:
        data["images"] = images
    return data


def cfg(images: dict[str, Any] | None = None, **chat: Any) -> AutomaConfig:
    return AutomaConfig.model_validate(cfg_data(images, **chat))


class Rig:
    def __init__(
        self,
        tmp: Path,
        images: list[bytes | Exception],
        *,
        config: AutomaConfig | None = None,
        **kw: Any,
    ) -> None:
        self.config = config or cfg()
        self.ledger = CostLedger(tmp / "ledger.db")
        self.guard = BudgetGuard(self.config.limits, self.ledger)
        self.provider = FakeImageProvider(images, **kw)
        self.gateway = ImageGateway(self.config, self.provider, guard=self.guard)
        root = tmp / "files"
        root.mkdir()
        self.workspace = Workspace(root)
        self.tool = GenerateImage(self.gateway, model="chat-img", context=CTX)

    def generate(self, prompt: str = "a cat") -> Any:
        return self.gateway.generate(ImageRequest(model="chat-img", prompt=prompt), context=CTX)

    def run(self, **args: Any) -> str:
        return self.tool.run(GenerateImageArgs(**args), self.workspace)


# --- catalog ------------------------------------------------------------------


def test_default_api_is_images_and_chat_reads() -> None:
    plain = ModelInfo.model_validate(
        {"family": "openai", "capabilities": ["image_out"], "input_per_mtok": 0,
         "output_per_mtok": 0, "per_image_usd": 0.04}
    )  # fmt: skip
    assert plain.image_api is ImageApi.IMAGES
    assert plain.image_max_tokens is None
    assert plain.image_worst_case_usd() == pytest.approx(0.04)
    chat = ModelInfo.model_validate(CHAT_IMG)
    assert chat.image_api is ImageApi.CHAT
    assert chat.image_max_tokens == 4000
    assert chat.image_worst_case_usd() == pytest.approx(WORST)


def test_chat_needs_max_tokens() -> None:
    bad = {k: v for k, v in CHAT_IMG.items() if k != "image_max_tokens"}
    with pytest.raises(ValidationError, match="image_max_tokens"):
        ModelInfo.model_validate(bad)


def test_max_tokens_only_for_chat() -> None:
    bad = {**CHAT_IMG, "image_api": "images"}
    with pytest.raises(ValidationError, match="only for image_api"):
        ModelInfo.model_validate(bad)


@pytest.mark.parametrize("field", ["image_api", "image_max_tokens"])
def test_image_fields_need_image_out(field: str) -> None:
    data = {**TEXT, field: "chat" if field == "image_api" else 100}
    with pytest.raises(ValidationError, match="image_out"):
        ModelInfo.model_validate(data)


@pytest.mark.parametrize("bad", [0, -1, MAX_IMAGE_CHAT_TOKENS + 1, 1.5, "100", True])
def test_max_tokens_is_a_bounded_integer(bad: Any) -> None:
    with pytest.raises(ValidationError):
        ModelInfo.model_validate({**CHAT_IMG, "image_max_tokens": bad})


@pytest.mark.parametrize("bad", ["completions", "CHAT", "", 1])
def test_unknown_api_is_rejected(bad: Any) -> None:
    with pytest.raises(ValidationError):
        ModelInfo.model_validate({**CHAT_IMG, "image_api": bad})


def test_per_image_price_is_a_floor_for_chat() -> None:
    info = ModelInfo.model_validate({**CHAT_IMG, "per_image_usd": 0.2})
    assert info.image_worst_case_usd() == pytest.approx(0.2)
    low = ModelInfo.model_validate({**CHAT_IMG, "per_image_usd": 0.001})
    assert low.image_worst_case_usd() == pytest.approx(WORST)


def test_non_image_model_has_no_worst_case() -> None:
    assert ModelInfo.model_validate(TEXT).image_worst_case_usd() is None


def test_images_section_accepts_a_chat_model_without_per_image_price() -> None:
    config = cfg({"model": "chat-img"})
    assert config.image_price() == pytest.approx(WORST)


def test_images_section_refuses_a_chat_model_above_the_per_call_cap() -> None:
    with pytest.raises(ValidationError, match="per_call_usd"):
        cfg({"model": "chat-img"}, image_max_tokens=10000)  # 0.3024 > 0.25


def test_chat_image_model_can_serve_a_text_tier() -> None:
    data = cfg_data()
    data["tiers"]["large"] = "chat-img"
    config = AutomaConfig.model_validate(data)
    assert config.price_for("chat-img").output_per_mtok == 30


def test_worst_case_input_bound_covers_the_longest_prompt() -> None:
    prompt = "x" * MAX_IMAGE_PROMPT_CHARS
    for size in ImageSize:
        text = chat_image_prompt(ImageRequest(model="m", prompt=prompt, size=size))
        req = ChatRequest(model="m", messages=[Message.user(text)], max_tokens=1)
        assert _estimate_input_tokens(req) <= IMAGE_CHAT_MAX_INPUT_TOKENS
        # Even at one character per token (no tokenizer is that bad for ASCII).
        assert len(text) + 8 <= IMAGE_CHAT_MAX_INPUT_TOKENS


# --- gateway ------------------------------------------------------------------


def test_gateway_estimate_uses_token_worst_case(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [])
    est = rig.gateway.estimate("chat-img", CTX, tier=Tier.MEDIUM)
    assert est.max_input_tokens == IMAGE_CHAT_MAX_INPUT_TOKENS
    assert est.max_output_tokens == 4000
    assert est.max_cost_usd == pytest.approx(WORST)
    assert rig.gateway.price_per_image("chat-img") == pytest.approx(WORST)
    assert rig.gateway.api_for("chat-img") is ImageApi.CHAT


def test_gateway_sends_chat_api_and_cap_from_the_catalog(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png()], usage=Usage(input_tokens=20, output_tokens=1290))
    # A caller cannot pick the endpoint or the token cap: the catalog decides.
    rig.gateway.generate(
        ImageRequest(model="chat-img", prompt="x", api=ImageApi.IMAGES, max_tokens=99999),
        context=CTX,
    )
    [sent] = rig.provider.requests
    assert sent.api is ImageApi.CHAT
    assert sent.max_tokens == 4000


def test_gateway_records_token_cost_and_counts(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png()], usage=Usage(input_tokens=20, output_tokens=1290))
    result = rig.generate()
    expected = (20 * 2 + 1290 * 30) / 1_000_000
    assert result.cost_usd == pytest.approx(expected)
    [entry] = rig.ledger.entries()
    assert (entry.input_tokens, entry.output_tokens) == (20, 1290)
    assert entry.cost_usd == pytest.approx(expected, abs=1e-9)
    assert entry.model == "chat-img"
    assert rig.guard.pending_nusd() == 0


def test_gateway_charges_worst_case_without_usage(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png()])  # no usage reported
    result = rig.generate()
    assert result.cost_usd == pytest.approx(WORST)
    [entry] = rig.ledger.entries()
    assert (entry.input_tokens, entry.output_tokens) == (IMAGE_CHAT_MAX_INPUT_TOKENS, 4000)


def test_gateway_charges_worst_case_with_zero_output(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png()], usage=Usage(input_tokens=20, output_tokens=0))
    assert rig.generate().cost_usd == pytest.approx(WORST)


def test_gateway_charges_max_input_when_input_missing(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png()], usage=Usage(input_tokens=0, output_tokens=1000))
    expected = (IMAGE_CHAT_MAX_INPUT_TOKENS * 2 + 1000 * 30) / 1_000_000
    assert rig.generate().cost_usd == pytest.approx(expected)
    [entry] = rig.ledger.entries()
    assert entry.input_tokens == IMAGE_CHAT_MAX_INPUT_TOKENS


def test_gateway_takes_higher_reported_cost(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        [png()],
        usage=Usage(input_tokens=20, output_tokens=100),
        reported_cost_usd=0.09,
    )
    assert rig.generate().cost_usd == pytest.approx(0.09)


def test_gateway_ignores_lower_reported_cost(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        [png()],
        usage=Usage(input_tokens=20, output_tokens=1000),
        reported_cost_usd=0.000001,
    )
    assert rig.generate().cost_usd == pytest.approx((40 + 30000) / 1_000_000)


def test_gateway_per_image_price_is_a_floor(tmp_path: Path) -> None:
    config = cfg(per_image_usd=0.05)
    rig = Rig(tmp_path, [png()], config=config, usage=Usage(input_tokens=10, output_tokens=10))
    assert rig.generate().cost_usd == pytest.approx(0.05)


def test_gateway_records_more_than_the_estimate_when_usage_exceeds_it(tmp_path: Path) -> None:
    # A gateway that ignores max_tokens is still charged what it reports.
    rig = Rig(tmp_path, [png()], usage=Usage(input_tokens=20, output_tokens=6000))
    assert rig.generate().cost_usd == pytest.approx((40 + 180000) / 1_000_000)
    assert rig.guard.pending_nusd() == 0


def test_gateway_releases_on_unbilled_failure(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [ProviderError("no connect", retryable=True)])
    with pytest.raises(ProviderError):
        rig.generate()
    assert rig.ledger.entries() == []
    assert rig.guard.pending_nusd() == 0


def test_gateway_records_worst_case_on_maybe_billed_failure(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [ProviderError("timeout", maybe_billed=True)])
    with pytest.raises(ProviderError):
        rig.generate()
    [entry] = rig.ledger.entries()
    assert entry.cost_usd == pytest.approx(WORST)
    assert (entry.input_tokens, entry.output_tokens) == (IMAGE_CHAT_MAX_INPUT_TOKENS, 4000)
    assert rig.guard.pending_nusd() == 0


def test_gateway_never_retries(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [ProviderError("busy", retryable=True, status=503), png()])
    with pytest.raises(ProviderError):
        rig.generate()
    assert len(rig.provider.requests) == 1


def test_budget_cap_blocks_before_sending(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png()] * 20)  # no usage: each call charged WORST
    n = math.floor(1.0 / WORST)
    for _ in range(n):
        rig.generate()
    with pytest.raises(BudgetExceeded):
        rig.generate()
    assert len(rig.provider.requests) == n


def test_per_call_cap_applies_to_the_token_worst_case(tmp_path: Path) -> None:
    config = cfg(image_max_tokens=10000)  # worst 0.3024 > per_call 0.25
    ledger = CostLedger(tmp_path / "l.db")
    provider = FakeImageProvider([png()])
    gw = ImageGateway(config, provider, guard=BudgetGuard(config.limits, ledger))
    with pytest.raises(BudgetExceeded):
        gw.generate(ImageRequest(model="chat-img", prompt="x"), context=CTX)
    assert provider.requests == []


def test_images_api_model_is_unchanged(tmp_path: Path) -> None:
    data = cfg_data()
    data["models"]["img"] = {
        "family": "openai",
        "capabilities": ["image_out"],
        "input_per_mtok": 5,
        "output_per_mtok": 5,
        "per_image_usd": 0.04,
    }
    config = AutomaConfig.model_validate(data)
    ledger = CostLedger(tmp_path / "l.db")
    provider = FakeImageProvider([png()], usage=Usage(input_tokens=99, output_tokens=99))
    gw = ImageGateway(config, provider, guard=BudgetGuard(config.limits, ledger))
    r = gw.generate(ImageRequest(model="img", prompt="x"), context=CTX)
    assert r.cost_usd == pytest.approx(0.04)
    [sent] = provider.requests
    assert sent.api is ImageApi.IMAGES and sent.max_tokens is None
    [entry] = ledger.entries()
    assert (entry.input_tokens, entry.output_tokens) == (0, 1)


def test_unpriced_images_model_still_refused(tmp_path: Path) -> None:
    data = cfg_data()
    data["models"]["img-unpriced"] = {
        "family": "openai",
        "capabilities": ["image_out"],
        "input_per_mtok": 0,
        "output_per_mtok": 0,
    }
    config = AutomaConfig.model_validate(data)
    gw = ImageGateway(
        config, FakeImageProvider([]), guard=BudgetGuard(config.limits, CostLedger(tmp_path / "l"))
    )
    with pytest.raises(ConfigError, match="per_image_usd"):
        gw.price_per_image("img-unpriced")


# --- wire format ----------------------------------------------------------------


def test_chat_wire_body() -> None:
    req = ImageRequest(
        model="chat-img", prompt="a cat", size=ImageSize.LANDSCAPE,
        api=ImageApi.CHAT, max_tokens=4000,
    )  # fmt: skip
    body = chat_image_request_to_wire(req)
    assert body["model"] == "chat-img"
    assert body["max_tokens"] == 4000
    assert body["modalities"] == ["image", "text"]
    assert body["stream"] is False
    [msg] = body["messages"]
    assert msg["role"] == "user"
    assert msg["content"].startswith("a cat")
    assert "1536x1024" in msg["content"]


def test_chat_wire_body_needs_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        chat_image_request_to_wire(ImageRequest(model="m", prompt="x", api=ImageApi.CHAT))


def _chat(message: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", **message}}], **extra}


def test_parse_images_field() -> None:
    image = png()
    data = _chat(
        {
            "content": "here",
            "images": [{"type": "image_url", "image_url": {"url": data_uri(image)}}],
        },
        usage={"prompt_tokens": 12, "completion_tokens": 1290, "cost": 0.04},
        model="chat-img-v2",
    )
    r = chat_image_response_from_wire(data, fallback_model="chat-img")
    assert r.data == image
    assert r.usage == Usage(input_tokens=12, output_tokens=1290, reported_cost_usd=0.04)
    assert r.reported_cost_usd == 0.04
    assert r.model == "chat-img-v2"


def test_parse_content_parts() -> None:
    image = jpeg()
    data = _chat(
        {
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "image_url", "image_url": {"url": data_uri(image, "image/jpeg")}},
            ]
        }
    )
    r = chat_image_response_from_wire(data, fallback_model="m")
    assert r.data == image
    assert r.model == "m"
    assert r.usage == Usage()


def test_parse_markdown_data_uri_in_text() -> None:
    image = png()
    data = _chat({"content": f"Here you go:\n![img]({data_uri(image)})\nEnjoy"})
    assert chat_image_response_from_wire(data, fallback_model="m").data == image


def test_parse_takes_the_first_of_several_images() -> None:
    a, b = png(20, 20), png(30, 30)
    data = _chat(
        {
            "images": [
                {"type": "image_url", "image_url": {"url": data_uri(a)}},
                {"type": "image_url", "image_url": {"url": data_uri(b)}},
            ]
        }
    )
    assert chat_image_response_from_wire(data, fallback_model="m").data == a


def test_parse_accepts_plain_string_image_url() -> None:
    image = png()
    data = _chat({"images": [{"type": "image_url", "image_url": data_uri(image)}]})
    assert chat_image_response_from_wire(data, fallback_model="m").data == image


@pytest.mark.parametrize(
    "message",
    [
        {"content": "sorry, I can only describe images"},
        {"content": None},
        {"images": [{"type": "image_url", "image_url": {"url": "https://evil.example/a.png"}}]},
        {"images": [{"type": "image_url", "image_url": {"url": "data:image/gif;base64,R0lG"}}]},
        {"images": [{"type": "image_url", "image_url": {"url": "data:image/svg+xml;base64,PHN2"}}]},
        {"images": [{"type": "text", "text": "x"}]},
        {"content": [{"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}]},
    ],
)
def test_parse_without_an_inline_png_or_jpeg_fails(message: dict[str, Any]) -> None:
    with pytest.raises(ProviderError, match="no inline PNG/JPEG"):
        chat_image_response_from_wire(_chat(message), fallback_model="m")


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"choices": []},
        {"choices": "x"},
        {"choices": [{}]},
        {"choices": [{"message": "text"}]},
        [],
        None,
    ],
)
def test_parse_malformed(data: Any) -> None:
    with pytest.raises(ProviderError, match="malformed"):
        chat_image_response_from_wire(data, fallback_model="m")


def test_parse_bad_base64_is_malformed() -> None:
    data = _chat(
        {"images": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,ab=c="}}]}
    )
    with pytest.raises(ProviderError, match="malformed"):
        chat_image_response_from_wire(data, fallback_model="m")


def test_parse_bounds_the_image_size() -> None:
    big = data_uri(b"\x89PNG" + b"\x00" * (MAX_IMAGE_BYTES + 10))
    data = _chat({"images": [{"type": "image_url", "image_url": {"url": big}}]})
    with pytest.raises(ProviderError, match="larger"):
        chat_image_response_from_wire(data, fallback_model="m")


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": -5, "completion_tokens": True, "cost": -1},
        {"prompt_tokens": "12", "completion_tokens": 1.5, "cost": float("inf")},
        "nope",
    ],
)
def test_parse_ignores_bogus_usage(usage: Any) -> None:
    data = _chat({"images": [{"type": "image_url", "image_url": {"url": data_uri(png())}}]})
    data["usage"] = usage
    r = chat_image_response_from_wire(data, fallback_model="m")
    assert r.usage == Usage()
    assert r.reported_cost_usd is None


# --- transport ------------------------------------------------------------------


def _provider(handler: Any) -> OpenAICompatImageProvider:
    return OpenAICompatImageProvider(
        "https://gw.example/v1", "sk-secret-key-123456", transport=httpx.MockTransport(handler)
    )


def _chat_req() -> ImageRequest:
    return ImageRequest(model="chat-img", prompt="a cat", api=ImageApi.CHAT, max_tokens=4000)


def test_transport_posts_to_chat_completions() -> None:
    seen: list[httpx.Request] = []
    image = png()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json=_chat(
                {"images": [{"type": "image_url", "image_url": {"url": data_uri(image)}}]},
                usage={"prompt_tokens": 9, "completion_tokens": 1290},
            ),
        )

    resp = _provider(handler).generate(_chat_req())
    assert resp.data == image
    assert resp.usage == Usage(input_tokens=9, output_tokens=1290)
    [req] = seen
    assert req.url == "https://gw.example/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer sk-secret-key-123456"
    body = json.loads(req.content)
    assert body["max_tokens"] == 4000
    assert body["messages"][0]["content"].startswith("a cat")


def test_transport_chat_without_image_is_maybe_billed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat({"content": "I cannot draw sk-secret"}))

    with pytest.raises(ProviderError) as info:
        _provider(handler).generate(_chat_req())
    assert info.value.maybe_billed
    assert "sk-secret" not in str(info.value)


def test_transport_chat_http_error_never_echoes_the_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad key sk-secret-key-123456"}})

    with pytest.raises(ProviderError) as info:
        _provider(handler).generate(_chat_req())
    assert "sk-secret" not in str(info.value)
    assert info.value.status == 400
    assert not info.value.maybe_billed


def test_transport_chat_error_object_in_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "quota sk-secret"}})

    with pytest.raises(ProviderError, match="error object") as info:
        _provider(handler).generate(_chat_req())
    assert "sk-secret" not in str(info.value)


def test_transport_images_path_unchanged() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png()).decode()}]})

    _provider(handler).generate(ImageRequest(model="img", prompt="x"))
    assert seen == ["https://gw.example/v1/images/generations"]


# --- generate_image tool: same format/dimension checks --------------------------


def test_tool_writes_a_png_from_a_chat_model(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png(64, 32)], usage=Usage(input_tokens=10, output_tokens=1000))
    out = rig.run(prompt="a cat", path="img/cat.png")
    assert out.startswith("wrote img/cat.png: PNG 64x32")
    assert (rig.workspace.root / "img/cat.png").read_bytes() == png(64, 32)


def test_tool_refuses_format_mismatch_from_a_chat_model(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [jpeg()])
    with pytest.raises(ToolError, match="not PNG"):
        rig.run(prompt="a cat", path="cat.png")
    assert not (rig.workspace.root / "cat.png").exists()
    # It was paid for, so it is in the ledger.
    assert len(rig.ledger.entries()) == 1


def test_tool_refuses_bad_dimensions_from_a_chat_model(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png(4, 4)])
    with pytest.raises(ToolError, match="dimensions"):
        rig.run(prompt="a cat", path="cat.png")


def test_tool_hides_provider_text(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [ProviderError("echo sk-secret", status=500)])
    with pytest.raises(ToolError) as info:
        rig.run(prompt="a cat", path="cat.png")
    assert "sk-secret" not in str(info.value)


# --- CLI ------------------------------------------------------------------------


def test_models_cli_shows_chat_image_api(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from automa.cli import app

    toml = tmp_path / "automa.toml"
    toml.write_text(
        'workspace = "ws"\n'
        '[tiers]\nsmall = "txt"\nmedium = "txt"\n'
        "[limits]\nproject_usd = 1.0\ndaily_usd = 1.0\nper_call_usd = 0.25\n"
        '[models."txt"]\nfamily = "anthropic"\ncapabilities = ["text"]\n'
        "input_per_mtok = 1\noutput_per_mtok = 2\n"
        '[models."chat-img"]\nfamily = "google"\ncapabilities = ["text", "image_out"]\n'
        'input_per_mtok = 2\noutput_per_mtok = 30\nimage_api = "chat"\nimage_max_tokens = 4000\n',
        encoding="utf-8",
    )
    runner = CliRunner()
    res = runner.invoke(app, ["models", "--config", str(toml)])
    assert res.exit_code == 0, res.output
    assert "images via chat (max 4000 tokens, up to $0.1224 per image)" in res.output
    res = runner.invoke(app, ["models", "--config", str(toml), "--json"])
    [entry] = [e for e in json.loads(res.output) if e["id"] == "chat-img"]
    assert entry["image_api"] == "chat"
    assert entry["image_max_tokens"] == 4000
    assert entry["image_worst_case_usd"] == pytest.approx(WORST)


def test_describe_images_line_for_chat_model() -> None:
    from automa.cli import _describe_images

    line = _describe_images(cfg({"model": "chat-img"}))
    assert line.startswith("chat-img (chat-based, up to $0.1224 per image")
    assert "max 4000 output tokens" in line
