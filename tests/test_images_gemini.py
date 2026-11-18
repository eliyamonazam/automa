"""Images through Google's native route on the gateway (``image_api = "gemini"``, D115).

Found by real runs on 1xAi: the OpenAI-style chat route made the image but failed
with "Unhandled generated data mime type: image/jpeg", and /v1/images said Google
image models are only served at /gemini/v1beta/models/{model}:generateContent.
A probe on the owner's laptop then got an image there with a Bearer key; the reply
below mirrors its shape (usage numbers verbatim). No network: MockTransport only.
"""

from __future__ import annotations

import base64
import json
import struct
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from automa.budget import BudgetGuard, CostLedger
from automa.config import AutomaConfig, ImageApi, ModelInfo
from automa.llm.gateway import CallContext
from automa.llm.images import (
    ImageGateway,
    ImageRequest,
    OpenAICompatImageProvider,
    gemini_image_request_to_wire,
    gemini_image_response_from_wire,
)
from automa.llm.provider import ProviderError

KEY = "1xai-secret-key-123456"


def jpeg(width: int = 300, height: int = 260) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof + b"\xff\xd9"


def reply(image: bytes = b"", mime: str = "image/jpeg", **usage: Any) -> dict[str, Any]:
    meta = {"promptTokenCount": 15, "candidatesTokenCount": 1475, "totalTokenCount": 1490}
    meta.update(usage)
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": mime,
                                "data": base64.b64encode(image or jpeg()).decode(),
                            },
                            "thoughtSignature": "x" * 2000,
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": meta,
        "modelVersion": "gemini-3.1-flash-image",
    }


MODEL = {
    "family": "google",
    "capabilities": ["text", "image_out"],
    "input_per_mtok": 0.501548,
    "output_per_mtok": 60.030961,
    "image_api": "gemini",
    "image_max_tokens": 3000,
}


def config() -> AutomaConfig:
    return AutomaConfig.model_validate(
        {
            "base_url": "https://1xai.ir/v1",
            "tiers": {"small": "txt"},
            "models": {
                "txt": {
                    "family": "x",
                    "capabilities": ["text"],
                    "input_per_mtok": 1,
                    "output_per_mtok": 1,
                },
                "gemini-3.1-flash-image": MODEL,
            },
            "images": {"model": "gemini-3.1-flash-image"},
            "limits": {"project_usd": 1.0, "daily_usd": 1.0, "per_call_usd": 0.5},
        }
    )


def provider(handler: Any, base: str = "https://1xai.ir/v1") -> OpenAICompatImageProvider:
    return OpenAICompatImageProvider(base, KEY, transport=httpx.MockTransport(handler))


def req() -> ImageRequest:
    return ImageRequest(
        model="gemini-3.1-flash-image", prompt="kites", api=ImageApi.GEMINI, max_tokens=3000
    )


# --- config ------------------------------------------------------------------------


def test_gemini_model_needs_a_token_cap() -> None:
    with pytest.raises(ValidationError, match="image_max_tokens"):
        ModelInfo.model_validate({**MODEL, "image_max_tokens": None})


def test_gemini_worst_case_is_token_priced() -> None:
    info = ModelInfo.model_validate(MODEL)
    worst = info.image_worst_case_usd()
    assert worst is not None and worst >= 3000 * 60.030961 / 1_000_000


# --- wire format ---------------------------------------------------------------------


def test_request_body_asks_for_an_image() -> None:
    body = gemini_image_request_to_wire(req())
    assert body["contents"][0]["parts"][0]["text"].startswith("kites")
    assert body["generationConfig"] == {
        "responseModalities": ["IMAGE", "TEXT"],
        "maxOutputTokens": 3000,
    }


def test_reply_is_parsed_with_real_usage() -> None:
    image = jpeg()
    r = gemini_image_response_from_wire(reply(image), fallback_model="m")
    assert r.data == image and r.model == "gemini-3.1-flash-image"
    assert r.usage is not None and r.usage.input_tokens == 15 and r.usage.output_tokens == 1475


def test_hidden_tokens_count_as_output() -> None:
    r = gemini_image_response_from_wire(
        reply(candidatesTokenCount=1000, totalTokenCount=1515, thoughtsTokenCount=200),
        fallback_model="m",
    )
    assert r.usage is not None and r.usage.output_tokens == 1500  # total - prompt


@pytest.mark.parametrize(
    "data",
    [
        {"candidates": []},
        {"candidates": [{"content": {"parts": [{"text": "no image, sorry"}]}}]},
        {"promptFeedback": {"blockReason": "SAFETY"}},
    ],
)
def test_reply_without_an_image_is_an_error(data: dict[str, Any]) -> None:
    with pytest.raises(ProviderError):
        gemini_image_response_from_wire(data, fallback_model="m")


def test_non_image_mime_is_ignored() -> None:
    with pytest.raises(ProviderError, match="no inline PNG/JPEG"):
        gemini_image_response_from_wire(reply(mime="image/svg+xml"), fallback_model="m")


# --- transport -----------------------------------------------------------------------


def test_posts_to_the_native_route_with_a_bearer_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=reply())

    r = provider(handler).generate(req())
    assert r.data[:2] == b"\xff\xd8"
    (sent,) = seen
    assert (
        str(sent.url)
        == "https://1xai.ir/gemini/v1beta/models/gemini-3.1-flash-image:generateContent"
    )
    assert sent.headers["Authorization"] == f"Bearer {KEY}"
    assert "key=" not in str(sent.url)
    assert json.loads(sent.content)["generationConfig"]["maxOutputTokens"] == 3000


def test_root_without_v1_suffix() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=reply())

    provider(handler, base="https://gw.example/api").generate(req())
    assert seen == [
        "https://gw.example/api/gemini/v1beta/models/gemini-3.1-flash-image:generateContent"
    ]


@pytest.mark.parametrize("bad", ["a/b", "../x", "x?y", "x y", "x#y"])
def test_unsafe_model_ids_are_refused(bad: str) -> None:
    calls: list[int] = []
    with pytest.raises(ProviderError):
        provider(lambda r: calls.append(1) or httpx.Response(200, json=reply())).generate(
            req().model_copy(update={"model": bad})
        )
    assert calls == []


def test_error_message_is_kept_and_redacted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=[{"error": {"message": f"bad thing {KEY}", "code": 400}}])

    with pytest.raises(ProviderError) as ei:
        provider(handler).generate(req())
    assert ei.value.status == 400 and KEY not in str(ei.value)


# --- budget ----------------------------------------------------------------------------


def test_gateway_charges_the_real_tokens(tmp_path: Any) -> None:
    cfg = config()
    with CostLedger(":memory:") as ledger:
        guard = BudgetGuard(cfg.limits, ledger)
        gw = ImageGateway(cfg, provider(lambda r: httpx.Response(200, json=reply())), guard=guard)
        result = gw.generate(
            ImageRequest(model="gemini-3.1-flash-image", prompt="kites"),
            context=CallContext(project_id="p", task_id="t"),
        )
        expected = (15 * 0.501548 + 1475 * 60.030961) / 1_000_000
        assert result.cost_usd == pytest.approx(expected, rel=1e-6)
        assert guard.pending_nusd() == 0 and len(ledger.entries()) == 1
