"""Image generation through the gateway's OpenAI-compatible ``/images/generations`` (D100).

``ImageGateway`` is the only path for an image call, mirroring ``LLMGateway``:

* the model must be catalogued with the ``image_out`` capability **and** a
  ``per_image_usd`` price, so the BudgetGuard can bound the call before it is sent;
* the mandatory ``CallGuard`` authorizes the worst case (one image at its price)
  first, then the actual cost is recorded, or the reservation released if the
  provider failed before anything could have been billed (``maybe_billed`` failures
  are recorded at the worst case, as for chat calls, D91);
* the recorded cost is never below the configured price (a lower ``usage.cost``
  from the gateway is ignored, a higher one is taken).

One request is exactly one image; image calls are never retried automatically
(each retry would be a new paid image). In the ledger an image call has
``input_tokens = 0`` and ``output_tokens = 1`` (one image).

``FakeImageProvider`` serves tests; ``OpenAICompatImageProvider`` talks HTTP and
takes an ``httpx`` transport so tests never touch the network.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field

from automa.config import AutomaConfig, Capability, ConfigError, Tier
from automa.llm.gateway import CallContext, CallEstimate, CallGuard, LLMGateway
from automa.llm.openai_compat import _RETRYABLE_STATUS, _retry_after
from automa.llm.provider import ProviderError
from automa.llm.types import ChatResponse, LLMResult, Message, Usage

MAX_IMAGE_PROMPT_CHARS = 1000
# Decoded image bytes accepted from a provider (a 1536x1024 PNG is ~3 MiB).
MAX_IMAGE_BYTES = 8 * 1024 * 1024
# Raw HTTP body accepted (base64 is 4/3 of the image plus JSON framing).
MAX_IMAGE_RESPONSE_BYTES = MAX_IMAGE_BYTES * 4 // 3 + 64 * 1024

_GUARD_METHODS = ("authorize", "record", "release")


class ImageSize(StrEnum):
    """Sizes accepted by OpenAI-compatible image endpoints."""

    SQUARE = "1024x1024"
    PORTRAIT = "1024x1536"
    LANDSCAPE = "1536x1024"


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=MAX_IMAGE_PROMPT_CHARS)
    size: ImageSize = ImageSize.SQUARE


class ImageResponse(BaseModel):
    """One generated image as raw bytes (not yet checked to be PNG/JPEG)."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    model: str
    data: bytes = Field(min_length=1, max_length=MAX_IMAGE_BYTES)
    reported_cost_usd: float | None = Field(default=None, ge=0)


class ImageResult(BaseModel):
    """What ``ImageGateway.generate`` returns: the image plus its recorded cost."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    model: str
    data: bytes
    cost_usd: float = Field(ge=0)


@runtime_checkable
class ImageProvider(Protocol):
    """Anything that can generate one image. Raises ``ProviderError`` on failure."""

    def generate(self, request: ImageRequest) -> ImageResponse: ...


class FakeImageProvider:
    """Returns canned images (or raises) in order, for tests. Records every request."""

    def __init__(
        self,
        images: Sequence[bytes | Exception] = (),
        *,
        reported_cost_usd: float | None = None,
    ) -> None:
        self._queue = list(images)
        self._reported = reported_cost_usd
        self.requests: list[ImageRequest] = []

    def generate(self, request: ImageRequest) -> ImageResponse:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("FakeImageProvider has no more images")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return ImageResponse(model=request.model, data=item, reported_cost_usd=self._reported)


def image_request_to_wire(request: ImageRequest) -> dict[str, Any]:
    """JSON body for ``POST /images/generations`` (always one base64 image)."""
    return {
        "model": request.model,
        "prompt": request.prompt,
        "n": 1,
        "size": request.size.value,
        "response_format": "b64_json",
    }


def image_response_from_wire(data: Any, *, fallback_model: str) -> ImageResponse:
    """Parse an images response. Raises ``ProviderError`` (no body text) on bad shape."""
    try:
        items = data["data"]
        if not isinstance(items, list) or len(items) != 1:
            raise ValueError("expected exactly one image")
        b64 = items[0]["b64_json"]
        if not isinstance(b64, str) or not b64:
            raise ValueError("missing b64_json")
        if len(b64) > MAX_IMAGE_BYTES * 4 // 3 + 4:
            raise ProviderError("image larger than the accepted size")
        raw = base64.b64decode(b64, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ProviderError("image larger than the accepted size")
        usage = data.get("usage") or {}
        cost = usage.get("cost") if isinstance(usage, dict) else None
        reported: float | None = None
        if isinstance(cost, int | float) and not isinstance(cost, bool):
            reported = float(cost) if 0 <= cost < float("inf") else None
        model = data.get("model")
        return ImageResponse(
            model=model if isinstance(model, str) and model else fallback_model,
            data=raw,
            reported_cost_usd=reported,
        )
    except ProviderError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, binascii.Error) as exc:
        raise ProviderError(f"malformed image response: {type(exc).__name__}") from None


class OpenAICompatImageProvider:
    """Calls ``{base_url}/images/generations`` with a bearer key.

    The key is only in the ``Authorization`` header; the gateway's error text is
    never put in an error message (it may echo the prompt or a key). Pass
    ``transport`` (e.g. ``httpx.MockTransport``) in tests.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout_s,
            transport=transport,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatImageProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def generate(self, request: ImageRequest) -> ImageResponse:
        try:
            with self._client.stream(
                "POST", "images/generations", json=image_request_to_wire(request)
            ) as resp:
                status = resp.status_code
                retry_after = _retry_after(resp)
                body = self._read_bounded(resp) if status < 400 else b""
        except ProviderError:
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            raise ProviderError(
                f"could not connect: {type(exc).__name__}", retryable=True
            ) from None
        except httpx.TimeoutException:
            raise ProviderError("request timed out", retryable=True, maybe_billed=True) from None
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"network error: {type(exc).__name__}", retryable=True, maybe_billed=True
            ) from None
        if status >= 400:
            raise ProviderError(
                f"image endpoint returned HTTP {status}",
                status=status,
                retryable=status in _RETRYABLE_STATUS,
                retry_after_s=retry_after,
            )
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ProviderError(
                "image endpoint returned non-JSON body", maybe_billed=True
            ) from None
        if isinstance(data, dict) and data.get("error"):
            raise ProviderError("image endpoint returned an error object", status=status)
        try:
            return image_response_from_wire(data, fallback_model=request.model)
        except ProviderError as e:
            # The endpoint answered 2xx: the image may have been billed anyway.
            raise ProviderError(str(e), status=status, maybe_billed=True) from None

    @staticmethod
    def _read_bounded(resp: httpx.Response) -> bytes:
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_bytes():
            total += len(chunk)
            if total > MAX_IMAGE_RESPONSE_BYTES:
                raise ProviderError(
                    "image response larger than the accepted size", maybe_billed=True
                )
            chunks.append(chunk)
        return b"".join(chunks)


class ImageGateway:
    """Checks, prices and guards every image call (D100)."""

    def __init__(self, config: AutomaConfig, provider: ImageProvider, *, guard: CallGuard):
        if guard is None:
            raise TypeError("ImageGateway requires a CallGuard; calls are never unguarded")
        missing = [name for name in _GUARD_METHODS if not callable(getattr(guard, name, None))]
        if missing:
            raise TypeError(f"guard is not a CallGuard (missing: {', '.join(missing)})")
        self._config = config
        self._provider = provider
        self._guard = guard

    @property
    def config(self) -> AutomaConfig:
        return self._config

    def shares_guard_with(self, gateway: LLMGateway) -> bool:
        """Whether ``gateway`` spends through the very same ``CallGuard`` (D101)."""
        return isinstance(gateway, LLMGateway) and gateway._guard is self._guard

    def price_per_image(self, model: str) -> float:
        """USD per image of ``model``; ``ConfigError`` unless it is a priced image model."""
        if not isinstance(model, str) or not model:
            raise ConfigError("image model must be a non-empty model id")
        info = self._config.model_info(model)
        if info is None:
            raise ConfigError(f"image model '{model}' is not in the [models] catalog")
        if not info.has(Capability.IMAGE_OUT):
            raise ConfigError(f"model '{model}' has no 'image_out' capability")
        if info.per_image_usd is None:
            raise ConfigError(f"no per_image_usd price configured for image model '{model}'")
        return info.per_image_usd

    def estimate(self, model: str, context: CallContext, *, tier: Tier) -> CallEstimate:
        return CallEstimate(
            tier=tier,
            model=model,
            max_input_tokens=0,
            max_output_tokens=1,
            max_cost_usd=self.price_per_image(model),
            context=context,
        )

    def generate(
        self,
        request: ImageRequest,
        *,
        context: CallContext,
        tier: Tier = Tier.MEDIUM,
    ) -> ImageResult:
        """Generate one image. Raises ``ConfigError``, a budget error or ``ProviderError``."""
        estimate = self.estimate(request.model, context, tier=tier)
        self._guard.authorize(estimate)
        try:
            response = self._provider.generate(request)
        except ProviderError as e:
            if e.maybe_billed:
                self._guard.record(estimate, _image_ledger_result(estimate, request.model, None))
            else:
                self._guard.release(estimate)
            raise
        except BaseException:
            self._guard.release(estimate)
            raise
        cost = estimate.max_cost_usd
        if response.reported_cost_usd is not None:
            cost = max(cost, response.reported_cost_usd)
        self._guard.record(
            estimate,
            _image_ledger_result(estimate, response.model, response.reported_cost_usd, cost=cost),
        )
        return ImageResult(model=request.model, data=response.data, cost_usd=cost)


def _image_ledger_result(
    estimate: CallEstimate,
    response_model: str,
    reported_cost_usd: float | None,
    *,
    cost: float | None = None,
) -> LLMResult:
    """The ledger row of an image call: 0 input tokens, 1 output unit (one image)."""
    response = ChatResponse(
        model=response_model,
        message=Message.assistant(""),
        usage=Usage(input_tokens=0, output_tokens=1, reported_cost_usd=reported_cost_usd),
    )
    return LLMResult(
        tier=estimate.tier,
        response=response,
        cost_usd=estimate.max_cost_usd if cost is None else cost,
    )
