"""HTTP provider for any OpenAI-compatible Chat Completions endpoint."""

from __future__ import annotations

import json
import math
import re
from typing import Any

import httpx

from automa.llm.provider import ProviderError
from automa.llm.types import (
    ChatRequest,
    ChatResponse,
    FinishReason,
    Message,
    Role,
    ToolCall,
    Usage,
)

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_FINISH = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


def _message_to_wire(msg: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in msg.tool_calls
        ]
    if msg.tool_call_id is not None:
        out["tool_call_id"] = msg.tool_call_id
    return out


def request_to_wire(request: ChatRequest) -> dict[str, Any]:
    """Build the JSON body for ``POST /chat/completions``."""
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [_message_to_wire(m) for m in request.messages],
        "max_tokens": request.max_tokens,
        "stream": False,
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in request.tools
        ]
    return body


def response_from_wire(data: Any, *, fallback_model: str) -> ChatResponse:
    """Parse a Chat Completions response body. Raises ``ProviderError`` on bad shape."""
    try:
        choice = data["choices"][0]
        raw_msg = choice["message"]
        tool_calls = tuple(
            ToolCall(
                id=str(tc["id"]),
                name=str(tc["function"]["name"]),
                arguments=tc["function"].get("arguments") or "{}",
            )
            for tc in (raw_msg.get("tool_calls") or ())
        )
        content = raw_msg.get("content")
        if content is None and not tool_calls:
            content = ""
        message = Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)
        raw_usage = data.get("usage") or {}
        cost = raw_usage.get("cost")
        input_tokens = int(raw_usage.get("prompt_tokens") or 0)
        output_tokens = int(raw_usage.get("completion_tokens") or 0)
        total = raw_usage.get("total_tokens")
        if isinstance(total, int | float) and not isinstance(total, bool):
            # Some gateways leave hidden "thinking" tokens out of completion_tokens but
            # count (and bill) them in total_tokens (seen with Gemini on 1xAi: prompt 4,
            # completion 0, total 21). Count the gap as output, the dearer side, so
            # the BudgetGuard never under-counts (D107).
            output_tokens = max(output_tokens, int(total) - input_tokens)
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reported_cost_usd=float(cost) if cost is not None else None,
        )
        finish = _FINISH.get(str(choice.get("finish_reason")), FinishReason.OTHER)
        return ChatResponse(
            model=str(data.get("model") or fallback_model),
            message=message,
            usage=usage,
            finish_reason=finish,
        )
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise ProviderError(f"malformed completion response: {type(exc).__name__}") from None


class OpenAICompatProvider:
    """Calls ``{base_url}/chat/completions`` with a bearer key.

    The API key is only placed in the ``Authorization`` header and never appears in
    error messages. Pass ``transport`` (e.g. ``httpx.MockTransport``) in tests.
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
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def complete(self, request: ChatRequest) -> ChatResponse:
        try:
            resp = self._client.post("chat/completions", json=request_to_wire(request))
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            # Never reached the provider: nothing can have been billed.
            raise ProviderError(
                f"could not connect: {type(exc).__name__}", retryable=True
            ) from None
        except httpx.TimeoutException:
            # Sent, but no answer in time: the provider may still bill it (D91).
            raise ProviderError("request timed out", retryable=True, maybe_billed=True) from None
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"network error: {type(exc).__name__}", retryable=True, maybe_billed=True
            ) from None
        if resp.status_code >= 400:
            suffix = self._error_suffix(resp)
            # "max_tokens ... limit was reached": the model produced (and the gateway
            # probably billed) a full reply before failing, so count it (D117).
            ran_out = resp.status_code == 400 and "max_tokens" in suffix.lower()
            raise ProviderError(
                f"provider returned HTTP {resp.status_code}{suffix}",
                status=resp.status_code,
                retryable=resp.status_code in _RETRYABLE_STATUS,
                maybe_billed=ran_out,
                retry_after_s=_retry_after(resp),
            )
        try:
            data = resp.json()
        except json.JSONDecodeError:
            raise ProviderError("provider returned non-JSON body") from None
        if isinstance(data, dict) and data.get("error"):
            # Some gateways return 200 with an error object.
            raise ProviderError(
                f"provider returned an error object{self._error_suffix(resp)}",
                status=resp.status_code,
            )
        return response_from_wire(data, fallback_model=request.model)

    def _error_suffix(self, resp: httpx.Response) -> str:
        """``: <gateway message>`` (redacted, single line, bounded) or ``""``."""
        return error_suffix(resp, self._client.headers.get("Authorization", ""))


def error_suffix(resp: httpx.Response, authorization: str = "") -> str:
    """``: <gateway message>`` from an error response: the API key (from the
    ``Authorization`` header value) and anything token-like redacted, one line,
    at most ``_MAX_ERROR_MESSAGE`` characters; ``""`` when there is no message."""
    message = _error_message(resp)
    if not message:
        return ""
    key = authorization.removeprefix("Bearer ")
    if key:
        message = message.replace(key, "[redacted]")
    message = _SECRETISH.sub("[redacted]", message)
    message = " ".join(message.split())  # no newlines / control layout
    message = "".join(ch for ch in message if ch.isprintable())
    if len(message) > _MAX_ERROR_MESSAGE:
        message = message[:_MAX_ERROR_MESSAGE] + " [...]"
    return f": {message}" if message else ""


_MAX_ERROR_MESSAGE = 300
# Anything that looks like a bearer token or API key in a gateway's error text.
_SECRETISH = re.compile(r"(?i)(bearer\s+\S+|\b(?:sk|1xai|key)[-_][A-Za-z0-9_\-]{4,})")


def _error_message(resp: httpx.Response) -> str:
    """Best-effort human message from an error body (OpenAI-style or plain text)."""
    try:
        data = resp.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return resp.text[: _MAX_ERROR_MESSAGE * 2] if resp.text else ""
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            for field in ("message", "detail", "error", "code", "type"):
                value = err.get(field)
                if isinstance(value, str) and value.strip():
                    return value
        elif isinstance(err, str):
            return err
        detail = data.get("detail") or data.get("message")
        if isinstance(detail, str):
            return detail
    return ""


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None
