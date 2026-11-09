"""HTTP provider for any OpenAI-compatible Chat Completions endpoint."""

from __future__ import annotations

import json
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
        "temperature": request.temperature,
        "stream": False,
    }
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
        usage = Usage(
            input_tokens=int(raw_usage.get("prompt_tokens") or 0),
            output_tokens=int(raw_usage.get("completion_tokens") or 0),
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
        except httpx.TimeoutException:
            raise ProviderError("request timed out", retryable=True) from None
        except httpx.HTTPError as exc:
            raise ProviderError(f"network error: {type(exc).__name__}", retryable=True) from None
        if resp.status_code >= 400:
            raise ProviderError(
                f"provider returned HTTP {resp.status_code}",
                status=resp.status_code,
                retryable=resp.status_code in _RETRYABLE_STATUS,
            )
        try:
            data = resp.json()
        except json.JSONDecodeError:
            raise ProviderError("provider returned non-JSON body") from None
        if isinstance(data, dict) and data.get("error"):
            # Some gateways return 200 with an error object.
            raise ProviderError("provider returned an error object", status=resp.status_code)
        return response_from_wire(data, fallback_model=request.model)
