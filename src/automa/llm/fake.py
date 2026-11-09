"""Deterministic in-memory provider for tests. Never touches the network."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable

from automa.llm.provider import ProviderError
from automa.llm.types import ChatRequest, ChatResponse, FinishReason, Message, Usage

FakeScript = str | ChatResponse | BaseException | Callable[[ChatRequest], ChatResponse | str]


def approx_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token, at least 1 for non-empty text)."""
    return (len(text) + 3) // 4


def _request_tokens(request: ChatRequest) -> int:
    return sum(approx_tokens(m.content or "") for m in request.messages)


class FakeProvider:
    """Replays a script of responses in order and records every request.

    Script items:
    - ``str``: an assistant reply; usage is approximated from text length.
    - ``ChatResponse``: returned as-is.
    - ``BaseException``: raised (e.g. ``ProviderError`` or ``KeyboardInterrupt``).
    - callable: called with the request; may return a ``str`` or ``ChatResponse``.

    Running past the end of the script raises ``ProviderError`` so a test never
    silently loops.
    """

    def __init__(self, script: Iterable[FakeScript] = (), *, default: str | None = None):
        self._script: deque[FakeScript] = deque(script)
        self._default = default
        self.requests: list[ChatRequest] = []

    def push(self, *items: FakeScript) -> None:
        self._script.extend(items)

    @property
    def remaining(self) -> int:
        return len(self._script)

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self._script:
            item = self._script.popleft()
        elif self._default is not None:
            item = self._default
        else:
            raise ProviderError("FakeProvider script exhausted")
        if isinstance(item, BaseException):
            raise item
        if callable(item) and not isinstance(item, ChatResponse):
            item = item(request)
        if isinstance(item, ChatResponse):
            return item
        return ChatResponse(
            model=request.model,
            message=Message.assistant(item),
            usage=Usage(input_tokens=_request_tokens(request), output_tokens=approx_tokens(item)),
            finish_reason=FinishReason.STOP,
        )
