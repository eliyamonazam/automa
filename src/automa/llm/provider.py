"""Provider interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from automa.llm.types import ChatRequest, ChatResponse


class ProviderError(Exception):
    """A provider call failed. ``retryable`` marks transient failures (429, 5xx, network)."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@runtime_checkable
class Provider(Protocol):
    """Anything that can execute one chat completion."""

    def complete(self, request: ChatRequest) -> ChatResponse: ...
