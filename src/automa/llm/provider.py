"""Provider interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from automa.llm.types import ChatRequest, ChatResponse


class ProviderError(Exception):
    """A provider call failed.

    ``retryable`` marks transient failures (429, 5xx, network). ``maybe_billed`` marks
    failures after the request was sent where the provider may still have charged
    for it (e.g. a read timeout): the gateway then records the worst-case cost
    instead of releasing the reservation (D91). ``retry_after_s`` carries a
    gateway ``Retry-After`` hint.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        maybe_billed: bool = False,
        retry_after_s: float | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.maybe_billed = maybe_billed
        self.retry_after_s = retry_after_s


@runtime_checkable
class Provider(Protocol):
    """Anything that can execute one chat completion."""

    def complete(self, request: ChatRequest) -> ChatResponse: ...
