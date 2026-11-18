"""``LLMGateway``: the single path through which every model call goes.

Responsibilities:
- resolve a capability tier to a concrete model (roles never name models), or
  accept an explicitly requested, priced model (the cross-family verifier, D96);
- bound every call (``max_tokens`` is mandatory) and estimate its worst-case cost;
- ask the mandatory ``CallGuard`` (``automa.budget.BudgetGuard`` in production)
  for permission before sending, report the actual cost after the call, and
  release the reservation if the provider fails;
- compute cost conservatively from configured prices.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from automa.config import AutomaConfig, Capability, ConfigError, ModelPrice, Tier
from automa.llm.provider import Provider, ProviderError
from automa.llm.types import (
    ChatRequest,
    ChatResponse,
    FinishReason,
    LLMResult,
    Message,
    ToolSpec,
    Usage,
)

# Indirection so tests can make retry waits instant.
_sleep = time.sleep


# Output cap used when a model ran out of room (D117): doubled from the caller's
# value at most twice, never above this.
MAX_GROWN_OUTPUT_TOKENS = 16_000


def _hit_output_limit(e: ProviderError) -> bool:
    """A 400 saying the reply did not fit in ``max_tokens`` (seen live on 1xAi with
    Claude: "Could not finish the message because max_tokens or model output limit
    was reached. Please try again with higher max_tokens.")."""
    text = str(e).lower()
    return (
        e.status == 400
        and "max_tokens" in text
        and ("limit was reached" in text or "higher max_tokens" in text)
    )


def _rejects_temperature(e: ProviderError) -> bool:
    """A 400 whose message blames ``temperature`` (e.g. "`temperature` is deprecated
    for this model", seen live with gpt-5.5 on 1xAi)."""
    return e.status == 400 and "temperature" in str(e).lower() and not e.maybe_billed


def _unanswered(request: ChatRequest, tier: Tier, estimate: CallEstimate) -> LLMResult:
    """A placeholder result for a sent-but-unanswered call, charged at the worst case."""
    response = ChatResponse(
        model=request.model,
        message=Message.assistant(""),
        usage=Usage(input_tokens=0, output_tokens=0),
        finish_reason=FinishReason.STOP,
    )
    return LLMResult(tier=tier, response=response, cost_usd=estimate.max_cost_usd)


# Per-message framing overhead in tokens, and a deliberately pessimistic
# characters-per-token ratio so the pre-call estimate errs on the high side.
_MESSAGE_OVERHEAD_TOKENS = 8
_CHARS_PER_TOKEN_PESSIMISTIC = 2


class CallContext(BaseModel):
    """Who is spending: used by the ledger and per-project caps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str | None = None
    task_id: str | None = None


class CallEstimate(BaseModel):
    """Pre-call worst-case estimate handed to the guard."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    tier: Tier
    model: str
    max_input_tokens: int = Field(ge=0)
    max_output_tokens: int = Field(gt=0)
    max_cost_usd: float = Field(ge=0)
    context: CallContext


class CallGuard(Protocol):
    """Budget hook. ``authorize`` must raise to block a call; it is never skipped.

    Exactly one of ``record`` (call completed) or ``release`` (call failed before
    returning usage) follows every successful ``authorize``.
    """

    def authorize(self, estimate: CallEstimate) -> None: ...

    def record(self, estimate: CallEstimate, result: LLMResult) -> object: ...

    def release(self, estimate: CallEstimate) -> None: ...


def compute_cost(usage: Usage, price: ModelPrice) -> float:
    """Cost in USD of ``usage`` at ``price`` (per million tokens)."""
    return (
        usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000


def _estimate_input_tokens(request: ChatRequest) -> int:
    chars = 0
    for m in request.messages:
        chars += len(m.content or "")
        for tc in m.tool_calls:
            chars += len(tc.name) + len(tc.arguments)
    for t in request.tools:
        chars += len(t.name) + len(t.description) + len(json.dumps(t.parameters))
    return (
        math.ceil(chars / _CHARS_PER_TOKEN_PESSIMISTIC)
        + _MESSAGE_OVERHEAD_TOKENS * len(request.messages)
        + _MESSAGE_OVERHEAD_TOKENS * len(request.tools)
    )


def estimate_max_cost(request: ChatRequest, price: ModelPrice) -> tuple[int, float]:
    """Pessimistic (input_tokens, cost_usd) upper estimate for ``request``."""
    input_tokens = _estimate_input_tokens(request)
    usage = Usage(input_tokens=input_tokens, output_tokens=request.max_tokens)
    return input_tokens, compute_cost(usage, price)


_GUARD_METHODS = ("authorize", "record", "release")


class LLMGateway:
    """Resolves tiers, bounds and prices every call, and enforces the guard hook."""

    def __init__(
        self,
        config: AutomaConfig,
        provider: Provider,
        *,
        guard: CallGuard,
    ):
        if guard is None:
            raise TypeError("LLMGateway requires a CallGuard; calls are never unguarded")
        missing = [name for name in _GUARD_METHODS if not callable(getattr(guard, name, None))]
        if missing:
            # Fail at construction, not at the first call: e.g. a CostLedger (which has
            # ``record`` but no ``authorize``) must never be usable as a guard.
            raise TypeError(f"guard is not a CallGuard (missing: {', '.join(missing)})")
        self._config = config
        self._provider = provider
        self._guard = guard
        # Models that rejected the temperature parameter (HTTP 400); later calls to
        # them omit it (D111).
        self._no_temperature: set[str] = set()

    @classmethod
    def from_config(cls, config: AutomaConfig, *, guard: CallGuard) -> LLMGateway:
        """Build a gateway that talks to the configured OpenAI-compatible endpoint."""
        from automa.llm.openai_compat import OpenAICompatProvider

        provider = OpenAICompatProvider(
            config.base_url, config.require_api_key(), timeout_s=config.provider.timeout_s
        )
        return cls(config, provider, guard=guard)

    @property
    def config(self) -> AutomaConfig:
        return self._config

    def build_request(
        self,
        messages: Sequence[Message],
        *,
        tier: Tier,
        max_tokens: int,
        temperature: float = 0.2,
        tools: Sequence[ToolSpec] = (),
        model: str | None = None,
    ) -> ChatRequest:
        model = self._config.model_for(tier) if model is None else self.check_model(model)
        return ChatRequest(
            model=model,
            messages=tuple(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tuple(tools),
        )

    def check_model(self, model: str) -> str:
        """Validate an explicitly requested model (D96) and return it.

        The model must have a configured price (so the BudgetGuard can bound the
        call) and, if it is catalogued, the ``text`` capability. Raises ``ConfigError``
        before anything is authorized or sent.
        """
        if not isinstance(model, str) or not model:
            raise ConfigError("model must be a non-empty model id")
        self._config.price_for(model)  # raises ConfigError if unpriced
        info = self._config.model_info(model)
        if info is not None and not info.has(Capability.TEXT):
            raise ConfigError(f"model '{model}' has no 'text' capability")
        return model

    def estimate(self, request: ChatRequest, tier: Tier, context: CallContext) -> CallEstimate:
        price = self._config.price_for(request.model)
        input_tokens, cost = estimate_max_cost(request, price)
        return CallEstimate(
            tier=tier,
            model=request.model,
            max_input_tokens=input_tokens,
            max_output_tokens=request.max_tokens,
            max_cost_usd=cost,
            context=context,
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tier: Tier,
        max_tokens: int,
        temperature: float = 0.2,
        tools: Sequence[ToolSpec] = (),
        context: CallContext | None = None,
        model: str | None = None,
    ) -> LLMResult:
        """Run one chat completion at ``tier``. Raises if the guard refuses.

        ``model`` overrides the tier's model with a specific priced model (D96); the
        tier is still recorded, and the call is guarded exactly the same way.

        Transient provider failures are retried per ``config.provider`` (D91). Every
        attempt is a separate call: authorized by the guard first (so a retry can
        never exceed a cap) and then recorded or released.
        """
        request = self.build_request(
            messages,
            tier=tier,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            model=model,
        )
        if request.model in self._no_temperature:
            request = request.model_copy(update={"temperature": None})
        estimate = self.estimate(request, tier, context or CallContext())
        policy = self._config.provider
        retry = 0
        grown = 0
        while True:
            try:
                return self._attempt(request, tier, estimate)
            except ProviderError as e:
                if (
                    _hit_output_limit(e)
                    and grown < 2
                    and request.max_tokens < MAX_GROWN_OUTPUT_TOKENS
                ):
                    # The reply needed more room (e.g. a whole report in one
                    # write_file call). Retry with a larger cap; the new worst case
                    # is authorized by the guard like any call (D117).
                    grown += 1
                    request = request.model_copy(
                        update={"max_tokens": min(request.max_tokens * 2, MAX_GROWN_OUTPUT_TOKENS)}
                    )
                    estimate = self.estimate(request, tier, estimate.context)
                    continue
                if _rejects_temperature(e) and request.temperature is not None:
                    # One more guarded attempt without the parameter; remembered for
                    # this model so it is never sent again (D111).
                    self._no_temperature.add(request.model)
                    request = request.model_copy(update={"temperature": None})
                    continue
                if not e.retryable or retry >= policy.max_retries:
                    raise
                retry += 1
                _sleep(policy.delay(retry, e.retry_after_s))

    def _attempt(self, request: ChatRequest, tier: Tier, estimate: CallEstimate) -> LLMResult:
        self._guard.authorize(estimate)
        try:
            response = self._provider.complete(request)
        except ProviderError as e:
            if e.maybe_billed:
                # The request was sent and may have been charged: count the worst
                # case rather than nothing, so the cap can never be undercut (D91).
                self._guard.record(estimate, _unanswered(request, tier, estimate))
            else:
                self._guard.release(estimate)
            raise
        except BaseException:
            self._guard.release(estimate)
            raise
        try:
            cost = self._actual_cost(response.usage, request.model, estimate)
            result = LLMResult(tier=tier, response=response, cost_usd=cost)
        except Exception:
            # The provider answered, so money was (probably) spent. Never leave it
            # unrecorded: charge the worst case, then surface the original error.
            fallback = LLMResult(tier=tier, response=response, cost_usd=estimate.max_cost_usd)
            self._guard.record(estimate, fallback)
            raise
        self._guard.record(estimate, result)
        return result

    def _actual_cost(self, usage: Usage, model: str, estimate: CallEstimate) -> float:
        # Always price against the *requested* model: a gateway may report a
        # different model id we have no price for, and we must not under-count.
        price = self._config.price_for(model)
        if usage.input_tokens == 0 and usage.output_tokens == 0:
            # No token usage reported: assume the worst case rather than zero, even if
            # the gateway claims a (lower) cost. A real completion always has input.
            computed = estimate.max_cost_usd
        elif usage.input_tokens == 0:
            # Output reported but input missing: charge the estimated maximum input.
            computed = compute_cost(
                Usage(input_tokens=estimate.max_input_tokens, output_tokens=usage.output_tokens),
                price,
            )
        else:
            computed = compute_cost(usage, price)
        if usage.reported_cost_usd is not None:
            return max(computed, usage.reported_cost_usd)
        return computed
