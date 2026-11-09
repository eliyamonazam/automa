"""``LLMGateway``: the single path through which every model call goes.

Responsibilities:
- resolve a capability tier to a concrete model (roles never name models);
- bound every call (``max_tokens`` is mandatory) and estimate its worst-case cost;
- ask the optional ``CallGuard`` for permission before sending, and report the
  actual cost after the call (the budget layer plugs in here in M1);
- compute cost conservatively from configured prices.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from automa.config import AutomaConfig, ModelPrice, Tier
from automa.llm.provider import Provider
from automa.llm.types import ChatRequest, LLMResult, Message, ToolSpec, Usage

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
    """Budget hook. ``authorize`` must raise to block a call; it is never skipped."""

    def authorize(self, estimate: CallEstimate) -> None: ...

    def record(self, estimate: CallEstimate, result: LLMResult) -> None: ...


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


class LLMGateway:
    """Resolves tiers, bounds and prices every call, and enforces the guard hook."""

    def __init__(
        self,
        config: AutomaConfig,
        provider: Provider,
        *,
        guard: CallGuard | None = None,
    ):
        self._config = config
        self._provider = provider
        self._guard = guard

    @classmethod
    def from_config(cls, config: AutomaConfig, *, guard: CallGuard | None = None) -> LLMGateway:
        """Build a gateway that talks to the configured OpenAI-compatible endpoint."""
        from automa.llm.openai_compat import OpenAICompatProvider

        provider = OpenAICompatProvider(config.base_url, config.require_api_key())
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
    ) -> ChatRequest:
        model = self._config.model_for(tier)
        return ChatRequest(
            model=model,
            messages=tuple(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tuple(tools),
        )

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
    ) -> LLMResult:
        """Run one chat completion at ``tier``. Raises if the guard refuses."""
        request = self.build_request(
            messages, tier=tier, max_tokens=max_tokens, temperature=temperature, tools=tools
        )
        estimate = self.estimate(request, tier, context or CallContext())
        if self._guard is not None:
            self._guard.authorize(estimate)
        response = self._provider.complete(request)
        cost = self._actual_cost(response.usage, request.model, estimate)
        result = LLMResult(tier=tier, response=response, cost_usd=cost)
        if self._guard is not None:
            self._guard.record(estimate, result)
        return result

    def _actual_cost(self, usage: Usage, model: str, estimate: CallEstimate) -> float:
        # Always price against the *requested* model: a gateway may report a
        # different model id we have no price for, and we must not under-count.
        computed = compute_cost(usage, self._config.price_for(model))
        if usage.input_tokens == 0 and usage.output_tokens == 0 and usage.reported_cost_usd is None:
            # No usage reported at all: assume the worst case rather than zero.
            return estimate.max_cost_usd
        if usage.reported_cost_usd is not None:
            return max(computed, usage.reported_cost_usd)
        return computed
