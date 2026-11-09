"""Provider-neutral data types for chat completions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from automa.config import Tier


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(_Strict):
    """A tool invocation requested by the model. ``arguments`` is the raw JSON string."""

    id: str
    name: str
    arguments: str = "{}"


class Message(_Strict):
    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Message:
        if self.role is Role.TOOL and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.tool_calls and self.role is not Role.ASSISTANT:
            raise ValueError("only assistant messages may carry tool_calls")
        if self.content is None and not self.tool_calls:
            raise ValueError("message needs content or tool_calls")
        return self

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Message:
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)


class ToolSpec(_Strict):
    """A tool the model may call. ``parameters`` is a JSON Schema object."""

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})


class ChatRequest(_Strict):
    """A fully resolved request sent to a provider (concrete model, bounded output)."""

    model: str = Field(min_length=1)
    messages: tuple[Message, ...] = Field(min_length=1)
    max_tokens: int = Field(gt=0, le=200_000)
    temperature: float = Field(default=0.2, ge=0, le=2)
    tools: tuple[ToolSpec, ...] = ()


class Usage(_Strict):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    # Cost reported by the gateway itself (e.g. OpenRouter ``usage.cost``), if any.
    reported_cost_usd: float | None = Field(default=None, ge=0)


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


class ChatResponse(_Strict):
    """What a provider returns."""

    model: str
    message: Message
    usage: Usage
    finish_reason: FinishReason = FinishReason.STOP


class LLMResult(_Strict):
    """What the gateway returns: the provider response plus resolved tier and cost."""

    tier: Tier
    response: ChatResponse
    cost_usd: float = Field(ge=0)

    @property
    def text(self) -> str:
        return self.response.message.content or ""

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.response.message.tool_calls
