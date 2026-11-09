"""Model gateway: one entry point for every model call (see docs/ARCHITECTURE.md)."""

from automa.llm.fake import FakeProvider
from automa.llm.gateway import (
    CallContext,
    CallEstimate,
    CallGuard,
    LLMGateway,
    compute_cost,
    estimate_max_cost,
)
from automa.llm.openai_compat import OpenAICompatProvider
from automa.llm.provider import Provider, ProviderError
from automa.llm.types import (
    ChatRequest,
    ChatResponse,
    FinishReason,
    LLMResult,
    Message,
    Role,
    ToolCall,
    ToolSpec,
    Usage,
)

__all__ = [
    "CallContext",
    "CallEstimate",
    "CallGuard",
    "ChatRequest",
    "ChatResponse",
    "FakeProvider",
    "FinishReason",
    "LLMGateway",
    "LLMResult",
    "Message",
    "OpenAICompatProvider",
    "Provider",
    "ProviderError",
    "Role",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "compute_cost",
    "estimate_max_cost",
]
