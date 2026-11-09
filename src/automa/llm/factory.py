"""Build the production provider from configuration.

Kept inside ``automa.llm`` so that no module outside this package imports a concrete
provider (enforced by ``tests/test_guard_bypass.py``). The returned provider must only
ever be wrapped in an ``LLMGateway``; it is never called directly.
"""

from __future__ import annotations

from automa.config import AutomaConfig
from automa.llm.openai_compat import OpenAICompatProvider
from automa.llm.provider import Provider


def provider_from_config(config: AutomaConfig) -> Provider:
    """OpenAI-compatible HTTP provider for ``config``. Raises ``ConfigError`` without a key."""
    return OpenAICompatProvider(config.base_url, config.require_api_key())
