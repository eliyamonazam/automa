"""Three-layer memory: user preferences, per-project knowledge, general lessons (M7, D48).

This package never calls a model and never grants a tool, risk level or budget.
``memory_prompt`` renders memory as a bounded, data-only system-prompt section
(D50); the post-project review step is a separate roadmap item.
"""

from automa.memory.models import (
    MAX_MEMORY_TEXT_CHARS,
    MEMORY_ID_PATTERN,
    MemoryEntry,
    MemoryScope,
    MemorySource,
    clean_memory_text,
    normalize_text,
)
from automa.memory.prompt import (
    MAX_MEMORY_PROMPT_CHARS,
    check_memory_project,
    memory_prompt,
    select_memory,
)
from automa.memory.store import (
    MAX_ENTRIES_PER_LAYER,
    MAX_MEMORY_FILE_BYTES,
    MEMORY_DIRNAME,
    MemoryContext,
    MemoryFull,
    MemoryNotFound,
    MemoryStore,
    MemoryStoreError,
    validate_memory_id,
)

__all__ = [
    "MAX_ENTRIES_PER_LAYER",
    "MAX_MEMORY_PROMPT_CHARS",
    "MAX_MEMORY_FILE_BYTES",
    "MAX_MEMORY_TEXT_CHARS",
    "MEMORY_DIRNAME",
    "MEMORY_ID_PATTERN",
    "MemoryContext",
    "MemoryEntry",
    "MemoryStoreError",
    "MemoryFull",
    "MemoryNotFound",
    "MemoryScope",
    "MemorySource",
    "MemoryStore",
    "check_memory_project",
    "clean_memory_text",
    "memory_prompt",
    "normalize_text",
    "select_memory",
    "validate_memory_id",
]
