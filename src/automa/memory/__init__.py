"""Three-layer memory: user preferences, per-project knowledge, general lessons (M7, D48).

This package is storage only: it never calls a model and never grants a tool,
risk level or budget. Using memory in prompts and the post-project review step
are separate roadmap items.
"""

from automa.memory.models import (
    MAX_MEMORY_TEXT_CHARS,
    MEMORY_ID_PATTERN,
    MemoryEntry,
    MemoryScope,
    MemorySource,
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
    "validate_memory_id",
]
