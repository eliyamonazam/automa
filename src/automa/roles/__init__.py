"""Worker roles as data files. See docs/ARCHITECTURE.md ("Worker Agents و نقش‌ها")."""

from automa.roles.models import (
    MAX_INSTRUCTIONS_CHARS,
    MAX_ROLE_SKILLS,
    MAX_ROLE_STEPS,
    ROLE_SCHEMA_VERSION,
    Role,
)
from automa.roles.registry import (
    BUILTIN_ROLES_DIR,
    MAX_ROLE_FILE_BYTES,
    RoleError,
    RoleRegistry,
    parse_role,
)

__all__ = [
    "BUILTIN_ROLES_DIR",
    "MAX_INSTRUCTIONS_CHARS",
    "MAX_ROLE_FILE_BYTES",
    "MAX_ROLE_SKILLS",
    "MAX_ROLE_STEPS",
    "ROLE_SCHEMA_VERSION",
    "Role",
    "RoleError",
    "RoleRegistry",
    "parse_role",
]
