"""Skills as ``SKILL.md`` files. See docs/ARCHITECTURE.md ("Skill Engine")."""

from automa.skills.models import (
    MAX_SKILL_BODY_CHARS,
    MAX_SKILL_DESCRIPTION_CHARS,
    SKILL_NAME_PATTERN,
    Skill,
)
from automa.skills.registry import (
    MAX_SKILL_FILE_BYTES,
    SKILL_FILE,
    SkillError,
    SkillRegistry,
    parse_skill,
    split_frontmatter,
)

__all__ = [
    "MAX_SKILL_BODY_CHARS",
    "MAX_SKILL_DESCRIPTION_CHARS",
    "MAX_SKILL_FILE_BYTES",
    "SKILL_FILE",
    "SKILL_NAME_PATTERN",
    "Skill",
    "SkillError",
    "SkillRegistry",
    "parse_skill",
    "split_frontmatter",
]
