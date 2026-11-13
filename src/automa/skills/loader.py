"""Where skills come from: the built-in set and the workspace's ``skills/`` folder.

Both sources are loaded fail-closed (see ``SkillRegistry.from_dir``). A workspace
skill may not reuse a built-in skill's name: shadowing would silently change what
a built-in role is told, so a clash is an error the user has to resolve.
"""

from __future__ import annotations

from pathlib import Path

from automa.skills.registry import SkillError, SkillRegistry

BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent / "builtin"
WORKSPACE_SKILLS_DIRNAME = "skills"


def workspace_skills_dir(workspace: str | Path) -> Path:
    """``<workspace>/skills``: where the user's own skills live."""
    return Path(workspace) / WORKSPACE_SKILLS_DIRNAME


def builtin_skills(directory: Path = BUILTIN_SKILLS_DIR) -> SkillRegistry:
    """The skills shipped with Automa (the directory must exist)."""
    return SkillRegistry.from_dir(directory)


def workspace_skills(workspace: str | Path) -> SkillRegistry:
    """The user's skills; a missing ``skills/`` folder is an empty set (a symlink is an error)."""
    try:
        return SkillRegistry.from_optional_dir(workspace_skills_dir(workspace))
    except SkillError as e:
        raise SkillError(f"workspace skills: {e}") from None


def merge_skills(builtin: SkillRegistry, workspace: SkillRegistry) -> SkillRegistry:
    """One set from both sources; a name in both is an error, never an override."""
    clash = sorted(builtin.names() & workspace.names())
    if clash:
        raise SkillError(
            f"workspace skills {clash} have the same name as built-in skills; "
            "rename the folders under skills/"
        )
    return SkillRegistry([*builtin, *workspace])


def load_skills(workspace: str | Path, *, builtin_dir: Path = BUILTIN_SKILLS_DIR) -> SkillRegistry:
    """All skills available to roles in ``workspace``: built-in plus ``<workspace>/skills/``."""
    return merge_skills(builtin_skills(builtin_dir), workspace_skills(workspace))
