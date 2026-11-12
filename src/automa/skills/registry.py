"""Loading skills from a directory of ``<name>/SKILL.md`` folders.

The frontmatter is a deliberately small subset of YAML (one ``key: value``
per line, optionally quoted; no nesting, lists or multi-line values), parsed
without a YAML dependency. Anything outside the subset is an error, not a
guess. Loading fails closed like ``RoleRegistry``: a malformed, oversized,
symlinked or inconsistent skill rejects the whole registry.

``allowed-tools`` (part of some ``SKILL.md`` dialects) is rejected on purpose:
in Automa a skill never grants tools; permissions come from the role only.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import ValidationError

from automa.skills.models import Skill

SKILL_FILE = "SKILL.md"
MAX_SKILL_FILE_BYTES = 64 * 1024
_FIELDS = {"name": "name", "description": "description", "license": "license"}
_PERMISSION_KEYS = {"allowed-tools", "allowed_tools", "tools"}


class SkillError(Exception):
    """A skill file or the skill set is invalid."""


def _read_skill_file(path: Path, source: str) -> str:
    # O_NONBLOCK: opening a FIFO must not hang; it is then rejected as non-regular.
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise SkillError(f"{source}: missing {SKILL_FILE}") from None
    except OSError as e:
        raise SkillError(f"{source}: cannot open skill file ({e.strerror})") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SkillError(f"{source}: not a regular file")
        if st.st_size > MAX_SKILL_FILE_BYTES:
            raise SkillError(f"{source}: skill file larger than {MAX_SKILL_FILE_BYTES} bytes")
        with os.fdopen(fd, "rb", closefd=True) as f:
            fd = -1
            data = f.read(MAX_SKILL_FILE_BYTES + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > MAX_SKILL_FILE_BYTES:
        raise SkillError(f"{source}: skill file larger than {MAX_SKILL_FILE_BYTES} bytes")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise SkillError(f"{source}: skill file is not valid UTF-8") from None


def _unquote(value: str, source: str, lineno: int) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        inner = value[1:-1]
        quote = value[0]
        if quote == "'":
            if "'" in inner.replace("''", ""):
                raise SkillError(f"{source}: line {lineno}: unescaped quote in value")
            return inner.replace("''", "'")
        if '"' in inner or "\\" in inner:
            raise SkillError(f"{source}: line {lineno}: escapes in double quotes are not supported")
        return inner
    if value[:1] in "\"'":
        raise SkillError(f"{source}: line {lineno}: unterminated quoted value")
    if value[:1] in "[{&*!|>%@`" or value.startswith("- "):
        raise SkillError(f"{source}: line {lineno}: only plain single-line values are supported")
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def split_frontmatter(text: str, *, source: str = "<skill>") -> tuple[dict[str, str], str]:
    """Split ``SKILL.md`` text into (frontmatter fields, body). Strict subset of YAML."""
    if text.startswith("﻿"):
        text = text[1:]
    lines = text.split("\n")
    if not lines or lines[0].rstrip("\r") != "---":
        raise SkillError(f"{source}: must start with a '---' frontmatter line")
    fields: dict[str, str] = {}
    end: int | None = None
    for i in range(1, len(lines)):
        line = lines[i].rstrip("\r")
        lineno = i + 1
        if line.rstrip() == "---":
            end = i
            break
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] in " \t":
            raise SkillError(
                f"{source}: line {lineno}: nested or indented values are not supported"
            )
        key, sep, raw = line.partition(":")
        key = key.strip()
        if not sep or not key:
            raise SkillError(f"{source}: line {lineno}: expected 'key: value'")
        if key in _PERMISSION_KEYS:
            raise SkillError(
                f"{source}: line {lineno}: {key!r} is not allowed — a skill never grants tools; "
                "tools come from the role"
            )
        if key not in _FIELDS:
            raise SkillError(f"{source}: line {lineno}: unknown field {key!r}")
        if key in fields:
            raise SkillError(f"{source}: line {lineno}: duplicate field {key!r}")
        value = raw.strip()
        if not value:
            raise SkillError(f"{source}: line {lineno}: empty value for {key!r}")
        fields[key] = _unquote(value, source, lineno)
    if end is None:
        raise SkillError(f"{source}: frontmatter is not closed with '---'")
    body = "\n".join(lines[end + 1 :]).replace("\r\n", "\n").strip()
    return fields, body


def parse_skill(text: str, *, expected_name: str, source: str = "<skill>") -> Skill:
    """Parse one ``SKILL.md``; ``name`` must equal ``expected_name`` (the directory name)."""
    fields, body = split_frontmatter(text, source=source)
    data: dict[str, str] = {_FIELDS[k]: v for k, v in fields.items()}
    data["body"] = body
    try:
        skill = Skill.model_validate(data)
    except ValidationError as e:
        lines = [
            f"- {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in e.errors()[:20]
        ]
        raise SkillError(f"{source}: invalid skill\n" + "\n".join(lines)) from None
    if skill.name != expected_name:
        raise SkillError(
            f"{source}: name {skill.name!r} does not match directory name {expected_name!r}"
        )
    return skill


class SkillRegistry:
    """An immutable set of skills, keyed by name. May be empty."""

    def __init__(self, skills: Iterable[Skill] = ()):
        by_name: dict[str, Skill] = {}
        for skill in skills:
            if not isinstance(skill, Skill):
                raise TypeError("SkillRegistry accepts only Skill instances")
            if skill.name in by_name:
                raise SkillError(f"duplicate skill {skill.name!r}")
            by_name[skill.name] = skill
        self._skills = dict(sorted(by_name.items()))

    @classmethod
    def from_dir(cls, directory: Path) -> SkillRegistry:
        """Load every ``<name>/SKILL.md`` under ``directory``.

        Hidden entries and plain files (e.g. a README) at the top level are
        ignored; symlinks and other special entries are errors.
        """
        directory = Path(directory)
        if directory.is_symlink() or not directory.is_dir():
            raise SkillError(f"skills directory not found: {directory}")
        skills: list[Skill] = []
        for entry in sorted(os.scandir(directory), key=lambda e: e.name):
            if entry.name.startswith("."):
                continue
            if entry.is_symlink():
                raise SkillError(f"{entry.name}: symlinks are not allowed in the skills directory")
            if entry.is_file(follow_symlinks=False):
                continue
            if not entry.is_dir(follow_symlinks=False):
                raise SkillError(f"{entry.name}: not a directory")
            source = f"{entry.name}/{SKILL_FILE}"
            text = _read_skill_file(directory / entry.name / SKILL_FILE, source)
            skills.append(parse_skill(text, expected_name=entry.name, source=source))
        return cls(skills)

    @classmethod
    def from_optional_dir(cls, directory: Path) -> SkillRegistry:
        """Like ``from_dir``, but a missing directory is an empty registry (a symlink is not)."""
        directory = Path(directory)
        if not directory.is_symlink() and not directory.exists():
            return cls()
        return cls.from_dir(directory)

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError:
            raise SkillError(f"unknown skill {name!r}") from None

    def names(self) -> frozenset[str]:
        return frozenset(self._skills)

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)
