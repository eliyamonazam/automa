"""Loading roles from a directory of ``<name>.toml`` files.

Loading fails closed: a malformed, oversized, symlinked or inconsistent role
file is an error for the whole registry, never silently skipped. When the
available tools are given, every tool a role lists must exist and must be
reachable under the role's ``max_risk`` (an unreachable tool is a mistake in
the role file, not something to discover at run time). Likewise, when the
available skills are given, every skill a role lists must exist.
"""

from __future__ import annotations

import os
import stat
import tomllib
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from pydantic import ValidationError

from automa.roles.models import Role
from automa.skills.registry import SkillRegistry
from automa.tools.base import RiskLevel, Tool

ROLE_SUFFIX = ".toml"
MAX_ROLE_FILE_BYTES = 64 * 1024
BUILTIN_ROLES_DIR = Path(__file__).resolve().parent / "builtin"


class RoleError(Exception):
    """A role file or the role set is invalid."""


def _read_role_file(path: Path) -> str:
    # O_NONBLOCK: opening a FIFO must not hang; it is then rejected as non-regular.
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as e:
        raise RoleError(f"{path.name}: cannot open role file ({e.strerror})") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise RoleError(f"{path.name}: not a regular file")
        if st.st_size > MAX_ROLE_FILE_BYTES:
            raise RoleError(f"{path.name}: role file larger than {MAX_ROLE_FILE_BYTES} bytes")
        with os.fdopen(fd, "rb", closefd=True) as f:
            fd = -1
            data = f.read(MAX_ROLE_FILE_BYTES + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > MAX_ROLE_FILE_BYTES:
        raise RoleError(f"{path.name}: role file larger than {MAX_ROLE_FILE_BYTES} bytes")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise RoleError(f"{path.name}: role file is not valid UTF-8") from None


def parse_role(text: str, *, expected_name: str, source: str = "<role>") -> Role:
    """Parse one role file's text; ``name`` must equal ``expected_name`` (the file stem)."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise RoleError(f"{source}: invalid TOML ({e})") from None
    try:
        role = Role.model_validate(data)
    except ValidationError as e:
        lines = [
            f"- {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in e.errors()[:20]
        ]
        raise RoleError(f"{source}: invalid role\n" + "\n".join(lines)) from None
    if role.name != expected_name:
        raise RoleError(f"{source}: name {role.name!r} does not match file name {expected_name!r}")
    return role


def _check_tools(role: Role, tools: Mapping[str, RiskLevel]) -> None:
    unknown = sorted(set(role.tools) - set(tools))
    if unknown:
        raise RoleError(f"role {role.name!r}: unknown tools {unknown}")
    above = sorted(
        n
        for n in role.tools
        if tools[n] is not RiskLevel.EXTERNAL and tools[n].rank > role.max_risk.rank
    )
    if above:
        raise RoleError(
            f"role {role.name!r}: tools {above} are above max_risk {role.max_risk.value!r}"
        )


class RoleRegistry:
    """An immutable set of roles, keyed by name."""

    def __init__(
        self,
        roles: Iterable[Role],
        *,
        tools: Iterable[Tool] | None = None,
        skills: SkillRegistry | None = None,
    ):
        by_name: dict[str, Role] = {}
        for role in roles:
            if not isinstance(role, Role):
                raise TypeError("RoleRegistry accepts only Role instances")
            if role.name in by_name:
                raise RoleError(f"duplicate role {role.name!r}")
            by_name[role.name] = role
        if not by_name:
            raise RoleError("no roles defined")
        if tools is not None:
            risks = {t.name: t.risk for t in tools}
            for role in by_name.values():
                _check_tools(role, risks)
        if skills is not None:
            if not isinstance(skills, SkillRegistry):
                raise TypeError("skills must be a SkillRegistry")
            for role in by_name.values():
                missing = sorted(set(role.skills) - skills.names())
                if missing:
                    raise RoleError(f"role {role.name!r}: unknown skills {missing}")
        self._roles = dict(sorted(by_name.items()))

    @classmethod
    def from_dir(
        cls,
        directory: Path,
        *,
        tools: Iterable[Tool] | None = None,
        skills: SkillRegistry | None = None,
    ) -> RoleRegistry:
        directory = Path(directory)
        if directory.is_symlink() or not directory.is_dir():
            raise RoleError(f"roles directory not found: {directory}")
        roles: list[Role] = []
        for entry in sorted(os.scandir(directory), key=lambda e: e.name):
            if entry.name.startswith(".") or not entry.name.endswith(ROLE_SUFFIX):
                continue
            path = directory / entry.name
            stem = entry.name[: -len(ROLE_SUFFIX)]
            roles.append(parse_role(_read_role_file(path), expected_name=stem, source=entry.name))
        return cls(roles, tools=tools, skills=skills)

    @classmethod
    def builtin(
        cls, *, tools: Iterable[Tool] | None = None, skills: SkillRegistry | None = None
    ) -> RoleRegistry:
        """The roles shipped with Automa (``src/automa/roles/builtin/``)."""
        return cls.from_dir(BUILTIN_ROLES_DIR, tools=tools, skills=skills)

    def get(self, name: str) -> Role:
        try:
            return self._roles[name]
        except KeyError:
            raise RoleError(f"unknown role {name!r}") from None

    def names(self) -> frozenset[str]:
        return frozenset(self._roles)

    def __contains__(self, name: object) -> bool:
        return name in self._roles

    def __iter__(self) -> Iterator[Role]:
        return iter(self._roles.values())

    def __len__(self) -> int:
        return len(self._roles)
