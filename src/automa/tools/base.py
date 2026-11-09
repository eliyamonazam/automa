"""Tool contract: name, description, typed parameters and a risk level.

A ``Tool`` never decides whether it may run; the ``ToolExecutor`` asks the
``PermissionPolicy`` first and is the only code that calls ``Tool.run``.
Tools only touch the filesystem through the ``Workspace`` they are given.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from automa.tools.workspace import Workspace

TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_TOOL_OUTPUT_CHARS = 20_000


class RiskLevel(StrEnum):
    """Ordered from least to most dangerous."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    EXTERNAL = "external"

    @property
    def rank(self) -> int:
        return _RANK[self]


_RANK = {RiskLevel.READ: 0, RiskLevel.WRITE: 1, RiskLevel.EXECUTE: 2, RiskLevel.EXTERNAL: 3}


class ToolArgs(BaseModel):
    """Base class for tool parameter models: strict, no unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ToolError(Exception):
    """A tool could not do what was asked. The message is shown to the model."""


class ToolResult(BaseModel):
    """What a tool call returns to the caller (and, as text, to the model)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str
    ok: bool
    content: str = Field(default="", max_length=MAX_TOOL_OUTPUT_CHARS + 100)
    error: str | None = None
    denied: bool = False

    def as_text(self) -> str:
        return self.content if self.ok else f"ERROR: {self.error}"


class Tool(ABC):
    """Subclasses set the class attributes and implement ``run``."""

    name: ClassVar[str]
    description: ClassVar[str]
    risk: ClassVar[RiskLevel]
    args_model: ClassVar[type[ToolArgs]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return
        name = getattr(cls, "name", None)
        if not isinstance(name, str) or not TOOL_NAME_RE.fullmatch(name):
            raise TypeError(f"{cls.__name__}: invalid tool name {name!r}")
        if not isinstance(getattr(cls, "risk", None), RiskLevel):
            raise TypeError(f"{cls.__name__}: risk must be a RiskLevel")
        args = getattr(cls, "args_model", None)
        if not (isinstance(args, type) and issubclass(args, ToolArgs)):
            raise TypeError(f"{cls.__name__}: args_model must subclass ToolArgs")
        if not getattr(cls, "description", "").strip():
            raise TypeError(f"{cls.__name__}: description is required")

    @abstractmethod
    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        """Do the work and return text for the model; raise ``ToolError`` on failure."""

    def spec(self) -> dict[str, Any]:
        """OpenAI-compatible function schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }
