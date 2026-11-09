"""File tools confined to the project workspace: ``read_file``, ``write_file``, ``list_dir``."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from automa.plan.models import MAX_PATH_CHARS
from automa.tools.base import MAX_TOOL_OUTPUT_CHARS, RiskLevel, Tool, ToolArgs
from automa.tools.workspace import MAX_LIST_ENTRIES, MAX_WRITE_BYTES, Workspace


class ReadFileArgs(ToolArgs):
    path: str = Field(min_length=1, max_length=MAX_PATH_CHARS, description="Relative path")


class ReadFile(Tool):
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = "Read a UTF-8 text file from the project workspace."
    risk: ClassVar[RiskLevel] = RiskLevel.READ
    args_model: ClassVar[type[ToolArgs]] = ReadFileArgs

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        if not isinstance(args, ReadFileArgs):
            raise TypeError("arguments were not validated by the executor")
        text, truncated = workspace.read_text(args.path, max_bytes=MAX_TOOL_OUTPUT_CHARS)
        if truncated or len(text) > MAX_TOOL_OUTPUT_CHARS:
            text = text[:MAX_TOOL_OUTPUT_CHARS] + "\n[... truncated]"
        return text


class WriteFileArgs(ToolArgs):
    path: str = Field(min_length=1, max_length=MAX_PATH_CHARS, description="Relative path")
    content: str = Field(max_length=MAX_WRITE_BYTES, description="Full new file content")


class WriteFile(Tool):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "Create or overwrite a UTF-8 text file in the project workspace "
        "(parent directories are created)."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.WRITE
    args_model: ClassVar[type[ToolArgs]] = WriteFileArgs

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        if not isinstance(args, WriteFileArgs):
            raise TypeError("arguments were not validated by the executor")
        n = workspace.write_text(args.path, args.content)
        return f"wrote {n} bytes to {args.path}"


class ListDirArgs(ToolArgs):
    path: str = Field(
        default="", max_length=MAX_PATH_CHARS, description="Relative directory; empty = root"
    )


class ListDir(Tool):
    name: ClassVar[str] = "list_dir"
    description: ClassVar[str] = "List the entries of a directory in the project workspace."
    risk: ClassVar[RiskLevel] = RiskLevel.READ
    args_model: ClassVar[type[ToolArgs]] = ListDirArgs

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        if not isinstance(args, ListDirArgs):
            raise TypeError("arguments were not validated by the executor")
        entries, total = workspace.list_dir(args.path, max_entries=MAX_LIST_ENTRIES)
        if not entries:
            return "(empty)"
        lines = []
        for e in entries:
            if e.kind == "dir":
                lines.append(f"{e.name}/")
            elif e.kind == "file":
                lines.append(f"{e.name}\t{e.size}")
            else:
                lines.append(f"{e.name}\t[{e.kind}]")
        if total > len(entries):
            lines.append(f"[... {total - len(entries)} more entries]")
        return "\n".join(lines)


FILE_TOOLS: tuple[type[Tool], ...] = (ReadFile, WriteFile, ListDir)
