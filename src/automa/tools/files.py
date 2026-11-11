"""File tools confined to the project workspace.

``read_file``, ``write_file``, ``list_dir`` and ``delete_file`` (irreversible: it
always needs the user's approval through the Approval Gate).
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, field_validator

from automa.plan.models import MAX_PATH_CHARS
from automa.tools.base import MAX_TOOL_OUTPUT_CHARS, RiskLevel, Tool, ToolArgs
from automa.tools.workspace import (
    MAX_LIST_ENTRIES,
    MAX_WRITE_BYTES,
    Workspace,
    WorkspaceError,
    normalize,
)


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

    def writes(self, args: ToolArgs) -> tuple[str, ...]:
        return _written(args)


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


class DeleteFileArgs(ToolArgs):
    path: str = Field(min_length=1, max_length=MAX_PATH_CHARS, description="Relative path")

    @field_validator("path")
    @classmethod
    def _safe_path(cls, v: str) -> str:
        # Validated (and normalized) before the Approval Gate records a request, so
        # the user is never asked to approve a path that could not be deleted, and
        # two spellings of one path share one approval.
        try:
            return normalize(v)
        except WorkspaceError as e:
            raise ValueError(str(e)) from None


class DeleteFile(Tool):
    name: ClassVar[str] = "delete_file"
    description: ClassVar[str] = (
        "Delete one existing file from the project workspace. Irreversible: the "
        "project pauses until the user approves this exact call."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.WRITE
    irreversible: ClassVar[bool] = True
    args_model: ClassVar[type[ToolArgs]] = DeleteFileArgs

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        if not isinstance(args, DeleteFileArgs):
            raise TypeError("arguments were not validated by the executor")
        n = workspace.delete_file(args.path)
        return f"deleted {args.path} ({n} bytes)"

    def writes(self, args: ToolArgs) -> tuple[str, ...]:
        return _written(args)


def _written(args: ToolArgs) -> tuple[str, ...]:
    path = getattr(args, "path", None)
    if not isinstance(path, str):
        return ()
    try:
        return (normalize(path),)
    except WorkspaceError:
        return ()  # the tool itself rejects the path


FILE_TOOLS: tuple[type[Tool], ...] = (ReadFile, WriteFile, ListDir, DeleteFile)
