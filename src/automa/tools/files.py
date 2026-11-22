"""File tools confined to the project workspace.

``read_file``, ``write_file`` (replace or append), ``edit_file`` (exact text
replacement, D165), ``list_dir`` and ``delete_file`` (irreversible: it
always needs the user's approval through the Approval Gate).
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, field_validator

from automa.plan.models import MAX_PATH_CHARS
from automa.tools.base import MAX_TOOL_OUTPUT_CHARS, RiskLevel, Tool, ToolArgs, ToolError
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
    append: bool = Field(
        default=False,
        description=(
            "Add content to the end of the file instead of replacing it (to write a "
            "long file in parts)"
        ),
    )


class WriteFile(Tool):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "Create or overwrite a UTF-8 text file in the project workspace "
        "(parent directories are created). With append=true the content is added "
        "to the end instead. To change part of an existing file use edit_file."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.WRITE
    args_model: ClassVar[type[ToolArgs]] = WriteFileArgs

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        if not isinstance(args, WriteFileArgs):
            raise TypeError("arguments were not validated by the executor")
        if args.append:
            old = _read_whole(workspace, args.path, missing_ok=True)
            n = workspace.write_text(args.path, old + args.content)
            return (
                f"appended {len(args.content.encode('utf-8'))} bytes to {args.path} ({n} bytes now)"
            )
        n = workspace.write_text(args.path, args.content)
        return f"wrote {n} bytes to {args.path}"

    def writes(self, args: ToolArgs) -> tuple[str, ...]:
        return _written(args)


def _read_whole(workspace: Workspace, path: str, *, missing_ok: bool = False) -> str:
    """The whole text of ``path`` (refused if larger than a write may be)."""
    try:
        text, truncated = workspace.read_text(path, max_bytes=MAX_WRITE_BYTES)
    except WorkspaceError as e:
        if missing_ok and str(e).startswith("file not found"):
            return ""
        raise
    if truncated:
        raise WorkspaceError(f"{path} is larger than {MAX_WRITE_BYTES} bytes")
    return text


class EditFileArgs(ToolArgs):
    path: str = Field(min_length=1, max_length=MAX_PATH_CHARS, description="Relative path")
    old_text: str = Field(
        min_length=1,
        max_length=MAX_WRITE_BYTES,
        description="Exact text to replace (copied from the file, unique)",
    )
    new_text: str = Field(max_length=MAX_WRITE_BYTES, description="Text to put in its place")
    replace_all: bool = Field(default=False, description="Replace every occurrence")


class EditFile(Tool):
    """Change part of a file without writing it all again (D165): fewer output
    tokens, and no truncated rewrites of long files."""

    name: ClassVar[str] = "edit_file"
    description: ClassVar[str] = (
        "Replace exact text in an existing UTF-8 file of the project workspace. "
        "old_text must appear exactly once (or set replace_all). Prefer this over "
        "rewriting a whole file for a small change."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.WRITE
    args_model: ClassVar[type[ToolArgs]] = EditFileArgs

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        if not isinstance(args, EditFileArgs):
            raise TypeError("arguments were not validated by the executor")
        text = _read_whole(workspace, args.path)
        count = text.count(args.old_text)
        if count == 0:
            raise ToolError(
                f"old_text not found in {args.path}; read the file and copy the exact text"
            )
        if count > 1 and not args.replace_all:
            raise ToolError(
                f"old_text appears {count} times in {args.path}; add surrounding text "
                "to make it unique or set replace_all"
            )
        new = (
            text.replace(args.old_text, args.new_text)
            if args.replace_all
            else text.replace(args.old_text, args.new_text, 1)
        )
        n = workspace.write_text(args.path, new)
        done = count if args.replace_all else 1
        return f"edited {args.path}: {done} replacement(s), {n} bytes now"

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


FILE_TOOLS: tuple[type[Tool], ...] = (ReadFile, WriteFile, EditFile, ListDir, DeleteFile)
