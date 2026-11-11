"""Confined filesystem access to one project's ``files/`` directory.

Every path is validated as a safe relative path (no ``..``, not absolute, no NUL)
and then opened component by component with ``dir_fd`` + ``O_NOFOLLOW``, so a
symlink anywhere on the path (even one swapped in after validation) is refused
and nothing outside the root can be read or written. Writes are atomic.
"""

from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from automa.plan.models import MAX_PATH_CHARS, _safe_relpath
from automa.tools.base import ToolError

MAX_READ_BYTES = 256 * 1024
MAX_WRITE_BYTES = 1024 * 1024
MAX_LIST_ENTRIES = 500
MAX_DEPTH = 32

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIR_FLAGS = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC


class WorkspaceError(ToolError):
    """A path or file operation was refused or failed."""


@dataclass(frozen=True)
class DirEntry:
    name: str
    kind: str  # "file" | "dir" | "symlink" | "other"
    size: int


def normalize(path: str) -> str:
    """Validate and normalize a relative path, or raise ``WorkspaceError``."""
    if not isinstance(path, str) or not path:
        raise WorkspaceError("path must be a non-empty string")
    if len(path) > MAX_PATH_CHARS:
        raise WorkspaceError(f"path longer than {MAX_PATH_CHARS} characters")
    try:
        norm = _safe_relpath(path)
    except ValueError as e:
        raise WorkspaceError(f"invalid path {path!r}: {e}") from None
    if norm.count("/") >= MAX_DEPTH:
        raise WorkspaceError(f"path deeper than {MAX_DEPTH} levels")
    return norm


class Workspace:
    """A project's working directory. ``root`` must exist and must not be a symlink."""

    def __init__(self, root: Path) -> None:
        if not os.supports_dir_fd >= {os.open, os.stat}:  # pragma: no cover - non-POSIX
            raise WorkspaceError("this platform cannot confine file access (no dir_fd)")
        self._root = Path(os.path.abspath(root))
        try:
            st = os.lstat(self._root)
        except OSError as e:
            raise WorkspaceError(f"workspace root does not exist: {self._root}") from e
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise WorkspaceError(f"workspace root must be a real directory: {self._root}")

    @property
    def root(self) -> Path:
        return self._root

    # -- low level ---------------------------------------------------------------

    @contextmanager
    def _dir(self, parts: list[str], *, create: bool = False) -> Iterator[int]:
        """Yield an fd for the directory ``root/parts...``, never following symlinks."""
        try:
            fd = os.open(self._root, _DIR_FLAGS)
        except OSError as e:
            raise WorkspaceError("workspace root is not accessible") from e
        try:
            for i, part in enumerate(parts):
                if create:
                    try:
                        os.mkdir(part, 0o755, dir_fd=fd)
                    except FileExistsError:
                        pass
                    except OSError as e:
                        raise WorkspaceError(f"cannot create directory {part!r}: {e}") from e
                try:
                    nfd = os.open(part, _DIR_FLAGS, dir_fd=fd)
                except FileNotFoundError:
                    raise WorkspaceError(
                        f"directory not found: {'/'.join(parts[: i + 1])}"
                    ) from None
                except OSError:
                    raise WorkspaceError(
                        f"not a directory (or a symlink): {'/'.join(parts[: i + 1])}"
                    ) from None
                os.close(fd)
                fd = nfd
            yield fd
        finally:
            os.close(fd)

    # -- operations --------------------------------------------------------------

    def dir_path(self, path: str = "") -> Path:
        """Absolute path of an existing directory inside the root (no symlinks on the way).

        Used to give a subprocess its working directory. ``""`` means the root.
        """
        if path in ("", "."):
            with self._dir([]):
                return self._root
        norm = normalize(path)
        with self._dir(norm.split("/")):
            return self._root.joinpath(*norm.split("/"))

    def read_text(self, path: str, *, max_bytes: int = MAX_READ_BYTES) -> tuple[str, bool]:
        """Return ``(text, truncated)``. Only regular UTF-8 files are readable."""
        norm = normalize(path)
        *dirs, name = norm.split("/")
        with self._dir(dirs) as dfd:
            try:
                fd = os.open(name, os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC, dir_fd=dfd)
            except FileNotFoundError:
                raise WorkspaceError(f"file not found: {norm}") from None
            except OSError:
                raise WorkspaceError(f"cannot open {norm} (symlink or not a file)") from None
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise WorkspaceError(f"not a regular file: {norm}")
                chunks: list[bytes] = []
                remaining = max_bytes + 1
                while remaining > 0:
                    chunk = os.read(fd, min(remaining, 65536))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
            finally:
                os.close(fd)
        data = b"".join(chunks)
        truncated = len(data) > max_bytes
        data = data[:max_bytes]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            # A cut in the middle of a multi-byte character is fine; anything else is not.
            if not (
                truncated and e.start >= len(data) - 3 and e.reason == "unexpected end of data"
            ):
                raise WorkspaceError(f"{norm} is not a UTF-8 text file") from None
            text = data[: e.start].decode("utf-8")
        if "\x00" in text:
            raise WorkspaceError(f"{norm} looks like a binary file")
        return text, truncated

    def write_text(self, path: str, content: str, *, max_bytes: int = MAX_WRITE_BYTES) -> int:
        """Atomically create or replace a regular file. Returns bytes written."""
        norm = normalize(path)
        if "\x00" in content:
            raise WorkspaceError("content must not contain NUL characters")
        data = content.encode("utf-8")
        if len(data) > max_bytes:
            raise WorkspaceError(f"content larger than {max_bytes} bytes")
        *dirs, name = norm.split("/")
        with self._dir(dirs, create=True) as dfd:
            try:
                st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
            except FileNotFoundError:
                st = None
            if st is not None and not stat.S_ISREG(st.st_mode):
                raise WorkspaceError(f"refusing to overwrite non-regular file: {norm}")
            tmp = f".{name[:200]}.{secrets.token_hex(6)}.tmp"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC
            fd = os.open(tmp, flags, 0o644, dir_fd=dfd)
            try:
                try:
                    view = memoryview(data)
                    while view:
                        n = os.write(fd, view)
                        view = view[n:]
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.replace(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd)
            except BaseException:
                try:
                    os.unlink(tmp, dir_fd=dfd)
                except OSError:
                    pass
                raise
        return len(data)

    def delete_file(self, path: str) -> int:
        """Delete one existing regular file (never a directory or symlink). Returns its size."""
        norm = normalize(path)
        *dirs, name = norm.split("/")
        with self._dir(dirs) as dfd:
            try:
                st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
            except FileNotFoundError:
                raise WorkspaceError(f"file not found: {norm}") from None
            if not stat.S_ISREG(st.st_mode):
                raise WorkspaceError(f"refusing to delete non-regular file: {norm}")
            try:
                # unlink never follows a symlink, even one swapped in after the check.
                os.unlink(name, dir_fd=dfd)
            except FileNotFoundError:
                raise WorkspaceError(f"file not found: {norm}") from None
        return st.st_size

    def list_dir(
        self, path: str = "", *, max_entries: int = MAX_LIST_ENTRIES
    ) -> tuple[list[DirEntry], int]:
        """``(entries, total)`` of a directory (``""`` = root), sorted by name.

        At most ``max_entries`` are returned; ``total`` is the full count. Symlinks are
        reported as ``symlink`` and never followed.
        """
        parts = [] if path in ("", ".") else normalize(path).split("/")
        entries: list[DirEntry] = []
        with self._dir(parts) as dfd, os.scandir(dfd) as it:
            for e in it:
                st = e.stat(follow_symlinks=False)
                if stat.S_ISLNK(st.st_mode):
                    kind = "symlink"
                elif stat.S_ISDIR(st.st_mode):
                    kind = "dir"
                elif stat.S_ISREG(st.st_mode):
                    kind = "file"
                else:
                    kind = "other"
                entries.append(DirEntry(e.name, kind, st.st_size if kind == "file" else 0))
        entries.sort(key=lambda d: d.name)
        return entries[:max_entries], len(entries)
