"""Read-only browser of a project's ``files/`` directory for the web UI (D128).

Every access goes through :class:`automa.tools.workspace.Workspace`: paths are
validated relative paths (no ``..``, not absolute, no NUL) opened component by
component with ``O_NOFOLLOW``, so a symlink anywhere on the way is refused and
nothing outside ``files/`` is ever read. Listings show symlinks and special files
but never follow or link them.

Routes (all ``GET``/``HEAD``, also in the read-only server):

* ``/projects/<pid>/files[?path=<dir>]`` — one directory;
* ``/projects/<pid>/files/view?path=<file>`` — text (UTF-8, capped, shown escaped),
  a PNG/JPEG image (only when the extension *and* the file signature agree), or a
  "binary" line with the download link;
* ``/projects/<pid>/files/raw?path=<image>`` — the image bytes with a checked type;
* ``/projects/<pid>/files/download?path=<file>`` — any regular file as
  ``application/octet-stream`` with ``Content-Disposition: attachment``.

Nothing here writes, creates a directory or calls a model.
"""

from __future__ import annotations

import os
from enum import StrEnum
from urllib.parse import parse_qsl, quote, unquote, urlencode

from pydantic import BaseModel, ConfigDict

from automa.project.store import ProjectStore, ProjectStoreError, validate_project_id
from automa.tools.base import ToolError
from automa.tools.images import image_format_for, image_info
from automa.tools.workspace import (
    MAX_LIST_ENTRIES,
    FileTooLarge,
    Workspace,
    WorkspaceError,
    normalize,
)
from automa.webui.refusal import Refusal

MAX_TEXT_BYTES = 256 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_QUERY_CHARS = 4096
IMAGE_TYPES = {"png": "image/png", "jpeg": "image/jpeg"}
# The raw/download responses are never documents: no scripts, no plugins, sandboxed.
FILE_CONTENT_SECURITY_POLICY = "default-src 'none'; sandbox; frame-ancestors 'none'"
_PREFIX = "/projects/"


class FilesRoute(StrEnum):
    LIST = "list"
    VIEW = "view"
    RAW = "raw"
    DOWNLOAD = "download"


class FilesRefused(Refusal):
    """A files request that is answered with an error page (D147: a ``code``
    translates it)."""


class _View(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileEntryView(_View):
    name: str
    path: str  # relative to files/
    kind: str  # "file" | "dir" | "symlink" | "other"
    size: int


class FileListing(_View):
    project_id: str
    dir: str  # "" = files/ itself
    parent: str | None  # None at the root
    entries: tuple[FileEntryView, ...]
    total: int
    missing: bool = False  # files/ does not exist yet


class FileContent(_View):
    project_id: str
    path: str
    kind: str  # "text" | "image" | "binary"
    size: int
    text: str | None = None
    truncated: bool = False
    image_format: str | None = None
    width: int | None = None
    height: int | None = None
    note: str | None = None
    # Why a file is not shown (D146): "too_large" | "not_image" | "not_text"; the
    # page translates it, ``note`` keeps the English text.
    note_code: str | None = None


def parse_files_path(path: str) -> tuple[str, FilesRoute] | None:
    """``(project_id, route)`` for a files path, else ``None``."""
    if not path.startswith(_PREFIX):
        return None
    parts = path[len(_PREFIX) :].split("/")
    if len(parts) < 2 or parts[1] != "files":
        return None
    if len(parts) == 2:
        route = FilesRoute.LIST
    elif len(parts) == 3 and parts[2] in ("view", "raw", "download"):
        route = FilesRoute(parts[2])
    else:
        return None
    try:
        pid = validate_project_id(unquote(parts[0]))
    except ProjectStoreError:
        return None
    return pid, route


def parse_path_query(query: str, *, allow_root: bool) -> str:
    """The normalized ``path`` of the query string (``""`` = root, lists only)."""
    if len(query) > MAX_QUERY_CHARS:
        raise FilesRefused.coded(414, "address_too_long")
    if not query:
        if allow_root:
            return ""
        raise FilesRefused.coded(400, "missing_path")
    try:
        pairs = parse_qsl(
            query, keep_blank_values=True, strict_parsing=True, max_num_fields=1, errors="strict"
        )
    except (ValueError, UnicodeDecodeError):
        raise FilesRefused.coded(400, "invalid_query") from None
    if len(pairs) != 1 or pairs[0][0] != "path":
        raise FilesRefused.coded(400, "only_path")
    raw = pairs[0][1]
    if raw in ("", ".") and allow_root:
        return ""
    try:
        return normalize(raw)
    except WorkspaceError as e:
        raise FilesRefused(400, str(e), lead="lead.files") from None


def files_href(project_id: str, route: FilesRoute, path: str = "") -> str:
    """The address of a files page (query-encoded ``path``)."""
    base = _PREFIX + quote(project_id, safe="") + "/files"
    if route is not FilesRoute.LIST:
        base += "/" + route.value
    return base + ("?" + urlencode({"path": path}) if path else "")


def image_source(host: str, project_id: str, scheme: str = "http") -> str:
    """The only image source a file view may load: this server's raw path (D128).

    ``host`` must be an already-checked ``Host`` value (loopback ``addr:port``, or the
    public name with ``scheme="https"``, D187). CSP matches a source without a trailing
    slash as one exact path; the query is ignored.
    """
    if scheme not in ("http", "https"):
        raise ValueError(f"invalid scheme: {scheme!r}")
    return scheme + "://" + host + _PREFIX + quote(project_id, safe="") + "/files/raw"


def _workspace(store: ProjectStore, project_id: str) -> Workspace | None:
    """The project's ``files/`` workspace; ``None`` when it does not exist yet."""
    if project_id not in store.list_ids():
        raise FilesRefused.coded(404, "no_project", id=project_id)
    root = store.files_dir(project_id)
    try:
        os.lstat(root)
    except FileNotFoundError:
        return None
    except OSError as e:
        raise FilesRefused(
            500, f"files/ is not accessible: {e.strerror}", lead="lead.files"
        ) from None
    try:
        return Workspace(root)
    except WorkspaceError as e:
        raise FilesRefused(500, str(e), lead="lead.files") from None


def _require(store: ProjectStore, project_id: str) -> Workspace:
    ws = _workspace(store, project_id)
    if ws is None:
        raise FilesRefused.coded(404, "no_files")
    return ws


def _refused(e: WorkspaceError) -> FilesRefused:
    if isinstance(e, FileTooLarge):
        return FilesRefused(413, str(e), lead="lead.files")
    return FilesRefused(404, str(e), lead="lead.files")


def build_listing(store: ProjectStore, project_id: str, directory: str = "") -> FileListing:
    """One directory of ``files/`` (``""`` = root), sorted by name, capped."""
    ws = _workspace(store, project_id)
    parent = None
    if directory:
        parent = directory.rsplit("/", 1)[0] if "/" in directory else ""
    if ws is None:
        if directory:
            raise FilesRefused.coded(404, "no_files")
        return FileListing(
            project_id=project_id, dir="", parent=None, entries=(), total=0, missing=True
        )
    try:
        entries, total = ws.list_dir(directory, max_entries=MAX_LIST_ENTRIES)
    except WorkspaceError as e:
        raise _refused(e) from None
    except OSError as e:
        raise FilesRefused(
            404, f"cannot list {directory or 'files/'}: {e.strerror}", lead="lead.files"
        ) from None
    prefix = directory + "/" if directory else ""
    return FileListing(
        project_id=project_id,
        dir=directory,
        parent=parent,
        entries=tuple(
            FileEntryView(name=e.name, path=prefix + e.name, kind=e.kind, size=e.size)
            for e in entries
        ),
        total=total,
    )


def decode_text(data: bytes, truncated: bool) -> str | None:
    """``data`` as UTF-8 text, or ``None`` if it is not text (NUL, invalid UTF-8).

    A multi-byte character cut by the size limit is dropped, not an error.
    """
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as e:
        if not (truncated and e.start >= len(data) - 3 and e.reason == "unexpected end of data"):
            return None
        decoded = data[: e.start].decode("utf-8")
    if "\x00" in decoded:
        return None
    return decoded


def checked_image(path: str, data: bytes) -> tuple[str, int, int] | None:
    """``(format, width, height)`` when the extension and the signature agree."""
    expected = image_format_for(path)
    if expected is None:
        return None
    try:
        fmt, width, height = image_info(data)
    except ToolError:
        return None
    return (fmt, width, height) if fmt == expected else None


def _size(ws: Workspace, path: str) -> int:
    try:
        return ws.file_size(path)
    except WorkspaceError as e:
        raise _refused(e) from None


def read_file(store: ProjectStore, project_id: str, path: str) -> FileContent:
    """What the view page of ``path`` shows (text, checked image or binary)."""
    ws = _require(store, project_id)
    size = _size(ws, path)
    base = {"project_id": project_id, "path": path, "size": size}
    if image_format_for(path) is not None:
        if size > MAX_IMAGE_BYTES:
            note = f"image larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MiB: download it"
            return FileContent(kind="binary", note=note, note_code="too_large", **base)
        try:
            data, _ = ws.read_bytes(path, max_bytes=MAX_IMAGE_BYTES)
        except WorkspaceError as e:
            raise _refused(e) from None
        info = checked_image(path, data)
        if info is not None:
            fmt, width, height = info
            return FileContent(kind="image", image_format=fmt, width=width, height=height, **base)
        note = "the extension says image but the content is not that image type"
        return FileContent(kind="binary", note=note, note_code="not_image", **base)
    try:
        data, truncated = ws.read_bytes(path, max_bytes=MAX_TEXT_BYTES, truncate=True)
    except WorkspaceError as e:
        raise _refused(e) from None
    decoded = decode_text(data, truncated)
    if decoded is None:
        return FileContent(
            kind="binary", note="not a UTF-8 text file", note_code="not_text", **base
        )
    return FileContent(kind="text", text=decoded, truncated=truncated, **base)


def image_bytes(store: ProjectStore, project_id: str, path: str) -> tuple[bytes, str]:
    """``(bytes, content type)`` of a checked PNG/JPEG image, else ``FilesRefused``."""
    ws = _require(store, project_id)
    if image_format_for(path) is None:
        raise FilesRefused.coded(415, "image_types")
    try:
        data, _ = ws.read_bytes(path, max_bytes=MAX_IMAGE_BYTES)
    except WorkspaceError as e:
        raise _refused(e) from None
    info = checked_image(path, data)
    if info is None:
        raise FilesRefused.coded(415, "bad_image")
    return data, IMAGE_TYPES[info[0]]


def download_bytes(store: ProjectStore, project_id: str, path: str) -> bytes:
    """The bytes of a regular file for download (capped)."""
    ws = _require(store, project_id)
    try:
        data, _ = ws.read_bytes(path, max_bytes=MAX_DOWNLOAD_BYTES)
    except WorkspaceError as e:
        raise _refused(e) from None
    return data


def ascii_filename(path: str) -> str:
    """A plain ASCII file name for ``Content-Disposition`` (fallback ``download``)."""
    name = path.rsplit("/", 1)[-1]
    safe = "".join(c if c.isascii() and (c.isalnum() or c in "._-") else "_" for c in name)
    safe = safe.lstrip(".")[:120]
    return safe or "download"


def content_disposition(path: str, *, attachment: bool) -> str:
    """``attachment``/``inline`` with an ASCII and an RFC 5987 UTF-8 file name."""
    name = path.rsplit("/", 1)[-1]
    kind = "attachment" if attachment else "inline"
    encoded = quote(name, safe="")
    return f"{kind}; filename=\"{ascii_filename(path)}\"; filename*=UTF-8''{encoded}"
