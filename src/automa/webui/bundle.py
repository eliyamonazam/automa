"""Download the whole product as one zip (D158).

``GET /projects/<pid>/download.zip`` returns every regular file of the project's
``files/`` plus ``DELIVERY.md`` (the delivery report, D157) so the user gets the
product and its "read me first" in one click, in read-only mode too.

Safety: files are listed and read through ``Workspace`` (no symlink is followed or
included, nothing outside ``files/``), the zip is built in memory with a cap on the
number of files and total bytes (over the cap → refused, never a silently partial
zip), and entry names are the checked relative paths. Nothing is written to disk.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime

from automa.delivery import render_markdown, try_build_delivery
from automa.project.store import ProjectStore, ProjectStoreError, validate_project_id
from automa.tools.workspace import Workspace, WorkspaceError
from automa.webui.files import FilesRefused

MAX_BUNDLE_FILES = 2_000
MAX_BUNDLE_BYTES = 50 * 1024 * 1024
MAX_BUNDLE_DEPTH = 12
DELIVERY_NAME = "DELIVERY.md"
_ZIP_RE = re.compile(r"^/projects/([^/]+)/download\.zip$")


def parse_bundle_path(path: str) -> str | None:
    """The project id of ``/projects/<pid>/download.zip``, else ``None``."""
    m = _ZIP_RE.match(path)
    if m is None:
        return None
    try:
        return validate_project_id(m.group(1))
    except ProjectStoreError:
        return None


def bundle_href(project_id: str) -> str:
    return f"/projects/{validate_project_id(project_id)}/download.zip"


def bundle_name(project_id: str) -> str:
    return f"{validate_project_id(project_id)}.zip"


def _walk(ws: Workspace) -> list[tuple[str, int]]:
    """``(path, size)`` of every regular file, sorted; symlinks and others skipped."""
    found: list[tuple[str, int]] = []
    pending: list[tuple[str, int]] = [("", 0)]
    while pending:
        directory, depth = pending.pop()
        entries, total = ws.list_dir(directory, max_entries=MAX_BUNDLE_FILES + 1)
        if total > MAX_BUNDLE_FILES:
            raise FilesRefused.coded(413, "bundle_too_large")
        for e in entries:
            path = f"{directory}/{e.name}" if directory else e.name
            if e.kind == "file":
                found.append((path, e.size))
            elif e.kind == "dir" and depth + 1 < MAX_BUNDLE_DEPTH:
                pending.append((path, depth + 1))
        if len(found) > MAX_BUNDLE_FILES:
            raise FilesRefused.coded(413, "bundle_too_large")
    return sorted(found)


def build_bundle(
    store: ProjectStore, project_id: str, *, lang: str = "fa", now: datetime | None = None
) -> bytes:
    """The zip bytes of ``files/`` plus ``DELIVERY.md``; ``FilesRefused`` if too big."""
    if project_id not in store.list_ids():
        raise FilesRefused.coded(404, "no_project", id=project_id)
    root = store.files_dir(project_id)
    files: list[tuple[str, int]] = []
    ws: Workspace | None = None
    if root.is_dir() and not root.is_symlink():
        try:
            ws = Workspace(root)
            files = _walk(ws)
        except WorkspaceError as e:
            raise FilesRefused(500, str(e), lead="lead.files") from None
    if sum(size for _, size in files) > MAX_BUNDLE_BYTES:
        raise FilesRefused.coded(413, "bundle_too_large")
    stamp = (now or datetime.now()).timetuple()[:6]
    out = io.BytesIO()
    total = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        delivery = try_build_delivery(store, project_id)
        if delivery is not None:
            name = DELIVERY_NAME
            if any(p == DELIVERY_NAME for p, _ in files):
                name = "DELIVERY-automa.md"
            zf.writestr(zipfile.ZipInfo(name, stamp), render_markdown(delivery, lang))
        for path, _size in files:
            assert ws is not None  # noqa: S101 - files is only filled with a workspace
            try:
                data, _ = ws.read_bytes(path, max_bytes=MAX_BUNDLE_BYTES - total)
            except WorkspaceError:
                # Changed while zipping (grew, became a symlink, vanished): refuse.
                raise FilesRefused.coded(409, "bundle_changed") from None
            total += len(data)
            info = zipfile.ZipInfo(path, stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return out.getvalue()
