"""Preview a project's HTML result in the browser, isolated (D136).

``/projects/<pid>/preview/<path>`` serves files from the project's ``files/`` so a
generated page can be opened with its own relative CSS, scripts and images:

* ``.html``/``.htm`` — the page, with ``Content-Security-Policy: sandbox
  allow-scripts`` and no ``allow-same-origin``: the page runs in an opaque origin, so
  it cannot read this server's pages, cookies or CSRF tokens, cannot submit forms
  (no ``allow-forms``, ``form-action 'none'``) and cannot reach the network
  (``connect-src 'none'``, no external sources at all);
* ``.css``, ``.js`` — as their types, for the page's own relative references;
* ``.png``, ``.jpg``, ``.jpeg`` — only files that really are such images.

Everything else is refused. Paths are normalized by the Workspace (no ``..``,
no absolute paths, symlinks never followed). Responses are read-only and cached
nowhere.
"""

from __future__ import annotations

from urllib.parse import quote, unquote

from automa.project.store import ProjectStore, ProjectStoreError, validate_project_id
from automa.tools.workspace import WorkspaceError, normalize
from automa.webui.files import (
    IMAGE_TYPES,
    MAX_IMAGE_BYTES,
    FilesRefused,
    _refused,
    _require,
    checked_image,
)

PREVIEW_SEGMENT = "preview"
MAX_PREVIEW_BYTES = 2 * 1024 * 1024
HTML_SUFFIXES = (".html", ".htm")
_TEXT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# The previewed page: scripts may run, but only inside an opaque, network-less sandbox.
PREVIEW_PAGE_CSP = (
    "sandbox allow-scripts allow-modals allow-downloads; "
    "default-src 'none'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "font-src 'self' data:; "
    "media-src 'self' data: blob:; "
    "connect-src 'none'; "
    "form-action 'none'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'"
)
# Assets are never documents.
PREVIEW_ASSET_CSP = "default-src 'none'; sandbox; frame-ancestors 'none'"

PREVIEW_HEADERS: tuple[tuple[str, str], ...] = (
    ("X-Frame-Options", "DENY"),
    ("X-Content-Type-Options", "nosniff"),
    ("Cache-Control", "no-store"),
    ("Referrer-Policy", "no-referrer"),
    # The sandboxed page has an opaque origin; its own images/CSS are requested
    # cross-origin, so assets must allow that (they are the user's files, loopback only).
    ("Cross-Origin-Resource-Policy", "cross-origin"),
)

_PREFIX = "/projects/"


def parse_preview_path(path: str) -> tuple[str, str] | None:
    """``(project_id, relative path)`` for a preview URL path, else ``None``."""
    if not path.startswith(_PREFIX):
        return None
    head, _, rest = path[len(_PREFIX) :].partition("/")
    segment, _, rel = rest.partition("/")
    if segment != PREVIEW_SEGMENT or not rel:
        return None
    try:
        pid = validate_project_id(unquote(head))
        rel_path = normalize(unquote(rel))
    except (ProjectStoreError, WorkspaceError, ValueError):
        return None
    if not rel_path or rel_path == ".":
        return None
    return pid, rel_path


def preview_href(project_id: str, path: str) -> str:
    return f"{_PREFIX}{quote(project_id, safe='')}/{PREVIEW_SEGMENT}/{quote(path, safe='/')}"


def is_previewable(path: str) -> bool:
    return path.lower().endswith(HTML_SUFFIXES)


def preview_response(store: ProjectStore, project_id: str, path: str) -> tuple[bytes, str, str]:
    """``(body, content type, CSP)`` for a preview path, else ``FilesRefused``."""
    lower = path.lower()
    ws = _require(store, project_id)
    if lower.endswith(_IMAGE_SUFFIXES):
        try:
            data, _ = ws.read_bytes(path, max_bytes=MAX_IMAGE_BYTES)
        except WorkspaceError as e:
            raise _refused(e) from None
        info = checked_image(path, data)
        if info is None:
            raise FilesRefused.coded(415, "preview_bad_image")
        return data, IMAGE_TYPES[info[0]], PREVIEW_ASSET_CSP
    if lower.endswith(HTML_SUFFIXES):
        mime, csp = "text/html; charset=utf-8", PREVIEW_PAGE_CSP
    elif any(lower.endswith(s) for s in _TEXT_TYPES):
        mime = next(t for s, t in _TEXT_TYPES.items() if lower.endswith(s))
        csp = PREVIEW_ASSET_CSP
    else:
        raise FilesRefused.coded(415, "preview_types")
    try:
        data, _ = ws.read_bytes(path, max_bytes=MAX_PREVIEW_BYTES)
    except WorkspaceError as e:
        raise _refused(e) from None
    return data, mime, csp
