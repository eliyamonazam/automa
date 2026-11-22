"""Fixed static assets of the web UI (M16, D161): the Vazirmatn font and its licence.

Only the names in :data:`ASSETS` are served, read from files shipped inside the
package (never from the workspace, never a path taken from the request). The font
is the variable Vazirmatn (SIL Open Font License 1.1, ``OFL.txt`` next to it), so
every page looks the same on every computer without loading anything from the
internet: the page CSP allows fonts from ``'self'`` only.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from importlib import resources
from types import MappingProxyType

STATIC_PREFIX = "/static/"
FONT_PATH = STATIC_PREFIX + "vazirmatn.woff2"
LICENSE_PATH = STATIC_PREFIX + "OFL.txt"

# request path -> (file inside automa/webui/static, Content-Type)
ASSETS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        FONT_PATH: ("vazirmatn.woff2", "font/woff2"),
        LICENSE_PATH: ("OFL.txt", "text/plain; charset=utf-8"),
    }
)

# The one @font-face of every page: the local variable font (weights 100-900).
FONT_FACE = (
    '@font-face{font-family:"Vazirmatn";src:url('
    + FONT_PATH
    + ') format("woff2");font-weight:100 900;font-style:normal;font-display:swap}'
)

# Headers of an asset response: loads nothing itself, cannot be framed or sniffed,
# may be cached (the files only change with a new version of Automa).
STATIC_HEADERS: tuple[tuple[str, str], ...] = (
    ("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"),
    ("X-Content-Type-Options", "nosniff"),
    ("Cache-Control", "public, max-age=86400"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
)


def is_static_path(path: str) -> bool:
    """Whether ``path`` is under ``/static/`` (served or refused here, never routed on)."""
    return path.startswith(STATIC_PREFIX)


@cache
def _read(name: str) -> bytes:
    return resources.files("automa.webui").joinpath("static", name).read_bytes()


def static_asset(path: str) -> tuple[bytes, str] | None:
    """The bytes and Content-Type of the asset at ``path``; ``None`` if it is not one
    of :data:`ASSETS` (exact match: no other file, no traversal, no case folding)."""
    entry = ASSETS.get(path)
    if entry is None:
        return None
    name, mime = entry
    return _read(name), mime
