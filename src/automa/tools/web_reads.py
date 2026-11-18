"""Opt-in automatic approval of plain web reads (``[web] auto_approve_reads``, D93).

``fetch_url`` stays an ``external`` tool: by default every call needs the user's
approval. When the user opts in, the executor may run a ``fetch_url`` call without
asking, but only if the URL cannot plausibly carry project data out:

* https on the default port, to a plain ASCII host name (no IP, no punycode);
* no query string and no userinfo;
* a short path with no long opaque run of characters (an encoded blob);
* at most ``max_per_task`` such reads per task executor.

Anything else falls back to the normal Approval Gate. Every automatic approval is
logged as a ``tool.auto_approved`` event with host and path.
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from automa.tools.web import describe_fetch_request

MAX_AUTO_PATH_CHARS = 160
MAX_AUTO_READS_PER_TASK = 100
# A run of URL-safe characters this long looks like encoded data, not a page name.
_OPAQUE_RUN = re.compile(r"[A-Za-z0-9+/=_\-%]{40,}")


class AutoReadPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_per_task: int = Field(default=20, ge=1, le=MAX_AUTO_READS_PER_TASK)


def auto_read_refusal(url: object) -> str | None:
    """Why ``url`` may not be fetched without asking, or ``None`` if it may."""
    view = describe_fetch_request(url)
    if not view.valid:
        return view.error or "invalid URL"
    if view.scheme != "https":
        return "only https pages are read without asking"
    if view.warnings:
        return view.warnings[0]
    assert isinstance(url, str)  # noqa: S101 - describe_fetch_request validated it
    parts = urlsplit(url)
    if parts.query:
        return "URLs with a query string need approval"
    if parts.username is not None or parts.password is not None:
        return "URLs with credentials need approval"
    if "." not in (parts.hostname or "").strip("."):
        return "single-label host names (e.g. localhost) need approval"
    raw = parts.path or "/"
    path = unquote(raw)
    if len(raw) > MAX_AUTO_PATH_CHARS or len(path) > MAX_AUTO_PATH_CHARS:
        return "long paths need approval"
    if _OPAQUE_RUN.search(raw) or _OPAQUE_RUN.search(path):
        return "paths that look like encoded data need approval"
    return None
