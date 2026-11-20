"""Structured warnings on a ``fetch_url`` / ``github_read`` request view (D148).

A request view (``describe_fetch_request``, ``describe_github_request``) keeps its
English ``warnings`` text exactly as the CLI prints it; next to it, ``warning_codes``
carries the same warnings as a code and parameters, so the web UI can show them in
the owner's language without matching English strings. The English text of every
warning is made from :data:`WARNING_TEXT`, the single source the web UI's ``en``
templates also come from. Pure: no files, no network.
"""

from __future__ import annotations

from collections.abc import Mapping
from string import Formatter
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict

WARNING_TEXT: Mapping[str, str] = MappingProxyType(
    {
        "fetch.http": "not encrypted (http): anyone on the network can read or change it",
        "fetch.ip_host": "the host is an IP address, not a name",
        "fetch.non_ascii_host": "non-ASCII host name; it is sent as {wire_host}",
        "fetch.punycode": "punycode host name (xn--): it may look like another name",
        "fetch.port": "non-default port {port}",
        "fetch.query": (
            "the query string ({count} parameter(s)) is sent to {wire_host}: "
            "check it carries no secrets or project data"
        ),
        "fetch.long_path": "very long path: check it carries no project data",
        "github.no_ref": (
            "no ref: reads the default branch as it is when the task runs, "
            "which may differ from now"
        ),
        "github.root_path": "path is the repository root: a directory listing",
    }
)


class RequestWarning(BaseModel):
    """One warning as a code and its parameters (values are technical: hosts, ports).

    ``port`` is kept as text so a translated page shows it with the same digits as the
    URL; ``count`` is a plain number."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    params: dict[str, str | int] = {}

    def english(self) -> str:
        return WARNING_TEXT[self.code].format(**self.params)


def warn(code: str, **params: str | int) -> RequestWarning:
    """A warning; an unknown code or missing/extra parameter raises at once."""
    if code not in WARNING_TEXT:
        raise ValueError(f"unknown request warning: {code!r}")
    fields = {f for _, f, _, _ in Formatter().parse(WARNING_TEXT[code]) if f}
    if set(params) != fields:
        raise ValueError(f"{code}: parameters {sorted(params)} != {sorted(fields)}")
    return RequestWarning(code=code, params=dict(params))
