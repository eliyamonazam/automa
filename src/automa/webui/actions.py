"""Action mode of the local web UI: approve/reject Approval Gate requests (D123).

Only active with ``automa web --allow-actions``; the default stays read-only (D78).
Every ``POST`` must pass, in this order, before anything on disk is read or changed:

1. the ``Host`` check of the server (DNS-rebinding defence, unchanged);
2. a known action path ``/projects/<pid>/approvals/<id>/approve|reject``;
3. same-origin proof: ``Sec-Fetch-Site`` (if sent) is ``same-origin`` and ``Origin``
   (or, only when ``Origin`` is missing, ``Referer``) is exactly this server; a request
   with neither, or with ``Origin: null``, is refused;
4. ``Content-Type: application/x-www-form-urlencoded`` and a small, well-formed body
   (each field at most once, valid UTF-8);
5. the random per-server-session CSRF token, compared in constant time.

Then the request must still be pending, its fingerprint must equal the one shown in
the form (so a different request can never be approved by a stale page), and an
approval needs the explicit "I reviewed the full arguments" checkbox — the browser
equivalent of the ``automa approve`` confirmation. The decision itself is the same
:func:`automa.tools.approval_decisions.record_decision` the CLI uses.
"""

from __future__ import annotations

import hmac
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit

from automa.project.store import ProjectStore, ProjectStoreError, validate_project_id
from automa.tools.approval_decisions import record_decision
from automa.tools.approvals import MAX_NOTE_CHARS, ApprovalError, ApprovalRequest

MAX_FORM_BYTES = 8_192
FORM_CONTENT_TYPE = "application/x-www-form-urlencoded"
CONFIRM_VALUE = "yes"
_FORM_FIELDS = frozenset({"csrf", "fingerprint", "confirm", "note"})
_ACTION_RE = re.compile(r"^/projects/([^/]+)/approvals/([^/]+)/(approve|reject)$")


class ActionRefused(Exception):
    """A refused action, with the HTTP status and a short, safe message."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class ActionSettings:
    """Per-server-session secrets and the exact origins of this server."""

    csrf_token: str
    origins: frozenset[str]


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def allowed_origins(hosts: frozenset[str]) -> frozenset[str]:
    """``http://<host>`` for every accepted ``Host`` value (lower-case)."""
    return frozenset(f"http://{h.lower()}" for h in hosts)


@dataclass(frozen=True)
class ActionTarget:
    project_id: str
    approval_id: str
    approve: bool


def parse_action_path(path: str) -> ActionTarget | None:
    """The target of a ``POST`` path, or ``None`` if it is not an action path."""
    m = _ACTION_RE.fullmatch(path)
    if m is None:
        return None
    try:
        pid = validate_project_id(m.group(1))
    except ProjectStoreError:
        return None
    return ActionTarget(project_id=pid, approval_id=m.group(2), approve=m.group(3) == "approve")


def _origin_of(url: str) -> str | None:
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme != "http" or not parts.netloc or "@" in parts.netloc:
        return None
    return f"http://{parts.netloc.lower()}"


def check_same_origin(headers: Mapping[str, str], settings: ActionSettings) -> None:
    """Refuse (403) anything that does not prove it came from this server's own page."""
    site = headers.get("sec-fetch-site")
    if site is not None and site.strip().lower() != "same-origin":
        raise ActionRefused(403, "cross-site request refused")
    origin = headers.get("origin")
    if origin is not None:
        if origin.strip().lower() not in settings.origins:
            raise ActionRefused(403, "request from another origin refused")
        return
    referer = headers.get("referer")
    if referer is None:
        raise ActionRefused(403, "request without Origin or Referer refused")
    if _origin_of(referer.strip()) not in settings.origins:
        raise ActionRefused(403, "request from another origin refused")


def parse_form(
    headers: Mapping[str, str],
    body: bytes,
    *,
    fields: frozenset[str] = _FORM_FIELDS,
    max_bytes: int = MAX_FORM_BYTES,
) -> dict[str, str]:
    """The fields of a small urlencoded form; anything unusual is refused.

    ``fields`` are the only accepted names (each at most once)."""
    ctype = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if ctype != FORM_CONTENT_TYPE:
        raise ActionRefused(415, "only HTML form submissions are accepted")
    if len(body) > max_bytes:
        raise ActionRefused(413, "form is too large")
    try:
        raw = body.decode("ascii")
        pairs = parse_qsl(
            raw,
            keep_blank_values=True,
            strict_parsing=bool(raw),
            encoding="utf-8",
            errors="strict",
            max_num_fields=len(fields),
        )
    except (UnicodeDecodeError, ValueError):
        raise ActionRefused(400, "malformed form") from None
    form: dict[str, str] = {}
    for key, value in pairs:
        if key not in fields or key in form:
            raise ActionRefused(400, "unexpected form field")
        form[key] = value
    return form


def check_token(form: Mapping[str, str], settings: ActionSettings) -> None:
    token = form.get("csrf", "")
    if not token or not hmac.compare_digest(
        token.encode("utf-8"), settings.csrf_token.encode("utf-8")
    ):
        raise ActionRefused(403, "missing or wrong form token; reload the page")


def decide_from_form(
    store: ProjectStore, target: ActionTarget, form: Mapping[str, str]
) -> ApprovalRequest:
    """Record the decision of an already origin- and token-checked form."""
    if target.project_id not in store.list_ids():
        raise ActionRefused(404, "no such project")
    try:
        req = store.approvals(target.project_id).get(target.approval_id)
    except ApprovalError:
        raise ActionRefused(404, "no such approval request") from None
    if req.decided:
        raise ActionRefused(409, f"request {req.id} is already {req.status.value}")
    if not hmac.compare_digest(form.get("fingerprint", "").encode(), req.fingerprint.encode()):
        raise ActionRefused(409, "the request differs from the page you saw; reload it")
    if target.approve and form.get("confirm") != CONFIRM_VALUE:
        raise ActionRefused(400, "tick the box to confirm you reviewed the full arguments")
    note = form.get("note")
    if note is not None and len(note.strip()) > MAX_NOTE_CHARS:
        raise ActionRefused(400, f"note is longer than {MAX_NOTE_CHARS} characters")
    try:
        return record_decision(
            store,
            target.project_id,
            req.id,
            approve=target.approve,
            note=note,
            via="web",
        )
    except (ApprovalError, ProjectStoreError) as e:
        raise ActionRefused(409, str(e)) from None
