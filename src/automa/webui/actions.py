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
from automa.webui.refusal import Refusal

MAX_FORM_BYTES = 8_192
FORM_CONTENT_TYPE = "application/x-www-form-urlencoded"
CONFIRM_VALUE = "yes"
_FORM_FIELDS = frozenset({"csrf", "fingerprint", "confirm", "note"})
_ACTION_RE = re.compile(r"^/projects/([^/]+)/approvals/([^/]+)/(approve|reject)$")


class ActionRefused(Refusal):
    """A refused action, with the HTTP status and a short, safe message (D147: a
    ``code`` translates it)."""


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
        raise ActionRefused.coded(403, "cross_site")
    origin = headers.get("origin")
    if origin is not None:
        if origin.strip().lower() not in settings.origins:
            raise ActionRefused.coded(403, "other_origin")
        return
    referer = headers.get("referer")
    if referer is None:
        raise ActionRefused.coded(403, "no_origin")
    if _origin_of(referer.strip()) not in settings.origins:
        raise ActionRefused.coded(403, "other_origin")


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
        raise ActionRefused.coded(415, "form_only")
    if len(body) > max_bytes:
        raise ActionRefused.coded(413, "form_too_large")
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
        raise ActionRefused.coded(400, "malformed_form") from None
    form: dict[str, str] = {}
    for key, value in pairs:
        if key not in fields or key in form:
            raise ActionRefused.coded(400, "unexpected_field")
        # A browser sends textarea line breaks as CRLF (HTML spec); store LF (D131).
        form[key] = value.replace("\r\n", "\n").replace("\r", "\n")
    return form


def check_token(form: Mapping[str, str], settings: ActionSettings) -> None:
    token = form.get("csrf", "")
    if not token or not hmac.compare_digest(
        token.encode("utf-8"), settings.csrf_token.encode("utf-8")
    ):
        raise ActionRefused.coded(403, "bad_token")


def decide_from_form(
    store: ProjectStore,
    target: ActionTarget,
    form: Mapping[str, str],
    *,
    by: str | None = None,
) -> ApprovalRequest:
    """Record the decision of an already origin- and token-checked form; ``by`` is
    the signed-in account that decided (D186), kept in the event."""
    if target.project_id not in store.list_ids():
        raise ActionRefused.coded(404, "no_such_project")
    try:
        req = store.approvals(target.project_id).get(target.approval_id)
    except ApprovalError:
        raise ActionRefused.coded(404, "no_such_request") from None
    if req.decided:
        raise ActionRefused.coded(409, "already_decided", id=req.id, status=req.status.value)
    if not hmac.compare_digest(form.get("fingerprint", "").encode(), req.fingerprint.encode()):
        raise ActionRefused.coded(409, "request_changed")
    if target.approve and form.get("confirm") != CONFIRM_VALUE:
        raise ActionRefused.coded(400, "confirm_arguments")
    note = form.get("note")
    if note is not None and len(note.strip()) > MAX_NOTE_CHARS:
        raise ActionRefused.coded(400, "note_too_long", max=MAX_NOTE_CHARS)
    try:
        return record_decision(
            store,
            target.project_id,
            req.id,
            approve=target.approve,
            note=note,
            via="web",
            by=by,
        )
    except (ApprovalError, ProjectStoreError) as e:
        raise ActionRefused(409, str(e), lead="lead.store") from None
