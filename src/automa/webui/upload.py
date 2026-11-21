"""A strict ``multipart/form-data`` parser for answering a question (D159).

Only what the answer form sends is accepted: the listed text fields (each at most
once, UTF-8, small) and file parts under the listed file field (at most
``max_files``; each file's bytes and name). Anything else (an unknown field, a
nested multipart, a missing boundary, a body over the size cap) is refused with
an ``ActionRefused`` before anything is stored. Pure: no files, no network.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from automa.project.store import ProjectStoreError, validate_project_id
from automa.webui.actions import ActionRefused

MULTIPART = "multipart/form-data"
MAX_TEXT_FIELD_BYTES = 64 * 1024
MAX_PARTS = 40
MAX_UPLOAD_BYTES = 60 * 1024 * 1024  # an answer's whole body (text + files)
_BOUNDARY_RE = re.compile(r'boundary="?([0-9A-Za-z\'()+_,\-./:=? ]{1,70})"?(?:;|$)')
_DISP_RE = re.compile(r'(\w+)\*?="((?:[^"\\]|\\.)*)"|(\w+)=([^;]+)')


@dataclass
class Upload:
    fields: dict[str, str] = field(default_factory=dict)
    files: list[tuple[str, bytes]] = field(default_factory=list)


def _disposition(value: str) -> dict[str, str]:
    kind, _, rest = value.partition(";")
    if kind.strip().lower() != "form-data":
        raise ActionRefused.coded(400, "malformed_form")
    params: dict[str, str] = {}
    for m in _DISP_RE.finditer(rest):
        if m.group(1):
            params[m.group(1).lower()] = m.group(2).replace('\\"', '"')
        else:
            params[m.group(3).lower()] = m.group(4).strip()
    return params


def parse_multipart(
    headers: Mapping[str, str],
    body: bytes,
    *,
    fields: frozenset[str],
    file_field: str,
    max_bytes: int,
    max_files: int,
) -> Upload:
    """The text fields and files of a ``multipart/form-data`` body."""
    ctype = headers.get("content-type", "")
    if ctype.split(";", 1)[0].strip().lower() != MULTIPART:
        raise ActionRefused.coded(415, "form_only")
    if len(body) > max_bytes:
        raise ActionRefused.coded(413, "form_too_large")
    m = _BOUNDARY_RE.search(ctype)
    if m is None:
        raise ActionRefused.coded(400, "malformed_form")
    delim = b"--" + m.group(1).strip().encode("ascii")
    if not body.startswith(delim):
        raise ActionRefused.coded(400, "malformed_form")
    chunks = body.split(b"\r\n" + delim)
    first = chunks[0][len(delim) :]
    parts = [first, *chunks[1:]]
    out = Upload()
    closed = False
    for i, raw in enumerate(parts):
        if raw.startswith(b"--"):
            closed = True
            break
        if i >= MAX_PARTS or not raw.startswith(b"\r\n"):
            raise ActionRefused.coded(400, "malformed_form")
        head, sep, data = raw[2:].partition(b"\r\n\r\n")
        if not sep:
            raise ActionRefused.coded(400, "malformed_form")
        part_headers: dict[str, str] = {}
        try:
            for line in head.decode("utf-8").split("\r\n"):
                name, colon, value = line.partition(":")
                if not colon:
                    raise ActionRefused.coded(400, "malformed_form")
                key = name.strip().lower()
                if key in part_headers:
                    raise ActionRefused.coded(400, "malformed_form")
                part_headers[key] = value.strip()
        except UnicodeDecodeError:
            raise ActionRefused.coded(400, "malformed_form") from None
        disp = _disposition(part_headers.get("content-disposition", ""))
        name = disp.get("name", "")
        if part_headers.get("content-type", "").lower().startswith("multipart/"):
            raise ActionRefused.coded(400, "malformed_form")
        if "filename" in disp:
            if name != file_field:
                raise ActionRefused.coded(400, "unexpected_field")
            filename = disp["filename"]
            if not filename and not data:
                continue  # the browser's empty file input
            if len(out.files) >= max_files:
                raise ActionRefused.coded(400, "unexpected_field")
            out.files.append((filename or "file", data))
            continue
        if name not in fields or name in out.fields or len(data) > MAX_TEXT_FIELD_BYTES:
            raise ActionRefused.coded(400, "unexpected_field")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ActionRefused.coded(400, "malformed_form") from None
        out.fields[name] = text.replace("\r\n", "\n").replace("\r", "\n")
    if not closed:
        raise ActionRefused.coded(400, "malformed_form")
    return out


REQUEST_ANSWER_FIELDS = frozenset({"csrf", "answer", "confirm"})
DECLINE_FIELDS = frozenset({"csrf", "reason"})
FILE_FIELD = "files"
_REQUEST_ACTION_RE = re.compile(r"^/projects/([^/]+)/requests/(req-[0-9a-f]{12})/(answer|decline)$")


def request_action_href(project_id: str, request_id: str, action: str) -> str:
    return f"/projects/{project_id}/requests/{request_id}/{action}"


def parse_request_action(path: str) -> tuple[str, str, str] | None:
    """``(project_id, request_id, action)`` of an answer/decline path, else ``None``."""
    m = _REQUEST_ACTION_RE.match(path)
    if m is None:
        return None
    try:
        return validate_project_id(m.group(1)), m.group(2), m.group(3)
    except ProjectStoreError:
        return None
