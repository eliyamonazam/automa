"""Notifications for approvals and run reports (core only, D79).

A notification leaves the machine (a chat service or an email server), so its content
is deliberately small:

* only ids, names, statuses, counts and the cost of the run;
* **never** tool arguments, task text, run ``detail`` (it may carry provider error
  text) or model output — the user reviews those locally with ``automa approvals``;
* control, format and bidi characters shown as visible escapes, lengths capped;
* a notification never approves anything: it only names the terminal command.

The channel is a ``Notifier``. This module ships only ``FakeNotifier`` (for tests); no
real service (Telegram, email) is chosen or wired yet. ``deliver`` never raises for a
channel failure and never shows the channel's error text (it may contain a token).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from automa.budget.ledger import nusd_to_usd
from automa.orchestrator import RunReport, StopReason
from automa.project.store import PROJECT_ID_PATTERN
from automa.tools.approvals import ApprovalRequest

MAX_TITLE_CHARS = 120
MAX_BODY_CHARS = 3000  # below Telegram's 4096-character message limit
MAX_REASON_CHARS = 200
MAX_LISTED_TASKS = 10
NOTIFIER_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_ALLOWED_FORMAT_CHARS = frozenset({"‌", "‍"})  # Persian joiners
_ELLIPSIS = "…"


def visible(value: str, *, newlines: bool = True) -> str:
    """``value`` with control/format characters as visible ``\\xNN``/``\\uNNNN``."""
    out = []
    for ch in value:
        if (newlines and ch == "\n") or ch in _ALLOWED_FORMAT_CHARS:
            out.append(ch)
        elif ch.isprintable() and unicodedata.category(ch) != "Cf":
            out.append(ch)
        else:
            code = ord(ch)
            out.append(f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}")
    return "".join(out)


def _cap(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + _ELLIPSIS


def clean(value: str, limit: int, *, newlines: bool = True) -> str:
    """Visible-escaped and capped to ``limit`` characters."""
    return _cap(visible(value, newlines=newlines), limit)


class NotificationKind(StrEnum):
    APPROVAL_PENDING = "approval_pending"
    RUN_FINISHED = "run_finished"


class Notification(BaseModel):
    """One message to the owner. Text is cleaned and capped on construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: NotificationKind
    project_id: str = Field(pattern=PROJECT_ID_PATTERN)
    title: str = Field(min_length=1)
    body: str = ""

    @field_validator("title")
    @classmethod
    def _title(cls, v: str) -> str:
        return clean(v, MAX_TITLE_CHARS, newlines=False)

    @field_validator("body")
    @classmethod
    def _body(cls, v: str) -> str:
        return clean(v, MAX_BODY_CHARS)

    def text(self) -> str:
        """Plain text form: title, blank line, body."""
        return f"{self.title}\n\n{self.body}" if self.body else self.title


class NotifierError(Exception):
    """The channel failed. Its message is never shown or recorded."""


@runtime_checkable
class Notifier(Protocol):
    """A delivery channel. ``name`` is recorded in events; nothing else about it is."""

    @property
    def name(self) -> str: ...

    def send(self, notification: Notification) -> None:
        """Deliver once; raise ``NotifierError`` on failure."""
        ...


class FakeNotifier:
    """Records every notification it is given, for tests."""

    def __init__(self, *, name: str = "fake", error: Exception | None = None) -> None:
        if not NOTIFIER_NAME_RE.fullmatch(name):
            raise ValueError("invalid notifier name")
        self._name = name
        self._error = error
        self.sent: list[Notification] = []

    @property
    def name(self) -> str:
        return self._name

    def send(self, notification: Notification) -> None:
        if self._error is not None:
            raise self._error
        self.sent.append(notification)


class DeliveryResult(BaseModel):
    """Outcome of ``deliver``. ``error`` is only an exception type name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    notifier: str
    kind: NotificationKind
    project_id: str
    sent: bool
    error: str | None = None

    def event_data(self) -> dict[str, object]:
        """Data for a ``notify.sent`` / ``notify.failed`` event (no message text)."""
        data: dict[str, object] = {
            "notifier": self.notifier,
            "kind": self.kind.value,
            "sent": self.sent,
        }
        if self.error is not None:
            data["error"] = self.error
        return data


def _notifier_name(notifier: object) -> str:
    try:
        name = getattr(notifier, "name", "")
    except Exception:  # noqa: BLE001 - a broken notifier must not break the caller
        return "invalid"
    return name if isinstance(name, str) and NOTIFIER_NAME_RE.fullmatch(name) else "invalid"


def deliver(notifier: Notifier, notification: Notification) -> DeliveryResult:
    """Send ``notification``; a channel failure is reported, never raised.

    ``KeyboardInterrupt``/``SystemExit`` still propagate. The exception text is
    dropped: HTTP and SMTP libraries often include URLs or credentials in it.
    """
    name = _notifier_name(notifier)

    def result(sent: bool, error: str | None = None) -> DeliveryResult:
        return DeliveryResult(
            notifier=name,
            kind=notification.kind,
            project_id=notification.project_id,
            sent=sent,
            error=error,
        )

    if name == "invalid":
        return result(False, "InvalidNotifier")
    try:
        notifier.send(notification)
    except Exception as e:  # noqa: BLE001 - see docstring
        return result(False, type(e).__name__[:64])
    return result(True)


def _time(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def approval_notification(req: ApprovalRequest) -> Notification:
    """A pending Approval Gate request: who asks, which tool — never the arguments."""
    lines = [
        f"project: {req.project_id}",
        f"task: {req.task_id or '-'}",
        f"tool: {req.tool} (risk {req.risk.value})",
        f"reason: {clean(req.reason, MAX_REASON_CHARS, newlines=False)}",
        f"request: {req.id}",
        f"created: {_time(req.created_at)}",
        f"status: {req.status.value}",
        "",
        "Arguments are not included in notifications.",
        f"Review them: automa approvals {req.project_id}",
        f"Decide in a terminal: automa approve {req.project_id} {req.id}",
    ]
    return Notification(
        kind=NotificationKind.APPROVAL_PENDING,
        project_id=req.project_id,
        title=f"Approval needed: {req.tool} in {req.project_id}",
        body="\n".join(lines),
    )


def _tasks(label: str, ids: list[str]) -> str:
    shown = ", ".join(ids[:MAX_LISTED_TASKS])
    more = f" (+{len(ids) - MAX_LISTED_TASKS} more)" if len(ids) > MAX_LISTED_TASKS else ""
    return f"{label} ({len(ids)}): {shown or '-'}{more}"


def run_notification(report: RunReport, *, cost_nusd: int | None = None) -> Notification:
    """End of one ``automa run``: outcome and counts — never ``report.detail``."""
    if cost_nusd is not None and cost_nusd < 0:
        raise ValueError("cost_nusd must not be negative")
    lines = [
        f"project: {report.project_id}",
        f"status: {report.status.value}",
        f"stop: {report.reason.value}",
        f"attempts this run: {report.attempts_run}",
        _tasks("passed", report.passed),
        _tasks("failed", report.failed),
        _tasks("blocked", report.blocked),
    ]
    if report.task_id is not None:
        lines.append(f"last task: {report.task_id}")
    if cost_nusd is not None:
        lines.append(f"cost this run: ${nusd_to_usd(cost_nusd):.4f}")
    lines.append("")
    if report.reason is StopReason.APPROVAL and report.approval_id is not None:
        lines.append(
            f"Decide in a terminal: automa approve {report.project_id} {report.approval_id}"
        )
    lines.append(f"Details: automa status {report.project_id}")
    return Notification(
        kind=NotificationKind.RUN_FINISHED,
        project_id=report.project_id,
        title=f"Run {report.reason.value}: {report.project_id}",
        body="\n".join(lines),
    )
