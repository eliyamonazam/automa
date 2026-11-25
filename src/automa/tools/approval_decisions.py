"""Recording one Approval Gate decision, shared by ``automa approve``/``reject`` and the
web UI's action mode (D123).

Exactly one place decides a pending request and writes the ``approval.decided`` event,
so the terminal and the browser cannot drift apart. Nothing here asks for confirmation;
callers show the full request and get an explicit confirmation first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from automa.events import Event
from automa.tools.approvals import ApprovalRequest

if TYPE_CHECKING:
    from automa.project.store import ProjectStore

DECIDED_EVENT = "approval.decided"


def record_decision(
    store: ProjectStore,
    project_id: str,
    approval_id: str,
    *,
    approve: bool,
    note: str | None = None,
    via: str | None = None,
    by: str | None = None,
) -> ApprovalRequest:
    """Decide a pending request once and log ``approval.decided``.

    ``note`` is stripped (empty → ``None``). ``via`` (e.g. ``"web"``) is added to the
    event only when given, so terminal decisions keep their original event shape;
    ``by`` (the signed-in web account, D186) likewise.
    Raises ``ApprovalError`` if the request is missing, already decided or invalid.
    """
    if note is not None:
        note = note.strip() or None
    decided = store.approvals(project_id).decide(approval_id, approve=approve, note=note)
    data: dict[str, object] = {
        "approval": decided.id,
        "task": decided.task_id,
        "tool": decided.tool,
        "status": decided.status.value,
        "note": decided.note is not None,
    }
    if via is not None:
        data["via"] = via
    if by is not None:
        data["by"] = by
    store.events(project_id).emit(
        Event(ts=decided.decided_at or decided.created_at, type=DECIDED_EVENT, data=data)
    )
    return decided
