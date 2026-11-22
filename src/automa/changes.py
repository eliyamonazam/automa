"""Change requests after delivery: go on with the same project (D169).

Like asking Claude for a change after seeing the result: on a ``done`` or
``failed`` project the user writes what to change (and may attach files). The
Executive adds the tasks the change needs to the plan (next plan version; the
tasks that ran stay as they are, passed), updates the deliverable, and the
project becomes ``paused``, ready for ``automa run`` (the web UI starts it).

Every request is kept in ``<project>/changes.json`` (oldest first) and shown on
the project page; the final review (D168) sees them too. The Executive call goes
through the same BudgetGuard as every call.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from automa.agents.executive import ExecutiveAgent
from automa.delivery import render_markdown, try_build_delivery
from automa.ideafiles import describe, list_files
from automa.plan import Plan
from automa.project.store import ProjectStatus, ProjectStore
from automa.tools.workspace import Workspace, WorkspaceError
from automa.userinput import MAX_ANSWER_FILE_BYTES, MAX_ANSWER_FILES, safe_file_name

CHANGES_FILE = "changes.json"
MAX_CHANGE_CHARS = 5_000
MAX_CHANGES = 200
CHANGE_INPUTS = "inputs/change-{n}"


class ChangeError(Exception):
    """The change cannot be requested now (message shown to the user)."""


class ChangeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    at: datetime
    text: str = Field(min_length=1, max_length=MAX_CHANGE_CHARS)
    plan_version: int = Field(ge=1)
    new_tasks: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    via: str = Field(default="cli", max_length=20)


_LIST = TypeAdapter(list[ChangeRecord])


def load_changes(project_dir: Path) -> list[ChangeRecord]:
    path = project_dir / CHANGES_FILE
    if path.is_symlink() or not path.is_file():
        return []
    try:
        return _LIST.validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return []


def _save_changes(project_dir: Path, changes: list[ChangeRecord]) -> None:
    path = project_dir / CHANGES_FILE
    if path.is_symlink():
        raise ChangeError(f"{path} must not be a symlink")
    data = json.dumps(
        [c.model_dump(mode="json") for c in changes[-MAX_CHANGES:]], ensure_ascii=False, indent=1
    )
    fd, tmp = tempfile.mkstemp(prefix=".changes.", suffix=".tmp", dir=project_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def can_change(store: ProjectStore, project_id: str) -> bool:
    try:
        status = store.load(project_id).status
        return status in (ProjectStatus.DONE, ProjectStatus.FAILED) and store.has_plan(project_id)
    except Exception:  # noqa: BLE001 - a page must never break on this question
        return False


def check_change(text: str, files: Sequence[tuple[str, bytes]] = ()) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ChangeError("write what should change")
    if len(text) > MAX_CHANGE_CHARS:
        raise ChangeError(f"the change request is longer than {MAX_CHANGE_CHARS} characters")
    if len(files) > MAX_ANSWER_FILES:
        raise ChangeError(f"at most {MAX_ANSWER_FILES} files")
    for name, data in files:
        if len(data) > MAX_ANSWER_FILE_BYTES:
            raise ChangeError(f"{safe_file_name(name)} is too large")
    return text


def request_change(
    agent: ExecutiveAgent,
    store: ProjectStore,
    project_id: str,
    text: str,
    files: Sequence[tuple[str, bytes]] = (),
    *,
    via: str = "cli",
    now: datetime | None = None,
) -> Plan:
    """Plan a change on a done/failed project; returns the new plan (D169).

    Raises ``ChangeError`` for a request that cannot be made now; Executive and
    budget errors propagate (the project keeps its status).
    """
    text = check_change(text, files)
    if not can_change(store, project_id):
        raise ChangeError(f"project {project_id} must be done or failed to request a change")
    project_dir = store.project_dir(project_id)
    changes = load_changes(project_dir)
    saved: list[str] = []
    if files:
        root = store.files_dir(project_id)
        root.mkdir(parents=True, exist_ok=True)
        try:
            ws = Workspace(root)
            folder = CHANGE_INPUTS.format(n=len(changes) + 1)
            for name, data in files:
                path = f"{folder}/{safe_file_name(name)}"
                ws.write_bytes(path, data, max_bytes=MAX_ANSWER_FILE_BYTES)
                saved.append(path)
        except WorkspaceError as e:
            raise ChangeError(f"could not save the files: {e}") from None
    plan = store.load_plan(project_id)
    delivery = try_build_delivery(store, project_id)
    request = text
    if saved:
        request += "\n\nFiles the user attached for this change: " + ", ".join(saved)
    new = agent.plan_change(
        plan,
        request,
        project_id=project_id,
        files=describe(list_files(store.files_dir(project_id))),
        delivery=render_markdown(delivery, "en") if delivery is not None else "",
    )
    before = store.task_states(project_id)
    store.save_plan(project_id, new)
    # Tasks that already ran keep their state in the new plan version.
    for state in before.values():
        store.save_task_state(project_id, state.model_copy(update={"plan_version": new.version}))
    old_ids = {t.id for t in plan.tasks}
    record = ChangeRecord(
        at=now or datetime.now(UTC),
        text=text,
        plan_version=new.version,
        new_tasks=tuple(t.id for t in new.tasks if t.id not in old_ids),
        files=tuple(saved),
        via=via,
    )
    _save_changes(project_dir, [*changes, record])
    store.set_status(project_id, ProjectStatus.PAUSED)
    return new
