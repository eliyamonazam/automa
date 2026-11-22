""" "What is it doing right now" for a running project (D171).

A plain sentence from the latest events of the current run: which step is being
worked on (or checked, or the final review), and the last thing done, such as
"wrote index.html" or "read a page on example.com", with how long ago. Built
only from what the events already record (the ``audit`` fields of each tool:
file paths, program names, web hosts; never arguments, file contents or queries).
Read-only and pure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from automa.events import Event

# Last actions shown, by tool: the i18n key and the audit field it names.
_ACTIONS: dict[str, tuple[str, str | None]] = {
    "write_file": ("activity.wrote", "path"),
    "edit_file": ("activity.edited", "path"),
    "read_file": ("activity.read", "path"),
    "list_dir": ("activity.listed", None),
    "delete_file": ("activity.deleted", "path"),
    "run_command": ("activity.ran", "program"),
    "fetch_url": ("activity.fetched", "host"),
    "web_search": ("activity.searched", None),
    "github_read": ("activity.github", None),
    "generate_image": ("activity.image", None),
}
_MAX_NAME = 80


class Activity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: str  # "working" | "checking" | "reviewing"
    task_title: str | None = None
    action_key: str | None = None
    action_arg: str = ""
    at: datetime | None = None


def _short(value: object) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= _MAX_NAME else text[: _MAX_NAME - 1] + "…"


def current_activity(
    events: Sequence[Event], titles: Mapping[str, str] | None = None
) -> Activity | None:
    """The activity of the latest run (``None`` before any step of it); ``titles``
    maps task ids to their titles."""
    start = 0
    for i in range(len(events) - 1, -1, -1):
        if events[i].type == "orchestrator.started":
            start = i
            break
    run = events[start:]
    titles = titles or {}
    phase: str | None = None
    task: str | None = None
    action: tuple[str, str] | None = None
    at: datetime | None = None
    for e in run:
        if e.type in ("orchestrator.attempt", "worker.started"):
            phase, task, action = "working", str(e.data.get("task") or ""), None
            at = e.ts
        elif e.type in ("verifier.checks", "verifier.evaluator"):
            phase, task = "checking", str(e.data.get("task") or task or "")
            at = e.ts
        elif e.type == "final_review.started":
            phase, task, action, at = "reviewing", None, None, e.ts
        elif e.type == "tool.call" and phase == "working":
            tool = str(e.data.get("tool") or "")
            if tool in _ACTIONS and e.data.get("ok") is not False:
                key, field = _ACTIONS[tool]
                arg = _short(e.data.get(field, "")) if field else ""
                action = (key, arg)
                at = e.ts
    if phase is None:
        return None
    return Activity(
        phase=phase,
        task_title=_short(titles.get(task) or task) if task else None,
        action_key=action[0] if action else None,
        action_arg=action[1] if action else "",
        at=at,
    )
