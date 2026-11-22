"""What a plan depends on beyond the workspace, warned about before "Start" (M16, D164).

The Executive marks each task's ``needs`` (``live_web_data``, ``login``,
``paid_api``, ``gui``). This module turns them into:

* :func:`plan_warnings`: one warning per need kind with the tasks that have it, in
  canonical order, for the project page and ``automa new``; the warning text and
  the cheaper, honest alternative ("give me the CSV yourself") are fixed
  translations chosen by code, never model text;
* :func:`worker_note`: the lines added to a worker's task message, so the worker
  knows what the plan allowed and never fills a gap with invented data.

Pure: no files, no model calls. Grants nothing: the Permission Policy, the
sandbox and the BudgetGuard decide what a tool may do, exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass

from automa.plan.models import TASK_NEEDS, Plan, Task, TaskNeed


@dataclass(frozen=True)
class FeasibilityWarning:
    """One need kind and the titles of the tasks (plan order) that have it."""

    need: TaskNeed
    tasks: tuple[str, ...]


def plan_warnings(plan: Plan) -> tuple[FeasibilityWarning, ...]:
    """The plan's needs grouped by kind (canonical order); empty if none."""
    out: list[FeasibilityWarning] = []
    for need in TASK_NEEDS:
        titles = tuple(t.title for t in plan.tasks if need in t.needs)
        if titles:
            out.append(FeasibilityWarning(need=need, tasks=titles))
    return tuple(out)


# English, for the CLI and the worker (the web page has its own fa/en messages).
WARNING_TEXT: dict[TaskNeed, str] = {
    "live_web_data": (
        "needs live data from websites: the tools may not get reliable data from "
        "some sites (heavy JavaScript, blocking, login) and retries cost money. "
        "Cheaper: give the data yourself as an Excel/CSV file."
    ),
    "login": (
        "needs an account behind a login: Automa never takes your password. "
        "Export the data from that account yourself and give it as a file."
    ),
    "paid_api": (
        "needs a paid service: it works only if you set up its key yourself (never "
        "in the idea text). Otherwise give the data as a file."
    ),
    "gui": (
        "needs a desktop program with a window: Automa cannot click in it. It will "
        "make the files or script, and you run that step yourself."
    ),
}

_WORKER_ALLOWED: dict[TaskNeed, str] = {
    "live_web_data": (
        "- live_web_data: this task may fetch live data from the internet. Use only "
        "data you really fetched; if a source fails, is blocked or needs JavaScript "
        "or a login, do not invent or estimate numbers: ask the user (ask_user) for "
        "the data as a file (Excel/CSV), or say clearly in your summary what is missing."
    ),
    "login": (
        "- login: never ask for or store a password; ask the user (ask_user) to "
        "export the data from that account and give it as a file."
    ),
    "paid_api": (
        "- paid_api: call the paid service only if its key is already configured; "
        "never ask for the key in a question. Without it, ask the user (ask_user) "
        "for the data as a file instead."
    ),
    "gui": (
        "- gui: you cannot operate a desktop program; produce files or a script the "
        "user runs, and explain that step in plain words."
    ),
}

NO_WEB_NOTE = (
    "The plan did not mark this task as needing live web data: do not download data "
    "from the internet with scripts; use the given files or ask the user (ask_user)."
)


def worker_note(task: Task) -> str:
    """The ``## What this task depends on`` section of a worker message, or ``""``
    for a task without needs (its message stays exactly as before)."""
    if not task.needs:
        return ""
    lines = ["## What this task depends on (from the plan)"]
    lines.extend(_WORKER_ALLOWED[n] for n in task.needs)
    if "live_web_data" not in task.needs:
        lines.append(NO_WEB_NOTE)
    return "\n".join(lines)


def cli_lines(plan: Plan) -> list[str]:
    """Warning lines for ``automa new``/``automa plan`` output; empty if none."""
    lines: list[str] = []
    for w in plan_warnings(plan):
        lines.append(f"warning: {', '.join(w.tasks)}: {WARNING_TEXT[w.need]}")
    return lines
