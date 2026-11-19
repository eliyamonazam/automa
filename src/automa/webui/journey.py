"""The project's journey as steps a non-programmer can follow (D141).

A project page shows one row of steps — idea, questions, plan, each task of the plan,
result — each *done*, *now*, *todo* or *fail*, an overall percentage for the progress
bar, and one plain sentence saying what is happening now and what to do next.

Pure: built only from the page's existing view models (no files, no network). Task
titles are recorded data and stay as written; everything else comes from
:data:`MESSAGES`, checked like the main translation table (same fields in both
languages).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from automa.l10n import LANGUAGES, Language, localize_digits
from automa.webui.i18n import check_messages, template_fields
from automa.webui.views import ProjectView

StepState = Literal["done", "now", "todo", "fail"]
Tone = Literal["info", "busy", "ok", "bad"]

_DONE_TASK = frozenset({"passed"})
_NOW_TASK = frozenset({"running", "verifying"})
_FAIL_TASK = frozenset({"failed", "blocked"})

_RAW: dict[str, dict[Language, str]] = {
    "nav.back": {"en": "← all projects", "fa": "→ همهٔ پروژه‌ها"},
    "cost.live": {
        "en": "This run so far, still running; estimated {range}",
        "fa": "این اجرا تا این لحظه، هنوز در حال کار؛ پیش‌بینی: {range}",
    },
    "cost.below": {
        "en": "This run below the estimated range: {range}",
        "fa": "این اجرا کمتر از بازهٔ پیش‌بینی خرج کرد: {range}",
    },
    "cost.within": {
        "en": "This run within the estimated range: {range}",
        "fa": "این اجرا داخل بازهٔ پیش‌بینی بود: {range}",
    },
    "cost.above": {
        "en": "This run above the estimated range: {range}",
        "fa": "این اجرا از بازهٔ پیش‌بینی بیشتر خرج کرد: {range}",
    },
    "cost.nothing": {"en": "Nothing spent yet.", "fa": "هنوز هیچ خرجی نشده."},
    "cost.planning": {"en": "Planning: {amount}", "fa": "برنامه‌ریزی: {amount}"},
    "cost.tasks": {"en": "Tasks: {amount}", "fa": "انجام کارها: {amount}"},
    "cost.each": {"en": "Cost of each task", "fa": "هزینهٔ هر مرحله"},
    "step.idea": {"en": "Idea", "fa": "ایده"},
    "step.questions": {"en": "Questions", "fa": "سؤال‌ها"},
    "step.plan": {"en": "Plan", "fa": "برنامه"},
    "step.result": {"en": "Result", "fa": "نتیجه"},
    "now.questions": {
        "en": "Waiting for your answers to a few questions.",
        "fa": "منتظر جواب تو به چند تا سؤاله.",
    },
    "now.planning": {
        "en": "Making the plan…",
        "fa": "داره برنامه می‌ریزه…",
    },
    "now.ready": {
        "en": "The plan is ready. Check the cost and press Start.",
        "fa": "برنامه آماده‌ست. هزینه رو ببین و «شروع» رو بزن.",
    },
    "now.approval": {
        "en": "Waiting for you: {count} request(s) need your OK below.",
        "fa": "منتظر توئه: {count} درخواست پایین صفحه تأیید تو رو می‌خواد.",
    },
    "now.working": {
        "en": "Working on step {step} of {total}: {title}",
        "fa": "داره روی مرحلهٔ {step} از {total} کار می‌کنه: {title}",
    },
    "now.paused": {
        "en": "Paused after {done} of {total} steps. Press Start to continue.",
        "fa": "بعد از {done} مرحله از {total} متوقف شد. برای ادامه «شروع» رو بزن.",
    },
    "now.done": {
        "en": "Done! The result is in Files.",
        "fa": "تموم شد! نتیجه توی «فایل‌ها»ست.",
    },
    "now.failed": {
        "en": "Stopped: the step “{title}” did not pass its check.",
        "fa": "متوقف شد: مرحلهٔ «{title}» در بررسی قبول نشد.",
    },
    "now.failed_plain": {
        "en": "Stopped before finishing.",
        "fa": "قبل از تموم شدن متوقف شد.",
    },
    "progress.percent": {"en": "{percent}% done", "fa": "{percent}٪ انجام شده"},
}

check_messages(_RAW)
MESSAGES: Mapping[str, Mapping[Language, str]] = MappingProxyType(
    {k: MappingProxyType(dict(v)) for k, v in _RAW.items()}
)


def jt(key: str, lang: Language, **params: object) -> str:
    """Like :func:`automa.webui.i18n.t` for this module's table."""
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language: {lang!r}")
    template = MESSAGES[key][lang]
    if set(params) != template_fields(template):
        raise ValueError(f"{key}: parameters {sorted(params)}")
    filled = {
        k: localize_digits(str(v), lang) if isinstance(v, int) and not isinstance(v, bool) else v
        for k, v in params.items()
    }
    for v in filled.values():
        if not isinstance(v, str):
            raise TypeError("message parameters are str or int")
    return template.format(**filled)


@dataclass(frozen=True)
class Step:
    name: str  # plain text (a task title is recorded data; the caller escapes it)
    state: StepState
    task_id: str | None = None


@dataclass(frozen=True)
class Journey:
    steps: tuple[Step, ...]
    percent: int  # 0..100
    now: str  # plain text
    tone: Tone


def _task_state(state: str | None) -> StepState:
    if state in _DONE_TASK:
        return "done"
    if state in _NOW_TASK:
        return "now"
    if state in _FAIL_TASK:
        return "fail"
    return "todo"


def build_journey(
    view: ProjectView,
    *,
    lang: Language = "en",
    questions_waiting: bool = False,
    running: bool = False,
) -> Journey:
    """The steps, percentage and "now" sentence of ``view``.

    ``questions_waiting``: the Executive asked questions not yet answered;
    ``running``: a background run of this project is alive."""
    has_plan = view.plan_version is not None
    status = view.status
    steps: list[Step] = [Step(jt("step.idea", lang), "done")]
    if questions_waiting:
        steps.append(Step(jt("step.questions", lang), "now"))
    else:
        steps.append(Step(jt("step.questions", lang), "done" if has_plan else "todo"))
    if has_plan:
        steps.append(Step(jt("step.plan", lang), "done"))
    elif questions_waiting:
        steps.append(Step(jt("step.plan", lang), "todo"))
    else:
        steps.append(
            Step(jt("step.plan", lang), "now" if status in ("new", "planning") else "todo")
        )
        steps[1] = Step(steps[1].name, "done")
    tasks = [Step(r.title or r.id, _task_state(r.state), r.id) for r in view.tasks]
    if running and tasks and not any(s.state in ("now", "fail") for s in tasks):
        first = next((i for i, s in enumerate(tasks) if s.state == "todo"), None)
        if first is not None:
            tasks[first] = Step(tasks[first].name, "now", tasks[first].task_id)
    steps.extend(tasks)
    finished = status == "done" and has_plan
    steps.append(Step(jt("step.result", lang), "done" if finished else "todo"))

    done = sum(1 for s in steps if s.state == "done")
    percent = 100 if finished else min(99, round(100 * done / len(steps)))

    total = len(tasks)
    current = next((i for i, s in enumerate(tasks) if s.state == "now"), None)
    failed = next((s for s in tasks if s.state == "fail"), None)
    passed = sum(1 for s in tasks if s.state == "done")
    if view.pending:
        now, tone = jt("now.approval", lang, count=len(view.pending)), "busy"
    elif questions_waiting:
        now, tone = jt("now.questions", lang), "busy"
    elif finished:
        now, tone = jt("now.done", lang), "ok"
    elif status == "failed" or (failed is not None and not running):
        now = (
            jt("now.failed", lang, title=failed.name)
            if failed is not None
            else jt("now.failed_plain", lang)
        )
        tone = "bad"
    elif not has_plan:
        now, tone = jt("now.planning", lang), "busy"
    elif current is not None and (running or status == "running"):
        now = jt("now.working", lang, step=current + 1, total=total, title=tasks[current].name)
        tone = "busy"
    elif passed > 0 or status == "paused":
        now, tone = jt("now.paused", lang, done=passed, total=total), "info"
    else:
        now, tone = jt("now.ready", lang), "info"
    return Journey(steps=tuple(steps), percent=percent, now=now, tone=tone)
