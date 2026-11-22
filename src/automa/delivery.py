"""The delivery report: what the user got, in plain terms (D157).

Built only from what is on disk (project, plan, task states, the last verdict of
every task and the files in ``files/``). It answers the user's questions after a
run without reading logs: *Is my product ready? Which files are it? How do I use
it? What exactly was checked? If something is missing, why and what now?*

* ``state`` is ``ready`` only for a ``done`` project whose main files all exist;
  a failed project or a missing main file is ``partial``; otherwise the work is
  ``in_progress`` (or ``not_started`` before any attempt).
* Main files come from the plan's ``deliverable`` (D155). Older plans and plans
  without one fall back to the files checked by the *final* tasks (those no other
  task depends on); ``derived`` tells the reader that the list was inferred.
* Every acceptance criterion is listed with what the last verdict said about it,
  so "done" means "these named things were checked", nothing vaguer.

Pure and read-only: no writes, no model calls, no network. ``render_markdown``
turns a report into a ``DELIVERY.md`` text (the web UI adds it to the zip).
"""

from __future__ import annotations

import os
import stat
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from automa.final_review import load_review
from automa.plan import Plan, TaskStatus
from automa.plan.models import AcceptanceCriterion, Verdict, checked_paths
from automa.project.store import ProjectStatus, ProjectStore, ProjectStoreError

MAX_ISSUE_CHARS = 400
MAX_ISSUES_PER_TASK = 5
MAX_DERIVED_FILES = 10


class DeliveryState(StrEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    READY = "ready"
    PARTIAL = "partial"


class _View(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DeliveryFile(_View):
    path: str
    exists: bool
    size: int = 0
    task_id: str | None = None

    @property
    def is_html(self) -> bool:
        return self.path.lower().endswith((".html", ".htm"))


class CriterionView(_View):
    description: str
    passed: bool | None  # None = not judged yet
    deterministic: bool


class TaskDelivery(_View):
    id: str
    title: str
    status: TaskStatus
    attempts: int
    criteria: tuple[CriterionView, ...]
    issues: tuple[str, ...] = ()


class Delivery(_View):
    project_id: str
    goal: str
    project_status: ProjectStatus
    state: DeliveryState
    summary: str
    how_to_use: str
    derived: bool  # True when the plan had no deliverable (summary/main files inferred)
    main_files: tuple[DeliveryFile, ...]
    tasks: tuple[TaskDelivery, ...]
    # Final review of the whole product (D168): None = not reviewed (yet).
    reviewed: bool | None = None
    review_issues: tuple[str, ...] = ()

    @property
    def passed_tasks(self) -> int:
        return sum(t.status is TaskStatus.PASSED for t in self.tasks)

    @property
    def missing_files(self) -> tuple[DeliveryFile, ...]:
        return tuple(f for f in self.main_files if not f.exists)

    @property
    def problem_tasks(self) -> tuple[TaskDelivery, ...]:
        return tuple(t for t in self.tasks if t.status in (TaskStatus.FAILED, TaskStatus.BLOCKED))


def _short(text: str, limit: int = MAX_ISSUE_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def final_tasks(plan: Plan) -> list[str]:
    """Ids of tasks no other task depends on (plan order): where the product ends."""
    needed = {d for t in plan.tasks for d in t.depends_on}
    return [t.id for t in plan.tasks if t.id not in needed]


def _file(store: ProjectStore, project_id: str, path: str, task_id: str | None) -> DeliveryFile:
    """Whether ``files/<path>`` is a regular file; no component may be a symlink."""
    current = store.files_dir(project_id)
    try:
        if stat.S_ISLNK(os.lstat(current).st_mode):
            return DeliveryFile(path=path, exists=False, task_id=task_id)
        for part in path.split("/"):
            current = current / part
            st = os.lstat(current)
            if stat.S_ISLNK(st.st_mode):
                return DeliveryFile(path=path, exists=False, task_id=task_id)
    except OSError:
        return DeliveryFile(path=path, exists=False, task_id=task_id)
    if not stat.S_ISREG(st.st_mode):
        return DeliveryFile(path=path, exists=False, task_id=task_id)
    return DeliveryFile(path=path, exists=True, size=st.st_size, task_id=task_id)


def _criteria(
    task_criteria: list[AcceptanceCriterion], verdict: Verdict | None
) -> tuple[CriterionView, ...]:
    by_check = {c.criterion: c.passed for c in verdict.checks} if verdict else {}
    flagged = {i.criterion for i in verdict.issues if i.criterion is not None} if verdict else set()
    views = []
    for n, criterion in enumerate(task_criteria):
        passed: bool | None
        if verdict is None:
            passed = None
        elif verdict.passed:
            passed = True
        elif n in flagged or by_check.get(n) is False:
            passed = False
        elif n in by_check:
            passed = by_check[n]
        else:
            passed = None  # a failed verdict that does not say anything about this one
        views.append(
            CriterionView(
                description=criterion.description,
                passed=passed,
                deterministic=criterion.deterministic,
            )
        )
    return tuple(views)


def build_delivery(store: ProjectStore, project_id: str) -> Delivery | None:
    """The delivery report of a planned project; ``None`` before there is a plan."""
    project = store.load(project_id)
    if not store.has_plan(project_id):
        return None
    plan = store.load_plan(project_id)
    states = store.task_states(project_id)
    tasks: list[TaskDelivery] = []
    started = False
    for task in plan.tasks:
        state = states[task.id]
        history = store.task_history(project_id, task.id)
        verdict = history[-1][1] if history else None
        started = started or bool(history)
        issues: tuple[str, ...] = ()
        if verdict is not None and not verdict.passed:
            issues = tuple(_short(i.description) for i in verdict.issues[:MAX_ISSUES_PER_TASK])
            if not issues:
                issues = tuple(
                    _short(c.detail or task.acceptance_criteria[c.criterion].description)
                    for c in verdict.checks
                    if not c.passed
                )[:MAX_ISSUES_PER_TASK]
        elif state.status is TaskStatus.BLOCKED and state.error:
            issues = (_short(state.error),)
        tasks.append(
            TaskDelivery(
                id=task.id,
                title=task.title,
                status=state.status,
                attempts=len(history),
                criteria=_criteria(task.acceptance_criteria, verdict),
                issues=issues,
            )
        )

    deliverable = plan.deliverable
    if deliverable is not None:
        paths = list(deliverable.main_files)
        summary, how_to_use, derived = deliverable.summary, deliverable.how_to_use, False
    else:
        paths = []
        for tid in final_tasks(plan):
            for p in sorted(checked_paths(plan.task(tid))):
                if p not in paths:
                    paths.append(p)
        paths = paths[:MAX_DERIVED_FILES]
        summary, how_to_use, derived = project.goal, "", True
    owner = {}
    for p in paths:
        t = plan.task_for_file(p)
        owner[p] = t.id if t is not None else None
    files = tuple(_file(store, project_id, p, owner[p]) for p in paths)

    review = load_review(store.project_dir(project_id))
    if review is not None and review.plan_version != plan.version:
        review = None
    reviewed = None if review is None or review.unavailable else review.passed
    review_issues = review.issues if review is not None and not review.passed else ()
    if project.status is ProjectStatus.DONE:
        state_ = (
            DeliveryState.READY
            if all(f.exists for f in files) and reviewed is not False
            else DeliveryState.PARTIAL
        )
    elif project.status is ProjectStatus.FAILED:
        state_ = DeliveryState.PARTIAL
    elif started:
        state_ = DeliveryState.IN_PROGRESS
    else:
        state_ = DeliveryState.NOT_STARTED
    return Delivery(
        project_id=project_id,
        goal=project.goal,
        project_status=project.status,
        state=state_,
        summary=summary,
        how_to_use=how_to_use,
        derived=derived,
        main_files=files,
        tasks=tuple(tasks),
        reviewed=reviewed,
        review_issues=tuple(review_issues),
    )


def try_build_delivery(store: ProjectStore, project_id: str) -> Delivery | None:
    """``build_delivery`` that returns ``None`` instead of raising on bad data."""
    try:
        return build_delivery(store, project_id)
    except (ProjectStoreError, OSError, ValueError, KeyError):
        return None


_MD = {
    "fa": {
        "title": "تحویل پروژه",
        "ready": "✅ محصول آماده است.",
        "partial": "⚠️ محصول کامل نشد؛ جزئیات پایین است.",
        "in_progress": "⏳ کار هنوز در جریان است.",
        "not_started": "کار هنوز شروع نشده.",
        "what": "چه چیزی ساخته شد",
        "files": "فایل‌های اصلی",
        "missing": "(ساخته نشد)",
        "how": "روش استفاده",
        "checks": "چه چیزهایی بررسی شد",
        "problems": "مشکل‌ها",
        "derived": "(برنامه شرح محصول نداشت؛ این فهرست از روی مراحل آخر برداشت شده.)",
        "goal": "درخواست",
        "review": "بازبینی نهایی",
    },
    "en": {
        "title": "Project delivery",
        "ready": "✅ The product is ready.",
        "partial": "⚠️ The product is not complete; details below.",
        "in_progress": "⏳ The work is still in progress.",
        "not_started": "The work has not started yet.",
        "what": "What was built",
        "files": "Main files",
        "missing": "(not made)",
        "how": "How to use it",
        "checks": "What was checked",
        "problems": "Problems",
        "derived": "(The plan had no product description; this list comes from the final steps.)",
        "goal": "Request",
        "review": "Final review",
    },
}


def render_markdown(delivery: Delivery, lang: str = "fa") -> str:
    """``DELIVERY.md`` text of ``delivery`` (UI language ``fa`` or ``en``)."""
    w = _MD.get(lang, _MD["en"])
    mark = {True: "✅", False: "❌", None: "•"}
    out = [f"# {w['title']}", "", w[delivery.state.value], ""]
    out += [f"## {w['what']}", "", delivery.summary.strip(), ""]
    if delivery.derived:
        out += [w["derived"], ""]
    if delivery.main_files:
        out += [f"## {w['files']}", ""]
        for f in delivery.main_files:
            out.append(f"- `{f.path}`" + ("" if f.exists else f" {w['missing']}"))
        out.append("")
    if delivery.how_to_use.strip():
        out += [f"## {w['how']}", "", delivery.how_to_use.strip(), ""]
    out += [f"## {w['checks']}", ""]
    for t in delivery.tasks:
        out.append(f"### {t.title}")
        for c in t.criteria:
            out.append(f"- {mark[c.passed]} {c.description}")
        out.append("")
    problems = delivery.problem_tasks
    if problems or delivery.review_issues:
        out += [f"## {w['problems']}", ""]
        for t in problems:
            out.append(f"- **{t.title}**: " + ("; ".join(t.issues) or t.status.value))
        for issue in delivery.review_issues:
            out.append(f"- {w['review']}: {issue}")
        out.append("")
    if delivery.goal.strip() != delivery.summary.strip():
        out += [f"## {w['goal']}", "", delivery.goal.strip(), ""]
    return "\n".join(out).rstrip() + "\n"
