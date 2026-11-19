"""Read-only view models of the web UI (D77).

Built only from what is already on disk: ``project.json``, ``plan.json``, stored
task results and approval requests, plus spending the caller read from the Ledger.
Nothing here writes, creates a file or directory, calls a model or touches the
network. Approval requests are shown, never decided.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from automa.budget import nusd_to_usd
from automa.plan import Plan
from automa.project.store import Project, ProjectStore, ProjectStoreError
from automa.tools.approvals import ApprovalError, ApprovalRequest, ApprovalStatus
from automa.tools.github import GitHubRead, describe_github_request
from automa.tools.web import FetchUrl, describe_fetch_request

GOAL_PREVIEW_CHARS = 120
_READ_ERRORS = (ProjectStoreError, ApprovalError)


class _View(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectRow(_View):
    id: str
    status: str
    goal_preview: str
    created_at: datetime
    cost_usd: float
    pending_approvals: int


class Overview(_View):
    projects: tuple[ProjectRow, ...] = ()
    errors: tuple[str, ...] = ()  # projects that could not be read (shown, not hidden)


class TaskRow(_View):
    order: int
    id: str
    title: str
    role: str
    depends_on: tuple[str, ...] = ()
    attempts: int = 0
    last_result: str | None = None  # status of the latest stored attempt
    state: str | None = None  # tasks/<id>/state.json status; None = not started (D127)
    model: str | None = None  # model of the current/last attempt (D97)


class ApprovalView(_View):
    id: str
    task_id: str | None
    tool: str
    risk: str
    reason: str
    created_at: datetime
    arguments: dict[str, object]
    fingerprint: str = ""  # echoed by the action-mode form (D123)
    request_lines: tuple[str, ...] = ()  # readable fetch_url / github_read target
    warnings: tuple[str, ...] = ()


class ProjectView(_View):
    id: str
    goal: str
    status: str
    created_at: datetime
    updated_at: datetime
    cost_usd: float
    plan_version: int | None = None
    tasks: tuple[TaskRow, ...] = ()
    pending: tuple[ApprovalView, ...] = ()
    errors: tuple[str, ...] = ()


def _preview(text: str, limit: int = GOAL_PREVIEW_CHARS) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def approval_view(req: ApprovalRequest) -> ApprovalView:
    """The readable view of one request, with the same target views as the CLI (D73, D76)."""
    lines: list[str] = []
    warnings: tuple[str, ...] = ()
    if req.tool == FetchUrl.name:
        fv = describe_fetch_request(req.arguments.get("url"))
        if fv.valid:
            lines = [f"host: {fv.host}", f"port: {fv.port}", f"path: {fv.path}"]
            if fv.query_parameters:
                lines.append(f"query parameters: {fv.query_parameters}")
        else:
            lines = [f"invalid URL ({fv.error or ''}); the tool will refuse it"]
        warnings = fv.warnings
    elif req.tool == GitHubRead.name:
        gv = describe_github_request(req.arguments)
        if gv.valid:
            lines = [f"api url: {gv.api_url}", f"kind: {gv.kind}", f"repo: {gv.repo}"]
        else:
            lines = [f"invalid request ({gv.error or ''}); the tool will refuse it"]
        warnings = gv.warnings
    return ApprovalView(
        id=req.id,
        task_id=req.task_id,
        tool=req.tool,
        risk=req.risk.value,
        reason=req.reason,
        created_at=req.created_at,
        arguments=dict(req.arguments),
        fingerprint=req.fingerprint,
        request_lines=tuple(lines),
        warnings=tuple(warnings),
    )


def task_rows(store: ProjectStore, project_id: str, plan: Plan) -> list[TaskRow]:
    rows = []
    for i, task in enumerate(plan.topological_order(), start=1):
        attempts = store.attempt_count(project_id, task.id)
        last = None
        if attempts:
            last = store.load_task_result(project_id, task.id, attempts).status.value
        try:
            ts = store.load_task_state(project_id, task.id)
        except _READ_ERRORS:  # one unreadable state must not hide the plan
            state, model = "unreadable", None
        else:
            state = ts.status.value if ts is not None else None
            model = ts.model if ts is not None else None
        rows.append(
            TaskRow(
                order=i,
                id=task.id,
                title=task.title,
                role=task.role,
                depends_on=tuple(task.depends_on),
                attempts=attempts,
                last_result=last,
                state=state,
                model=model,
            )
        )
    return rows


def _pending(store: ProjectStore, project_id: str) -> list[ApprovalRequest]:
    return store.approvals(project_id).list(ApprovalStatus.PENDING)


def build_overview(store: ProjectStore, spent_nusd: dict[str, int]) -> Overview:
    """All projects, oldest first. A corrupt project becomes an error line, not a crash."""
    rows: list[ProjectRow] = []
    errors: list[str] = []
    projects: list[Project] = []
    for pid in store.list_ids():
        try:
            projects.append(store.load(pid))
        except _READ_ERRORS as e:  # one bad project must not hide the rest
            errors.append(f"{pid}: {e}")
    projects.sort(key=lambda p: (p.created_at, p.id))
    for p in projects:
        try:
            pending = len(_pending(store, p.id))
        except _READ_ERRORS as e:
            errors.append(f"{p.id}: approvals unreadable ({e})")
            pending = 0
        rows.append(
            ProjectRow(
                id=p.id,
                status=p.status.value,
                goal_preview=_preview(p.goal),
                created_at=p.created_at,
                cost_usd=nusd_to_usd(spent_nusd.get(p.id, 0)),
                pending_approvals=pending,
            )
        )
    return Overview(projects=tuple(rows), errors=tuple(errors))


def build_project_view(store: ProjectStore, project_id: str, spent_nusd: int) -> ProjectView:
    """One project. Raises ``ProjectStoreError`` if the project itself is missing/corrupt;
    an unreadable plan, task result or approvals directory becomes an error line."""
    project = store.load(project_id)
    errors: list[str] = []
    plan_version = None
    tasks: list[TaskRow] = []
    if store.has_plan(project_id):
        try:
            plan = store.load_plan(project_id)
            plan_version = plan.version
            tasks = task_rows(store, project_id, plan)
        except _READ_ERRORS as e:
            errors.append(f"plan: {e}")
    pending: list[ApprovalView] = []
    try:
        pending = [approval_view(r) for r in _pending(store, project_id)]
    except _READ_ERRORS as e:
        errors.append(f"approvals: {e}")
    return ProjectView(
        id=project.id,
        goal=project.goal,
        status=project.status.value,
        created_at=project.created_at,
        updated_at=project.updated_at,
        cost_usd=nusd_to_usd(spent_nusd),
        plan_version=plan_version,
        tasks=tuple(tasks),
        pending=tuple(pending),
        errors=tuple(errors),
    )
