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
from automa.delivery import try_build_delivery
from automa.l10n import Language
from automa.plan import Plan
from automa.plan.models import ResultStatus
from automa.project.store import Project, ProjectStore, ProjectStoreError
from automa.tools.approvals import ApprovalError, ApprovalRequest, ApprovalStatus
from automa.tools.github import GitHubRead, describe_github_request
from automa.tools.missing import missing_program
from automa.tools.request_warnings import RequestWarning
from automa.tools.web import FetchUrl, describe_fetch_request
from automa.userinput import UserRequestError
from automa.webui.cards import CardState, card_state
from automa.webui.i18n import t

GOAL_PREVIEW_CHARS = 120
_READ_ERRORS = (ProjectStoreError, ApprovalError)


class _View(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


BENCH_PREFIX = "bench-"
MAX_TITLE_CHARS = 60


def project_title(goal: str) -> str:
    """A short human title from a goal: its first non-empty line, cut at a word."""
    line = next((ln.strip() for ln in goal.replace("\r", "\n").splitlines() if ln.strip()), "")
    line = " ".join(line.split())
    if len(line) <= MAX_TITLE_CHARS:
        return line
    cut = line[:MAX_TITLE_CHARS].rsplit(" ", 1)[0] or line[:MAX_TITLE_CHARS]
    return cut.rstrip(" ,.:;،؛-") + "…"


class ProjectRow(_View):
    id: str
    status: str
    title: str = ""
    goal_preview: str
    created_at: datetime
    cost_usd: float
    pending_approvals: int
    # Home-page card (M16, D160): delivery state, step progress and what waits for the user.
    delivery: str | None = None  # DeliveryState value; None before a plan / unreadable
    tasks_total: int = 0
    tasks_passed: int = 0
    open_asks: int = 0  # unanswered ask_user questions (D159)
    questions_waiting: bool = False  # the Executive's questions before the plan (D124)
    # D186: approvals only the owner decides (a member's page) and, on the owner's
    # page, the member account whose project asks for an approval.
    owner_decides: bool = False
    member: str | None = None

    @property
    def card(self) -> CardState:
        """The home-page card state, always derived from the fields above."""
        return card_state(
            self.status,
            self.delivery,
            open_asks=self.open_asks,
            pending_approvals=0 if self.owner_decides else self.pending_approvals,
            owner_approvals=self.pending_approvals if self.owner_decides else 0,
            questions_waiting=self.questions_waiting,
        )


class Overview(_View):
    projects: tuple[ProjectRow, ...] = ()
    errors: tuple[str, ...] = ()  # projects that could not be read (shown, not hidden)
    hidden_tests: int = 0  # benchmark projects left out (shown with ?tests=1)
    showing_tests: bool = False


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
    # A safe program name when the latest attempt failed on "program not found" (D152).
    missing_program: str | None = None


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
    # Part of cost_usd spent on calls without a task (questions, plan, review; D133).
    planning_usd: float = 0.0
    plan_version: int | None = None
    tasks: tuple[TaskRow, ...] = ()
    pending: tuple[ApprovalView, ...] = ()
    errors: tuple[str, ...] = ()


def _preview(text: str, limit: int = GOAL_PREVIEW_CHARS) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def approval_view(req: ApprovalRequest, lang: Language = "en") -> ApprovalView:
    """The readable view of one request, with the same target views as the CLI (D73, D76).

    The target lines' labels are in ``lang`` (D146); the values (host, path, repo...)
    are shown exactly as the tools describe them. The tools' warnings come from their
    codes, in ``lang`` (D148); in English that is the tools' own text."""
    lines: list[str] = []
    codes: tuple[RequestWarning, ...] = ()
    if req.tool == FetchUrl.name:
        fv = describe_fetch_request(req.arguments.get("url"))
        if fv.valid:
            lines = [
                t("target.host", lang, value=str(fv.host)),
                t("target.port", lang, value=str(fv.port)),
                t("target.path", lang, value=str(fv.path)),
            ]
            if fv.query_parameters:
                lines.append(t("target.query", lang, value=fv.query_parameters))
        else:
            lines = [t("target.bad_url", lang, error=fv.error or "")]
        codes = fv.warning_codes
    elif req.tool == GitHubRead.name:
        gv = describe_github_request(req.arguments)
        if gv.valid:
            lines = [
                t("target.api_url", lang, value=str(gv.api_url)),
                t("target.kind", lang, value=str(gv.kind)),
                t("target.repo", lang, value=str(gv.repo)),
            ]
        else:
            lines = [t("target.bad_request", lang, error=gv.error or "")]
        codes = gv.warning_codes
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
        warnings=tuple(t(f"warn.{w.code}", lang, **w.params) for w in codes),
    )


def task_rows(store: ProjectStore, project_id: str, plan: Plan) -> list[TaskRow]:
    rows = []
    for i, task in enumerate(plan.topological_order(), start=1):
        attempts = store.attempt_count(project_id, task.id)
        last = None
        missing = None
        if attempts:
            result = store.load_task_result(project_id, task.id, attempts)
            last = result.status.value
            if result.status is not ResultStatus.COMPLETED:
                missing = missing_program(result.error)
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
                missing_program=missing,
            )
        )
    return rows


def _pending(store: ProjectStore, project_id: str) -> list[ApprovalRequest]:
    return store.approvals(project_id).list(ApprovalStatus.PENDING)


def _questions_waiting(store: ProjectStore, project_id: str) -> bool:
    """The Executive asked questions before the plan and they are not answered (D124)."""
    try:
        if store.has_plan(project_id) or store.has_clarifications(project_id):
            return False
        return bool(store.load_questions(project_id))
    except (ProjectStoreError, OSError):
        return False


def build_overview(
    store: ProjectStore,
    spent_nusd: dict[str, int],
    *,
    include_tests: bool = True,
    owner: str | None = None,
    members: frozenset[str] = frozenset(),
) -> Overview:
    """All projects, newest first (D138). Benchmark projects (``bench-…``) are left
    out unless ``include_tests``; their number is kept. A corrupt project becomes an
    error line, not a crash.

    ``owner`` (D182): only that account's projects (a member's page); nothing about
    the others is shown, not even an unreadable one's id; its approvals are the
    owner's to decide (D186). ``members`` (the owner's page): the member account
    names, so a member's project says whose it is."""
    rows: list[ProjectRow] = []
    errors: list[str] = []
    projects: list[Project] = []
    for pid in store.list_ids():
        try:
            projects.append(store.load(pid))
        except _READ_ERRORS as e:  # one bad project must not hide the rest
            if owner is None:
                errors.append(f"{pid}: {e}")
    if owner is not None:
        projects = [p for p in projects if p.owner == owner]
    projects.sort(key=lambda p: (p.created_at, p.id), reverse=True)
    hidden = 0
    if not include_tests:
        kept = [p for p in projects if not p.id.startswith(BENCH_PREFIX)]
        hidden = len(projects) - len(kept)
        projects = kept
    for p in projects:
        try:
            pending = len(_pending(store, p.id))
        except _READ_ERRORS as e:
            errors.append(f"{p.id}: approvals unreadable ({e})")
            pending = 0
        try:
            asks = len(store.user_requests(p.id).pending())
        except (UserRequestError, ProjectStoreError, OSError) as e:
            errors.append(f"{p.id}: questions unreadable ({e})")
            asks = 0
        delivery = try_build_delivery(store, p.id)
        questions = delivery is None and _questions_waiting(store, p.id)
        state = delivery.state.value if delivery is not None else None
        rows.append(
            ProjectRow(
                id=p.id,
                status=p.status.value,
                title=project_title(p.goal) or p.id,
                goal_preview=_preview(p.goal),
                created_at=p.created_at,
                cost_usd=nusd_to_usd(spent_nusd.get(p.id, 0)),
                pending_approvals=pending,
                delivery=state,
                tasks_total=len(delivery.tasks) if delivery is not None else 0,
                tasks_passed=delivery.passed_tasks if delivery is not None else 0,
                open_asks=asks,
                questions_waiting=questions,
                owner_decides=owner is not None,
                member=p.owner if owner is None and p.owner in members else None,
            )
        )
    return Overview(
        projects=tuple(rows),
        errors=tuple(errors),
        hidden_tests=hidden,
        showing_tests=include_tests,
    )


def build_project_view(
    store: ProjectStore,
    project_id: str,
    spent_nusd: int,
    *,
    planning_nusd: int = 0,
    lang: Language = "en",
) -> ProjectView:
    """One project. Raises ``ProjectStoreError`` if the project itself is missing/corrupt;
    an unreadable plan, task result or approvals directory becomes an error line.

    ``planning_nusd`` is the part of ``spent_nusd`` on calls without a task (D133);
    it is clamped to ``[0, spent_nusd]`` so the split never reads more than the total.
    ``lang`` is the language of the approval target lines (D146)."""
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
        pending = [approval_view(r, lang) for r in _pending(store, project_id)]
    except _READ_ERRORS as e:
        errors.append(f"approvals: {e}")
    return ProjectView(
        id=project.id,
        goal=project.goal,
        status=project.status.value,
        created_at=project.created_at,
        updated_at=project.updated_at,
        cost_usd=nusd_to_usd(spent_nusd),
        planning_usd=nusd_to_usd(min(max(planning_nusd, 0), max(spent_nusd, 0))),
        plan_version=plan_version,
        tasks=tuple(tasks),
        pending=tuple(pending),
        errors=tuple(errors),
    )
