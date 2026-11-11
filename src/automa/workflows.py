"""Workflow templates: plans of successful projects, saved for reuse (M6, D38).

A template is a frozen copy of the plan of a project that finished with **every**
task ``passed`` (the store only marks a task passed with a stored passing verdict),
plus the source goal and a few statistics. Templates live in
``<workspace>/workflows/<id>.json`` and are written once (never overwritten).

``WorkflowStore.suggest(goal)`` ranks stored templates by a deterministic word
overlap with a new goal (no model call, no cost). ``WorkflowStore.use`` writes a
template's plan as the plan of a new project (D40): roles and starting tiers are
checked against the current configuration first, and the Executive is not called.
Users can also define templates themselves (``WorkflowStore.create``) and edit any
template (``WorkflowStore.revise``); both write a *new* ``user`` template (an edit
links to its ``parent``), so stored files stay write-once and the proven statistics
of a project template are never attached to a plan nobody has run (D41).
Nothing here runs a task.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
from collections.abc import Callable, Collection, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from automa.config import Tier
from automa.events import Event
from automa.plan import Plan, TaskStatus
from automa.plan.models import _format_errors
from automa.project import ProjectStatus, ProjectStore, ProjectStoreError, validate_project_id
from automa.project.store import MAX_GOAL_CHARS, _write_json_exclusive

WORKFLOW_SCHEMA_VERSION = 1
WORKFLOWS_DIRNAME = "workflows"
WORKFLOW_ID_PATTERN = r"^wf-[a-z0-9][a-z0-9-]{0,58}[a-z0-9]$"
_WORKFLOW_ID_RE = re.compile(WORKFLOW_ID_PATTERN)
MAX_SUGGESTIONS = 20
DEFAULT_MIN_SCORE = 0.2

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class WorkflowError(Exception):
    """A template cannot be saved, read or matched."""


class WorkflowNotFound(WorkflowError):
    pass


class WorkflowIncompatible(WorkflowError):
    """A template's plan does not fit the current roles or configured tiers."""


def validate_workflow_id(workflow_id: str) -> str:
    if not isinstance(workflow_id, str) or not _WORKFLOW_ID_RE.fullmatch(workflow_id):
        raise WorkflowError(f"invalid workflow id: {workflow_id!r}")
    return workflow_id


class WorkflowStats(BaseModel):
    """What the source project took: tasks, worker attempts and model cost of the results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: int = Field(ge=1)
    attempts: int = Field(ge=0)
    cost_nusd: int = Field(ge=0)


class WorkflowOrigin(StrEnum):
    PROJECT = "project"  # copied from a finished project where every task passed
    USER = "user"  # written or edited by the user; never run as this exact plan


class WorkflowTemplate(BaseModel):
    """Contents of ``workflows/<id>.json``. Immutable.

    ``project`` templates carry their source project, plan version and stats;
    ``user`` templates carry none of these (nothing proves them) and may name the
    template they were edited from in ``parent``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: int = WORKFLOW_SCHEMA_VERSION
    id: str = Field(pattern=WORKFLOW_ID_PATTERN)
    origin: WorkflowOrigin = WorkflowOrigin.PROJECT
    goal: str = Field(min_length=1, max_length=MAX_GOAL_CHARS)
    source_project: str | None = None
    source_plan_version: int | None = Field(default=None, ge=1)
    parent: str | None = Field(default=None, pattern=WORKFLOW_ID_PATTERN)
    created_at: datetime
    plan: Plan
    stats: WorkflowStats | None = None

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != WORKFLOW_SCHEMA_VERSION:
            raise ValueError(f"unsupported workflow schema version {v}")
        return v

    @field_validator("source_project")
    @classmethod
    def _project_id(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            return validate_project_id(v)
        except ProjectStoreError as e:
            raise ValueError(str(e)) from None

    @model_validator(mode="after")
    def _origin_fields(self) -> WorkflowTemplate:
        if self.origin is WorkflowOrigin.PROJECT:
            if self.source_project is None or self.source_plan_version is None:
                raise ValueError("a project workflow needs source_project and source_plan_version")
            if self.stats is None:
                raise ValueError("a project workflow needs stats")
            if self.parent is not None:
                raise ValueError("a project workflow cannot have a parent")
            if self.stats.tasks != len(self.plan.tasks):
                raise ValueError("stats.tasks does not match the plan")
        else:
            if self.source_project is not None or self.source_plan_version is not None:
                raise ValueError("a user workflow has no source project")
            if self.stats is not None:
                raise ValueError("a user workflow has no stats (it was never run as is)")
        if self.parent is not None and self.parent == self.id:
            raise ValueError("a workflow cannot be its own parent")
        # Project templates keep their plan exactly as run (D38), so only user
        # templates are required to carry their own goal in the plan.
        if self.origin is WorkflowOrigin.USER and self.plan.goal != self.goal:
            raise ValueError("plan.goal must equal goal")
        if self.plan.version != 1:
            raise ValueError("a workflow plan has version 1")
        return self

    @property
    def task_count(self) -> int:
        return len(self.plan.tasks)

    @field_validator("created_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return v.astimezone(UTC)


class WorkflowMatch(BaseModel):
    """One suggestion: a template and its similarity (0..1) to the new goal."""

    model_config = ConfigDict(frozen=True)

    template: WorkflowTemplate
    score: float = Field(ge=0.0, le=1.0)
    shared: tuple[str, ...] = ()


# --- similarity ----------------------------------------------------------------

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
# Arabic code points that Persian text often uses interchangeably with Persian ones.
_NORMALIZE = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "‌": " "})
_STOPWORDS = frozenset(
    {
        # English
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
        "is", "it", "its", "make", "me", "my", "of", "on", "or", "that", "the", "this",
        "to", "with", "write", "create", "build", "please", "some", "using", "new",
        # Persian
        "از", "با", "به", "برای", "در", "را", "که", "و", "یا", "این", "آن", "یک",
        "رو", "تا", "هم", "بر", "کن", "بکن", "بساز", "بنویس", "ساخت", "نوشتن", "می",
    }
)  # fmt: skip


def goal_terms(text: str) -> frozenset[str]:
    """Normalized content words of ``text`` (case-folded, without stopwords and numbers)."""
    words = _WORD_RE.findall(text.translate(_NORMALIZE).casefold())
    return frozenset(w for w in words if len(w) > 1 and not w.isdigit() and w not in _STOPWORDS)


def similarity(a: str, b: str) -> tuple[float, frozenset[str]]:
    """Jaccard overlap of the content words of two goals, and the shared words."""
    ta, tb = goal_terms(a), goal_terms(b)
    if not ta or not tb:
        return 0.0, frozenset()
    shared = ta & tb
    return len(shared) / len(ta | tb), shared


# --- using a template -------------------------------------------------------------


def check_compatible(
    plan: Plan,
    *,
    role_tiers: Mapping[str, Tier],
    configured_tiers: Collection[Tier],
    label: str = "workflow",
) -> None:
    """Raise ``WorkflowIncompatible`` unless every task's role is known and the tier
    it would start at (its own tier, else its role's) has a configured model."""
    unknown = sorted({t.role for t in plan.tasks} - set(role_tiers))
    if unknown:
        raise WorkflowIncompatible(
            f"{label} uses unknown role(s): {', '.join(unknown)}; "
            f"known roles: {', '.join(sorted(role_tiers)) or '-'}"
        )
    missing = sorted(
        {
            (t.tier or role_tiers[t.role]).value
            for t in plan.tasks
            if (t.tier or role_tiers[t.role]) not in configured_tiers
        }
    )
    if missing:
        raise WorkflowIncompatible(
            f"{label} needs tier(s) with no configured model: {', '.join(missing)}"
        )


def user_plan(data: Any, goal: str) -> Plan:
    """Validate a user-written plan for a template with ``goal``.

    ``data`` is a JSON object (the shape of ``plan.json``); its ``goal`` and
    ``version`` are ignored and replaced by ``goal`` and 1. Every ``Plan`` rule
    applies (limits, safe paths, single-line commands, acyclic dependencies). Raises
    ``WorkflowError`` listing the problems; never returns a partially valid plan.
    """
    if not isinstance(data, Mapping):
        raise WorkflowError("plan must be a JSON object")
    try:
        return Plan.model_validate({**data, "goal": goal, "version": 1})
    except ValidationError as e:
        raise WorkflowError(f"plan failed validation:\n{_format_errors(e)}") from None


def plan_from_template(
    template: WorkflowTemplate,
    goal: str,
    *,
    role_tiers: Mapping[str, Tier],
    configured_tiers: Collection[Tier],
) -> Plan:
    """The template's plan for a new project with ``goal`` (version 1, re-validated).

    ``role_tiers`` maps every currently known role to its default tier. Every task's
    role must be known, and the tier each task would start at (its own tier, else its
    role's) must be configured; otherwise ``WorkflowIncompatible`` is raised and
    nothing is written. Task texts, inputs and checks are copied unchanged.
    """
    check_compatible(
        template.plan,
        role_tiers=role_tiers,
        configured_tiers=configured_tiers,
        label=f"workflow {template.id!r}",
    )
    try:
        return Plan.model_validate(
            {**template.plan.model_dump(mode="json"), "goal": goal, "version": 1}
        )
    except ValidationError as e:
        raise WorkflowError(f"workflow {template.id!r} does not fit goal {goal!r}") from e


# --- store -----------------------------------------------------------------------


def _new_id(now: datetime) -> str:
    return f"wf-{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


class WorkflowStore:
    """Saves, lists and suggests workflow templates under ``<workspace>/workflows``."""

    def __init__(self, root: str | Path, *, clock: Clock = _utc_now):
        self._root = Path(root)
        self._clock = clock
        self._lock = threading.Lock()

    @classmethod
    def for_workspace(cls, workspace: str | Path, *, clock: Clock = _utc_now) -> WorkflowStore:
        return cls(Path(workspace) / WORKFLOWS_DIRNAME, clock=clock)

    @property
    def root(self) -> Path:
        return self._root

    def path(self, workflow_id: str) -> Path:
        validate_workflow_id(workflow_id)
        return self._root / f"{workflow_id}.json"

    # -- reading -------------------------------------------------------------

    def _check_root(self) -> bool:
        if self._root.is_symlink():
            raise WorkflowError(f"{self._root}: workflows directory must not be a symlink")
        return self._root.is_dir()

    def load(self, workflow_id: str) -> WorkflowTemplate:
        path = self.path(workflow_id)
        self._check_root()
        if path.is_symlink():
            raise WorkflowError(f"{path}: workflow file must not be a symlink")
        if not path.is_file():
            raise WorkflowNotFound(f"no such workflow: {workflow_id!r}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise WorkflowError(f"{path}: unreadable workflow file") from e
        try:
            template = WorkflowTemplate.model_validate(raw)
        except ValidationError as e:
            raise WorkflowError(f"{path}: invalid workflow file") from e
        if template.id != workflow_id:
            raise WorkflowError(f"{path}: id {template.id!r} does not match the file name")
        return template

    def list_ids(self) -> list[str]:
        if not self._check_root():
            return []
        ids = []
        for child in self._root.iterdir():
            stem = child.name.removesuffix(".json")
            if child.name.endswith(".json") and _WORKFLOW_ID_RE.fullmatch(stem):
                ids.append(stem)
        return sorted(ids)

    def templates(self) -> list[WorkflowTemplate]:
        """All templates, newest first. Raises on any corrupt template (fail closed)."""
        templates = [self.load(wid) for wid in self.list_ids()]
        return sorted(templates, key=lambda t: (t.created_at, t.id), reverse=True)

    def find_by_project(self, project_id: str) -> WorkflowTemplate | None:
        validate_project_id(project_id)
        for t in self.templates():
            if t.source_project == project_id:
                return t
        return None

    # -- saving --------------------------------------------------------------

    def save_from_project(self, projects: ProjectStore, project_id: str) -> WorkflowTemplate:
        """Save the plan of a successful project as a new template.

        Only a project with status ``done`` whose current plan has every task
        ``passed`` qualifies. A project is saved at most once (``WorkflowError``).
        Logs ``workflow.saved`` in the project's event log.
        """
        project = projects.load(project_id)
        if project.status is not ProjectStatus.DONE:
            raise WorkflowError(
                f"project {project_id!r} is {project.status.value!r}; "
                "only a finished (done) project can be saved as a workflow"
            )
        plan = projects.load_plan(project_id)
        states = projects.task_states(project_id)
        unpassed = sorted(t for t, s in states.items() if s.status is not TaskStatus.PASSED)
        if unpassed:
            raise WorkflowError(
                f"project {project_id!r} has tasks that did not pass: {', '.join(unpassed)}"
            )
        attempts = sum(s.attempts for s in states.values())
        cost = sum(
            result.cost_nusd
            for task in plan.tasks
            for result, _ in projects.task_history(project_id, task.id)
        )
        now = self._now()
        with self._lock:
            if self.find_by_project(project_id) is not None:
                raise WorkflowError(f"project {project_id!r} is already saved as a workflow")
            template = self._write(
                {
                    "id": _new_id(now),
                    "origin": WorkflowOrigin.PROJECT,
                    "goal": project.goal,
                    "source_project": project_id,
                    "source_plan_version": plan.version,
                    "created_at": now,
                    "plan": {**plan.model_dump(mode="json"), "version": 1},
                    "stats": WorkflowStats(
                        tasks=len(plan.tasks), attempts=attempts, cost_nusd=cost
                    ).model_dump(),
                }
            )
        projects.events(project_id).emit(
            Event(
                ts=now,
                type="workflow.saved",
                project_id=project_id,
                data={"workflow_id": template.id, "tasks": template.task_count},
            )
        )
        return template

    def _write(self, fields: dict[str, Any]) -> WorkflowTemplate:
        """Validate a whole template and write it once. Caller holds ``_lock``."""
        try:
            template = WorkflowTemplate.model_validate(fields)
        except ValidationError as e:
            raise WorkflowError(f"invalid workflow:\n{_format_errors(e)}") from None
        self._check_root()
        self._root.mkdir(parents=True, exist_ok=True)
        try:
            _write_json_exclusive(self.path(template.id), template.model_dump(mode="json"))
        except ProjectStoreError as e:
            raise WorkflowError(str(e)) from None
        return template

    # -- user-defined and edited templates (D41) -------------------------------

    def create(
        self,
        goal: str,
        plan: Any,
        *,
        role_tiers: Mapping[str, Tier],
        configured_tiers: Collection[Tier],
    ) -> WorkflowTemplate:
        """Save a user-written plan as a new ``user`` template (no parent, no stats).

        ``plan`` is a ``Plan`` or a JSON object of that shape (see ``user_plan``). It
        must fit the current roles and configured tiers (``WorkflowIncompatible``);
        nothing is written on any error. No model is called.
        """
        return self._create_user(goal, plan, None, role_tiers, configured_tiers)

    def revise(
        self,
        workflow_id: str,
        *,
        goal: str | None = None,
        plan: Any = None,
        role_tiers: Mapping[str, Tier],
        configured_tiers: Collection[Tier],
    ) -> WorkflowTemplate:
        """Save an edited copy of template ``workflow_id`` as a new ``user`` template.

        ``goal`` and/or ``plan`` replace the parent's; the result must differ from it.
        The parent file is never changed, and the edit gets no stats (D41): even an
        edit of a proven project template is a plan that has not been run as is.
        """
        parent = self.load(workflow_id)
        if goal is None and plan is None:
            raise WorkflowError("nothing to change: give a new goal and/or plan")
        new_goal = parent.goal if goal is None else goal
        source = parent.plan if plan is None else plan
        new_plan = (
            user_plan(source.model_dump(mode="json"), new_goal)
            if isinstance(source, Plan)
            else user_plan(source, new_goal)
        )
        if new_goal == parent.goal and new_plan == parent.plan:
            raise WorkflowError(f"nothing changed compared to workflow {workflow_id!r}")
        return self._create_user(new_goal, new_plan, parent.id, role_tiers, configured_tiers)

    def _create_user(
        self,
        goal: str,
        plan: Any,
        parent: str | None,
        role_tiers: Mapping[str, Tier],
        configured_tiers: Collection[Tier],
    ) -> WorkflowTemplate:
        if not isinstance(goal, str) or not goal.strip():
            raise WorkflowError("goal must not be blank")
        goal = goal.strip()
        data = plan.model_dump(mode="json") if isinstance(plan, Plan) else plan
        validated = user_plan(data, goal)
        check_compatible(
            validated, role_tiers=role_tiers, configured_tiers=configured_tiers, label="plan"
        )
        now = self._now()
        with self._lock:
            return self._write(
                {
                    "id": _new_id(now),
                    "origin": WorkflowOrigin.USER,
                    "goal": goal,
                    "parent": parent,
                    "created_at": now,
                    "plan": validated.model_dump(mode="json"),
                }
            )

    def history(self, workflow_id: str) -> list[WorkflowTemplate]:
        """The template and its ancestors via ``parent``, newest first.

        Fails closed on a missing ancestor or a cycle (a hand-edited file).
        """
        chain: list[WorkflowTemplate] = []
        seen: set[str] = set()
        current: str | None = workflow_id
        while current is not None:
            if current in seen:
                raise WorkflowError(f"workflow {workflow_id!r} has a parent cycle")
            seen.add(current)
            try:
                template = self.load(current)
            except WorkflowNotFound:
                if not chain:
                    raise
                raise WorkflowError(
                    f"workflow {chain[-1].id!r} names a missing parent {current!r}"
                ) from None
            chain.append(template)
            current = template.parent
        return chain

    # -- using ---------------------------------------------------------------

    def use(
        self,
        projects: ProjectStore,
        project_id: str,
        workflow_id: str,
        *,
        role_tiers: Mapping[str, Tier],
        configured_tiers: Collection[Tier],
    ) -> Plan:
        """Write the plan of template ``workflow_id`` as the plan of ``project_id``.

        The project must exist and have no plan yet. The template is (re)loaded and
        checked with ``plan_from_template`` before anything changes. On success the
        project's status is ``planning`` (as after ``plan_project``; the Orchestrator
        moves it on) and ``workflow.used`` is logged; on any error the previous status
        is restored. No model is called and the template file is not changed.
        """
        project = projects.load(project_id)
        if projects.has_plan(project_id):
            raise WorkflowError(f"project {project_id!r} already has a plan")
        template = self.load(workflow_id)
        plan = plan_from_template(
            template, project.goal, role_tiers=role_tiers, configured_tiers=configured_tiers
        )
        previous = project.status
        projects.set_status(project_id, ProjectStatus.PLANNING)
        try:
            saved = projects.save_plan(project_id, plan)
        except BaseException:
            projects.set_status(project_id, previous)
            raise
        projects.events(project_id).emit(
            Event(
                ts=self._now(),
                type="workflow.used",
                project_id=project_id,
                data={"workflow_id": template.id, "tasks": len(saved.tasks)},
            )
        )
        return saved

    # -- suggesting ----------------------------------------------------------

    def suggest(
        self, goal: str, *, limit: int = 3, min_score: float = DEFAULT_MIN_SCORE
    ) -> list[WorkflowMatch]:
        """Templates whose goal is similar to ``goal``, best first (deterministic, free).

        Ties are broken by the newer template. Only scores >= ``min_score`` are kept.
        """
        if not 1 <= limit <= MAX_SUGGESTIONS:
            raise WorkflowError(f"limit must be between 1 and {MAX_SUGGESTIONS}")
        if not 0.0 < min_score <= 1.0:
            raise WorkflowError("min_score must be in (0, 1]")
        matches = []
        for template in self.templates():
            score, shared = similarity(goal, template.goal)
            if score >= min_score:
                matches.append(
                    WorkflowMatch(template=template, score=score, shared=tuple(sorted(shared)))
                )
        matches.sort(key=lambda m: m.score, reverse=True)  # stable: newest first on ties
        return matches[:limit]

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise WorkflowError("clock returned a naive datetime")
        return now.astimezone(UTC)
