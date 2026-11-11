"""Workflow templates: plans of successful projects, saved for reuse (M6, D38).

A template is a frozen copy of the plan of a project that finished with **every**
task ``passed`` (the store only marks a task passed with a stored passing verdict),
plus the source goal and a few statistics. Templates live in
``<workspace>/workflows/<id>.json`` and are written once (never overwritten).

``WorkflowStore.suggest(goal)`` ranks stored templates by a deterministic word
overlap with a new goal (no model call, no cost). Nothing here runs a template or
changes a project; using a suggestion is a later, explicit step.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from automa.events import Event
from automa.plan import Plan, TaskStatus
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


class WorkflowTemplate(BaseModel):
    """Contents of ``workflows/<id>.json``. Immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: int = WORKFLOW_SCHEMA_VERSION
    id: str = Field(pattern=WORKFLOW_ID_PATTERN)
    goal: str = Field(min_length=1, max_length=MAX_GOAL_CHARS)
    source_project: str
    source_plan_version: int = Field(ge=1)
    created_at: datetime
    plan: Plan
    stats: WorkflowStats

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != WORKFLOW_SCHEMA_VERSION:
            raise ValueError(f"unsupported workflow schema version {v}")
        return v

    @field_validator("source_project")
    @classmethod
    def _project_id(cls, v: str) -> str:
        try:
            return validate_project_id(v)
        except ProjectStoreError as e:
            raise ValueError(str(e)) from None

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


# --- store -----------------------------------------------------------------------


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
            template = WorkflowTemplate(
                id=f"wf-{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}",
                goal=project.goal,
                source_project=project_id,
                source_plan_version=plan.version,
                created_at=now,
                plan=plan.model_copy(update={"version": 1}),
                stats=WorkflowStats(tasks=len(plan.tasks), attempts=attempts, cost_nusd=cost),
            )
            # Re-validate the whole template (model_copy skips validation).
            template = WorkflowTemplate.model_validate(template.model_dump())
            self._check_root()
            self._root.mkdir(parents=True, exist_ok=True)
            try:
                _write_json_exclusive(self.path(template.id), template.model_dump(mode="json"))
            except ProjectStoreError as e:
                raise WorkflowError(str(e)) from None
        projects.events(project_id).emit(
            Event(
                ts=now,
                type="workflow.saved",
                project_id=project_id,
                data={"workflow_id": template.id, "tasks": template.stats.tasks},
            )
        )
        return template

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
