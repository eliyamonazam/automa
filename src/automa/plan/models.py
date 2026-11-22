"""Data models of the planning/execution loop: ``Plan``, ``Task``, ``TaskResult``, ``Verdict``.

These models are the contract between agents. Model (LLM) output is untrusted, so
every model is strict (``extra="forbid"``, frozen, no inf/NaN) and validated:

* ids are safe directory names (tasks live in ``tasks/<task-id>/``);
* file paths in checks and results are relative, normalized and stay inside the
  project's ``files/`` directory (no ``..``, no absolute paths, no drive letters);
* a ``Plan`` has unique task ids, only known dependencies, no self-dependencies and
  no cycles, so a topological order always exists;
* a ``Verdict`` is internally consistent: ``pass`` has no failed check and no issue,
  ``fail`` names at least one reason.

Checks in acceptance criteria are *descriptions* of deterministic checks. They are
executed later by the Verifier through the Tool Layer and its Permission Policy;
nothing here runs anything.
"""

from __future__ import annotations

import json
import re
import shlex
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from automa.config import Tier

PLAN_SCHEMA_VERSION = 1

TASK_ID_PATTERN = r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$"
ROLE_PATTERN = r"^[a-z][a-z0-9_-]{0,31}$"
INPUT_KEY_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"

MAX_TASKS = 100
MAX_TITLE_CHARS = 200
MAX_TEXT_CHARS = 10_000
MAX_LIST_ITEMS = 50
MAX_PATH_CHARS = 255
MAX_COMMAND_CHARS = 2_000
MAX_CHECK_TIMEOUT_S = 600
MAX_QUESTION_CHARS = 500
MAX_ANSWER_CHARS = 2_000
MAX_CLARIFICATIONS = 20
MAX_NEEDS = 5  # things the user must provide, asked with the questions (D163)
MAX_NEED_WHAT_CHARS = 300
MAX_NEED_TEXT_CHARS = 500

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class PlanError(Exception):
    """A plan (or other loop document) is invalid. ``details`` is model-readable feedback."""

    def __init__(self, message: str, details: str = ""):
        super().__init__(message)
        self.details = details or message


def _not_blank(v: str) -> str:
    if not v.strip():
        raise ValueError("must not be blank")
    return v


def _safe_relpath(v: str) -> str:
    """Normalize a path relative to the project's ``files/`` dir, or raise ``ValueError``."""
    if "\x00" in v:
        raise ValueError("path must not contain NUL")
    if "\\" in v:
        raise ValueError("path must use '/' separators")
    if v.startswith("/") or _WINDOWS_DRIVE_RE.match(v):
        raise ValueError("path must be relative")
    parts = [p for p in PurePosixPath(v).parts if p != "."]
    if not parts:
        raise ValueError("path must name a file or directory")
    if any(p == ".." for p in parts):
        raise ValueError("path must not contain '..'")
    return "/".join(parts)


Title = Annotated[str, Field(min_length=1, max_length=MAX_TITLE_CHARS), AfterValidator(_not_blank)]
Text = Annotated[str, Field(max_length=MAX_TEXT_CHARS)]
LongText = Annotated[
    str, Field(min_length=1, max_length=MAX_TEXT_CHARS), AfterValidator(_not_blank)
]
TaskId = Annotated[str, Field(pattern=TASK_ID_PATTERN)]
RelPath = Annotated[
    str, Field(min_length=1, max_length=MAX_PATH_CHARS), AfterValidator(_safe_relpath)
]


def _utc(v: datetime) -> datetime:
    if v.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return v.astimezone(UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_utc)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


# --- acceptance criteria -----------------------------------------------------


class FileExistsCheck(_Strict):
    """The file ``path`` (relative to ``files/``) exists."""

    type: Literal["file_exists"] = "file_exists"
    path: RelPath


class FileContainsCheck(_Strict):
    """The file ``path`` exists and contains ``text`` (plain substring)."""

    type: Literal["file_contains"] = "file_contains"
    path: RelPath
    text: Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS)]


class CommandCheck(_Strict):
    """``command`` (run via the Tool Layer, inside the project) exits with ``exit_code``."""

    type: Literal["command"] = "command"
    command: Annotated[
        str, Field(min_length=1, max_length=MAX_COMMAND_CHARS), AfterValidator(_not_blank)
    ]
    exit_code: int = Field(default=0, ge=0, le=255)
    timeout_s: int = Field(default=60, ge=1, le=MAX_CHECK_TIMEOUT_S)

    @field_validator("command")
    @classmethod
    def _single_line(cls, v: str) -> str:
        if "\x00" in v or "\n" in v or "\r" in v:
            raise ValueError("command must be a single line")
        return v


Check = Annotated[FileExistsCheck | FileContainsCheck | CommandCheck, Field(discriminator="type")]


class AcceptanceCriterion(_Strict):
    """One acceptance criterion. With ``check`` it is verified deterministically first;
    without, only by the model-based evaluation (see D4)."""

    description: Title
    check: Check | None = None

    @property
    def deterministic(self) -> bool:
        return self.check is not None


# --- plan --------------------------------------------------------------------


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    VERIFYING = "verifying"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


# What a task depends on that the tools may not reliably give (M16, D164): live data
# from specific websites, an account behind a login, a paid API, or a desktop GUI.
# Canonical order; the project page and ``automa new`` warn about these before
# "Start" and offer a cheaper, honest alternative. Data only: grants nothing.
TaskNeed = Literal["live_web_data", "login", "paid_api", "gui"]
TASK_NEEDS: tuple[TaskNeed, ...] = ("live_web_data", "login", "paid_api", "gui")


class Task(_Strict):
    """One unit of work, done by a worker with ``role`` and judged by its criteria.

    ``needs`` (D164) names what the task depends on beyond the workspace (see
    ``TaskNeed``); duplicates are dropped and the order is canonical."""

    id: TaskId
    title: Title
    description: Text = ""
    role: str = Field(pattern=ROLE_PATTERN)
    tier: Tier | None = None  # None = the role's default tier
    inputs: dict[Annotated[str, Field(pattern=INPUT_KEY_PATTERN)], Text] = Field(
        default_factory=dict, max_length=MAX_LIST_ITEMS
    )
    depends_on: list[TaskId] = Field(default_factory=list, max_length=MAX_TASKS)
    acceptance_criteria: list[AcceptanceCriterion] = Field(min_length=1, max_length=MAX_LIST_ITEMS)
    needs: list[TaskNeed] = Field(default_factory=list, max_length=MAX_LIST_ITEMS)

    @field_validator("needs")
    @classmethod
    def _canonical_needs(cls, v: list[TaskNeed]) -> list[TaskNeed]:
        return [n for n in TASK_NEEDS if n in v]

    @model_validator(mode="after")
    def _deps(self) -> Task:
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError(f"task {self.id!r}: duplicate entries in depends_on")
        if self.id in self.depends_on:
            raise ValueError(f"task {self.id!r} depends on itself")
        return self


MAX_MAIN_FILES = 20


class Deliverable(_Strict):
    """What the user gets at the end, stated before any work starts (D155).

    ``summary`` says in plain words what the finished product is, ``main_files``
    are the files that *are* the product (each one is checked by some task, see
    ``Plan``), and ``how_to_use`` says how to open, run or read it. All of it is
    data shown to the user; none of it grants a tool, a path or a budget.
    """

    summary: LongText
    main_files: list[RelPath] = Field(default_factory=list, max_length=MAX_MAIN_FILES)
    how_to_use: Text = ""

    @model_validator(mode="after")
    def _unique(self) -> Deliverable:
        if len(set(self.main_files)) != len(self.main_files):
            raise ValueError("deliverable.main_files has duplicate entries")
        return self


def checked_paths(task: Task) -> frozenset[str]:
    """Paths a task's deterministic file checks name (``file_exists``/``file_contains``)."""
    return frozenset(
        c.check.path
        for c in task.acceptance_criteria
        if isinstance(c.check, FileExistsCheck | FileContainsCheck)
    )


class Plan(_Strict):
    """Contents of ``plan.json``: the goal broken into a DAG of tasks.

    ``version`` starts at 1 and grows with every re-plan by the Executive.
    ``deliverable`` (optional, D155) names the finished product; every main file
    must be named by a ``file_exists``/``file_contains`` check of some task, so the
    product the user is promised is the product the verifier checks.
    """

    schema_version: int = PLAN_SCHEMA_VERSION
    version: int = Field(default=1, ge=1)
    goal: LongText
    tasks: list[Task] = Field(min_length=1, max_length=MAX_TASKS)
    deliverable: Deliverable | None = None

    @model_validator(mode="after")
    def _deliverable_checked(self) -> Plan:
        if self.deliverable is None:
            return self
        checked: set[str] = set()
        for t in self.tasks:
            checked |= checked_paths(t)
        unchecked = [p for p in self.deliverable.main_files if p not in checked]
        if unchecked:
            raise ValueError(
                "deliverable.main_files not checked by any task (add a file_exists "
                f"check to the task that makes them): {', '.join(unchecked)}"
            )
        return self

    def task_for_file(self, path: str) -> Task | None:
        """The first task (plan order) whose file checks name ``path``."""
        return next((t for t in self.tasks if path in checked_paths(t)), None)

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported plan schema version {v} (expected {PLAN_SCHEMA_VERSION})"
            )
        return v

    @model_validator(mode="after")
    def _graph(self) -> Plan:
        ids = [t.id for t in self.tasks]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate task ids: {', '.join(dupes)}")
        known = set(ids)
        for t in self.tasks:
            missing = [d for d in t.depends_on if d not in known]
            if missing:
                raise ValueError(f"task {t.id!r} depends on unknown task(s): {', '.join(missing)}")
        cycle = _find_cycle(self.tasks)
        if cycle:
            raise ValueError(f"dependency cycle: {' -> '.join(cycle)}")
        return self

    def task(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise KeyError(task_id)

    @property
    def task_ids(self) -> list[str]:
        return [t.id for t in self.tasks]

    def topological_order(self) -> list[Task]:
        """Tasks so that each comes after all its dependencies; ties keep plan order."""
        done: set[str] = set()
        order: list[Task] = []
        remaining = list(self.tasks)
        while remaining:
            ready = [t for t in remaining if all(d in done for d in t.depends_on)]
            # The plan validator guarantees acyclicity, so ``ready`` is never empty.
            for t in ready:
                order.append(t)
                done.add(t.id)
            remaining = [t for t in remaining if t.id not in done]
        return order

    def ready_tasks(self, passed: set[str] | frozenset[str]) -> list[Task]:
        """Tasks not in ``passed`` whose dependencies are all in ``passed`` (plan order)."""
        unknown = set(passed) - set(self.task_ids)
        if unknown:
            raise KeyError(f"unknown task id(s): {', '.join(sorted(unknown))}")
        return [
            t for t in self.tasks if t.id not in passed and all(d in passed for d in t.depends_on)
        ]

    def dependents(self, task_id: str) -> list[Task]:
        """Tasks that (directly or transitively) depend on ``task_id``, in plan order."""
        self.task(task_id)
        affected = {task_id}
        changed = True
        while changed:
            changed = False
            for t in self.tasks:
                if t.id not in affected and any(d in affected for d in t.depends_on):
                    affected.add(t.id)
                    changed = True
        return [t for t in self.tasks if t.id in affected and t.id != task_id]


def _find_cycle(tasks: list[Task]) -> list[str] | None:
    """Return one dependency cycle as ``[a, b, ..., a]``, or ``None``. Iterative DFS."""
    deps = {t.id: t.depends_on for t in tasks}
    white, grey, black = 0, 1, 2
    color = dict.fromkeys(deps, white)
    for start in deps:
        if color[start] != white:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        path: list[str] = [start]
        color[start] = grey
        while stack:
            node, i = stack[-1]
            if i < len(deps[node]):
                stack[-1] = (node, i + 1)
                nxt = deps[node][i]
                if color[nxt] == grey:
                    return path[path.index(nxt) :] + [nxt]
                if color[nxt] == white:
                    color[nxt] = grey
                    stack.append((nxt, 0))
                    path.append(nxt)
            else:
                color[node] = black
                stack.pop()
                path.pop()
    return None


# --- results and verdicts ----------------------------------------------------


class ResultStatus(StrEnum):
    COMPLETED = "completed"  # worker says it is done; still needs verification
    FAILED = "failed"  # worker gave up or hit an error
    STEP_LIMIT = "step_limit"  # worker hit its step cap


class TaskResult(_Strict):
    """What a worker produced in one attempt at a task (``tasks/<id>/``)."""

    task_id: TaskId
    attempt: int = Field(ge=1)
    status: ResultStatus
    summary: Text = ""
    files: list[RelPath] = Field(default_factory=list, max_length=1_000)
    tier: Tier
    model: str = Field(min_length=1, max_length=200)
    steps: int = Field(default=0, ge=0)
    cost_nusd: int = Field(default=0, ge=0)  # integer nano-USD, as in the ledger
    error: Text | None = None
    started_at: UtcDatetime
    finished_at: UtcDatetime

    @model_validator(mode="after")
    def _consistent(self) -> TaskResult:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at is before started_at")
        if len(set(self.files)) != len(self.files):
            raise ValueError("duplicate entries in files")
        if self.status is ResultStatus.FAILED and not (self.error and self.error.strip()):
            raise ValueError("a failed result needs an error")
        return self


class VerdictOutcome(StrEnum):
    PASS = "pass"  # noqa: S105 - verdict outcome, not a password
    FAIL = "fail"


# A catalogued model family, as in ``[models."<id>"] family`` (D94).
FAMILY_PATTERN = r"^[a-z][a-z0-9_-]{0,31}$"


class CheckResult(_Strict):
    """Outcome of one deterministic check (``criterion`` indexes ``acceptance_criteria``)."""

    criterion: int = Field(ge=0, lt=MAX_LIST_ITEMS)
    passed: bool
    detail: Text = ""


class Issue(_Strict):
    """A problem the Verifier found; goes back to the worker on the next attempt."""

    description: LongText
    criterion: int | None = Field(default=None, ge=0, lt=MAX_LIST_ITEMS)


class Verdict(_Strict):
    """The Verifier's judgment of one ``TaskResult`` (same ``task_id`` and ``attempt``).

    ``model_evaluated`` is false when the deterministic checks already failed, so the
    model-based evaluation was skipped (D4).
    """

    task_id: TaskId
    attempt: int = Field(ge=1)
    outcome: VerdictOutcome
    checks: list[CheckResult] = Field(default_factory=list, max_length=MAX_LIST_ITEMS)
    issues: list[Issue] = Field(default_factory=list, max_length=MAX_LIST_ITEMS)
    model_evaluated: bool = False
    # Which model did the model-based evaluation, and its catalogued family (D96).
    # Absent in verdicts written before D96 and when no model evaluated.
    evaluator_model: str | None = Field(default=None, min_length=1, max_length=200)
    evaluator_family: str | None = Field(default=None, pattern=FAMILY_PATTERN)
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _consistent(self) -> Verdict:
        if not self.model_evaluated and (
            self.evaluator_model is not None or self.evaluator_family is not None
        ):
            raise ValueError("an evaluator is only recorded when a model evaluated")
        if self.evaluator_family is not None and self.evaluator_model is None:
            raise ValueError("an evaluator family needs an evaluator model")
        indexes = [c.criterion for c in self.checks]
        if len(set(indexes)) != len(indexes):
            raise ValueError("duplicate check results for one criterion")
        failed_check = any(not c.passed for c in self.checks)
        if self.outcome is VerdictOutcome.PASS:
            if failed_check:
                raise ValueError("a passing verdict cannot have a failed check")
            if self.issues:
                raise ValueError("a passing verdict cannot have issues")
        elif not failed_check and not self.issues:
            raise ValueError("a failing verdict needs a failed check or an issue")
        return self

    @property
    def passed(self) -> bool:
        return self.outcome is VerdictOutcome.PASS

    def matches(self, task: Task, result: TaskResult) -> bool:
        """True if this verdict is about ``result`` of ``task`` and its indexes fit."""
        n = len(task.acceptance_criteria)
        return (
            self.task_id == task.id == result.task_id
            and self.attempt == result.attempt
            and all(c.criterion < n for c in self.checks)
            and all(i.criterion is None or i.criterion < n for i in self.issues)
        )


MAX_TASK_ATTEMPTS = 50


class TaskState(_Strict):
    """Orchestrator bookkeeping of one task (``tasks/<id>/state.json``).

    ``attempts`` is the number of ``TaskResult`` files stored for the task; the store
    only saves a state whose ``attempts`` matches the files on disk, and only marks a
    task ``passed`` if the verdict of its last attempt is a stored ``pass``. ``tier``
    is the tier of the next attempt (``None`` = task/role default), so model escalation
    survives a restart. ``revision`` gives optimistic concurrency like ``Project``.
    """

    schema_version: int = PLAN_SCHEMA_VERSION
    task_id: TaskId
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = Field(default=0, ge=0, le=MAX_TASK_ATTEMPTS)
    tier: Tier | None = None
    # The model of the current/last attempt (D97): from the role's model list or the
    # tier. Recorded for visibility only; it is recomputed from the config each attempt.
    model: str | None = Field(default=None, min_length=1, max_length=200)
    plan_version: int = Field(default=1, ge=1)
    error: Text | None = None
    updated_at: UtcDatetime
    revision: int = Field(default=0, ge=0)

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported task state schema_version {v}")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> TaskState:
        if self.status is TaskStatus.PASSED and self.attempts < 1:
            raise ValueError("a passed task needs at least one attempt")
        if self.status in (TaskStatus.FAILED, TaskStatus.BLOCKED) and not (
            self.error and self.error.strip()
        ):
            raise ValueError(f"a {self.status.value} task needs an error")
        return self


# --- parsing model output ----------------------------------------------------


def _format_errors(e: ValidationError) -> str:
    lines = []
    for err in e.errors(include_url=False, include_input=False)[:20]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"- {loc}: {err['msg']}")
    return "\n".join(lines)


def parse_plan(text: str) -> Plan:
    """Parse and validate a plan from (model) JSON text.

    Accepts a bare JSON object, optionally wrapped in one Markdown code fence. Raises
    ``PlanError`` whose ``details`` lists the problems in a form that can be sent back
    to the model for a retry. Never returns a partially valid plan.
    """
    raw = strip_code_fence(text)
    try:
        data: Any = json.loads(raw)
    except ValueError as e:
        raise PlanError("plan is not valid JSON", f"- <root>: invalid JSON ({e})") from e
    if not isinstance(data, dict):
        raise PlanError("plan must be a JSON object", "- <root>: expected a JSON object")
    try:
        plan = Plan.model_validate(data)
    except ValidationError as e:
        raise PlanError("plan failed validation", _format_errors(e)) from e
    problems = shell_syntax_problems(plan)
    if problems:
        raise PlanError("plan uses shell syntax in command checks", "\n".join(problems))
    return plan


# Command checks run as a plain argv (``shlex.split``, no shell; see the Tool
# Layer). Shell operators therefore never work: ``!`` becomes a program name,
# ``$(...)`` is passed literally, ``|`` is an argument. A check written that way
# can never pass, so every worker attempt on its task is wasted (D92). Catch it
# at planning time, when the Executive can still rewrite the check.
_SHELL_OPERATORS = frozenset(
    {"|", "||", "&", "&&", ";", ";;", "!", ">", ">>", "<", "<<", "<<<", "(", ")", "{", "}"}
)
_SHELL_HINT = (
    "commands run WITHOUT a shell (no pipes, redirects, '!', '&&', ';', '$(...)', "
    "backticks or $VARIABLES). Use one program with arguments, e.g. "
    "\"python3 -c '<code>'\" for anything complex; to require that something is "
    "absent, use the program's own exit code (e.g. grep -q PATTERN FILE with "
    "exit_code 1) instead of '!'."
)


def shell_construct(command: str) -> str | None:
    """The first shell-only construct in ``command`` (as the model wrote it), or None."""
    if "$(" in command or "`" in command:
        return "$(...)" if "$(" in command else "backticks"
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return None  # unbalanced quotes: reported when the check runs
    for tok in tokens:
        if tok in _SHELL_OPERATORS or (tok and set(tok) <= set("|&;<>()")):
            return tok
        if re.fullmatch(r"\$[A-Za-z_{][A-Za-z0-9_}]*", tok):
            return tok
    return None


def shell_syntax_problems(plan: Plan) -> list[str]:
    """Model-readable problems for command checks that need a shell (D92)."""
    problems = []
    for task in plan.tasks:
        for i, criterion in enumerate(task.acceptance_criteria):
            check = criterion.check
            if isinstance(check, CommandCheck):
                found = shell_construct(check.command)
                if found is not None:
                    problems.append(
                        f"- tasks[{task.id}].acceptance_criteria.{i}.check.command: "
                        f"uses {found!r}; {_SHELL_HINT}"
                    )
    return problems


_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n(.*)\n```$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """Strip surrounding whitespace and at most one Markdown code fence."""
    s = text.strip()
    m = _FENCE_RE.match(s)
    return m.group(1) if m else s


QuestionText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
]


class Clarification(BaseModel):
    """A clarifying question (asked by the Executive) and the user's answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: QuestionText
    answer: Annotated[str, StringConstraints(strip_whitespace=True, max_length=MAX_ANSWER_CHARS)]


NeedKind = Literal["file", "access", "example", "decision", "other"]


class NeedFromUser(BaseModel):
    """Something only the user can give (D163): asked together with the questions.

    Model output: shown escaped, never trusted. ``what``: the thing in plain words;
    ``why``: what it is needed for; ``required``: the product cannot be done right
    without it (otherwise it only improves the result); ``where``: how the user can
    get it (plain steps, for a non-technical person)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    what: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_NEED_WHAT_CHARS)
    ]
    why: Annotated[
        str, StringConstraints(strip_whitespace=True, max_length=MAX_NEED_TEXT_CHARS)
    ] = ""
    required: bool = False
    where: Annotated[
        str, StringConstraints(strip_whitespace=True, max_length=MAX_NEED_TEXT_CHARS)
    ] = ""
    kind: NeedKind = "other"


class NeedAnswer(BaseModel):
    """What the user gave for one :class:`NeedFromUser` (``needs.json``, D163)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    need: NeedFromUser
    answer: Annotated[
        str, StringConstraints(strip_whitespace=True, max_length=MAX_ANSWER_CHARS)
    ] = ""
    files: list[Annotated[str, StringConstraints(min_length=1, max_length=MAX_PATH_CHARS)]] = Field(
        default_factory=list, max_length=MAX_LIST_ITEMS
    )

    @property
    def provided(self) -> bool:
        return bool(self.answer or self.files)


class NeedsRecord(BaseModel):
    """On-disk record of the needs and what the user gave (``needs.json``, D163)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = PLAN_SCHEMA_VERSION
    items: list[NeedAnswer] = Field(default_factory=list, max_length=MAX_NEEDS)

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported needs schema_version {v}")
        return v

    def missing_required(self) -> list[NeedFromUser]:
        return [i.need for i in self.items if i.need.required and not i.provided]


class PendingQuestions(BaseModel):
    """Questions asked by the Executive and not answered yet (``questions.json``, D124).

    Written by the web UI between "new idea" and "answers"; the answers themselves go
    to ``clarifications.json`` as before. Model output: shown escaped, never trusted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = PLAN_SCHEMA_VERSION
    questions: list[QuestionText] = Field(default_factory=list, max_length=MAX_CLARIFICATIONS)
    needs: list[NeedFromUser] = Field(default_factory=list, max_length=MAX_NEEDS)  # D163

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported questions schema_version {v}")
        return v


class Clarifications(BaseModel):
    """On-disk record of the clarification round (``clarifications.json``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = PLAN_SCHEMA_VERSION
    items: list[Clarification] = Field(default_factory=list, max_length=MAX_CLARIFICATIONS)

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported clarifications schema_version {v}")
        return v
