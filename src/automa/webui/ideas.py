"""New idea from the browser (D124): goal → clarifying questions → plan and cost range.

Only active in action mode (``automa web --allow-actions``) and only when the server
was given an :class:`IdeaPlanner` (the CLI builds one from ``automa.toml``). The
browser flow mirrors ``automa new``:

1. ``POST /new`` (goal, optional "ask questions"): the planner is prepared first
   (configuration, roles and the API key), so a broken setup never leaves an orphan
   project; then the project is created and the Executive is asked its questions.
   Questions are stored in ``questions.json`` until answered.
2. ``POST /projects/<pid>/clarify`` (one answer per question): answers go to
   ``clarifications.json`` (the same record as the CLI) and the plan is made.

Every model call goes through the planner, which uses the same guarded gateway as the
CLI (``BudgetGuard`` and caps unchanged). At most one Executive call runs at a time in
this server (a second one gets ``409``). Running the plan is not started from here.
The request checks (Host, same origin, CSRF token, form shape) are done by the server
before anything in this module runs.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote, unquote

from automa.plan import Clarification
from automa.plan.models import MAX_ANSWER_CHARS, MAX_CLARIFICATIONS
from automa.project.store import (
    MAX_GOAL_CHARS,
    ProjectStatus,
    ProjectStore,
    ProjectStoreError,
    validate_project_id,
)
from automa.webui.actions import ActionRefused

NEW_PATH = "/new"
# A urlencoded Persian goal or answer takes ~6 bytes per character.
MAX_IDEA_FORM_BYTES = 131_072
IDEA_FIELDS = frozenset({"csrf", "goal", "ask"})
ANSWER_FIELDS = frozenset({"csrf", *(f"a{i}" for i in range(MAX_CLARIFICATIONS))})
ASK_VALUE = "yes"
_CLARIFY_RE = re.compile(r"^/projects/([^/]+)/clarify$")

# Kinds of planner failure and the HTTP status the page gets.
KIND_STATUS = {"config": 500, "budget": 409, "planning": 422, "provider": 502}


class IdeaError(Exception):
    """A planner failure with a short, safe message (``kind`` in ``KIND_STATUS``)."""

    def __init__(self, kind: str, message: str) -> None:
        if kind not in KIND_STATUS:
            raise ValueError(f"unknown idea error kind {kind!r}")
        super().__init__(message)
        self.kind = kind
        self.message = message


class IdeaPlanner(Protocol):
    """What the browser flow needs from the Executive (built by the CLI).

    Every method raises only :class:`IdeaError`. ``clarify`` and ``plan`` call the
    model through a gateway guarded by the ``BudgetGuard``."""

    def prepare(self) -> None: ...

    def clarify(self, project_id: str) -> list[str]: ...

    def plan(self, project_id: str, clarifications: Sequence[Clarification]) -> None: ...

    def cost_range(self, project_id: str) -> str: ...


def project_path(project_id: str) -> str:
    return "/projects/" + quote(project_id, safe="")


def clarify_path(project_id: str) -> str:
    return project_path(project_id) + "/clarify"


def parse_clarify_path(path: str) -> str | None:
    """The project id of a ``/projects/<pid>/clarify`` path, else ``None``."""
    m = _CLARIFY_RE.fullmatch(path)
    if m is None:
        return None
    try:
        return validate_project_id(m.group(1))
    except ProjectStoreError:
        return None


def is_idea_path(path: str) -> bool:
    return path == NEW_PATH or parse_clarify_path(path) is not None


def parse_idea(form: Mapping[str, str]) -> tuple[str, bool]:
    """``(goal, ask)`` of a checked new-idea form."""
    goal = form.get("goal", "").strip()
    if not goal:
        raise ActionRefused(400, "write the idea (goal) first")
    if len(goal) > MAX_GOAL_CHARS:
        raise ActionRefused(400, f"the goal is longer than {MAX_GOAL_CHARS} characters")
    ask = form.get("ask")
    if ask is not None and ask != ASK_VALUE:
        raise ActionRefused(400, "malformed form")
    return goal, ask == ASK_VALUE


def parse_answers(form: Mapping[str, str], questions: Sequence[str]) -> list[Clarification]:
    """One :class:`Clarification` per stored question; an unknown answer field is refused."""
    expected = {f"a{i}" for i in range(len(questions))}
    extra = set(form) - {"csrf"} - expected
    if extra:
        raise ActionRefused(400, "the questions differ from the page you saw; reload it")
    items: list[Clarification] = []
    for i, question in enumerate(questions):
        answer = form.get(f"a{i}", "").strip()
        if len(answer) > MAX_ANSWER_CHARS:
            raise ActionRefused(400, f"answer {i + 1} is longer than {MAX_ANSWER_CHARS} characters")
        items.append(Clarification(question=question, answer=answer))
    return items


def _refusal(e: IdeaError, project_id: str | None) -> ActionRefused:
    where = f" (project {project_id})" if project_id else " (nothing was created)"
    return ActionRefused(KIND_STATUS[e.kind], f"{e.message}{where}")


def thinking_path(project_id: str) -> str:
    return f"/projects/{quote(project_id, safe='')}/thinking"


def parse_thinking_path(path: str) -> str | None:
    """The project id of a ``/projects/<pid>/thinking`` path, else ``None``."""
    if not path.startswith("/projects/") or not path.endswith("/thinking"):
        return None
    raw = path[len("/projects/") : -len("/thinking")]
    if "/" in raw:
        return None
    try:
        return validate_project_id(unquote(raw))
    except ProjectStoreError:
        return None


@dataclass
class Job:
    """One background Executive step (D137): asking questions or planning."""

    phase: str  # "questions" | "planning"
    started: float
    done: bool = False
    next_path: str | None = None
    error: str | None = None


class IdeaFlow:
    """The two idea steps over a :class:`ProjectStore` and an :class:`IdeaPlanner`.

    With ``background=True`` (the real server, D137) the slow Executive calls run in
    a thread: the POST returns at once and the browser waits on a self-refreshing
    "thinking" page instead of a request that hangs for a minute.
    """

    def __init__(
        self, store: ProjectStore, planner: IdeaPlanner, *, background: bool = False
    ) -> None:
        self._store = store
        self._planner = planner
        self._busy = threading.Lock()
        self._background = background
        self._jobs: dict[str, Job] = {}
        self._threads: dict[str, threading.Thread] = {}

    def job(self, project_id: str) -> Job | None:
        return self._jobs.get(project_id)

    def wait(self, project_id: str, timeout: float = 10.0) -> None:
        """Wait for a background job (tests and shutdown)."""
        t = self._threads.get(project_id)
        if t is not None:
            t.join(timeout)

    def _spawn(self, project_id: str, phase: str, work: Callable[[], str]) -> str:
        """Run ``work`` (returns the next path) in a thread; the lock is released
        when it ends. Returns the thinking page path."""
        job = Job(phase=phase, started=time.monotonic())
        self._jobs[project_id] = job

        def runner() -> None:
            try:
                job.next_path = work()
            except ActionRefused as e:
                job.error = e.message
            except Exception as e:  # noqa: BLE001 - shown to the owner, never re-raised
                job.error = f"unexpected error: {type(e).__name__}"
            finally:
                job.done = True
                self._busy.release()

        thread = threading.Thread(target=runner, name=f"idea-{project_id}", daemon=True)
        self._threads[project_id] = thread
        thread.start()
        return thinking_path(project_id)

    @property
    def planner(self) -> IdeaPlanner:
        return self._planner

    def _acquire(self) -> None:
        if not self._busy.acquire(blocking=False):
            raise ActionRefused(409, "the Executive is already working on an idea; wait for it")

    def start(self, form: Mapping[str, str]) -> str:
        """Create the project and ask (or plan); returns where the browser goes next."""
        goal, ask = parse_idea(form)
        self._acquire()
        handed_off = False
        try:
            try:
                self._planner.prepare()
            except IdeaError as e:
                raise _refusal(e, None) from None
            try:
                project = self._store.create(goal)
            except ProjectStoreError as e:
                raise ActionRefused(400, str(e)) from None
            pid = project.id
            if self._background:
                handed_off = True
                return self._spawn(
                    pid, "questions" if ask else "planning", lambda: self._first_step(pid, ask)
                )
            return self._first_step(pid, ask)
        finally:
            if not handed_off:
                self._busy.release()

    def _first_step(self, pid: str, ask: bool) -> str:
        """Ask questions (or plan right away); returns the next path."""
        if ask:
            try:
                questions = self._planner.clarify(pid)
            except IdeaError as e:
                # Same statuses as ``automa new`` after a failed question round.
                if e.kind == "budget":
                    self._store.set_status(pid, ProjectStatus.PAUSED)
                elif e.kind == "planning":
                    self._store.set_status(pid, ProjectStatus.FAILED)
                raise _refusal(e, pid) from None
            if questions:
                self._store.save_questions(pid, questions)
                return clarify_path(pid)
        self._plan(pid, [])
        return project_path(pid)

    def pending_questions(self, project_id: str) -> list[str]:
        """The questions still waiting for answers, or ``ActionRefused``."""
        if project_id not in self._store.list_ids():
            raise ActionRefused(404, "no such project")
        try:
            if self._store.has_plan(project_id):
                raise ActionRefused(409, f"project {project_id} already has a plan")
            if self._store.has_clarifications(project_id):
                raise ActionRefused(409, f"project {project_id} was already answered")
            questions = self._store.load_questions(project_id)
        except ProjectStoreError as e:
            raise ActionRefused(500, str(e)) from None
        if not questions:
            raise ActionRefused(409, f"project {project_id} has no questions waiting")
        return questions

    def answer(self, project_id: str, form: Mapping[str, str]) -> str:
        """Record the answers and make the plan; returns the project page path."""
        self._acquire()
        handed_off = False
        try:
            questions = self.pending_questions(project_id)
            items = parse_answers(form, questions)
            try:
                self._planner.prepare()
            except IdeaError as e:
                raise _refusal(e, project_id) from None
            try:
                self._store.save_clarifications(project_id, items)
            except ProjectStoreError as e:
                raise ActionRefused(409, str(e)) from None
            if self._background:
                handed_off = True
                return self._spawn(
                    project_id, "planning", lambda: self._plan_then(project_id, items)
                )
            return self._plan_then(project_id, items)
        finally:
            if not handed_off:
                self._busy.release()

    def _plan_then(self, project_id: str, items: Sequence[Clarification]) -> str:
        self._plan(project_id, items)
        return project_path(project_id)

    def _plan(self, project_id: str, items: Sequence[Clarification]) -> None:
        try:
            self._planner.plan(project_id, items)
        except IdeaError as e:
            raise _refusal(e, project_id) from None
