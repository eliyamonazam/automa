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
   Needs from the user (D163, asked in the same Executive call) are answered on the
   same form with text and/or files; files go to ``files/inputs/idea/`` and the
   record to ``needs.json``; each need is also a clarification for the plan.

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

from automa.agents.executive import Questions
from automa.ideafiles import IdeaFileError, check_files, check_more, save_files
from automa.needs import need_clarifications
from automa.plan import Clarification
from automa.plan.models import (
    MAX_ANSWER_CHARS,
    MAX_CLARIFICATIONS,
    MAX_NEEDS,
    NeedAnswer,
    NeedFromUser,
    PendingQuestions,
)
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
NEED_FIELDS = frozenset(f"n{i}" for i in range(MAX_NEEDS))  # D163: text for need i
NEED_FILE_FIELDS = frozenset(f"nf{i}" for i in range(MAX_NEEDS))  # files for need i
ANSWER_FIELDS = frozenset({"csrf", *(f"a{i}" for i in range(MAX_CLARIFICATIONS)), *NEED_FIELDS})
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
    model through a gateway guarded by the ``BudgetGuard``. ``clarify`` returns the
    questions, or (D163) the whole round with the needs from the user."""

    def prepare(self) -> None: ...

    def clarify(self, project_id: str) -> Sequence[str] | Questions: ...

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
        raise ActionRefused.coded(400, "goal_missing")
    if len(goal) > MAX_GOAL_CHARS:
        raise ActionRefused.coded(400, "goal_too_long", max=MAX_GOAL_CHARS)
    ask = form.get("ask")
    if ask is not None and ask != ASK_VALUE:
        raise ActionRefused.coded(400, "malformed_form")
    return goal, ask == ASK_VALUE


def parse_answers(
    form: Mapping[str, str], questions: Sequence[str], needs: int = 0
) -> list[Clarification]:
    """One :class:`Clarification` per stored question; an unknown answer field is refused.

    ``needs``: how many needs (D163) the form shows; their text fields are allowed."""
    expected = {f"a{i}" for i in range(len(questions))} | {f"n{i}" for i in range(needs)}
    extra = set(form) - {"csrf"} - expected
    if extra:
        raise ActionRefused.coded(400, "questions_changed")
    items: list[Clarification] = []
    for i, question in enumerate(questions):
        answer = form.get(f"a{i}", "").strip()
        if len(answer) > MAX_ANSWER_CHARS:
            raise ActionRefused.coded(400, "answer_too_long", n=i + 1, max=MAX_ANSWER_CHARS)
        items.append(Clarification(question=question, answer=answer))
    return items


def parse_need_answers(
    form: Mapping[str, str],
    needs: Sequence[NeedFromUser],
    files: Sequence[tuple[str, str, bytes]],
) -> tuple[list[str], list[list[tuple[str, bytes]]]]:
    """The text and the files given for each need (D163).

    ``files``: ``(field, name, data)``; a file under a field of no shown need is
    refused (the questions changed)."""
    texts: list[str] = []
    for i in range(len(needs)):
        answer = form.get(f"n{i}", "").strip()
        if len(answer) > MAX_ANSWER_CHARS:
            raise ActionRefused.coded(400, "need_answer_too_long", n=i + 1, max=MAX_ANSWER_CHARS)
        texts.append(answer)
    per_need: list[list[tuple[str, bytes]]] = [[] for _ in needs]
    allowed = {f"nf{i}": i for i in range(len(needs))}
    for field, name, data in files:
        if field not in allowed:
            raise ActionRefused.coded(400, "questions_changed")
        per_need[allowed[field]].append((name, data))
    return texts, per_need


def split_round(result: Sequence[str] | Questions) -> tuple[list[str], list[NeedFromUser]]:
    """Questions and needs of a planner's ``clarify`` result (a plain list: no needs)."""
    if isinstance(result, Questions):
        return list(result.questions), list(result.needs_from_user)
    return list(result), []


def _file_refusal(e: IdeaFileError) -> ActionRefused:
    """A refused attached file (D162) as a translated refusal."""
    p = e.params
    if e.code == "idea_files_too_many":
        return ActionRefused.coded(400, "idea_files_too_many", max=str(p.get("max", "")))
    if e.code == "idea_file_too_large":
        return ActionRefused.coded(
            400, "idea_file_too_large", name=str(p.get("name", "")), mb=str(p.get("mb", ""))
        )
    if e.code == "idea_files_too_large":
        return ActionRefused.coded(400, "idea_files_too_large", mb=str(p.get("mb", "")))
    return ActionRefused.coded(400, "idea_file_not_saved", name=str(p.get("name", "")))


def _refusal(e: IdeaError, project_id: str | None) -> ActionRefused:
    where = f" (project {project_id})" if project_id else " (nothing was created)"
    return ActionRefused(KIND_STATUS[e.kind], f"{e.message}{where}", lead=f"idea.{e.kind}")


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
            raise ActionRefused.coded(409, "executive_busy")

    def start(self, form: Mapping[str, str], files: Sequence[tuple[str, bytes]] = ()) -> str:
        """Create the project and ask (or plan); returns where the browser goes next.

        ``files`` (D162): files attached to the idea; checked before the project is
        created and saved into its ``files/inputs/idea/`` before the first model call.
        """
        goal, ask = parse_idea(form)
        try:
            check_files(files)
        except IdeaFileError as e:
            raise _file_refusal(e) from None
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
                raise ActionRefused(400, str(e), lead="lead.store") from None
            pid = project.id
            if files:
                try:
                    save_files(self._store.files_dir(pid), files)
                except IdeaFileError as e:
                    raise _file_refusal(e) from None
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
                questions, needs = split_round(self._planner.clarify(pid))
            except IdeaError as e:
                # Same statuses as ``automa new`` after a failed question round.
                if e.kind == "budget":
                    self._store.set_status(pid, ProjectStatus.PAUSED)
                elif e.kind == "planning":
                    self._store.set_status(pid, ProjectStatus.FAILED)
                raise _refusal(e, pid) from None
            if questions or needs:
                self._store.save_questions(pid, questions, needs)
                return clarify_path(pid)
        self._plan(pid, [])
        return project_path(pid)

    def pending_questions(self, project_id: str) -> list[str]:
        """The questions still waiting for answers, or ``ActionRefused``."""
        return list(self.pending(project_id).questions)

    def pending(self, project_id: str) -> PendingQuestions:
        """The questions and needs (D163) still waiting for answers, or ``ActionRefused``."""
        if project_id not in self._store.list_ids():
            raise ActionRefused.coded(404, "no_such_project")
        try:
            if self._store.has_plan(project_id):
                raise ActionRefused.coded(409, "has_plan", id=project_id)
            if self._store.has_clarifications(project_id):
                raise ActionRefused.coded(409, "already_answered", id=project_id)
            pending = self._store.load_pending(project_id)
        except ProjectStoreError as e:
            raise ActionRefused(500, str(e), lead="lead.store") from None
        if pending is None or not (pending.questions or pending.needs):
            raise ActionRefused.coded(409, "no_questions", id=project_id)
        return pending

    def answer(
        self,
        project_id: str,
        form: Mapping[str, str],
        files: Sequence[tuple[str, str, bytes]] = (),
    ) -> str:
        """Record the answers and make the plan; returns the project page path.

        ``files`` (D163): ``(field, name, data)`` given for the needs (``nf<i>``)."""
        self._acquire()
        handed_off = False
        try:
            pending = self.pending(project_id)
            needs = list(pending.needs)
            items = parse_answers(form, pending.questions, len(needs))
            texts, per_need = parse_need_answers(form, needs, files)
            files_dir = self._store.files_dir(project_id)
            try:
                check_more(files_dir, [f for group in per_need for f in group])
            except IdeaFileError as e:
                raise _file_refusal(e) from None
            try:
                self._planner.prepare()
            except IdeaError as e:
                raise _refusal(e, project_id) from None
            answers: list[NeedAnswer] = []
            for need, text, group in zip(needs, texts, per_need, strict=True):
                try:
                    paths = save_files(files_dir, group)
                except IdeaFileError as e:
                    raise _file_refusal(e) from None
                answers.append(NeedAnswer(need=need, answer=text, files=paths))
            items = [*items, *need_clarifications(answers)]
            try:
                self._store.save_clarifications(project_id, items)
                if answers:
                    self._store.save_needs(project_id, answers)
            except ProjectStoreError as e:
                raise ActionRefused(409, str(e), lead="lead.store") from None
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
