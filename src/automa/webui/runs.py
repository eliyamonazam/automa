"""Start a planned project's run from the browser, in the background (D125).

Only active in action mode (``automa web --allow-actions``) and only when the server
was given a :class:`RunPreflight` (the CLI builds one from ``automa.toml``).

``POST /projects/<pid>/start`` (after the server's Host, same-origin and CSRF checks)
starts ``automa run <pid>`` as a **child process** — the very same command as in the
terminal, so the ``BudgetGuard``, the caps, the Permission Policy and the Approval Gate
are exactly those of the CLI. The child gets no terminal (``stdin`` is ``/dev/null``):

* if the upper end of the cost range is above ``[limits] confirm_above_usd`` (D121),
  the page shows a checkbox; only a ticked box adds ``--yes``. Without it the start is
  refused before anything runs;
* if the range was below the threshold, ``--yes`` is **not** passed: should the range
  grow above it before the child starts, the child stops with exit code 11 without
  spending anything (it never spends silently).

One background run per project and server (a second start gets ``409``), and never
while the project is marked ``running`` (e.g. by a terminal run). The child's output
goes to a new file ``<project>/web-runs/run-<UTC time>.log`` (created exclusively,
never through a symlink). A watcher thread records ``web.run_started`` and
``web.run_finished`` (with the exit code) in the project's events.

``POST /projects/<pid>/stop`` (same checks, D126) asks the running child to pause
after its current step: it writes the project's stop request
(``ProjectStore.request_stop``), which the Orchestrator honours between steps — no
model call is interrupted and nothing is killed. The child then exits with code 12
and the project is ``paused``; "Start run" or ``automa run <pid>`` resumes it. A stop
is only accepted while this server's child is running; when the child ends, any
request left over is removed (under the same lock), so it can never stop a later run.
"""

from __future__ import annotations

import os
import re
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from automa.budget.brake import BrakeError, continue_task, decline_task
from automa.events import Event
from automa.project.store import (
    ProjectStatus,
    ProjectStore,
    ProjectStoreError,
    validate_project_id,
)
from automa.retry import RetryError, can_retry
from automa.tools.command import spawn_automa_run
from automa.webui.actions import CONFIRM_VALUE, ActionRefused

START_FIELDS = frozenset({"csrf", "confirm"})
STOP_FIELDS = frozenset({"csrf"})
BRAKE_FIELDS = frozenset({"csrf", "task", "decision", "confirm"})
RETRY_FIELDS = frozenset({"csrf", "confirm"})
BRAKE_CONTINUE = "continue"
BRAKE_STOP = "stop"
MAX_TASK_ID = 200
RUNS_DIR = "web-runs"
STARTED_EVENT = "web.run_started"
FINISHED_EVENT = "web.run_finished"
STOP_EVENT = "web.stop_requested"
LOG_TAIL_BYTES = 8_192
LOG_TAIL_LINES = 12
_START_RE = re.compile(r"^/projects/([^/]+)/start$")
_STOP_RE = re.compile(r"^/projects/([^/]+)/stop$")
_BRAKE_RE = re.compile(r"^/projects/([^/]+)/brake$")
_RETRY_RE = re.compile(r"^/projects/([^/]+)/retry$")

# Meaning of each ``automa run`` exit code (see the ``automa.cli`` docstring).
EXIT_MEANINGS: Mapping[int, str] = {
    0: "finished: every task passed",
    1: "stopped: user error (see the log)",
    2: "stopped: configuration error (see the log)",
    3: "budget stop: the project is paused until you raise the cap or wait",
    4: "planning failed",
    5: "provider error (network or gateway); run again to resume",
    6: "finished, but some task failed for good",
    7: "paused: an approval is needed or another cause (see the log)",
    11: "not started: the cost range grew above confirm_above_usd; reload and confirm",
    12: "stopped on request: the project is paused; start again to resume",
    130: "interrupted: the project is paused; start again to resume",
}


def exit_meaning(code: int) -> str:
    if code < 0:
        return f"killed by signal {-code}"
    return EXIT_MEANINGS.get(code, "stopped (see the log)")


class RunPreflight(Protocol):
    """What the start button needs from the CLI (no model call, nothing written)."""

    def needs_confirm(self, project_id: str) -> bool:
        """Whether the upper end of the cost range is above ``confirm_above_usd``."""
        ...

    def command(self, project_id: str, *, yes: bool) -> list[str]:
        """The ``automa run`` command line for the child process."""
        ...


@runtime_checkable
class RetryPreflight(Protocol):
    """Optional part of the preflight for the "try again" button (D156)."""

    def grant_retry(self, project_id: str) -> list[str]:
        """Grant the failed tasks more attempts (as ``automa retry``); ids retried."""
        ...


@runtime_checkable
class SandboxPreflight(Protocol):
    """Optional part of the preflight: Docker needed but not usable (D185)."""

    def sandbox_refusal(self, project_id: str) -> str | None:
        """A refusal code (``member_needs_docker``/``docker_unavailable``) or ``None``."""
        ...


class RunProcess(Protocol):
    """The part of :class:`subprocess.Popen` used here."""

    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...

    def wait(self) -> int: ...


Spawn = Callable[[Sequence[str], Path], RunProcess]


def popen_spawn(argv: Sequence[str], log_path: Path) -> RunProcess:
    """Start the ``automa run`` child (see :func:`automa.tools.command.spawn_automa_run`)."""
    return spawn_automa_run(argv, log_path)


def default_command(project_id: str, *, yes: bool, config_path: Path | None) -> list[str]:
    """``<python> -m automa run <pid> [--yes] [--config <path>]`` (no shell)."""
    argv = [sys.executable, "-m", "automa", "run", validate_project_id(project_id)]
    if yes:
        argv.append("--yes")
    if config_path is not None:
        argv += ["--config", str(config_path.resolve())]
    return argv


def start_path(project_id: str) -> str:
    from urllib.parse import quote

    return "/projects/" + quote(project_id, safe="") + "/start"


def stop_path(project_id: str) -> str:
    from urllib.parse import quote

    return "/projects/" + quote(project_id, safe="") + "/stop"


def brake_path(project_id: str) -> str:
    from urllib.parse import quote

    return "/projects/" + quote(project_id, safe="") + "/brake"


def parse_brake_path(path: str) -> str | None:
    """The project id of a ``/projects/<pid>/brake`` path, else ``None`` (D154)."""
    return _parse(_BRAKE_RE, path)


def retry_path(project_id: str) -> str:
    return f"/projects/{validate_project_id(project_id)}/retry"


def parse_retry_path(path: str) -> str | None:
    return _parse(_RETRY_RE, path)


def parse_start_path(path: str) -> str | None:
    """The project id of a ``/projects/<pid>/start`` path, else ``None``."""
    return _parse(_START_RE, path)


def parse_stop_path(path: str) -> str | None:
    """The project id of a ``/projects/<pid>/stop`` path, else ``None``."""
    return _parse(_STOP_RE, path)


def _parse(pattern: re.Pattern[str], path: str) -> str | None:
    m = pattern.fullmatch(path)
    if m is None:
        return None
    try:
        return validate_project_id(m.group(1))
    except ProjectStoreError:
        return None


@dataclass(frozen=True)
class RunState:
    """A background run as shown on the project page."""

    started_at: datetime
    log_path: Path
    confirmed: bool
    exit_code: int | None
    tail: tuple[str, ...] = ()
    stop_requested: bool = False

    @property
    def running(self) -> bool:
        return self.exit_code is None

    @property
    def summary(self) -> str:
        if self.exit_code is None:
            if self.stop_requested:
                return "stopping: the run pauses after its current step"
            return "running in the background"
        return f"ended with exit code {self.exit_code}: {exit_meaning(self.exit_code)}"


@dataclass
class _Run:
    process: RunProcess
    started_at: datetime
    log_path: Path
    confirmed: bool
    exit_code: int | None = None
    stop_requested: bool = False


def _now() -> datetime:
    return datetime.now(UTC)


def read_tail(
    path: Path, *, max_bytes: int = LOG_TAIL_BYTES, lines: int = LOG_TAIL_LINES
) -> tuple[str, ...]:
    """The last ``lines`` lines of a log (at most ``max_bytes`` read; never a symlink)."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return ()
    try:
        size = os.fstat(fd).st_size
        os.lseek(fd, max(0, size - max_bytes), os.SEEK_SET)
        data = os.read(fd, max_bytes)
    except OSError:
        return ()
    finally:
        os.close(fd)
    decoded = data.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes and decoded:
        decoded = decoded[1:]  # the first line may be cut
    return tuple(decoded[-lines:])


class BackgroundRuns:
    """Background ``automa run`` children of one web server, one per project."""

    def __init__(
        self,
        store: ProjectStore,
        preflight: RunPreflight,
        *,
        spawn: Spawn = popen_spawn,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._store = store
        self._preflight = preflight
        self._spawn = spawn
        self._clock = clock
        self._lock = threading.Lock()
        self._runs: dict[str, _Run] = {}

    def _active(self, project_id: str) -> _Run | None:
        run = self._runs.get(project_id)
        if run is not None and run.exit_code is None:
            code = run.process.poll()
            if code is not None:
                run.exit_code = code
        return run if run is not None and run.exit_code is None else None

    def needs_confirm(self, project_id: str) -> bool:
        return self._preflight.needs_confirm(project_id)

    def state(self, project_id: str) -> RunState | None:
        """The last background run of ``project_id`` in this server, if any."""
        with self._lock:
            self._active(project_id)
            run = self._runs.get(project_id)
            if run is None:
                return None
            snapshot = (run.started_at, run.log_path, run.confirmed, run.exit_code)
            stopping = run.stop_requested and run.exit_code is None
        return RunState(*snapshot, tail=read_tail(snapshot[1]), stop_requested=stopping)

    def can_stop(self, project_id: str) -> bool:
        """Whether the stop button is shown (this server's child is still running)."""
        with self._lock:
            run = self._active(project_id)
            return run is not None and not run.stop_requested

    def stop(self, project_id: str, form: Mapping[str, str]) -> None:
        """Ask the running child to pause after its current step (D126).

        Raises :class:`ActionRefused` when no child of this server is running for
        the project. Asking twice is harmless (the second time changes nothing).
        """
        if project_id not in self._store.list_ids():
            raise ActionRefused.coded(404, "no_such_project")
        with self._lock:
            run = self._active(project_id)
            if run is None:
                raise ActionRefused.coded(409, "not_running_here", id=project_id)
            if run.stop_requested:
                return
            try:
                self._store.request_stop(project_id, via="web")
            except (OSError, ProjectStoreError) as e:
                raise ActionRefused.coded(500, "stop_failed", reason=str(e)) from None
            run.stop_requested = True
        self._emit(project_id, STOP_EVENT, {"log": run.log_path.name})

    def can_start(self, project_id: str) -> bool:
        """Whether the start button is shown (planned, not running, not done)."""
        with self._lock:
            if self._active(project_id) is not None:
                return False
        try:
            if not self._store.has_plan(project_id):
                return False
            status = self._store.load(project_id).status
        except ProjectStoreError:
            return False
        return status not in (ProjectStatus.RUNNING, ProjectStatus.DONE)

    def start(self, project_id: str, form: Mapping[str, str]) -> None:
        """Start ``automa run`` in the background, or raise :class:`ActionRefused`."""
        self._start(project_id, form)

    def brake(self, project_id: str, form: Mapping[str, str]) -> None:
        """Answer the task brake's question "continue?" (D154).

        ``decision=stop`` records ``brake.declined`` (allows nothing). ``continue``
        runs every check of :meth:`start` first (plan, status, one run per project,
        the cost confirmation box); only then is ``brake.continued`` recorded, exactly
        as ``automa brake continue`` does, and the very same ``automa run`` child is
        started, so the BudgetGuard and the brake itself apply unchanged. Either is
        refused unless the task's latest brake stop is still waiting for a decision.
        """
        decision = form.get("decision")
        task = form.get("task", "")
        if decision not in (BRAKE_CONTINUE, BRAKE_STOP) or not task or len(task) > MAX_TASK_ID:
            raise ActionRefused.coded(400, "malformed_form")
        if decision == BRAKE_STOP:
            if form.get("confirm") is not None:
                raise ActionRefused.coded(400, "malformed_form")
            if project_id not in self._store.list_ids():
                raise ActionRefused.coded(404, "no_such_project")
            with self._lock:
                if self._active(project_id) is not None:
                    raise ActionRefused.coded(409, "already_running", id=project_id)
                self._decide_brake(project_id, task, continuing=False)
            return
        self._start(
            project_id,
            {k: v for k, v in form.items() if k in START_FIELDS},
            before_spawn=lambda: self._decide_brake(project_id, task, continuing=True),
        )

    def _decide_brake(self, project_id: str, task: str, *, continuing: bool) -> None:
        """Record the brake decision (called under the lock)."""
        try:
            sink = self._store.events(project_id)
            recorded = sink.read()
        except (OSError, ValueError, ProjectStoreError) as e:
            raise ActionRefused.coded(500, "brake_failed", reason=str(e)) from None
        record = continue_task if continuing else decline_task
        try:
            record(recorded, sink, project_id, task, via="web", now=self._clock())
        except BrakeError:
            raise ActionRefused.coded(409, "not_braked", task=task, id=project_id) from None
        except (OSError, ValueError) as e:
            raise ActionRefused.coded(500, "brake_failed", reason=str(e)) from None

    def _start(
        self,
        project_id: str,
        form: Mapping[str, str],
        *,
        before_spawn: Callable[[], None] | None = None,
    ) -> None:
        confirm = form.get("confirm")
        if confirm is not None and confirm != CONFIRM_VALUE:
            raise ActionRefused.coded(400, "malformed_form")
        if project_id not in self._store.list_ids():
            raise ActionRefused.coded(404, "no_such_project")
        with self._lock:
            if self._active(project_id) is not None:
                raise ActionRefused.coded(409, "already_running", id=project_id)
            try:
                if not self._store.has_plan(project_id):
                    raise ActionRefused.coded(409, "no_plan", id=project_id)
                status = self._store.load(project_id).status
            except ProjectStoreError as e:
                raise ActionRefused(500, str(e), lead="lead.store") from None
            if status is ProjectStatus.RUNNING:
                raise ActionRefused.coded(409, "marked_running", id=project_id)
            if status is ProjectStatus.DONE:
                raise ActionRefused.coded(409, "already_done", id=project_id)
            self._check_sandbox(project_id)
            needs = self._preflight.needs_confirm(project_id)
            if needs and confirm != CONFIRM_VALUE:
                raise ActionRefused.coded(409, "confirm_cost")
            yes = needs and confirm == CONFIRM_VALUE
            if before_spawn is not None:
                before_spawn()
            run = self._spawn_locked(project_id, yes)
        self._started(project_id, run)

    def can_retry(self, project_id: str) -> bool:
        """Whether the "try again" button is shown (failed project, nothing running)."""
        if not isinstance(self._preflight, RetryPreflight):
            return False
        with self._lock:
            if self._active(project_id) is not None:
                return False
        return can_retry(self._store, project_id)

    def retry(self, project_id: str, form: Mapping[str, str]) -> None:
        """Give a failed project's failed tasks more attempts and run it (D156).

        Retrying spends more money, so the confirmation box is always required and
        the child gets ``--yes`` (the user confirmed this spending on the page). The
        grant itself is the same as ``automa retry``; the child is the same
        ``automa run``, so the BudgetGuard caps apply unchanged.
        """
        if form.get("confirm") != CONFIRM_VALUE:
            raise ActionRefused.coded(409, "confirm_cost")
        if not isinstance(self._preflight, RetryPreflight):
            raise ActionRefused.coded(404, "no_such_project")
        if project_id not in self._store.list_ids():
            raise ActionRefused.coded(404, "no_such_project")
        with self._lock:
            if self._active(project_id) is not None:
                raise ActionRefused.coded(409, "already_running", id=project_id)
            if not can_retry(self._store, project_id):
                raise ActionRefused.coded(409, "not_retryable", id=project_id)
            self._check_sandbox(project_id)
            try:
                self._preflight.grant_retry(project_id)
            except (RetryError, OSError, ProjectStoreError, ValueError) as e:
                raise ActionRefused.coded(500, "retry_failed", reason=str(e)) from None
            run = self._spawn_locked(project_id, True)
        self._started(project_id, run)

    def _check_sandbox(self, project_id: str) -> None:
        """Refuse before spawning when the run needs Docker and Docker is not usable.

        Nothing is spawned or granted then. A member's project always needs Docker
        (D185); the child ``automa run`` checks the same again (defence in depth).
        An unknown code from the preflight still refuses (fail closed)."""
        if not isinstance(self._preflight, SandboxPreflight):
            return
        code = self._preflight.sandbox_refusal(project_id)
        if code is None:
            return
        if code == "member_needs_docker":
            raise ActionRefused.coded(409, "member_needs_docker")
        raise ActionRefused.coded(409, "docker_unavailable")

    def _spawn_locked(self, project_id: str, yes: bool) -> _Run:
        """Start the ``automa run`` child (called under the lock)."""
        argv = self._preflight.command(project_id, yes=yes)
        started = self._clock()
        log_path = self._new_log(project_id, started)
        try:
            process = self._spawn(argv, log_path)
        except (OSError, ValueError) as e:
            reason = e.strerror if isinstance(e, OSError) and e.strerror else str(e)
            raise ActionRefused.coded(500, "start_failed", reason=reason) from None
        run = _Run(process, started, log_path, yes)
        self._runs[project_id] = run
        return run

    def _started(self, project_id: str, run: _Run) -> None:
        self._emit(
            project_id, STARTED_EVENT, {"confirmed": run.confirmed, "log": run.log_path.name}
        )
        watcher = threading.Thread(
            target=self._watch, args=(project_id, run), name=f"automa-run-{project_id}", daemon=True
        )
        watcher.start()

    def _new_log(self, project_id: str, started: datetime) -> Path:
        try:
            base = self._store.project_dir(project_id)
            runs = base / RUNS_DIR
            if runs.is_symlink():
                raise ActionRefused(
                    500, f"{RUNS_DIR} of project {project_id} is a symlink", lead="lead.run"
                )
            runs.mkdir(mode=0o700, exist_ok=True)
        except (OSError, ProjectStoreError) as e:
            raise ActionRefused(500, f"cannot prepare the run log: {e}", lead="lead.run") from None
        stamp = started.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return runs / f"run-{stamp}.log"

    def _watch(self, project_id: str, run: _Run) -> None:
        code = run.process.wait()
        with self._lock:
            run.exit_code = code
            if run.stop_requested:
                # A request the child never saw (it ended first) must not stop a later run.
                try:
                    self._store.clear_stop_request(project_id)
                except (OSError, ProjectStoreError):
                    pass
        self._emit(project_id, FINISHED_EVENT, {"exit_code": code, "log": run.log_path.name})

    def _emit(self, project_id: str, kind: str, data: dict[str, object]) -> None:
        try:
            self._store.events(project_id).emit(
                Event(ts=self._clock(), type=kind, project_id=project_id, data=data)
            )
        except (OSError, ProjectStoreError, ValueError):
            pass  # the event log is a record only; the run itself is unaffected
