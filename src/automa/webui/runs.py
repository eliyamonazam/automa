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
from typing import Protocol

from automa.events import Event
from automa.project.store import (
    ProjectStatus,
    ProjectStore,
    ProjectStoreError,
    validate_project_id,
)
from automa.tools.command import spawn_automa_run
from automa.webui.actions import CONFIRM_VALUE, ActionRefused

START_FIELDS = frozenset({"csrf", "confirm"})
RUNS_DIR = "web-runs"
STARTED_EVENT = "web.run_started"
FINISHED_EVENT = "web.run_finished"
LOG_TAIL_BYTES = 8_192
LOG_TAIL_LINES = 12
_START_RE = re.compile(r"^/projects/([^/]+)/start$")

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


def parse_start_path(path: str) -> str | None:
    """The project id of a ``/projects/<pid>/start`` path, else ``None``."""
    m = _START_RE.fullmatch(path)
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

    @property
    def running(self) -> bool:
        return self.exit_code is None

    @property
    def summary(self) -> str:
        if self.exit_code is None:
            return "running in the background"
        return f"ended with exit code {self.exit_code}: {exit_meaning(self.exit_code)}"


@dataclass
class _Run:
    process: RunProcess
    started_at: datetime
    log_path: Path
    confirmed: bool
    exit_code: int | None = None


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
        return RunState(*snapshot, tail=read_tail(snapshot[1]))

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
        confirm = form.get("confirm")
        if confirm is not None and confirm != CONFIRM_VALUE:
            raise ActionRefused(400, "malformed form")
        if project_id not in self._store.list_ids():
            raise ActionRefused(404, "no such project")
        with self._lock:
            if self._active(project_id) is not None:
                raise ActionRefused(409, f"project {project_id} is already running here")
            try:
                if not self._store.has_plan(project_id):
                    raise ActionRefused(409, f"project {project_id} has no plan yet")
                status = self._store.load(project_id).status
            except ProjectStoreError as e:
                raise ActionRefused(500, str(e)) from None
            if status is ProjectStatus.RUNNING:
                raise ActionRefused(
                    409,
                    f"project {project_id} is marked running (maybe in a terminal); "
                    "wait for it, or use automa run --force there after a crash",
                )
            if status is ProjectStatus.DONE:
                raise ActionRefused(409, f"project {project_id} is already done")
            needs = self._preflight.needs_confirm(project_id)
            if needs and confirm != CONFIRM_VALUE:
                raise ActionRefused(
                    409,
                    "the cost range is above confirm_above_usd: tick the box to confirm "
                    "it, then start (nothing was run)",
                )
            yes = needs and confirm == CONFIRM_VALUE
            argv = self._preflight.command(project_id, yes=yes)
            started = self._clock()
            log_path = self._new_log(project_id, started)
            try:
                process = self._spawn(argv, log_path)
            except (OSError, ValueError) as e:
                reason = e.strerror if isinstance(e, OSError) and e.strerror else str(e)
                raise ActionRefused(500, f"could not start automa run: {reason}") from None
            run = _Run(process, started, log_path, yes)
            self._runs[project_id] = run
        self._emit(project_id, STARTED_EVENT, {"confirmed": yes, "log": log_path.name})
        watcher = threading.Thread(
            target=self._watch, args=(project_id, run), name=f"automa-run-{project_id}", daemon=True
        )
        watcher.start()

    def _new_log(self, project_id: str, started: datetime) -> Path:
        try:
            base = self._store.project_dir(project_id)
            runs = base / RUNS_DIR
            if runs.is_symlink():
                raise ActionRefused(500, f"{RUNS_DIR} of project {project_id} is a symlink")
            runs.mkdir(mode=0o700, exist_ok=True)
        except (OSError, ProjectStoreError) as e:
            raise ActionRefused(500, f"cannot prepare the run log: {e}") from None
        stamp = started.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return runs / f"run-{stamp}.log"

    def _watch(self, project_id: str, run: _Run) -> None:
        code = run.process.wait()
        with self._lock:
            run.exit_code = code
        self._emit(project_id, FINISHED_EVENT, {"exit_code": code, "log": run.log_path.name})

    def _emit(self, project_id: str, kind: str, data: dict[str, object]) -> None:
        try:
            self._store.events(project_id).emit(
                Event(ts=self._clock(), type=kind, project_id=project_id, data=data)
            )
        except (OSError, ProjectStoreError, ValueError):
            pass  # the event log is a record only; the run itself is unaffected
