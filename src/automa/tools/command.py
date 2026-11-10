"""``run_command``: run one program inside the project workspace.

Risk level ``execute``. Safety properties (see docs/DECISIONS.md D18):

- no shell: ``argv`` is passed to ``exec`` as a list, so ``;``, ``|``, ``$()``
  and globs are plain characters;
- ``argv[0]`` is a bare program name (looked up on ``PATH``) or a safe relative
  path inside the workspace; absolute paths and ``..`` are refused; an optional
  per-instance allow-list restricts which programs may run;
- the working directory is a real directory inside ``files/`` (no symlinks);
- the environment is rebuilt from a small allow-list, so API keys and other
  secrets in the parent's environment never reach the child;
- stdin is closed, the child runs in its own session and the whole process
  group is killed on timeout;
- stdout/stderr are read with a hard byte cap; the rest is discarded.

This is *not* a sandbox: the program runs with the user's privileges and can
reach the filesystem and network. Container isolation is roadmap item M5.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Iterable
from typing import IO, ClassVar

from pydantic import Field, field_validator

from automa.plan.models import MAX_PATH_CHARS, _safe_relpath
from automa.tools.base import MAX_TOOL_OUTPUT_CHARS, RiskLevel, Tool, ToolArgs, ToolError
from automa.tools.workspace import Workspace

MAX_ARGV_ITEMS = 64
MAX_ARG_CHARS = 4_096
DEFAULT_TIMEOUT_S = 60.0
MAX_TIMEOUT_S = 600.0
MAX_STREAM_BYTES = 64 * 1024
DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"
ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "TZ")
_DRAIN_JOIN_S = 2.0


class RunCommandArgs(ToolArgs):
    argv: list[str] = Field(
        min_length=1,
        max_length=MAX_ARGV_ITEMS,
        description="Program and its arguments, e.g. ['python', '-m', 'pytest']. No shell.",
    )
    cwd: str = Field(
        default="",
        max_length=MAX_PATH_CHARS,
        description="Working directory relative to the workspace; empty = root",
    )
    timeout_s: float = Field(
        default=DEFAULT_TIMEOUT_S, gt=0, le=MAX_TIMEOUT_S, description="Seconds before kill"
    )

    @field_validator("argv")
    @classmethod
    def _check_argv(cls, v: list[str]) -> list[str]:
        for i, a in enumerate(v):
            if "\x00" in a:
                raise ValueError(f"argv[{i}] contains NUL")
            if len(a) > MAX_ARG_CHARS:
                raise ValueError(f"argv[{i}] longer than {MAX_ARG_CHARS} characters")
        program = v[0]
        if not program.strip() or program != program.strip():
            raise ValueError("argv[0] must be a program name")
        if "/" in program or "\\" in program:
            try:
                norm = _safe_relpath(program)
            except ValueError as e:
                raise ValueError(f"argv[0] must be a bare name or a relative path: {e}") from None
            v = [f"./{norm}", *v[1:]]
        return v


def clean_env(workspace_root: str, parent: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the child: an allow-list only, never secrets from the parent."""
    src = os.environ if parent is None else parent
    env = {k: src[k] for k in ENV_PASSTHROUGH if k in src and "\x00" not in src[k]}
    env.setdefault("PATH", DEFAULT_PATH)
    env.setdefault("LANG", "C.UTF-8")
    env["HOME"] = workspace_root
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class _CappedReader(threading.Thread):
    """Read a pipe to EOF, keeping at most ``cap`` bytes."""

    def __init__(self, stream: IO[bytes], cap: int) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._cap = cap
        self.data = bytearray()
        self.dropped = 0

    def run(self) -> None:
        try:
            while chunk := self._stream.read(8192):
                room = self._cap - len(self.data)
                if room > 0:
                    self.data += chunk[:room]
                self.dropped += max(0, len(chunk) - max(room, 0))
        except (OSError, ValueError):
            pass

    def text(self) -> str:
        out = self.data.decode("utf-8", errors="replace")
        if self.dropped:
            out += f"\n[... {self.dropped} more bytes discarded]"
        return out


def _wait_unreaped(pid: int, timeout_s: float) -> bool:
    """Wait until ``pid`` exits without reaping it; False on timeout."""
    deadline = time.monotonic() + timeout_s
    delay = 0.005
    while True:
        try:
            info = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
        except ChildProcessError:
            return True
        if info is not None:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, 0.1)


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass


class RunCommand(Tool):
    name: ClassVar[str] = "run_command"
    description: ClassVar[str] = (
        "Run a program (no shell) in the project workspace and return its exit code, "
        "stdout and stderr. Pipes, redirects and globs are not interpreted."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.EXECUTE
    args_model: ClassVar[type[ToolArgs]] = RunCommandArgs

    def __init__(self, allowed_programs: Iterable[str] | None = None) -> None:
        if not hasattr(os, "waitid") or not hasattr(os, "killpg"):  # pragma: no cover
            raise ToolError("run_command needs a POSIX platform (process groups)")
        if isinstance(allowed_programs, str):
            raise TypeError("allowed_programs must be a collection of names, not a string")
        self._allowed = None if allowed_programs is None else frozenset(allowed_programs)

    @property
    def allowed_programs(self) -> frozenset[str] | None:
        return self._allowed

    def audit(self, args: ToolArgs) -> dict[str, str]:
        if not isinstance(args, RunCommandArgs):
            return {}
        # Only the program name: arguments may contain data the model chose to pass.
        return {"program": args.argv[0], "cwd": args.cwd}

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        if not isinstance(args, RunCommandArgs):
            raise TypeError("arguments were not validated by the executor")
        program = args.argv[0]
        if self._allowed is not None and program not in self._allowed:
            allowed = ", ".join(sorted(self._allowed)) or "(none)"
            raise ToolError(f"program {program!r} is not allowed; allowed: {allowed}")
        cwd = workspace.dir_path(args.cwd)
        env = clean_env(str(workspace.root))
        try:
            proc = subprocess.Popen(  # noqa: S603 - argv list, no shell
                args.argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                close_fds=True,
                start_new_session=True,
            )
        except FileNotFoundError:
            raise ToolError(f"program not found: {program!r}") from None
        except PermissionError:
            raise ToolError(f"program is not executable: {program!r}") from None
        except OSError as e:
            raise ToolError(f"cannot start {program!r}: {e.strerror or e}") from None

        if proc.stdout is None or proc.stderr is None:  # pragma: no cover - PIPE requested
            _kill_group(proc)
            raise ToolError("could not capture output")
        readers = [
            _CappedReader(proc.stdout, MAX_STREAM_BYTES),
            _CappedReader(proc.stderr, MAX_STREAM_BYTES),
        ]
        for r in readers:
            r.start()
        try:
            timed_out = not _wait_unreaped(proc.pid, args.timeout_s)
        finally:
            # The child is not reaped yet, so its pid (= process group id) cannot be
            # reused: killing the group here is safe and also stops background
            # children a finished command left behind.
            _kill_group(proc)
            proc.wait()
            for r in readers:
                r.join(_DRAIN_JOIN_S)
            proc.stdout.close()
            proc.stderr.close()

        out, err = readers[0].text(), readers[1].text()
        if timed_out:
            tail = (err or out)[-500:]
            raise ToolError(
                f"timed out after {args.timeout_s:g}s; process killed"
                + (f"; last output: {tail}" if tail else "")
            )
        return _format(proc.returncode, out, err)


def _format(code: int, out: str, err: str) -> str:
    head = f"exit_code: {code}\n"
    budget = MAX_TOOL_OUTPUT_CHARS - len(head) - 64
    half = budget // 2
    if len(out) + len(err) > budget:
        # Keep the end of each stream: errors and summaries are usually last.
        if len(err) <= half:
            out = "[... truncated]\n" + out[-(budget - len(err)) :]
        elif len(out) <= half:
            err = "[... truncated]\n" + err[-(budget - len(out)) :]
        else:
            out = "[... truncated]\n" + out[-half:]
            err = "[... truncated]\n" + err[-half:]
    return f"{head}--- stdout ---\n{out}\n--- stderr ---\n{err}"
