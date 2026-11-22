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

Without a sandbox the program runs with the user's privileges and can reach the
filesystem and network. With ``RunCommand(sandbox=DockerSandbox(...))`` every call
runs in a fresh, isolated container instead (``tools/sandbox.py``, D32); if Docker is
unavailable the call fails and nothing runs on the host (no silent fallback).
"""

from __future__ import annotations

import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import IO, ClassVar, NamedTuple

from pydantic import Field, field_validator

from automa.plan.models import MAX_PATH_CHARS, _safe_relpath
from automa.tools.base import MAX_TOOL_OUTPUT_CHARS, RiskLevel, Tool, ToolArgs, ToolError
from automa.tools.sandbox import (
    DOCKER_RUN_ERROR,
    DockerSandbox,
    SandboxError,
    client_env,
    container_env,
    container_name,
    kill_argv,
    run_argv,
    version_argv,
)
from automa.tools.workspace import Workspace

MAX_ARGV_ITEMS = 64
MAX_ARG_CHARS = 4_096
DEFAULT_TIMEOUT_S = 60.0
MAX_TIMEOUT_S = 600.0
MAX_STREAM_BYTES = 64 * 1024
DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"
ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "TZ")
_DRAIN_JOIN_S = 2.0
_DOCKER_RM_TIMEOUT_S = 30.0
DOCKER_CHECK_TIMEOUT_S = 15.0
_CHECK_OUTPUT_CHARS = 300


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

    def __init__(
        self,
        allowed_programs: Iterable[str] | None = None,
        *,
        sandbox: DockerSandbox | None = None,
    ) -> None:
        if not hasattr(os, "waitid") or not hasattr(os, "killpg"):  # pragma: no cover
            raise ToolError("run_command needs a POSIX platform (process groups)")
        if isinstance(allowed_programs, str):
            raise TypeError("allowed_programs must be a collection of names, not a string")
        if sandbox is not None and not isinstance(sandbox, DockerSandbox):
            raise TypeError("sandbox must be a DockerSandbox or None")
        self._allowed = None if allowed_programs is None else frozenset(allowed_programs)
        self._sandbox = sandbox

    @property
    def sandbox(self) -> DockerSandbox | None:
        return self._sandbox

    @property
    def allowed_programs(self) -> frozenset[str] | None:
        return self._allowed

    def audit(self, args: ToolArgs) -> dict[str, str]:
        if not isinstance(args, RunCommandArgs):
            return {}
        # Only the program name: arguments may contain data the model chose to pass.
        out = {"program": args.argv[0], "cwd": args.cwd}
        if self._sandbox is not None:
            out["sandbox"] = "docker"
            out["image"] = self._sandbox.image
        return out

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        if not isinstance(args, RunCommandArgs):
            raise TypeError("arguments were not validated by the executor")
        program = args.argv[0]
        if self._allowed is not None and program not in self._allowed:
            allowed = ", ".join(sorted(self._allowed)) or "(none)"
            raise ToolError(f"program {program!r} is not allowed; allowed: {allowed}")
        cwd = workspace.dir_path(args.cwd)
        if self._sandbox is None:
            env = clean_env(str(workspace.root))
            code, out, err = _execute(args.argv, cwd, env, args.timeout_s, program)
            return _format(code, out, err)
        return self._run_in_docker(self._sandbox, args, workspace, cwd)

    def _run_in_docker(
        self, sandbox: DockerSandbox, args: RunCommandArgs, workspace: Workspace, cwd: Path
    ) -> str:
        rel = cwd.relative_to(workspace.root).as_posix()
        name = container_name(secrets.token_hex(8))
        try:
            popen_argv = run_argv(
                sandbox,
                args.argv,
                workspace_root=workspace.root,
                cwd="" if rel == "." else rel,
                name=name,
                environment=container_env(),
            )
        except SandboxError as e:
            raise ToolError(f"sandbox: {e}; command not run") from None
        env = client_env()
        code, out, err = _execute(
            popen_argv,
            workspace.root,
            env,
            args.timeout_s,
            args.argv[0],
            on_kill=lambda: _remove_container(sandbox, name),
            missing=f"docker client not found: {sandbox.docker!r}; command not run",
        )
        if code == DOCKER_RUN_ERROR:
            tail = err.strip()[-500:]
            raise ToolError(
                f"sandbox: docker could not start the container (image {sandbox.image!r})"
                + (f": {tail}" if tail else "")
            )
        return _format(code, out, err)


def check_docker(sandbox: DockerSandbox, *, timeout_s: float = DOCKER_CHECK_TIMEOUT_S) -> str:
    """Preflight: confirm the docker client exists and its daemon answers.

    Returns the daemon's version. Raises ``SandboxError`` otherwise, so a caller that
    asked for the Docker sandbox can stop before any work starts (fail closed). It
    starts no container and touches no workspace.
    """
    env = client_env()
    try:
        done = subprocess.run(  # noqa: S603 - argv list, no shell
            version_argv(sandbox),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            shell=False,
            close_fds=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        raise SandboxError(f"docker client not found: {sandbox.docker!r}") from None
    except subprocess.TimeoutExpired:
        raise SandboxError(f"docker did not answer within {timeout_s:g}s") from None
    except (OSError, subprocess.SubprocessError) as e:
        raise SandboxError(f"cannot run the docker client: {e.__class__.__name__}") from None
    out = done.stdout.decode("utf-8", "replace").strip()
    if done.returncode != 0 or not out:
        tail = done.stderr.decode("utf-8", "replace").strip()[-_CHECK_OUTPUT_CHARS:]
        raise SandboxError("docker daemon is not available" + (f": {tail}" if tail else ""))
    return out[:_CHECK_OUTPUT_CHARS]


# -- read-only git for repository snapshots (D82) -----------------------------------
# Not a tool: no model ever reaches it. It lives here so that this module stays the
# only place in Automa that starts processes.

GIT_READ_SUBCOMMANDS = frozenset({"rev-parse", "ls-tree", "cat-file"})
GIT_READ_TIMEOUT_S = 60.0


class GitReadError(RuntimeError):
    """A read-only git command could not run or failed."""


def git_env(parent: dict[str, str] | None = None) -> dict[str, str]:
    """A minimal environment for git: no outer ``GIT_*``, no system/global config."""
    source = os.environ if parent is None else parent
    return {
        "PATH": source.get("PATH") or DEFAULT_PATH,
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }


def run_git_read(
    repo: Path,
    args: Sequence[str],
    *,
    stdin: bytes | None = None,
    timeout_s: float = GIT_READ_TIMEOUT_S,
) -> bytes:
    """Run one read-only git subcommand (``GIT_READ_SUBCOMMANDS``) in ``repo``.

    No shell, minimal environment (``git_env``), fsmonitor off; returns stdout.
    Raises ``GitReadError`` for another subcommand, a missing git, a timeout or a
    non-zero exit (stderr is not repeated).
    """
    if not args or args[0] not in GIT_READ_SUBCOMMANDS:
        raise GitReadError("git subcommand not allowed")
    env = git_env()
    argv = ["git", "--no-pager", "-c", "core.fsmonitor=false", "-C", str(repo), *args]
    try:
        done = subprocess.run(  # noqa: S603 - argv list, no shell
            argv,
            env=env,
            input=stdin or b"",
            capture_output=True,
            shell=False,
            close_fds=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        raise GitReadError("git is not installed") from None
    except subprocess.TimeoutExpired:
        raise GitReadError(f"git {args[0]} timed out") from None
    except (OSError, subprocess.SubprocessError) as e:
        raise GitReadError(f"cannot run git: {e.__class__.__name__}") from None
    if done.returncode != 0:
        raise GitReadError(f"git {args[0]} failed (exit {done.returncode})")
    return done.stdout


#: git subcommands ``automa update`` may run (D170): reads, ``fetch`` and
#: ``merge --ff-only``. Never reset, push, stash, rebase or clean.
UPDATE_GIT_SUBCOMMANDS = frozenset(
    {"rev-parse", "symbolic-ref", "status", "fetch", "log", "merge-base", "rev-list", "diff"}
)


class CommandResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


def run_update_command(argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
    """Run one command of ``automa update`` (D170) in the user's own clone.

    Allowed: ``git <UPDATE_GIT_SUBCOMMANDS ...>``, ``git merge --ff-only --quiet
    origin/main`` and ``<this python> -m pip install --quiet -e <cwd>``; anything
    else is ``ValueError``. No shell. The user's own git config is kept (credential
    helpers, proxy) but prompts are off so nothing ever waits for input. A missing
    program or another start failure is exit code 127, a timeout 124."""
    args = list(argv)
    if args[:1] == ["git"] and len(args) >= 2:
        sub = args[1]
        allowed = sub in UPDATE_GIT_SUBCOMMANDS or args[1:] == [
            "merge",
            "--ff-only",
            "--quiet",
            "origin/main",
        ]
        if sub in {"fetch", "merge"} and any(a.startswith("--force") or a == "-f" for a in args):
            allowed = False
    else:
        allowed = args == [sys.executable, "-m", "pip", "install", "--quiet", "-e", str(cwd)]
    if not allowed:
        raise ValueError("command not allowed for automa update")
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["LC_ALL"] = "C"
    try:
        done = subprocess.run(  # noqa: S603 - allow-listed argv list, no shell
            args,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            close_fds=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(124, "", f"{args[0]} timed out")
    except (OSError, subprocess.SubprocessError) as e:
        return CommandResult(127, "", f"cannot run {args[0]}: {e.__class__.__name__}")
    return CommandResult(done.returncode, done.stdout or "", done.stderr or "")


AUTOMA_RUN_PREFIX = ("-m", "automa", "run")


def spawn_automa_run(argv: Sequence[str], log_path: Path) -> subprocess.Popen[bytes]:
    """Start ``<this python> -m automa run <pid> ...`` in the background (web UI, D125).

    Only this exact command can be started (anything else is ``ValueError``): the
    child is the same ``automa run`` as in the terminal, with its own BudgetGuard,
    caps, Policy and Approval Gate. No shell; no terminal (``stdin`` is
    ``/dev/null``); stdout and stderr go to ``log_path``, which must not exist yet
    (``O_EXCL``) and is never opened through a symlink (mode ``0600``). The child
    keeps the parent's environment (it needs the API key, as in the terminal) and
    stays in the server's process group, so Ctrl+C on ``automa web`` also
    interrupts it and the project is paused cleanly."""
    args = list(argv)
    if len(args) < 5 or args[0] != sys.executable or tuple(args[1:4]) != AUTOMA_RUN_PREFIX:
        raise ValueError("only '<python> -m automa run <project>' can be started")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(log_path, flags, 0o600)
    try:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        return subprocess.Popen(  # noqa: S603 - fixed argv list, no shell
            args,
            stdin=subprocess.DEVNULL,
            stdout=fd,
            stderr=subprocess.STDOUT,
            env=env,
            close_fds=True,
        )
    finally:
        os.close(fd)


def _remove_container(sandbox: DockerSandbox, name: str) -> None:
    """Best effort ``docker rm -f``: the container must not outlive a killed client."""
    env = client_env()
    try:
        subprocess.run(  # noqa: S603 - argv list, no shell
            kill_argv(sandbox, name),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            close_fds=True,
            timeout=_DOCKER_RM_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _execute(
    popen_argv: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_s: float,
    program: str,
    *,
    on_kill: Callable[[], None] | None = None,
    missing: str | None = None,
) -> tuple[int, str, str]:
    """Run ``popen_argv`` to completion or timeout; return ``(exit_code, stdout, stderr)``.

    ``on_kill`` runs when the process did not finish on its own (timeout or interrupt),
    before the process group is killed.
    """
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv list, no shell
            popen_argv,
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
        raise ToolError(missing or f"program not found: {program!r}") from None
    except PermissionError:
        raise ToolError(missing or f"program is not executable: {program!r}") from None
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
    finished = False
    timed_out = False
    try:
        timed_out = not _wait_unreaped(proc.pid, timeout_s)
        finished = not timed_out
    finally:
        if not finished and on_kill is not None:
            on_kill()
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
            f"timed out after {timeout_s:g}s; process killed"
            + (f"; last output: {tail}" if tail else "")
        )
    return proc.returncode, out, err


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
