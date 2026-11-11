"""Docker sandbox for ``run_command``: settings and the ``docker run`` command line.

This module never starts a process; it only builds argument lists. Spawning stays in
``tools/command.py`` (a structural test enforces that). See docs/DECISIONS.md D32.

Isolation of every container (none of it is configurable to be weaker):

- ``--network none``: no network at all;
- ``--read-only`` root filesystem, a small ``noexec`` tmpfs on ``/tmp``;
- ``--cap-drop ALL`` and ``no-new-privileges``;
- runs as the calling user's uid:gid, never root;
- limits on memory (no extra swap), CPUs and number of processes;
- the only mount is the project's ``files/`` directory at ``/workspace``;
- the environment contains only the values passed explicitly with ``--env K=V``,
  never anything from the parent process.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from automa.config import SandboxMode

CONTAINER_WORKDIR = "/workspace"
CONTAINER_NAME_PREFIX = "automa-run-"
DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_MEMORY_MB = 512
MAX_MEMORY_MB = 8192
DEFAULT_CPUS = 1.0
MAX_CPUS = 8.0
DEFAULT_PIDS_LIMIT = 128
MAX_PIDS_LIMIT = 4096
TMPFS_SIZE_MB = 64
# Docker reserves 125 for "docker run itself failed" (daemon, image, bad option).
DOCKER_RUN_ERROR = 125
# Variables the *docker client* may need to reach the daemon. They are given to the
# client process only; the container's environment is built from ``--env`` alone.
CLIENT_ENV_PASSTHROUGH = (
    "PATH",
    "HOME",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_CERT_PATH",
    "DOCKER_TLS_VERIFY",
    "XDG_RUNTIME_DIR",
)
CONTAINER_ENV_PASSTHROUGH = ("LANG", "LC_ALL", "TZ")

# A conservative subset of Docker's image reference grammar: optional registry/path,
# optional tag, optional sha256 digest. Never starts with '-' (it would be an option).
_IMAGE_RE = re.compile(
    r"^[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]{1,5})?"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*"
    r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?"
    r"(?:@sha256:[a-f0-9]{64})?$"
)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Characters that would change the meaning of ``--mount type=bind,source=...``.
_BAD_MOUNT_CHARS = frozenset(',"\n\r\x00')

__all__ = [
    "DockerSandbox",
    "SandboxError",
    "SandboxMode",
    "client_env",
    "container_env",
    "container_name",
    "kill_argv",
    "run_argv",
    "version_argv",
]

ImageRef = Annotated[str, Field(min_length=1, max_length=255)]


class SandboxError(ValueError):
    """The sandbox cannot be configured or used safely (fail closed: nothing runs)."""


class DockerSandbox(BaseModel):
    """Settings for running commands in a throw-away Docker container."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    image: ImageRef = DEFAULT_IMAGE
    docker: str = Field(
        default="docker",
        min_length=1,
        max_length=4096,
        description="docker client: a bare name on PATH or an absolute path",
    )
    memory_mb: int = Field(default=DEFAULT_MEMORY_MB, ge=16, le=MAX_MEMORY_MB)
    cpus: float = Field(default=DEFAULT_CPUS, gt=0, le=MAX_CPUS)
    pids_limit: int = Field(default=DEFAULT_PIDS_LIMIT, ge=8, le=MAX_PIDS_LIMIT)

    @field_validator("image")
    @classmethod
    def _check_image(cls, v: str) -> str:
        if not _IMAGE_RE.fullmatch(v):
            raise ValueError(f"invalid image reference: {v!r}")
        return v

    @field_validator("docker")
    @classmethod
    def _check_docker(cls, v: str) -> str:
        if "\x00" in v or v != v.strip() or v.startswith("-"):
            raise ValueError("docker must be a program name or an absolute path")
        if "/" in v and not v.startswith("/"):
            raise ValueError("docker must be a bare name or an absolute path")
        if "\\" in v:
            raise ValueError("docker must not contain backslashes")
        return v


def container_name(token: str) -> str:
    """Container name for one ``run_command`` call (``token`` is random hex)."""
    name = f"{CONTAINER_NAME_PREFIX}{token}"
    if not _NAME_RE.fullmatch(name):
        raise SandboxError(f"invalid container name: {name!r}")
    return name


def client_env(parent: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment of the docker *client* process: only what it needs to find the daemon."""
    src = os.environ if parent is None else parent
    env = {k: src[k] for k in CLIENT_ENV_PASSTHROUGH if k in src and "\x00" not in src[k]}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    return env


def container_env(parent: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment *inside* the container: an allow-list, never secrets from the parent."""
    src = os.environ if parent is None else parent
    env = {
        k: src[k]
        for k in CONTAINER_ENV_PASSTHROUGH
        if k in src and "\x00" not in src[k] and "\n" not in src[k]
    }
    env.setdefault("LANG", "C.UTF-8")
    env["HOME"] = CONTAINER_WORKDIR
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _mount_source(root: Path) -> str:
    src = str(root)
    if not root.is_absolute():
        raise SandboxError("workspace root must be an absolute path")
    if _BAD_MOUNT_CHARS & set(src):
        raise SandboxError("workspace path contains characters that cannot be mounted safely")
    return src


def _user(uid: int | None, gid: int | None) -> str:
    u = os.getuid() if uid is None else uid
    g = os.getgid() if gid is None else gid
    if u == 0:
        # Never run the container as root, even if Automa itself does.
        u, g = 65534, 65534
    return f"{u}:{g}"


def run_argv(
    sandbox: DockerSandbox,
    argv: Sequence[str],
    *,
    workspace_root: Path,
    cwd: str,
    name: str,
    environment: Mapping[str, str],
    uid: int | None = None,
    gid: int | None = None,
) -> list[str]:
    """``docker run`` argument list that runs ``argv`` in a fresh, isolated container.

    ``cwd`` is an already-validated path relative to the workspace ("" = root).
    """
    if not argv:
        raise SandboxError("empty argv")
    if not _NAME_RE.fullmatch(name):
        raise SandboxError(f"invalid container name: {name!r}")
    if cwd.startswith("/") or ".." in cwd.split("/") or "\x00" in cwd:
        raise SandboxError("cwd must be a safe relative path")
    workdir = CONTAINER_WORKDIR + (f"/{cwd}" if cwd else "")
    out = [
        sandbox.docker,
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        f"/tmp:rw,noexec,nosuid,nodev,size={TMPFS_SIZE_MB}m",  # noqa: S108 - inside container
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(sandbox.pids_limit),
        "--memory",
        f"{sandbox.memory_mb}m",
        "--memory-swap",
        f"{sandbox.memory_mb}m",
        "--cpus",
        f"{sandbox.cpus:g}",
        "--user",
        _user(uid, gid),
        "--mount",
        f"type=bind,source={_mount_source(workspace_root)},target={CONTAINER_WORKDIR}",
        "--workdir",
        workdir,
    ]
    for k, v in environment.items():
        if not _ENV_KEY_RE.fullmatch(k) or "\x00" in v or "\n" in v:
            raise SandboxError(f"invalid environment entry: {k!r}")
        out += ["--env", f"{k}={v}"]
    # Options end at the image: docker passes everything after it to the container.
    out += [sandbox.image, *argv]
    return out


def kill_argv(sandbox: DockerSandbox, name: str) -> list[str]:
    """``docker rm -f`` argument list: stop and remove one container by name."""
    if not _NAME_RE.fullmatch(name):
        raise SandboxError(f"invalid container name: {name!r}")
    return [sandbox.docker, "rm", "-f", name]


def version_argv(sandbox: DockerSandbox) -> list[str]:
    """``docker version`` argument list: succeeds only if the daemon answers."""
    return [sandbox.docker, "version", "--format", "{{.Server.Version}}"]
