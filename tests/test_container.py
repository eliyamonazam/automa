"""Guards for Automa's own container files: Dockerfile, .dockerignore, docker-compose.yml.

These are static checks (no Docker daemon needed). They pin the security properties of
the image and the compose service so a later change cannot quietly weaken them: no
secrets in the image, non-root user, no Docker socket, hardened runtime options.
A real build smoke test runs only with ``AUTOMA_DOCKER_TESTS=1``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
COMPOSE = ROOT / "docker-compose.yml"

SECRET_WORDS = ("KEY", "SECRET", "TOKEN", "PASSWORD")


def _instructions(text: str) -> list[tuple[str, str]]:
    """Dockerfile instructions as (KEYWORD, args), with continuation lines joined."""
    joined = re.sub(r"\\\n", " ", text)
    out: list[tuple[str, str]] = []
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, args = line.partition(" ")
        out.append((keyword.upper(), args.strip()))
    return out


@pytest.fixture(scope="module")
def dockerfile() -> list[tuple[str, str]]:
    return _instructions(DOCKERFILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compose_text() -> str:
    return COMPOSE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def service(compose_text: str) -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(compose_text)
    assert set(data["services"]) == {"automa"}
    svc: dict[str, Any] = data["services"]["automa"]
    return svc


# --- Dockerfile -------------------------------------------------------------------------


def test_files_exist() -> None:
    assert DOCKERFILE.is_file()
    assert DOCKERIGNORE.is_file()
    assert COMPOSE.is_file()


def _final_stage(instr: list[tuple[str, str]]) -> list[tuple[str, str]]:
    starts = [i for i, (k, _) in enumerate(instr) if k == "FROM"]
    assert starts, "no FROM"
    return instr[starts[-1] :]


def test_base_image_pinned(dockerfile: list[tuple[str, str]]) -> None:
    args = dict(a.split("=", 1) for k, a in dockerfile if k == "ARG" and "=" in a)
    image = args["PYTHON_IMAGE"]
    name, _, tag = image.partition(":")
    assert name == "python"
    assert tag and tag != "latest"
    # Same minimum version as pyproject (3.11+).
    major, minor = (int(x) for x in re.match(r"(\d+)\.(\d+)", tag).groups())  # type: ignore[union-attr]
    assert (major, minor) >= (3, 11)
    for keyword, a in dockerfile:
        if keyword == "FROM":
            assert a.split()[0] in ("${PYTHON_IMAGE}", "$PYTHON_IMAGE"), a


def test_runs_as_non_root(dockerfile: list[tuple[str, str]]) -> None:
    final = _final_stage(dockerfile)
    users = [a for k, a in final if k == "USER"]
    assert users, "final stage must set USER"
    user = users[-1].split(":")[0]
    assert user not in ("0", "root")
    # Nothing may run after switching user that could need root (and no switch back).
    after = final[[k for k, _ in final].index("USER") :]
    assert all(k != "RUN" for k, _ in after)


def test_entrypoint_is_cli(dockerfile: list[tuple[str, str]]) -> None:
    final = _final_stage(dockerfile)
    entry = [a for k, a in final if k == "ENTRYPOINT"]
    assert entry == ['["automa"]']


def test_workspace_env_points_inside_data(dockerfile: list[tuple[str, str]]) -> None:
    env = " ".join(a for k, a in _final_stage(dockerfile) if k == "ENV")
    assert "AUTOMA_WORKSPACE=/data/workspace" in env
    workdir = [a for k, a in _final_stage(dockerfile) if k == "WORKDIR"]
    assert workdir[-1] == "/data"


def test_no_secrets_in_dockerfile(dockerfile: list[tuple[str, str]]) -> None:
    for keyword, args in dockerfile:
        if keyword in ("ARG", "ENV", "LABEL"):
            assert not any(w in args.upper() for w in SECRET_WORDS), args
        if keyword in ("COPY", "ADD"):
            sources = args.split()[:-1]
            for src in sources:
                assert ".env" not in src and "automa.toml" not in src, args
                assert src != "." or "--from=" in args, "copy only what the build needs"
    assert all(k != "ADD" for k, _ in dockerfile), "use COPY, not ADD (no remote fetches)"


def test_no_docker_client_installed(dockerfile: list[tuple[str, str]]) -> None:
    # Inside the container [sandbox] mode = "docker" must fail closed, not reach a daemon.
    runs = " ".join(a for k, a in dockerfile if k == "RUN").lower()
    assert "docker" not in runs


def test_build_copies_only_package_sources(dockerfile: list[tuple[str, str]]) -> None:
    copied = [a for k, a in dockerfile if k == "COPY" and "--from=" not in a]
    sources = {s for a in copied for s in a.split()[:-1]}
    assert sources == {"pyproject.toml", "README.md", "src"}


# --- .dockerignore ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern",
    [".env", ".env.*", "automa.toml", "workspace/", "benchmarks/results/", "*.sqlite3", ".git/"],
)
def test_dockerignore_excludes(pattern: str) -> None:
    lines = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
    assert pattern in lines


def test_dockerignore_keeps_env_example_only() -> None:
    lines = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
    negations = [line for line in lines if line.startswith("!")]
    assert negations == ["!.env.example"]


# --- docker-compose.yml -----------------------------------------------------------------


def test_compose_never_mounts_docker_socket(compose_text: str) -> None:
    assert "docker.sock" not in compose_text
    assert "privileged" not in compose_text
    assert "network_mode" not in compose_text
    assert "pid:" not in compose_text


def test_compose_has_no_inline_secrets(compose_text: str, service: dict[str, Any]) -> None:
    assert "environment" not in service, "secrets come only from .env via env_file"
    for line in compose_text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        assert "API_KEY" not in line.upper()


def test_compose_reads_env_file(service: dict[str, Any]) -> None:
    assert service["env_file"] == [{"path": ".env", "required": True}]


def test_compose_hardening(service: dict[str, Any]) -> None:
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "cap_add" not in service
    assert "no-new-privileges:true" in service["security_opt"]
    assert not any("unconfined" in opt for opt in service["security_opt"])
    assert isinstance(service["pids_limit"], int) and 0 < service["pids_limit"] <= 4096
    assert service["mem_limit"]
    assert "ports" not in service and "expose" not in service
    assert "devices" not in service
    assert service.get("init") is True


def test_compose_user_is_not_root(service: dict[str, Any]) -> None:
    user = service["user"]
    defaults = re.findall(r"\$\{[A-Z_]+:-(\d+)\}", user)
    assert len(defaults) == 2, user
    assert all(int(d) != 0 for d in defaults)


def test_compose_mounts(service: dict[str, Any]) -> None:
    mounts = {m["target"]: m for m in service["volumes"]}
    assert set(mounts) == {"/data/workspace", "/data/automa.toml", "/data/benchmarks"}
    for m in mounts.values():
        assert m["type"] == "bind"
        # Long syntax: a missing source must be an error, not a root-owned new directory.
        assert m.get("bind", {}).get("create_host_path", False) is False
        assert m["source"].startswith("./") and ".." not in m["source"]
    assert mounts["/data/automa.toml"]["read_only"] is True
    assert mounts["/data/workspace"]["source"] == "./workspace"
    # .env is never mounted (the key goes through env_file only).
    assert not any(".env" in m["source"] for m in mounts.values())


def test_compose_tmpfs_only_tmp(service: dict[str, Any]) -> None:
    assert [t.split(":")[0] for t in service["tmpfs"]] == ["/tmp"]  # noqa: S108


def test_compose_build_and_image(service: dict[str, Any]) -> None:
    assert service["build"] == "."
    assert service["image"] == "automa:local"


def test_compose_targets_match_image(service: dict[str, Any], dockerfile: list) -> None:
    env = " ".join(a for k, a in _final_stage(dockerfile) if k == "ENV")
    targets = {m["target"] for m in service["volumes"]}
    ws = re.search(r"AUTOMA_WORKSPACE=(\S+)", env).group(1)  # type: ignore[union-attr]
    assert ws in targets


# --- optional real build ----------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("AUTOMA_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="real Docker test; set AUTOMA_DOCKER_TESTS=1",
)
def test_real_image_build_and_cli(tmp_path: Path) -> None:  # pragma: no cover
    docker = shutil.which("docker")
    assert docker
    tag = "automa:test-build"
    subprocess.run([docker, "build", "-t", tag, str(ROOT)], check=True, timeout=900)  # noqa: S603
    out = subprocess.run(  # noqa: S603
        [docker, "run", "--rm", "--read-only", "--cap-drop", "ALL", tag, "version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.stdout.strip()
    uid = subprocess.run(  # noqa: S603
        [docker, "run", "--rm", "--entrypoint", "id", tag, "-u"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert uid.stdout.strip() != "0"
