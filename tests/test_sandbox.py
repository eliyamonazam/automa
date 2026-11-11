"""Docker sandbox for ``run_command``.

No real Docker daemon is needed: a fake ``docker`` client (a small Python script) records
its argv and runs the containerised command locally. A real-Docker smoke test runs only
with ``AUTOMA_DOCKER_TESTS=1``.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from automa.events import MemoryEventSink
from automa.tools import (
    PermissionPolicy,
    RiskLevel,
    RunCommand,
    RunCommandArgs,
    ToolError,
    ToolExecutor,
    Workspace,
)
from automa.tools.sandbox import (
    CONTAINER_WORKDIR,
    DockerSandbox,
    SandboxError,
    client_env,
    container_env,
    container_name,
    kill_argv,
    run_argv,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "automa"
SECRET = "sk-test-secret-should-never-leak"  # noqa: S105 - fake value

FAKE_DOCKER = r"""#!{python}
import json, os, signal, subprocess, sys, time
LOG = {log!r}
MODE = {mode!r}
PIDFILE = LOG + ".pid"
argv = sys.argv[1:]
with open(LOG, "a") as f:
    f.write(json.dumps({{"argv": argv, "env": dict(os.environ)}}) + "\n")
if argv[:1] == ["rm"]:
    try:
        os.kill(int(open(PIDFILE).read()), signal.SIGKILL)
    except (OSError, ValueError):
        pass
    sys.exit(0)
if MODE == "fail125":
    sys.stderr.write("Unable to find image 'nope:latest' locally\n")
    sys.exit(125)
if MODE == "hang":
    # Simulates a container that is NOT in our process group: detach completely.
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    open(PIDFILE, "w").write(str(child.pid))
    time.sleep(30)
    sys.exit(0)
# exec mode: emulate "docker run [opts] IMAGE CMD..." locally.
i = 1
env = {{}}
mount = workdir = None
takes_value = {{"--name", "--network", "--tmpfs", "--cap-drop", "--security-opt",
               "--pids-limit", "--memory", "--memory-swap", "--cpus", "--user",
               "--mount", "--workdir", "--env"}}
while argv[i].startswith("-"):
    opt = argv[i]
    if opt in takes_value:
        val = argv[i + 1]
        if opt == "--env":
            k, _, v = val.partition("=")
            env[k] = v
        elif opt == "--mount":
            mount = dict(p.split("=", 1) for p in val.split(","))
        elif opt == "--workdir":
            workdir = val
        i += 2
    else:
        i += 1
image, cmd = argv[i], argv[i + 1:]
cwd = mount["source"] + workdir[len(mount["target"]):]
env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
try:
    sys.exit(subprocess.run(cmd, cwd=cwd, env=env).returncode)
except FileNotFoundError:
    sys.exit(127)
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "files"
    r.mkdir()
    return r


@pytest.fixture
def ws(root: Path) -> Workspace:
    return Workspace(root)


def fake_docker(tmp_path: Path, mode: str = "exec") -> tuple[DockerSandbox, Path]:
    log = tmp_path / f"docker-{mode}.log"
    script = tmp_path / f"fake-docker-{mode}"
    script.write_text(FAKE_DOCKER.format(python=sys.executable, log=str(log), mode=mode))
    script.chmod(0o755)
    return DockerSandbox(docker=str(script), image="automa/test:1"), log


def calls(log: Path) -> list[dict[str, object]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def run(tool: RunCommand, ws: Workspace, argv: list[str], **kw: object) -> str:
    return tool.run(RunCommandArgs(argv=argv, **kw), ws)  # type: ignore[arg-type]


def build(**kw: object) -> list[str]:
    params: dict[str, object] = {
        "workspace_root": Path("/srv/p/files"),
        "cwd": "",
        "name": "automa-run-abc",
        "environment": {"LANG": "C.UTF-8"},
        "uid": 1000,
        "gid": 1000,
    }
    params.update(kw)
    sb = params.pop("sandbox", DockerSandbox())
    argv = params.pop("argv", ["python", "-V"])
    return run_argv(sb, argv, **params)  # type: ignore[arg-type]


def opt(argv: list[str], name: str) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == name]


# --------------------------------------------------------------------------- argv


class TestRunArgv:
    def test_isolation_flags(self) -> None:
        a = build()
        assert a[:3] == ["docker", "run", "--rm"]
        assert opt(a, "--network") == ["none"]
        assert "--read-only" in a
        assert opt(a, "--cap-drop") == ["ALL"]
        assert opt(a, "--security-opt") == ["no-new-privileges"]
        assert opt(a, "--pids-limit") == ["128"]
        assert opt(a, "--memory") == ["512m"] and opt(a, "--memory-swap") == ["512m"]
        assert opt(a, "--cpus") == ["1"]
        assert opt(a, "--user") == ["1000:1000"]
        tmpfs = opt(a, "--tmpfs")[0]
        assert tmpfs.startswith("/tmp:") and "noexec" in tmpfs  # noqa: S108

    def test_only_mount_is_workspace(self) -> None:
        a = build()
        assert opt(a, "--mount") == [f"type=bind,source=/srv/p/files,target={CONTAINER_WORKDIR}"]
        for forbidden in ("-v", "--volume", "--privileged", "--pid", "--ipc", "--device"):
            assert forbidden not in a
        assert not any(x.startswith("--network=") or x == "host" for x in a)

    def test_image_then_command_last(self) -> None:
        a = build(argv=["--privileged", "-v", "/:/host"])
        i = a.index(DockerSandbox().image)
        assert a[i + 1 :] == ["--privileged", "-v", "/:/host"]
        # Model-chosen argv never appears before the image (where docker parses options).
        assert "--privileged" not in a[:i] and "/:/host" not in a[:i]

    def test_workdir(self) -> None:
        assert opt(build(), "--workdir") == ["/workspace"]
        assert opt(build(cwd="src/pkg"), "--workdir") == ["/workspace/src/pkg"]

    @pytest.mark.parametrize("cwd", ["/etc", "../x", "a/../../b", "a\x00"])
    def test_bad_cwd(self, cwd: str) -> None:
        with pytest.raises(SandboxError):
            build(cwd=cwd)

    def test_root_user_is_mapped_to_nobody(self) -> None:
        assert opt(build(uid=0, gid=0), "--user") == ["65534:65534"]

    def test_default_user_is_caller(self) -> None:
        a = build(uid=None, gid=None)
        expected = "65534:65534" if os.getuid() == 0 else f"{os.getuid()}:{os.getgid()}"
        assert opt(a, "--user") == [expected]

    def test_env_only_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", SECRET)
        a = build(environment={"LANG": "C.UTF-8", "HOME": "/workspace"})
        assert opt(a, "--env") == ["LANG=C.UTF-8", "HOME=/workspace"]
        assert SECRET not in " ".join(a)

    @pytest.mark.parametrize(
        "env", [{"A B": "1"}, {"-e": "1"}, {"A": "x\ny"}, {"A": "x\x00"}, {"": "1"}, {"A=B": "1"}]
    )
    def test_bad_env(self, env: dict[str, str]) -> None:
        with pytest.raises(SandboxError):
            build(environment=env)

    @pytest.mark.parametrize("root", ["/a,b/files", '/a"b', "/a\nb", "rel/files"])
    def test_bad_mount_source(self, root: str) -> None:
        with pytest.raises(SandboxError):
            build(workspace_root=Path(root))

    @pytest.mark.parametrize("name", ["", "-x", "a b", "a;b", "x" * 200])
    def test_bad_name(self, name: str) -> None:
        with pytest.raises(SandboxError):
            build(name=name)
        with pytest.raises(SandboxError):
            kill_argv(DockerSandbox(), name)

    def test_empty_argv(self) -> None:
        with pytest.raises(SandboxError):
            build(argv=[])

    def test_limits_from_settings(self) -> None:
        a = build(sandbox=DockerSandbox(memory_mb=256, cpus=0.5, pids_limit=64))
        assert opt(a, "--memory") == ["256m"] and opt(a, "--cpus") == ["0.5"]
        assert opt(a, "--pids-limit") == ["64"]

    def test_kill_argv(self) -> None:
        assert kill_argv(DockerSandbox(docker="/usr/bin/docker"), "automa-run-1") == [
            "/usr/bin/docker",
            "rm",
            "-f",
            "automa-run-1",
        ]

    def test_container_name(self) -> None:
        assert container_name("deadbeef") == "automa-run-deadbeef"
        with pytest.raises(SandboxError):
            container_name("a b")


class TestSettings:
    @pytest.mark.parametrize(
        "image",
        [
            "python:3.12-slim",
            "ghcr.io/org/img:1.0",
            "localhost:5000/a/b",
            "alpine@sha256:" + "a" * 64,
        ],
    )
    def test_good_images(self, image: str) -> None:
        assert DockerSandbox(image=image).image == image

    @pytest.mark.parametrize(
        "image",
        ["", "-v", "--privileged", "Upper", "a b", "a;b", "a\nb", "a:", "img:tag:x", "a/../b"],
    )
    def test_bad_images(self, image: str) -> None:
        with pytest.raises(ValidationError):
            DockerSandbox(image=image)

    @pytest.mark.parametrize("docker", ["", " docker", "-H", "bin/docker", "a\x00", "C:\\docker"])
    def test_bad_docker(self, docker: str) -> None:
        with pytest.raises(ValidationError):
            DockerSandbox(docker=docker)

    @pytest.mark.parametrize(
        "kw",
        [
            {"memory_mb": 0},
            {"memory_mb": 10**6},
            {"cpus": 0},
            {"cpus": float("inf")},
            {"cpus": 100},
            {"pids_limit": 1},
            {"pids_limit": 10**6},
            {"network": "host"},
            {"privileged": True},
            {"mounts": ["/:/host"]},
        ],
    )
    def test_cannot_weaken(self, kw: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            DockerSandbox(**kw)  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        sb = DockerSandbox()
        with pytest.raises(ValidationError):
            sb.image = "other"  # type: ignore[misc]


class TestEnv:
    def test_client_env_no_secrets(self) -> None:
        parent = {"PATH": "/bin", "DOCKER_HOST": "unix:///x", "OPENAI_API_KEY": SECRET, "A": "1"}
        assert client_env(parent) == {"PATH": "/bin", "DOCKER_HOST": "unix:///x"}

    def test_container_env_no_secrets(self) -> None:
        parent = {"PATH": "/host/bin", "LANG": "fa_IR.UTF-8", "AUTOMA_API_KEY": SECRET}
        env = container_env(parent)
        assert env == {
            "LANG": "fa_IR.UTF-8",
            "HOME": CONTAINER_WORKDIR,
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def test_container_env_drops_bad_values(self) -> None:
        env = container_env({"TZ": "a\nb", "LC_ALL": "x\x00"})
        assert "TZ" not in env and "LC_ALL" not in env and env["LANG"] == "C.UTF-8"


# --------------------------------------------------------------------------- RunCommand


class TestRunInSandbox:
    def test_runs_through_docker(self, tmp_path: Path, ws: Workspace) -> None:
        sb, log = fake_docker(tmp_path)
        out = run(RunCommand(sandbox=sb), ws, ["echo", "hello"])
        assert out.startswith("exit_code: 0") and "hello" in out
        (c,) = calls(log)
        argv = c["argv"]
        assert isinstance(argv, list) and argv[0] == "run"
        assert argv[-3:] == ["automa/test:1", "echo", "hello"]
        assert opt(["x", *argv], "--network") == ["none"]
        assert opt(["x", *argv], "--name")[0].startswith("automa-run-")

    def test_exit_code_passthrough(self, tmp_path: Path, ws: Workspace) -> None:
        sb, _ = fake_docker(tmp_path)
        out = run(RunCommand(sandbox=sb), ws, ["sh", "-c", "echo bad >&2; exit 3"])
        assert out.startswith("exit_code: 3") and "bad" in out

    def test_cwd_and_relative_program(self, tmp_path: Path, root: Path, ws: Workspace) -> None:
        (root / "sub").mkdir()
        script = root / "sub" / "hi.sh"
        script.write_text("#!/bin/sh\npwd\n")
        script.chmod(0o755)
        sb, log = fake_docker(tmp_path)
        tool = RunCommand(sandbox=sb)
        out = run(tool, ws, ["sub/hi.sh"])
        assert "exit_code: 0" in out and str(root) in out
        argv = ["x", *calls(log)[0]["argv"]]  # type: ignore[misc]
        assert opt(argv, "--workdir") == ["/workspace"]
        assert argv[-1] == "./sub/hi.sh"
        assert opt(argv, "--mount") == [f"type=bind,source={root},target=/workspace"]
        out = run(tool, ws, ["pwd"], cwd="sub")
        assert out.splitlines()[2] == str(root / "sub")
        assert opt(["x", *calls(log)[1]["argv"]], "--workdir") == ["/workspace/sub"]  # type: ignore[misc]

    def test_unique_container_names(self, tmp_path: Path, ws: Workspace) -> None:
        sb, log = fake_docker(tmp_path)
        tool = RunCommand(sandbox=sb)
        run(tool, ws, ["true"])
        run(tool, ws, ["true"])
        names = [opt(["x", *c["argv"]], "--name")[0] for c in calls(log)]  # type: ignore[misc]
        assert len(set(names)) == 2

    def test_secrets_reach_neither_client_nor_container(
        self, tmp_path: Path, ws: Workspace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", SECRET)
        monkeypatch.setenv("AUTOMA_SECRET_TOKEN", SECRET)
        sb, log = fake_docker(tmp_path)
        out = run(RunCommand(sandbox=sb), ws, ["env"])
        assert SECRET not in out
        assert "HOME=/workspace" in out
        (c,) = calls(log)
        assert SECRET not in json.dumps(c)

    def test_docker_error_is_tool_error(self, tmp_path: Path, ws: Workspace) -> None:
        sb, _ = fake_docker(tmp_path, "fail125")
        with pytest.raises(ToolError, match="could not start the container.*Unable to find"):
            run(RunCommand(sandbox=sb), ws, ["echo", "x"])

    def test_missing_docker_fails_closed(self, tmp_path: Path, root: Path, ws: Workspace) -> None:
        sb = DockerSandbox(docker=str(tmp_path / "no-such-docker"))
        with pytest.raises(ToolError, match="docker client not found.*not run"):
            run(RunCommand(sandbox=sb), ws, ["touch", "ran-on-host"])
        assert not (root / "ran-on-host").exists()

    def test_missing_docker_on_path_fails_closed(
        self, root: Path, ws: Workspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        empty = tmp_path / "emptybin"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))
        with pytest.raises(ToolError, match="docker client not found"):
            run(RunCommand(sandbox=DockerSandbox()), ws, ["touch", "ran-on-host"])
        assert not (root / "ran-on-host").exists()

    def test_timeout_removes_container(self, tmp_path: Path, ws: Workspace) -> None:
        sb, log = fake_docker(tmp_path, "hang")
        t0 = time.monotonic()
        with pytest.raises(ToolError, match="timed out"):
            run(RunCommand(sandbox=sb), ws, ["sleep", "30"], timeout_s=1.0)
        assert time.monotonic() - t0 < 15
        cs = calls(log)
        assert [c["argv"][0] for c in cs] == ["run", "rm"]  # type: ignore[index]
        name = opt(["x", *cs[0]["argv"]], "--name")[0]  # type: ignore[misc]
        assert cs[1]["argv"] == ["rm", "-f", name]
        # The detached "container" process was stopped by `docker rm -f`.
        pid = int(Path(str(log) + ".pid").read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _alive(pid):
            time.sleep(0.05)
        assert not _alive(pid)

    def test_no_rm_after_normal_exit(self, tmp_path: Path, ws: Workspace) -> None:
        sb, log = fake_docker(tmp_path)
        run(RunCommand(sandbox=sb), ws, ["true"])
        assert [c["argv"][0] for c in calls(log)] == ["run"]  # type: ignore[index]

    def test_allow_list_checked_before_docker(self, tmp_path: Path, ws: Workspace) -> None:
        sb, log = fake_docker(tmp_path)
        with pytest.raises(ToolError, match="not allowed"):
            run(RunCommand({"pytest"}, sandbox=sb), ws, ["sh", "-c", "id"])
        assert calls(log) == []

    def test_bad_cwd_checked_before_docker(self, tmp_path: Path, root: Path) -> None:
        sb, log = fake_docker(tmp_path)
        (tmp_path / "outside").mkdir()
        (root / "link").symlink_to(tmp_path / "outside")
        with pytest.raises(ToolError):
            run(RunCommand(sandbox=sb), Workspace(root), ["ls"], cwd="link")
        assert calls(log) == []

    def test_unmountable_workspace_not_run(self, tmp_path: Path) -> None:
        r = tmp_path / "a,b" / "files"
        r.mkdir(parents=True)
        sb, log = fake_docker(tmp_path)
        with pytest.raises(ToolError, match="sandbox:.*not run"):
            run(RunCommand(sandbox=sb), Workspace(r), ["true"])
        assert calls(log) == []

    def test_sandbox_type_checked(self) -> None:
        with pytest.raises(TypeError):
            RunCommand(sandbox={"image": "x"})  # type: ignore[arg-type]
        assert RunCommand().sandbox is None

    def test_spec_unchanged(self, tmp_path: Path) -> None:
        sb, _ = fake_docker(tmp_path)
        assert RunCommand(sandbox=sb).spec() == RunCommand().spec()

    def test_event_records_sandbox(self, tmp_path: Path, root: Path) -> None:
        sb, _ = fake_docker(tmp_path)
        sink = MemoryEventSink()
        ex = ToolExecutor(
            [RunCommand(sandbox=sb)],
            PermissionPolicy(allowed_tools=frozenset({"run_command"}), max_risk=RiskLevel.EXECUTE),
            Workspace(root),
            sink,
            "proj",
        )
        res = ex.execute("run_command", {"argv": ["echo", SECRET]})
        assert res.ok
        assert sink.events[-1].data == {
            "tool": "run_command",
            "risk": "execute",
            "ok": True,
            "program": "echo",
            "cwd": "",
            "sandbox": "docker",
            "image": "automa/test:1",
        }
        assert SECRET not in str([e.data for e in sink.events])

    def test_policy_still_applies(self, tmp_path: Path, root: Path) -> None:
        sb, log = fake_docker(tmp_path)
        ex = ToolExecutor(
            [RunCommand(sandbox=sb)],
            PermissionPolicy(allowed_tools=frozenset({"run_command"}), max_risk=RiskLevel.WRITE),
            Workspace(root),
            MemoryEventSink(),
            "proj",
        )
        assert ex.execute("run_command", {"argv": ["true"]}).denied
        assert calls(log) == []


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    except OSError:  # pragma: no cover
        return True
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def test_sandbox_module_never_spawns() -> None:
    tree = ast.parse((SRC / "tools" / "sandbox.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            mods = {a.name for a in node.names} | {getattr(node, "module", None) or ""}
            assert not mods & {"subprocess", "pty", "shutil"}


@pytest.mark.skipif(
    os.environ.get("AUTOMA_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="real Docker test; set AUTOMA_DOCKER_TESTS=1",
)
def test_real_docker_isolation(root: Path, ws: Workspace) -> None:  # pragma: no cover
    tool = RunCommand(sandbox=DockerSandbox(image="python:3.12-slim"))
    out = run(
        tool,
        ws,
        [
            "python",
            "-c",
            "import os,socket;open('made','w').write(str(os.getuid()));"
            "print('HOME', os.environ['HOME']);"
            "s=socket.socket();s.settimeout(2)\n"
            "try:\n s.connect(('1.1.1.1',80));print('NET')\nexcept OSError:print('NONET')",
        ],
        timeout_s=120,
    )
    assert "exit_code: 0" in out and "NONET" in out and "HOME /workspace" in out
    assert (root / "made").exists()
