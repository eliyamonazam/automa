"""``[sandbox]`` configuration and its wiring (D33).

No real Docker: a fake ``docker`` client (a small Python script) answers
``docker version`` and emulates ``docker run`` locally while recording its argv.
No real model calls: the CLI provider factory is replaced by a ``FakeProvider``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.agents import VerifierAgent, WorkerAgent
from automa.benchmarks import evaluate_benchmark
from automa.benchmarks.loader import BenchmarkSuite
from automa.budget import BudgetGuard, CostLedger
from automa.config import AutomaConfig, ConfigError, SandboxConfig, SandboxMode, load_config
from automa.events import MemoryEventSink
from automa.llm import (
    ChatResponse,
    FakeProvider,
    FinishReason,
    LLMGateway,
    Message,
    ToolCall,
    Usage,
)
from automa.llm import Role as MsgRole
from automa.orchestrator import Orchestrator
from automa.plan import Plan
from automa.project import ProjectStatus, ProjectStore
from automa.roles import RoleRegistry
from automa.tools import (
    DockerSandbox,
    RunCommand,
    SandboxError,
    check_docker,
    default_tools,
    tools_factory,
)
from automa.tools.sandbox import version_argv

REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO / "benchmarks"
PID = "proj-1"
SECRET = "sk-test-secret-should-never-leak"  # noqa: S105 - fake value

FAKE_DOCKER = r"""#!{python}
import json, os, subprocess, sys, time
LOG = {log!r}
MODE = {mode!r}
argv = sys.argv[1:]
with open(LOG, "a") as f:
    f.write(json.dumps({{"argv": argv, "env": dict(os.environ)}}) + "\n")
if argv[:1] == ["version"]:
    if MODE == "down":
        sys.stderr.write("Cannot connect to the Docker daemon\n")
        sys.exit(1)
    if MODE == "hang":
        time.sleep(30)
    if MODE == "empty":
        sys.exit(0)
    print("27.1.0")
    sys.exit(0)
if argv[:1] == ["rm"]:
    sys.exit(0)
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
cmd = argv[i + 1:]
cwd = mount["source"] + workdir[len(mount["target"]):]
env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
try:
    sys.exit(subprocess.run(cmd, cwd=cwd, env=env).returncode)
except FileNotFoundError:
    sys.exit(127)
"""

TOML = """
workspace = "{ws}"
[tiers]
small = "fake-small"
medium = "fake-medium"
large = "fake-large"
[prices."fake-small"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[prices."fake-medium"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[prices."fake-large"]
input_per_mtok = 1.0
output_per_mtok = 2.0
{sandbox}
"""


def make_fake_docker(tmp_path: Path, mode: str = "ok") -> tuple[Path, Path]:
    log = tmp_path / f"docker-{mode}.log"
    script = tmp_path / f"fake-docker-{mode}"
    script.write_text(FAKE_DOCKER.format(python=sys.executable, log=str(log), mode=mode))
    script.chmod(0o755)
    return script, log


def docker_calls(log: Path) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def subcommands(log: Path) -> list[str]:
    return [c["argv"][0] for c in docker_calls(log)]


def write_toml(path: Path, ws: Path, sandbox: str = "") -> Path:
    path.write_text(TOML.format(ws=ws.as_posix(), sandbox=sandbox), encoding="utf-8")
    return path


def cfg(tmp_path: Path, sandbox: str) -> AutomaConfig:
    return load_config(write_toml(tmp_path / "automa.toml", tmp_path / "ws", sandbox), environ={})


# -- configuration --------------------------------------------------------------------


def test_default_is_local_without_sandbox(tmp_path: Path) -> None:
    c = cfg(tmp_path, "")
    assert c.sandbox.mode is SandboxMode.LOCAL
    assert c.sandbox.docker_sandbox() is None
    assert "local" in c.sandbox.describe() and "not isolated" in c.sandbox.describe()


def test_docker_mode_with_defaults(tmp_path: Path) -> None:
    c = cfg(tmp_path, '[sandbox]\nmode = "docker"\n')
    assert c.sandbox.docker_sandbox() == DockerSandbox()
    assert "docker" in c.sandbox.describe() and "no network" in c.sandbox.describe()


def test_docker_mode_with_settings(tmp_path: Path) -> None:
    c = cfg(
        tmp_path,
        '[sandbox]\nmode = "docker"\nimage = "automa/test:1"\ndocker = "/usr/bin/docker"\n'
        "memory_mb = 256\ncpus = 0.5\npids_limit = 64\n",
    )
    sb = c.sandbox.docker_sandbox()
    assert sb == DockerSandbox(
        image="automa/test:1", docker="/usr/bin/docker", memory_mb=256, cpus=0.5, pids_limit=64
    )
    assert "automa/test:1" in c.sandbox.describe() and "256m" in c.sandbox.describe()


@pytest.mark.parametrize(
    "body",
    [
        'image = "python:3.12-slim"',
        "memory_mb = 256",
        'mode = "local"\ncpus = 1.0',
        'docker = "docker"',
    ],
)
def test_docker_settings_without_docker_mode_are_rejected(tmp_path: Path, body: str) -> None:
    with pytest.raises(ConfigError, match='need mode = "docker"'):
        cfg(tmp_path, f"[sandbox]\n{body}\n")


@pytest.mark.parametrize(
    "body",
    [
        'mode = "vm"',
        'mode = "docker"\nimage = "--privileged"',
        'mode = "docker"\nimage = "Bad Image"',
        'mode = "docker"\nmemory_mb = 999999',
        'mode = "docker"\ncpus = 0',
        'mode = "docker"\npids_limit = 1',
        'mode = "docker"\ndocker = "bin/docker"',
        'mode = "docker"\nnetwork = "host"',
        'mode = "docker"\nprivileged = true',
        'mode = "docker"\nmounts = ["/:/host"]',
    ],
)
def test_invalid_or_weakening_sandbox_settings_are_rejected(tmp_path: Path, body: str) -> None:
    with pytest.raises(ConfigError, match="sandbox"):
        cfg(tmp_path, f"[sandbox]\n{body}\n")


def test_secret_like_key_in_sandbox_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="secrets are not allowed"):
        cfg(tmp_path, f'[sandbox]\nmode = "docker"\ntoken = "{SECRET}"\n')


def test_sandbox_config_is_frozen() -> None:
    sc = SandboxConfig(mode=SandboxMode.DOCKER)
    with pytest.raises(ValueError):
        sc.mode = SandboxMode.LOCAL  # type: ignore[misc]


# -- tools ----------------------------------------------------------------------------


def test_default_tools_bind_the_sandbox() -> None:
    sb = DockerSandbox(image="automa/test:1")
    (rc,) = [t for t in default_tools(sb) if isinstance(t, RunCommand)]
    assert rc.sandbox == sb
    (local,) = [t for t in default_tools() if isinstance(t, RunCommand)]
    assert local.sandbox is None


def test_tools_factory_makes_fresh_sandboxed_tools() -> None:
    sb = DockerSandbox()
    make = tools_factory(sb)
    a, b = make(), make()
    assert all(x is not y for x, y in zip(a, b, strict=True))
    assert all(t.sandbox == sb for t in [*a, *b] if isinstance(t, RunCommand))


# -- preflight ------------------------------------------------------------------------


def test_version_argv() -> None:
    assert version_argv(DockerSandbox(docker="/x/docker")) == [
        "/x/docker",
        "version",
        "--format",
        "{{.Server.Version}}",
    ]


def test_check_docker_ok_and_client_env_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTOMA_API_KEY", SECRET)
    script, log = make_fake_docker(tmp_path)
    assert check_docker(DockerSandbox(docker=str(script))) == "27.1.0"
    (call,) = docker_calls(log)
    assert call["argv"][0] == "version"
    assert "AUTOMA_API_KEY" not in call["env"]
    assert SECRET not in json.dumps(call)


@pytest.mark.parametrize(("mode", "match"), [("down", "Cannot connect"), ("empty", "daemon")])
def test_check_docker_daemon_unavailable(tmp_path: Path, mode: str, match: str) -> None:
    script, _ = make_fake_docker(tmp_path, mode)
    with pytest.raises(SandboxError, match=match):
        check_docker(DockerSandbox(docker=str(script)))


def test_check_docker_missing_client(tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="not found"):
        check_docker(DockerSandbox(docker=str(tmp_path / "no-docker")))


def test_check_docker_timeout(tmp_path: Path) -> None:
    script, _ = make_fake_docker(tmp_path, "hang")
    with pytest.raises(SandboxError, match="did not answer"):
        check_docker(DockerSandbox(docker=str(script)), timeout_s=0.5)


# -- orchestrator ---------------------------------------------------------------------


def _orchestrator(config: AutomaConfig, **kw: Any) -> Orchestrator:
    store = ProjectStore.for_workspace(config.workspace)
    ledger = CostLedger(config.workspace.parent / "ledger.sqlite3")
    gateway = LLMGateway(config, FakeProvider(), guard=BudgetGuard(config.limits, ledger))
    return Orchestrator(
        store,
        WorkerAgent(gateway),
        VerifierAgent(gateway),
        RoleRegistry.builtin(tools=default_tools()),
        config,
        **kw,
    )


def test_orchestrator_default_tools_follow_config(tmp_path: Path) -> None:
    c = cfg(tmp_path, '[sandbox]\nmode = "docker"\nimage = "automa/test:1"\n')
    tools = _orchestrator(c)._make_tools()
    (rc,) = [t for t in tools if isinstance(t, RunCommand)]
    assert rc.sandbox == DockerSandbox(image="automa/test:1")


def test_orchestrator_refuses_unsandboxed_tools_in_docker_mode(tmp_path: Path) -> None:
    c = cfg(tmp_path, '[sandbox]\nmode = "docker"\n')
    with pytest.raises(ConfigError, match="refusing to run commands on the host"):
        _orchestrator(c, tools=default_tools)


def test_orchestrator_local_mode_accepts_local_tools(tmp_path: Path) -> None:
    c = cfg(tmp_path, "")
    tools = _orchestrator(c, tools=default_tools)._make_tools()
    assert any(isinstance(t, RunCommand) and t.sandbox is None for t in tools)


# -- benchmark evaluation -------------------------------------------------------------


def test_evaluate_benchmark_runs_checks_in_sandbox(tmp_path: Path) -> None:
    script, log = make_fake_docker(tmp_path)
    files = tmp_path / "files"
    files.mkdir()
    bench = BenchmarkSuite.from_dir(BENCH_DIR).get("cli-word-count")
    evaluate_benchmark(bench, files, MemoryEventSink(), sandbox=DockerSandbox(docker=str(script)))
    assert "run" in subcommands(log)
    run_call = next(c for c in docker_calls(log) if c["argv"][0] == "run")
    assert "--network" in run_call["argv"] and "none" in run_call["argv"]


# -- CLI ------------------------------------------------------------------------------


def _calls(*items: tuple[str, dict[str, Any]]) -> ChatResponse:
    tcs = tuple(
        ToolCall(id=f"x{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(items)
    )
    return ChatResponse(
        model="fake",
        message=Message(role=MsgRole.ASSISTANT, content=None, tool_calls=tcs),
        usage=Usage(input_tokens=100, output_tokens=20),
        finish_reason=FinishReason.TOOL_CALLS,
    )


TASK = {
    "id": "a",
    "title": "Write a",
    "role": "writer",
    "acceptance_criteria": [
        {
            "description": "a.txt is readable by a command",
            "check": {"type": "command", "command": "cat a.txt"},
        }
    ],
}


class Env:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.tmp = tmp_path
        self.ws = tmp_path / "ws"
        self.config = tmp_path / "automa.toml"
        self.provider = FakeProvider()
        monkeypatch.chdir(tmp_path)
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(cli, "make_provider", lambda config: self.provider)
        self.sandbox("")

    def sandbox(self, body: str) -> None:
        write_toml(self.config, self.ws, body)

    def docker(self, mode: str = "ok") -> Path:
        script, log = make_fake_docker(self.tmp, mode)
        self.sandbox(f'[sandbox]\nmode = "docker"\ndocker = "{script}"\nimage = "automa/test:1"\n')
        return log

    @property
    def store(self) -> ProjectStore:
        return ProjectStore.for_workspace(self.ws)

    def project(self) -> None:
        self.store.create("test goal", project_id=PID)
        self.store.save_plan(PID, Plan.model_validate({"goal": "test goal", "tasks": [TASK]}))

    def cli(self, *args: str) -> Any:
        return CliRunner().invoke(cli.app, [*args, "--config", str(self.config)])


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


def test_run_in_docker_mode_runs_check_commands_in_container(env: Env) -> None:
    log = env.docker()
    env.project()
    env.provider.push(_calls(("write_file", {"path": "a.txt", "content": "ok"})), "done")
    r = env.cli("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "sandbox: docker (image automa/test:1" in r.output
    subs = subcommands(log)
    assert subs[0] == "version"  # preflight before any work
    assert "run" in subs
    assert env.store.load(PID).status is ProjectStatus.DONE


@pytest.mark.parametrize("mode", ["down", "missing"])
def test_run_fails_closed_when_docker_unavailable(env: Env, mode: str) -> None:
    if mode == "missing":
        env.sandbox(f'[sandbox]\nmode = "docker"\ndocker = "{env.tmp / "nope"}"\n')
        log = env.tmp / "unused.log"
    else:
        log = env.docker(mode)
    env.project()
    env.provider.push(_calls(("write_file", {"path": "a.txt", "content": "ok"})), "done")
    before = env.store.load(PID)
    r = env.cli("run", PID)
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert "Docker is not usable" in r.output and "nothing was run" in r.output
    assert "run" not in subcommands(log)
    # nothing spent, nothing changed
    assert env.provider.remaining == 2
    assert env.store.load(PID) == before
    assert not (env.store.files_dir(PID) / "a.txt").exists()


def test_bench_run_fails_closed_when_docker_unavailable(env: Env) -> None:
    env.docker("down")
    results = env.tmp / "results"
    r = env.cli(
        "bench", "run", "cli-word-count", "--suite", str(BENCH_DIR), "--results", str(results)
    )
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert "Docker is not usable" in r.output
    assert not env.store.root.exists() or env.store.list_projects() == []
    assert not results.exists() or not any(results.iterdir())


def test_local_mode_runs_no_preflight(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(sandbox: DockerSandbox) -> str:
        raise AssertionError("no docker check in local mode")

    monkeypatch.setattr(cli, "docker_check", boom)
    env.project()
    env.provider.push(_calls(("write_file", {"path": "a.txt", "content": "ok"})), "done")
    r = env.cli("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "sandbox: local" in r.output


def test_status_shows_sandbox_mode(env: Env) -> None:
    env.project()
    r = env.cli("status", PID)
    assert r.exit_code == 0 and "sandbox: local" in r.output
    assert json.loads(env.cli("status", PID, "--json").output)["sandbox"] == {"mode": "local"}
    log = env.docker("down")
    r = env.cli("status", PID)
    assert r.exit_code == 0, r.output
    assert "sandbox: docker (image automa/test:1" in r.output
    data = json.loads(env.cli("status", PID, "--json").output)["sandbox"]
    assert data["mode"] == "docker" and data["image"] == "automa/test:1"
    listing = env.cli("status")
    assert listing.exit_code == 0 and "sandbox: docker" in listing.output
    assert docker_calls(log) == []  # status is read-only: it never contacts Docker
