"""Safe mode for a member's projects (M15, D185): ``run_command`` only in the Docker
sandbox (never on the host, refused with a clear message when Docker is not usable)
and no automatic approvals.

No real Docker: a fake ``docker`` client (from ``test_sandbox_config``) answers
``docker version`` and emulates ``docker run`` while recording its argv. No real
model calls: FakeProvider only.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from automa import cli
from automa.budget.accounts import config_for_project
from automa.config import AutomaConfig, SandboxMode
from automa.member_safety import is_member_safe, member_config
from automa.orchestrator import Orchestrator
from automa.project import ProjectStatus
from automa.tools import DockerSandbox, RunCommand, SandboxError
from automa.users import UserRole, UserStore
from automa.webui import i18n
from automa.webui import runs as wr
from test_sandbox_config import (
    PID,
    TASK,
    Env,
    _calls,
    docker_calls,
    make_fake_docker,
    subcommands,
)
from test_webui_runs import TOKEN, FakePreflight
from test_webui_runs import Env as WebEnv

PASSWORD = "a-long-password"  # noqa: S105 - test value


def _config(**extra: Any) -> AutomaConfig:
    data: dict[str, Any] = {
        "limits": {"project_usd": 5.0, "daily_usd": 2.0, "per_call_usd": 0.25},
        "workspace": "ws",
    }
    data.update(extra)
    return AutomaConfig.model_validate(data)


def _accounts(ws: Path) -> None:
    store = UserStore(ws, scrypt_n=2**14)
    store.add("ilia", PASSWORD, UserRole.OWNER)
    store.add("sara", PASSWORD, UserRole.MEMBER)


# -- member_config (pure) -----------------------------------------------------------------


def test_member_config_forces_docker_from_local_mode() -> None:
    config = _config()
    assert config.sandbox.mode is SandboxMode.LOCAL
    safe = member_config(config)
    assert safe.sandbox.mode is SandboxMode.DOCKER
    sandbox = safe.sandbox.docker_sandbox()
    assert sandbox is not None and sandbox == DockerSandbox()  # the safe defaults
    assert config.sandbox.mode is SandboxMode.LOCAL  # the original is untouched
    assert is_member_safe(safe) and not is_member_safe(config)


def test_member_config_keeps_the_owners_docker_settings() -> None:
    config = _config(sandbox={"mode": "docker", "image": "automa/test:1", "memory_mb": 256})
    safe = member_config(config)
    assert safe is config  # already safe: nothing to change
    sandbox = safe.sandbox.docker_sandbox()
    assert sandbox is not None and sandbox.image == "automa/test:1" and sandbox.memory_mb == 256


def test_member_config_turns_automatic_approvals_off() -> None:
    config = _config(web={"auto_approve_reads": True, "max_auto_reads_per_task": 7})
    safe = member_config(config)
    assert not safe.web.auto_approve_reads
    assert safe.web.max_auto_reads_per_task == 7
    assert config.web.auto_approve_reads  # the owner's own projects keep their setting


def test_member_config_changes_nothing_else() -> None:
    config = _config(web={"auto_approve_reads": True})
    safe = member_config(config)
    before = config.model_dump(exclude={"sandbox", "web"})
    assert safe.model_dump(exclude={"sandbox", "web"}) == before


def test_member_config_is_idempotent() -> None:
    once = member_config(_config(web={"auto_approve_reads": True}))
    assert member_config(once) is once


# -- config_for_project ------------------------------------------------------------------


def test_member_project_gets_the_safe_mode(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _accounts(ws)
    config = _config(workspace=str(ws), web={"auto_approve_reads": True})
    member, budget = config_for_project(config, "sara")
    assert budget.member and is_member_safe(member)
    assert member.limits.project_usd == 1.0  # the stricter cap of D183 is still there


@pytest.mark.parametrize("owner", [None, "ilia"])
def test_owner_and_cli_projects_keep_the_configuration(tmp_path: Path, owner: str | None) -> None:
    ws = tmp_path / "ws"
    _accounts(ws)
    config = _config(workspace=str(ws), web={"auto_approve_reads": True})
    same, budget = config_for_project(config, owner)
    assert not budget.member
    assert same.sandbox.mode is SandboxMode.LOCAL and same.web.auto_approve_reads


@pytest.mark.parametrize("setup", ["no_users_file", "deleted_account", "open_permissions"])
def test_unknown_owner_fails_closed_into_the_safe_mode(tmp_path: Path, setup: str) -> None:
    ws = tmp_path / "ws"
    owner = "ilia"
    if setup != "no_users_file":
        _accounts(ws)
    if setup == "deleted_account":
        owner = "gone"
    if setup == "open_permissions":
        (ws / "users.json").chmod(0o644)  # unreadable for the store
    config, budget = config_for_project(_config(workspace=str(ws)), owner)
    assert budget.member and is_member_safe(config)


def test_orchestrator_refuses_host_commands_for_a_member_config(tmp_path: Path) -> None:
    """Defence in depth: with the member config, a run_command without a sandbox
    is refused by the orchestrator itself, before any task runs."""
    from automa.agents import VerifierAgent, WorkerAgent
    from automa.budget import BudgetGuard, CostLedger
    from automa.config import ConfigError
    from automa.llm import FakeProvider, LLMGateway
    from automa.project import ProjectStore
    from automa.roles import RoleRegistry
    from automa.tools import default_tools

    config = member_config(_config(workspace=str(tmp_path / "ws")))
    store = ProjectStore.for_workspace(tmp_path / "ws")
    with CostLedger(":memory:") as ledger:
        gateway = LLMGateway(config, FakeProvider(), guard=BudgetGuard(config.limits, ledger))
        with pytest.raises(ConfigError, match="refusing to run commands on the host"):
            Orchestrator(
                store,
                WorkerAgent(gateway),
                VerifierAgent(gateway),
                RoleRegistry.builtin(tools=default_tools()),
                config,
                tools=lambda: [RunCommand()],  # no sandbox
            )._make_tools()


# -- automa run ----------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


def _docker_on_path(env: Env, monkeypatch: pytest.MonkeyPatch, mode: str = "ok") -> Path:
    """A fake client called plain ``docker`` first on PATH (the default sandbox's client)."""
    script, log = make_fake_docker(env.tmp, mode)
    bin_dir = env.tmp / f"bin-{mode}"
    bin_dir.mkdir()
    (bin_dir / "docker").symlink_to(script)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return log


def _project(env: Env, owner: str | None) -> None:
    from automa.plan import Plan

    _accounts(env.ws)
    env.store.create("test goal", project_id=PID, owner=owner)
    env.store.save_plan(PID, Plan.model_validate({"goal": "test goal", "tasks": [TASK]}))


def test_member_run_executes_commands_only_in_docker(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _docker_on_path(env, monkeypatch)
    _project(env, "sara")  # automa.toml says local mode
    env.provider.push(_calls(("write_file", {"path": "a.txt", "content": "ok"})), "done")
    r = env.cli("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "sandbox: docker" in r.output
    subs = subcommands(log)
    assert subs[0] == "version" and "run" in subs  # the check command ran in a container
    run = next(c["argv"] for c in docker_calls(log) if c["argv"][0] == "run")
    assert "--network" in run and run[run.index("--network") + 1] == "none"
    assert env.store.load(PID).status is ProjectStatus.DONE


def test_owner_run_with_local_mode_is_unchanged(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    log = _docker_on_path(env, monkeypatch)
    _project(env, "ilia")
    env.provider.push(_calls(("write_file", {"path": "a.txt", "content": "ok"})), "done")
    r = env.cli("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "sandbox: local" in r.output
    assert docker_calls(log) == []


@pytest.mark.parametrize("mode", ["down", "missing"])
def test_member_run_is_refused_without_docker(
    env: Env, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    if mode == "missing":
        empty = env.tmp / "empty-bin"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))
        log = env.tmp / "unused.log"
    else:
        log = _docker_on_path(env, monkeypatch, mode)
    _project(env, "sara")
    env.provider.push(_calls(("write_file", {"path": "a.txt", "content": "ok"})), "done")
    before = env.store.load(PID)
    r = env.cli("run", PID)
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert "member account" in r.output and "nothing was run" in r.output
    assert 'mode = "local"' not in r.output  # never suggests running on the host
    assert "run" not in subcommands(log)
    assert env.provider.remaining == 2  # no model call, nothing spent
    assert env.store.load(PID) == before
    assert not (env.store.files_dir(PID) / "a.txt").exists()


def test_member_run_has_no_automatic_reads(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """``auto_approve_reads`` of automa.toml never reaches a member's executor."""
    _docker_on_path(env, monkeypatch)
    env.config.write_text(
        env.config.read_text(encoding="utf-8") + "\n[web]\nauto_approve_reads = true\n",
        encoding="utf-8",
    )
    _project(env, "sara")
    seen: list[AutomaConfig] = []
    original = Orchestrator.__init__

    def spy(self: Orchestrator, *args: Any, **kwargs: Any) -> None:
        seen.append(args[4] if len(args) > 4 else kwargs["config"])
        original(self, *args, **kwargs)

    monkeypatch.setattr(Orchestrator, "__init__", spy)
    env.provider.push(_calls(("write_file", {"path": "a.txt", "content": "ok"})), "done")
    r = env.cli("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert seen and not seen[0].web.auto_approve_reads
    assert seen[0].sandbox.mode is SandboxMode.DOCKER


# -- web: the preflight of the CLI ---------------------------------------------------------


def _preflight(tmp_path: Path, **extra: Any) -> tuple[Any, Path]:
    from automa.plan import Plan
    from automa.project import ProjectStore

    ws = tmp_path / "ws"
    _accounts(ws)
    store = ProjectStore.for_workspace(ws)
    for pid, owner in (("p-sara", "sara"), ("p-ilia", "ilia"), ("p-cli", None)):
        store.create("g", project_id=pid, owner=owner)
        store.save_plan(pid, Plan.model_validate({"goal": "g", "tasks": [TASK]}))
    return cli._WebRunPreflight(_config(workspace=str(ws), **extra), None), ws


def _docker(monkeypatch: pytest.MonkeyPatch, ok: bool) -> list[DockerSandbox]:
    calls: list[DockerSandbox] = []

    def check(sandbox: DockerSandbox) -> str:
        calls.append(sandbox)
        if not ok:
            raise SandboxError("docker daemon is not available")
        return "27.0.0"

    monkeypatch.setattr(cli, "docker_check", check)
    return calls


def test_web_preflight_refuses_member_project_without_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _docker(monkeypatch, ok=False)
    preflight, _ = _preflight(tmp_path)
    assert isinstance(preflight, wr.SandboxPreflight)
    assert preflight.sandbox_refusal("p-sara") == "member_needs_docker"
    assert calls and calls[0] == DockerSandbox()
    calls.clear()
    assert preflight.sandbox_refusal("p-ilia") is None  # local mode: no check at all
    assert preflight.sandbox_refusal("p-cli") is None
    assert calls == []


def test_web_preflight_lets_member_project_start_with_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _docker(monkeypatch, ok=True)
    preflight, _ = _preflight(tmp_path)
    assert preflight.sandbox_refusal("p-sara") is None


def test_web_preflight_in_docker_mode_checks_every_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _docker(monkeypatch, ok=False)
    preflight, _ = _preflight(tmp_path, sandbox={"mode": "docker", "image": "automa/test:1"})
    assert preflight.sandbox_refusal("p-ilia") == "docker_unavailable"
    assert preflight.sandbox_refusal("p-cli") == "docker_unavailable"
    assert preflight.sandbox_refusal("p-sara") == "member_needs_docker"
    assert {c.image for c in calls} == {"automa/test:1"}  # the owner's settings kept


def test_web_preflight_unknown_project_is_left_to_the_start_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _docker(monkeypatch, ok=False)
    preflight, _ = _preflight(tmp_path)
    assert preflight.sandbox_refusal("p-nope") is None
    assert calls == []


# -- web: the start and retry buttons -------------------------------------------------------


class SandboxFake(FakePreflight):
    def __init__(self) -> None:
        super().__init__()
        self.refusal: str | None = "member_needs_docker"
        self.asked: list[str] = []

    def sandbox_refusal(self, project_id: str) -> str | None:
        self.asked.append(project_id)
        return self.refusal


@pytest.fixture
def web(tmp_path: Path) -> Iterator[WebEnv]:
    e = WebEnv(tmp_path)
    e.preflight.__class__ = SandboxFake
    e.preflight.refusal = "member_needs_docker"  # type: ignore[attr-defined]
    e.preflight.asked = []  # type: ignore[attr-defined]
    yield e
    for _, _, proc in e.spawned:
        if proc.code is None:
            proc.finish(0)


def test_start_is_refused_with_a_clear_persian_sentence(web: WebEnv) -> None:
    pid = web.project()
    r = web.start(pid)
    assert r.status == 409
    body = r.body.decode("utf-8")
    assert "Docker" in body
    assert web.spawned == [] and web.preflight.commands == []
    assert web.store.load(pid).status is not ProjectStatus.RUNNING
    assert web.preflight.asked == [pid]  # type: ignore[attr-defined]


def test_start_refusal_text_is_in_persian() -> None:
    fa = i18n.t("refusal.member_needs_docker", "fa")
    assert "Docker" in fa and "صاحب Automa" in fa and "چیزی اجرا نشد" in fa
    assert "محلی" not in fa and "local" not in fa  # never offers running on the host
    assert "چیزی اجرا نشد" in i18n.t("refusal.docker_unavailable", "fa")


@pytest.mark.parametrize("code", ["docker_unavailable", "something-else"])
def test_other_refusal_codes_still_refuse(web: WebEnv, code: str) -> None:
    web.preflight.refusal = code  # type: ignore[attr-defined]
    pid = web.project()
    assert web.start(pid).status == 409
    assert web.spawned == []


def test_start_with_docker_available_spawns(web: WebEnv) -> None:
    web.preflight.refusal = None  # type: ignore[attr-defined]
    pid = web.project()
    r = web.start(pid)
    assert r.status in (303, 200), r.status
    assert len(web.spawned) == 1


def test_sandbox_check_comes_before_the_cost_question(web: WebEnv) -> None:
    web.preflight.confirm = True
    pid = web.project()
    r = web.start(pid)  # no confirmation box ticked
    assert r.status == 409 and "Docker" in r.body.decode("utf-8")


def test_preflight_without_sandbox_part_is_unchanged(tmp_path: Path) -> None:
    e = WebEnv(tmp_path)
    assert not isinstance(e.preflight, wr.SandboxPreflight)
    pid = e.project()
    try:
        assert e.start(pid).status in (303, 200)
        assert len(e.spawned) == 1
    finally:
        for _, _, proc in e.spawned:
            proc.finish(0)


def test_retry_is_refused_before_anything_is_granted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Both(SandboxFake):
        def __init__(self) -> None:
            super().__init__()
            self.granted: list[str] = []

        def grant_retry(self, project_id: str) -> list[str]:
            self.granted.append(project_id)
            return ["build"]

    monkeypatch.setattr(wr, "can_retry", lambda store, pid: True)
    e = WebEnv(tmp_path)
    both = Both()
    background = wr.BackgroundRuns(e.store, both, spawn=e._spawn)
    pid = e.project(status=ProjectStatus.FAILED)
    assert background.can_retry(pid)
    with pytest.raises(wr.ActionRefused) as info:
        background.retry(pid, {"csrf": TOKEN, "confirm": wr.CONFIRM_VALUE})
    assert info.value.code == "member_needs_docker"
    assert both.granted == [] and e.spawned == []
    both.refusal = None
    background.retry(pid, {"csrf": TOKEN, "confirm": wr.CONFIRM_VALUE})
    assert both.granted == [pid] and len(e.spawned) == 1
    e.spawned[0][2].finish(0)
