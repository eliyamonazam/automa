"""Tests for stopping a run cleanly between steps (web UI "Stop", D126).

Store (``request_stop``/``stop_requested``/``clear_stop_request``), the Orchestrator
(sequential and parallel: the stop is honoured only between steps, the project is
paused with reason ``user`` and the request is always removed at the end of a run),
``automa run`` (exit code 12) and the web UI (``POST /projects/<pid>/stop`` with the
same Host / same-origin / CSRF checks). FakeProvider only; no network, no real model.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.benchmarks.results import BenchmarkStop
from automa.config import AutomaConfig
from automa.llm import ChatRequest, ChatResponse, FakeProvider
from automa.orchestrator import StopReason
from automa.plan import Plan, TaskStatus
from automa.project import ProjectNotFound, ProjectStatus, ProjectStore
from automa.project.store import STOP_REQUEST_FILE
from automa.webui import runs as wr
from test_orchestrator import PID, Rig, make_plan, task_dict, write
from test_parallel import routed_rig, statuses
from test_webui_runs import TOKEN, Env, _body

# --- store ---------------------------------------------------------------------------


def _store(tmp_path: Path) -> ProjectStore:
    store = ProjectStore.for_workspace(tmp_path / "ws")
    store.create("g", project_id=PID)
    return store


def test_request_stop_writes_once_and_clears(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert not store.stop_requested(PID)
    assert store.request_stop(PID, via="web") is True
    path = store.project_dir(PID) / STOP_REQUEST_FILE
    assert path.is_file() and store.stop_requested(PID)
    assert '"via": "web"' in path.read_text()
    assert store.request_stop(PID, via="web") is False  # one request is enough
    assert store.clear_stop_request(PID) is True
    assert not store.stop_requested(PID)
    assert store.clear_stop_request(PID) is False


def test_request_stop_unknown_project(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ProjectNotFound):
        store.request_stop("nope", via="web")


def test_symlink_counts_as_request_and_clear_keeps_target(tmp_path: Path) -> None:
    store = _store(tmp_path)
    target = tmp_path / "target.txt"
    target.write_text("keep")
    link = store.project_dir(PID) / STOP_REQUEST_FILE
    link.symlink_to(target)
    assert store.stop_requested(PID)  # stopping is always the safe side
    assert store.request_stop(PID, via="web") is False  # never written through a link
    assert target.read_text() == "keep"
    assert store.clear_stop_request(PID) is True
    assert not os.path.lexists(link) and target.read_text() == "keep"


def test_dangling_symlink_counts_as_request(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.project_dir(PID) / STOP_REQUEST_FILE).symlink_to(tmp_path / "missing")
    assert store.stop_requested(PID)


# --- orchestrator (sequential) -------------------------------------------------------


def _two_tasks() -> Plan:
    return make_plan(task_dict("a"), task_dict("b", deps=["a"]))


def test_stop_before_run_makes_no_call(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [*write("a"), *write("b")], _two_tasks())
    rig.store.request_stop(PID, via="web")
    report = rig.run()
    assert report.status is ProjectStatus.PAUSED
    assert report.reason is StopReason.USER
    assert report.task_id == "a" and report.attempts_run == 0
    assert rig.provider.requests == []
    assert rig.project_status() is ProjectStatus.PAUSED
    assert not rig.store.stop_requested(PID)
    paused = [e for e in rig.store.events(PID).read() if e.type == "orchestrator.paused"]
    assert paused[-1].data["reason"] == "user"


def _stop_on_first_call(rig: Rig, script: list[Any]) -> list[Any]:
    """The first response also writes a stop request (as the web UI would)."""
    first = script[0]

    def respond(request: ChatRequest) -> ChatResponse | str:
        rig.store.request_stop(PID, via="test")
        return first

    return [respond, *script[1:]]


def test_stop_during_step_finishes_step_then_pauses(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [], _two_tasks())
    rig.provider = FakeProvider(_stop_on_first_call(rig, [*write("a"), *write("b")]))
    rig.rebuild()
    report = rig.run()
    # Task a's step ran to the end (worker + verifier), nothing was cut off.
    assert rig.state("a").status is TaskStatus.PASSED
    assert rig.state("b").status is TaskStatus.PENDING
    assert report.reason is StopReason.USER and report.task_id == "b"
    assert report.attempts_run == 1 and report.passed == ["a"]
    assert not rig.store.stop_requested(PID)
    used = len(rig.provider.requests)
    # Resume: the old request is gone, so the run continues to the end.
    report = rig.run()
    assert report.status is ProjectStatus.DONE and report.reason is StopReason.COMPLETED
    assert len(rig.provider.requests) > used


def test_request_during_last_step_is_cleared_on_finish(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [], make_plan(task_dict("a")))
    rig.provider = FakeProvider(_stop_on_first_call(rig, write("a")))
    rig.rebuild()
    report = rig.run()
    assert report.status is ProjectStatus.DONE  # nothing left to stop
    assert not rig.store.stop_requested(PID)  # never stops a later run


def test_request_cleared_even_when_run_raises(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [], make_plan(task_dict("a")))

    def boom(request: ChatRequest) -> ChatResponse:
        rig.store.request_stop(PID, via="test")
        raise KeyboardInterrupt

    rig.provider = FakeProvider([boom])
    rig.rebuild()
    with pytest.raises(KeyboardInterrupt):
        rig.run()
    assert rig.project_status() is ProjectStatus.PAUSED
    assert not rig.store.stop_requested(PID)


def test_unreadable_stop_state_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = Rig(tmp_path, write("a"), make_plan(task_dict("a")))

    def broken(project_id: str) -> bool:
        raise OSError("disk gone")

    monkeypatch.setattr(rig.store, "stop_requested", broken)
    report = rig.run()
    assert report.reason is StopReason.USER
    assert rig.provider.requests == []


def test_no_request_runs_normally(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [*write("a"), *write("b")], _two_tasks())
    report = rig.run()
    assert report.status is ProjectStatus.DONE
    assert "orchestrator.paused" not in rig.event_types()


# --- orchestrator (parallel) ---------------------------------------------------------


def _three_independent() -> Plan:
    return make_plan(task_dict("a"), task_dict("b"), task_dict("c"))


def test_parallel_stop_before_run_makes_no_call(tmp_path: Path) -> None:
    rig = routed_rig(tmp_path, _three_independent(), {t: write(t) for t in ("a", "b", "c")})
    rig.store.request_stop(PID, via="web")
    report = rig.run(max_parallel=2)
    assert report.reason is StopReason.USER and report.status is ProjectStatus.PAUSED
    assert rig.provider.requests == []
    assert not rig.store.stop_requested(PID)


def test_parallel_stop_lets_running_steps_finish_and_schedules_nothing_new(
    tmp_path: Path,
) -> None:
    scripts: dict[str, list[Any]] = {t: write(t) for t in ("a", "b", "c")}
    rig = routed_rig(tmp_path, _three_independent(), scripts)
    first = scripts["a"][0]

    def stop_then(request: ChatRequest) -> ChatResponse:
        rig.store.request_stop(PID, via="test")
        return first

    scripts["a"][0] = stop_then
    rig.router.scripts["a"][0] = stop_then  # type: ignore[attr-defined]
    report = rig.run(max_parallel=2)
    assert report.reason is StopReason.USER
    st = statuses(rig)
    assert st["a"] is TaskStatus.PASSED  # its step was not cut off
    assert st["c"] is TaskStatus.PENDING  # never scheduled after the request
    assert "c" not in rig.router.order  # type: ignore[attr-defined]
    assert st["b"] in (TaskStatus.PASSED, TaskStatus.PENDING)
    assert not rig.store.stop_requested(PID)
    report = rig.run(max_parallel=2)
    assert report.status is ProjectStatus.DONE


# --- automa run (CLI) ----------------------------------------------------------------

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
[limits]
project_usd = 5.0
daily_usd = 2.0
per_call_usd = 0.25
"""


def _cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, ProjectStore]:
    ws = tmp_path / "ws"
    cfg = tmp_path / "automa.toml"
    cfg.write_text(TOML.format(ws=ws.as_posix()), encoding="utf-8")
    for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    store = ProjectStore.for_workspace(ws)
    store.create("test goal", project_id=PID)
    store.save_plan(PID, make_plan(task_dict("a")))
    return cfg, store


def test_cli_exit_code_for_user_stop() -> None:
    assert cli.RUN_EXIT_CODES[StopReason.USER] == cli.EXIT_STOPPED == 12
    assert BenchmarkStop.USER.value == StopReason.USER.value
    assert "stopped on request" in wr.exit_meaning(12)
    assert "paused" in wr.exit_meaning(12)


def test_cli_run_with_stop_request_exits_12(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, store = _cli_env(tmp_path, monkeypatch)
    provider = FakeProvider(write("a"))

    def factory(config: AutomaConfig) -> FakeProvider:
        return provider

    monkeypatch.setattr(cli, "make_provider", factory)
    store.request_stop(PID, via="web")
    r = CliRunner().invoke(cli.app, ["run", PID, "--config", str(cfg)])
    assert r.exit_code == cli.EXIT_STOPPED, r.output
    assert "paused (user)" in r.output
    assert "stopped on request" in r.output
    assert f"then: automa run {PID}" in r.output
    assert provider.requests == []
    assert store.load(PID).status is ProjectStatus.PAUSED
    assert not store.stop_requested(PID)
    # Resuming in the terminal finishes the project.
    r = CliRunner().invoke(cli.app, ["run", PID, "--config", str(cfg)])
    assert r.exit_code == cli.EXIT_OK, r.output
    assert store.load(PID).status is ProjectStatus.DONE


def test_real_child_automa_run_honours_stop_without_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, store = _cli_env(tmp_path, monkeypatch)
    # A key so the child gets past configuration; an unreachable gateway so that even
    # a bug could never reach a real model API.
    monkeypatch.setenv("AUTOMA_API_KEY", "test-not-a-key")
    monkeypatch.setenv("AUTOMA_BASE_URL", "http://127.0.0.1:9/v1")
    store.request_stop(PID, via="web")
    log = tmp_path / "run.log"
    argv = [sys.executable, "-m", "automa", "run", PID, "--config", str(cfg)]
    proc = wr.popen_spawn(argv, log)
    assert proc.wait() == cli.EXIT_STOPPED, log.read_text()
    assert "paused (user)" in log.read_text()
    assert store.load(PID).status is ProjectStatus.PAUSED
    assert not store.stop_requested(PID)


# --- web UI --------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path) -> Any:
    e = Env(tmp_path)
    yield e
    for _, _, proc in e.spawned:
        if proc.code is None:
            proc.finish(0)


def _stop(env: Env, pid: str, **extra: str) -> Any:
    return env.post(f"/projects/{pid}/stop", {"csrf": TOKEN, **extra})


def _started(env: Env) -> str:
    pid = env.project()
    assert env.start(pid).status == 303
    return pid


def test_stop_button_only_while_running(env: Env) -> None:
    pid = env.project()
    assert f'action="/projects/{pid}/stop"' not in _body(env.get(f"/projects/{pid}"))
    assert env.start(pid).status == 303
    body = _body(env.get(f"/projects/{pid}"))
    assert f'action="/projects/{pid}/stop"' in body
    assert f'name="csrf" value="{TOKEN}"' in body
    assert "<script" not in body
    env.spawned[0][2].finish(0)
    env.wait_event(pid, wr.FINISHED_EVENT)
    assert f'action="/projects/{pid}/stop"' not in _body(env.get(f"/projects/{pid}"))


def test_stop_writes_request_and_event(env: Env) -> None:
    pid = _started(env)
    r = _stop(env, pid)
    assert r.status == 303
    assert dict(r.headers)["Location"] == f"/projects/{pid}"
    assert env.store.stop_requested(pid)
    assert env.wait_event(pid, wr.STOP_EVENT)["data"] == {"log": env.spawned[0][1].name}
    body = _body(env.get(f"/projects/{pid}"))
    assert "stopping: the run pauses after its current step" in body
    assert f'action="/projects/{pid}/stop"' not in body  # already asked
    # Asking again changes nothing.
    assert _stop(env, pid).status == 303
    assert sum(1 for e in env.events(pid) if e["type"] == wr.STOP_EVENT) == 1
    # The child is never killed by the web UI.
    assert env.spawned[0][2].code is None


def test_leftover_request_removed_when_child_ends(env: Env) -> None:
    pid = _started(env)
    assert _stop(env, pid).status == 303
    env.spawned[0][2].finish(0)  # ended before it saw the request
    env.wait_event(pid, wr.FINISHED_EVENT)
    for _ in range(200):
        if not env.store.stop_requested(pid):
            break
        time.sleep(0.01)
    assert not env.store.stop_requested(pid)


def test_watcher_keeps_requests_it_did_not_write(env: Env) -> None:
    pid = _started(env)
    env.spawned[0][2].finish(12)
    env.wait_event(pid, wr.FINISHED_EVENT)
    # Not this server's request: left alone (only an Orchestrator run consumes it).
    env.store.request_stop(pid, via="other")
    assert env.store.stop_requested(pid)


def test_stop_without_running_child_refused(env: Env) -> None:
    pid = env.project()
    r = _stop(env, pid)
    assert r.status == 409
    assert "Ctrl+C" in _body(r)
    assert not env.store.stop_requested(pid)


def test_stop_after_child_ended_refused(env: Env) -> None:
    pid = _started(env)
    env.spawned[0][2].finish(0)
    env.wait_event(pid, wr.FINISHED_EVENT)
    assert _stop(env, pid).status == 409
    assert not env.store.stop_requested(pid)


def test_stop_terminal_run_refused(env: Env) -> None:
    pid = env.project(status=ProjectStatus.RUNNING)
    assert _stop(env, pid).status == 409
    assert not env.store.stop_requested(pid)


def test_stop_unknown_project(env: Env) -> None:
    assert _stop(env, "p-nothere").status == 404


@pytest.mark.parametrize("path", ["/projects/NOPE/stop", "/projects/../stop", "/projects//stop"])
def test_stop_bad_path(env: Env, path: str) -> None:
    assert env.post(path, {"csrf": TOKEN}).status == 404


def test_stop_query_string_not_accepted(env: Env) -> None:
    pid = _started(env)
    assert env.post(f"/projects/{pid}/stop?x=1", {"csrf": TOKEN}).status == 404
    assert not env.store.stop_requested(pid)


def test_get_stop_is_not_an_action(env: Env) -> None:
    pid = _started(env)
    assert env.get(f"/projects/{pid}/stop").status == 404
    assert not env.store.stop_requested(pid)


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Content-Type": "text/plain"},
    ],
)
def test_stop_cross_origin_or_bad_type_refused(env: Env, headers: dict[str, str]) -> None:
    pid = _started(env)
    r = env.post(f"/projects/{pid}/stop", {"csrf": TOKEN}, headers=headers)
    assert r.status in (400, 403, 415)
    assert not env.store.stop_requested(pid)


@pytest.mark.parametrize("token", ["", "x" * 43, TOKEN + "x"])
def test_stop_wrong_token_refused(env: Env, token: str) -> None:
    pid = _started(env)
    assert env.post(f"/projects/{pid}/stop", {"csrf": token}).status == 403
    assert not env.store.stop_requested(pid)


def test_stop_unknown_field_refused(env: Env) -> None:
    pid = _started(env)
    assert _stop(env, pid, confirm="yes").status == 400
    assert not env.store.stop_requested(pid)


def test_stop_wrong_host_refused(env: Env) -> None:
    pid = _started(env)
    hdrs = [
        ("Origin", "http://127.0.0.1:8765"),
        ("Content-Type", "application/x-www-form-urlencoded"),
    ]
    r = env.app.handle(
        "POST", f"/projects/{pid}/stop", "evil.example:8765", hdrs, b"csrf=" + TOKEN.encode()
    )
    assert r.status == 421
    assert not env.store.stop_requested(pid)


def test_read_only_server_has_no_stop(tmp_path: Path) -> None:
    e = Env(tmp_path, actions=False)
    pid = e.project()
    assert _stop(e, pid).status == 405
    assert not e.store.stop_requested(pid)


def test_action_server_without_runs_has_no_stop(tmp_path: Path) -> None:
    e = Env(tmp_path, runs=False)
    pid = e.project()
    assert _stop(e, pid).status == 404
    assert not e.store.stop_requested(pid)


def test_restart_after_stop(env: Env) -> None:
    pid = _started(env)
    assert _stop(env, pid).status == 303
    env.spawned[0][2].finish(12)  # the child paused on request
    env.wait_event(pid, wr.FINISHED_EVENT)
    body = _body(env.get(f"/projects/{pid}"))
    assert "stopped on request" in body
    assert "Start run" in body
    assert env.start(pid).status == 303
    assert len(env.spawned) == 2
