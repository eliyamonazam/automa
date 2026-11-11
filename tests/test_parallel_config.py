"""Parallel execution configuration and CLI (``[orchestrator] max_parallel``, ``--parallel``; D36).

Local only: ``FakeProvider`` behind the real gateway/guard/ledger, orchestrator,
agents, tools and store (the ``Env`` of ``test_cli_run``). Responses are routed per
task so the tests do not depend on thread scheduling. No network, no real model calls.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from automa import cli, orchestrator
from automa.config import (
    MAX_PARALLEL,
    ConfigError,
    OrchestratorConfig,
    describe_parallel,
    load_config,
)
from automa.llm import ChatRequest, ChatResponse
from automa.orchestrator import OrchestratorSettings
from automa.project import ProjectStatus
from test_cli_run import PID, Env, calls, task_dict, write
from test_parallel import WAIT, Router

BASE = 'workspace = "{ws}"\n'


def cfg(tmp_path: Path, body: str) -> Any:
    path = tmp_path / "c.toml"
    path.write_text(BASE.format(ws=(tmp_path / "ws").as_posix()) + body, encoding="utf-8")
    return load_config(path, env_file=None, environ={})


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


def set_parallel(env: Env, value: int) -> None:
    text = env.config.read_text(encoding="utf-8")
    env.config.write_text(text + f"\n[orchestrator]\nmax_parallel = {value}\n", encoding="utf-8")


def route(env: Env, scripts: dict[str, list[Any]]) -> Router:
    router = Router(scripts)
    env.script(*([router] * router.total()))
    return router


def started(env: Env) -> list[dict[str, Any]]:
    return [e.data for e in env.store.events(PID).read() if e.type == "orchestrator.started"]


# --- [orchestrator] in automa.toml ----------------------------------------------------


def test_default_is_sequential_without_section(tmp_path: Path) -> None:
    c = cfg(tmp_path, "")
    assert c.orchestrator == OrchestratorConfig()
    assert c.orchestrator.max_parallel == 1
    assert c.orchestrator.describe() == "1 (sequential)"


@pytest.mark.parametrize("value", [1, 2, MAX_PARALLEL])
def test_max_parallel_is_loaded(tmp_path: Path, value: int) -> None:
    assert cfg(tmp_path, f"[orchestrator]\nmax_parallel = {value}\n").orchestrator.max_parallel == (
        value
    )


@pytest.mark.parametrize(
    "body",
    [
        "max_parallel = 0",
        "max_parallel = -1",
        f"max_parallel = {MAX_PARALLEL + 1}",
        "max_parallel = 1000",
        "max_parallel = 1.5",
        'max_parallel = "many"',
        "max_attempts = 3",  # unknown keys are rejected, not ignored
    ],
)
def test_invalid_orchestrator_settings_are_rejected(tmp_path: Path, body: str) -> None:
    with pytest.raises(ConfigError, match="orchestrator"):
        cfg(tmp_path, f"[orchestrator]\n{body}\n")


def test_secret_like_key_in_orchestrator_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="secrets are not allowed"):
        cfg(tmp_path, '[orchestrator]\ntoken = "sk-test-not-a-real-key"\n')


def test_orchestrator_config_is_frozen() -> None:
    oc = OrchestratorConfig(max_parallel=2)
    with pytest.raises(ValueError):
        oc.max_parallel = 3  # type: ignore[misc]


def test_bounds_match_orchestrator_settings() -> None:
    assert orchestrator.MAX_PARALLEL is MAX_PARALLEL
    for value in (0, 1, MAX_PARALLEL, MAX_PARALLEL + 1):
        ok_config = ok_settings = True
        try:
            OrchestratorConfig(max_parallel=value)
        except ValueError:
            ok_config = False
        try:
            OrchestratorSettings(max_parallel=value)
        except ValueError:
            ok_settings = False
        assert ok_config == ok_settings


def test_describe_parallel_warns_about_shared_files() -> None:
    assert describe_parallel(1) == "1 (sequential)"
    text = describe_parallel(3)
    assert "up to 3 tasks" in text and "files/" in text and "cannot be written by another" in text
    assert "run_command" in text  # the unchecked path is named


def test_example_config_documents_the_section() -> None:
    example = Path(__file__).resolve().parents[1] / "automa.toml.example"
    text = example.read_text(encoding="utf-8")
    assert "[orchestrator]" in text and "max_parallel = 1" in text
    # the example (with its placeholder prices) still loads, defaults intact
    assert load_config(example, env_file=None, environ={}).orchestrator.max_parallel == 1


# --- automa run --------------------------------------------------------------------


def test_run_default_is_sequential(env: Env) -> None:
    env.project(task_dict("a"), task_dict("b"))
    env.script(*write("a"), *write("b"))
    r = env.run(PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "parallel: 1 (sequential)" in r.output
    assert "max_parallel" not in started(env)[0]
    types = [e.type for e in env.store.events(PID).read()]
    assert "orchestrator.scheduled" not in types


def barrier_scripts(*tids: str) -> dict[str, list[Any]]:
    """Each task's first call only returns once all ``tids`` are in a call at once."""
    barrier = threading.Barrier(len(tids), timeout=WAIT)

    def first(tid: str) -> Any:
        def respond(_: ChatRequest) -> ChatResponse:
            barrier.wait()
            return calls(("write_file", {"path": f"{tid}.txt", "content": "ok"}))

        return respond

    return {tid: [first(tid), "done"] for tid in tids}


def test_config_max_parallel_runs_tasks_concurrently(env: Env) -> None:
    set_parallel(env, 2)
    env.project(task_dict("a"), task_dict("b"))
    route(env, barrier_scripts("a", "b"))
    r = env.run(PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "parallel: up to 2 tasks at a time" in r.output
    assert started(env)[0]["max_parallel"] == 2
    assert env.status() is ProjectStatus.DONE


def test_parallel_option_overrides_config(env: Env) -> None:
    set_parallel(env, 1)
    env.project(task_dict("a"), task_dict("b"), task_dict("c"))
    route(env, barrier_scripts("a", "b", "c"))
    r = env.run(PID, "--parallel", "3")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "parallel: up to 3 tasks at a time" in r.output
    assert started(env)[0]["max_parallel"] == 3


def test_parallel_option_can_force_sequential(env: Env) -> None:
    set_parallel(env, 4)
    env.project(task_dict("a"), task_dict("b"))
    env.script(*write("a"), *write("b"))
    r = env.run(PID, "--parallel", "1")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "parallel: 1 (sequential)" in r.output
    assert "max_parallel" not in started(env)[0]


def test_parallel_and_max_attempts_combine(env: Env) -> None:
    env.project(task_dict("a"), task_dict("b"))
    router = route(
        env,
        {"a": [*write("a", "no"), *write("a", "no")], "b": write("b")},
    )
    r = env.run(PID, "--parallel", "2", "--max-attempts", "2")
    assert r.exit_code != cli.EXIT_OK
    assert "failed: a" in r.output and "passed: b" in r.output
    assert router.total() == 0  # exactly two attempts for a, none more


@pytest.mark.parametrize("value", ["0", "-1", str(MAX_PARALLEL + 1), "100"])
def test_run_invalid_parallel_touches_nothing(env: Env, value: str) -> None:
    env.project(task_dict("a"))
    r = env.run(PID, "--parallel", value)
    assert r.exit_code == cli.EXIT_USER
    assert "invalid --parallel" in r.output
    assert env.factory_calls == 0 and env.provider.requests == []
    assert env.status() is ProjectStatus.NEW
    assert not (env.ws / cli.LEDGER_FILENAME).exists()


def test_invalid_config_value_fails_before_any_call(env: Env) -> None:
    set_parallel(env, MAX_PARALLEL + 1)
    env.project(task_dict("a"))
    r = env.run(PID)
    assert r.exit_code != cli.EXIT_OK
    assert "orchestrator" in r.output
    assert env.factory_calls == 0 and env.provider.requests == []
    assert env.status() is ProjectStatus.NEW


def test_parallel_budget_cap_still_holds(env: Env) -> None:
    """A per-call cap below one call's worst case stops everything, in parallel too."""
    env.write_config(per_call=0.000001)
    set_parallel(env, 2)
    env.project(task_dict("a"), task_dict("b"))
    r = env.run(PID)
    assert r.exit_code == cli.EXIT_BUDGET, r.output
    assert env.provider.requests == []
    assert "$0.0000 this run" in r.output
    assert env.status() is ProjectStatus.PAUSED


# --- automa status -----------------------------------------------------------------


def status(env: Env, *args: str) -> Any:
    return CliRunner().invoke(cli.app, ["status", *args, "--config", str(env.config)])


def test_status_shows_parallel_setting(env: Env) -> None:
    env.project(task_dict("a"))
    r = status(env)
    assert r.exit_code == 0, r.output
    assert "parallel: 1 (sequential)" in r.output
    set_parallel(env, 3)
    for args in ((), (PID,)):
        r = status(env, *args)
        assert r.exit_code == 0, r.output
        assert "parallel: up to 3 tasks at a time" in r.output


def test_status_json_includes_max_parallel(env: Env) -> None:
    env.project(task_dict("a"))
    set_parallel(env, 2)
    r = status(env, PID, "--json")
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["max_parallel"] == 2
