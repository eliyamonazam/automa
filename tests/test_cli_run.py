"""Tests for ``automa run <project>`` (run and resume).

Local only: the provider factory is replaced by a ``FakeProvider``; the real
``LLMGateway`` + ``BudgetGuard`` + ``CostLedger``, ``Orchestrator``, agents, tool
executor and ``ProjectStore`` are used. No network, no real model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.config import AutomaConfig
from automa.llm import (
    ChatResponse,
    FakeProvider,
    FinishReason,
    Message,
    ProviderError,
    ToolCall,
    Usage,
)
from automa.llm import Role as MsgRole
from automa.llm.factory import provider_from_config
from automa.orchestrator import StopReason
from automa.plan import Plan, TaskStatus
from automa.project import ProjectStatus, ProjectStore

PID = "proj-1"

TOML = """
workspace = "{ws}"
[tiers]
{tiers}
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
per_call_usd = {per_call}
"""
ALL_TIERS = 'small = "fake-small"\nmedium = "fake-medium"\nlarge = "fake-large"'


def task_dict(tid: str, *deps: str, **kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": tid,
        "title": f"Write {tid}",
        "role": "writer",
        "depends_on": list(deps),
        "acceptance_criteria": [
            {
                "description": f"{tid}.txt says ok",
                "check": {"type": "file_contains", "path": f"{tid}.txt", "text": "ok"},
            }
        ],
    }
    d.update(kw)
    return d


def calls(*items: tuple[str, dict[str, Any]]) -> ChatResponse:
    tcs = tuple(
        ToolCall(id=f"x{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(items)
    )
    return ChatResponse(
        model="fake",
        message=Message(role=MsgRole.ASSISTANT, content=None, tool_calls=tcs),
        usage=Usage(input_tokens=100, output_tokens=20),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def write(tid: str, content: str = "ok") -> list[Any]:
    """Worker script of one attempt: write ``<tid>.txt`` then finish."""
    return [calls(("write_file", {"path": f"{tid}.txt", "content": content})), "done"]


class Env:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.ws = tmp_path / "ws"
        self.config = tmp_path / "automa.toml"
        self.write_config()
        self.provider = FakeProvider()
        self.factory_calls = 0
        monkeypatch.chdir(tmp_path)
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)

        def factory(config: AutomaConfig) -> FakeProvider:
            self.factory_calls += 1
            return self.provider

        monkeypatch.setattr(cli, "make_provider", factory)

    def write_config(self, *, tiers: str = ALL_TIERS, per_call: float = 0.25) -> None:
        self.config.write_text(
            TOML.format(ws=self.ws.as_posix(), tiers=tiers, per_call=per_call), encoding="utf-8"
        )

    @property
    def store(self) -> ProjectStore:
        return ProjectStore.for_workspace(self.ws)

    def project(self, *tasks: dict[str, Any], plan: bool = True) -> None:
        self.store.create("test goal", project_id=PID)
        if plan:
            self.store.save_plan(
                PID, Plan.model_validate({"goal": "test goal", "tasks": list(tasks)})
            )

    def script(self, *items: Any) -> None:
        self.provider.push(*items)

    def run(self, *args: str) -> Any:
        return CliRunner().invoke(cli.app, ["run", *args, "--config", str(self.config)])

    def status(self) -> ProjectStatus:
        return self.store.load(PID).status

    def task_status(self, tid: str) -> TaskStatus:
        return self.store.task_states(PID)[tid].status


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


# -- exit code table ---------------------------------------------------------------


def test_every_stop_reason_has_an_exit_code() -> None:
    assert set(cli.RUN_EXIT_CODES) == set(StopReason)
    assert cli.RUN_EXIT_CODES[StopReason.COMPLETED] == cli.EXIT_OK
    assert cli.RUN_EXIT_CODES[StopReason.BUDGET] == cli.EXIT_BUDGET
    # every non-success outcome is a non-zero exit
    assert all(code != 0 for r, code in cli.RUN_EXIT_CODES.items() if r is not StopReason.COMPLETED)


# -- refusals before any model call -----------------------------------------------


def test_run_without_key_uses_real_factory_and_touches_nothing(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.project(task_dict("a"))
    monkeypatch.setattr(cli, "make_provider", provider_from_config)
    r = env.run(PID)
    assert r.exit_code == cli.EXIT_CONFIG
    assert "AUTOMA_API_KEY" in r.output
    assert env.status() is ProjectStatus.NEW
    assert env.task_status("a") is TaskStatus.PENDING
    assert not (env.ws / cli.LEDGER_FILENAME).exists()


def test_run_unknown_project(env: Env) -> None:
    r = env.run("nope")
    assert r.exit_code == cli.EXIT_USER
    assert env.factory_calls == 0


@pytest.mark.parametrize("bad", ["../x", "A B", "x-"])
def test_run_rejects_unsafe_id(env: Env, bad: str) -> None:
    assert env.run(bad).exit_code == cli.EXIT_USER


def test_run_project_without_plan(env: Env) -> None:
    env.project(plan=False)
    r = env.run(PID)
    assert r.exit_code == cli.EXIT_USER
    assert "no plan" in r.output
    assert env.factory_calls == 0
    assert env.status() is ProjectStatus.NEW


@pytest.mark.parametrize("value", ["0", "51", "-1"])
def test_run_invalid_max_attempts(env: Env, value: str) -> None:
    env.project(task_dict("a"))
    r = env.run(PID, "--max-attempts", value)
    assert r.exit_code == cli.EXIT_USER
    assert "max-attempts" in r.output
    assert env.provider.requests == []


def test_run_refuses_project_marked_running_without_force(env: Env) -> None:
    env.project(task_dict("a"))
    env.store.set_status(PID, ProjectStatus.RUNNING)
    r = env.run(PID)
    assert r.exit_code == cli.EXIT_USER
    assert "--force" in r.output
    assert env.factory_calls == 0 and env.provider.requests == []
    assert env.status() is ProjectStatus.RUNNING
    env.script(*write("a"))
    r = env.run(PID, "--force")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert env.status() is ProjectStatus.DONE


# -- running ---------------------------------------------------------------------


def test_run_completes_project(env: Env) -> None:
    env.project(task_dict("b", "a"), task_dict("a"))
    env.script(*write("a"), *write("b"))
    r = env.run(PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "status: done (completed)" in r.output
    assert "passed: b, a" in r.output
    assert "attempts this run: 2" in r.output
    assert "cost: $" in r.output and "this run" in r.output
    assert env.status() is ProjectStatus.DONE
    files = env.store.files_dir(PID)
    assert (files / "a.txt").read_text() == "ok" and (files / "b.txt").read_text() == "ok"
    assert env.provider.remaining == 0


def test_run_done_project_again_spends_nothing(env: Env) -> None:
    env.project(task_dict("a"))
    env.script(*write("a"))
    assert env.run(PID).exit_code == cli.EXIT_OK
    n = len(env.provider.requests)
    r = env.run(PID)
    assert r.exit_code == cli.EXIT_OK
    assert "$0.0000 this run" in r.output
    assert len(env.provider.requests) == n


def test_run_failed_task_blocks_dependents(env: Env) -> None:
    env.project(task_dict("a"), task_dict("b", "a"), task_dict("c"))
    env.script(*write("a", "bad"), *write("c"))
    r = env.run(PID, "--max-attempts", "1")
    assert r.exit_code == cli.EXIT_TASKS_FAILED, r.output
    assert "failed: a" in r.output and "blocked: b" in r.output and "passed: c" in r.output
    assert env.status() is ProjectStatus.FAILED
    assert env.task_status("b") is TaskStatus.BLOCKED


def test_run_retries_with_feedback_up_to_max_attempts(env: Env) -> None:
    env.project(task_dict("a"))
    env.script(*write("a", "bad"), *write("a"))
    r = env.run(PID, "--max-attempts", "2")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "attempts this run: 2" in r.output
    assert len(env.store.task_history(PID, "a")) == 2


# -- stops and resume ---------------------------------------------------------------


def test_provider_error_pauses_and_run_resumes(env: Env) -> None:
    env.project(task_dict("a"), task_dict("b", "a"))
    env.script(*write("a"), ProviderError("down"))
    r = env.run(PID)
    assert r.exit_code == cli.EXIT_PROVIDER, r.output
    assert "paused at task b" in r.output and "automa run proj-1" in r.output
    assert env.status() is ProjectStatus.PAUSED
    assert env.task_status("a") is TaskStatus.PASSED
    # resume: task a is not redone
    env.script(*write("b"))
    r = env.run(PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "attempts this run: 1" in r.output
    assert len(env.store.task_history(PID, "a")) == 1
    assert env.status() is ProjectStatus.DONE


def test_budget_stop_before_provider_then_resume_after_raising_cap(env: Env) -> None:
    env.project(task_dict("a"))
    env.write_config(per_call=0.000001)
    env.script(*write("a"))
    r = env.run(PID)
    assert r.exit_code == cli.EXIT_BUDGET, r.output
    assert "raise the cap" in r.output
    assert env.provider.requests == []  # blocked before the provider
    assert env.status() is ProjectStatus.PAUSED
    assert env.store.task_history(PID, "a") == []
    env.write_config()
    r = env.run(PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert env.status() is ProjectStatus.DONE


def test_unmapped_tier_pauses_with_config_exit(env: Env) -> None:
    env.project(task_dict("a", tier="small"))
    env.write_config(tiers='medium = "fake-medium"\nlarge = "fake-large"')
    r = env.run(PID)
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert env.status() is ProjectStatus.PAUSED
    assert env.provider.requests == []


def test_ctrl_c_pauses_and_is_resumable(env: Env) -> None:
    env.project(task_dict("a"))
    env.script(KeyboardInterrupt())
    r = env.run(PID)
    assert r.exit_code == cli.EXIT_INTERRUPTED
    assert "interrupted" in r.output and "cost:" in r.output
    assert env.status() is ProjectStatus.PAUSED  # never left "running"
    env.script(*write("a"))
    assert env.run(PID).exit_code == cli.EXIT_OK
    assert env.status() is ProjectStatus.DONE


def test_cost_lines_match_ledger(env: Env) -> None:
    env.project(task_dict("a"), task_dict("b", "a"))
    env.script(*write("a"), ProviderError("down"))
    first = env.run(PID)
    env.script(*write("b"))
    second = env.run(PID)
    from automa.budget import CostLedger

    with CostLedger(env.ws / cli.LEDGER_FILENAME) as ledger:
        total = ledger.spent_nusd(project_id=PID)
    assert f"{cli._usd(total)} total" in second.output
    assert "total" in first.output


def test_status_reads_run_results(env: Env) -> None:
    env.project(task_dict("a"))
    env.script(*write("a"))
    env.run(PID)
    r = CliRunner().invoke(cli.app, ["status", PID, "--config", str(env.config)])
    assert r.exit_code == 0 and "status: done" in r.output
