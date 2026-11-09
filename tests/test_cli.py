"""Tests for the CLI (``automa new``, ``automa status``).

Local only: the provider factory is replaced by a ``FakeProvider``; the real
``LLMGateway`` + ``BudgetGuard`` + ``CostLedger`` are used. No network, no real calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.config import AutomaConfig
from automa.llm import FakeProvider, ProviderError
from automa.llm.factory import provider_from_config
from automa.project import ProjectStatus, ProjectStore

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
project_usd = {project}
daily_usd = {daily}
per_call_usd = {per_call}
"""


def task(tid: str, *deps: str) -> dict[str, Any]:
    return {
        "id": tid,
        "title": f"Task {tid}",
        "role": "developer",
        "depends_on": list(deps),
        "acceptance_criteria": [{"description": "works"}],
    }


def plan_json(*tasks: dict[str, Any]) -> str:
    return json.dumps({"goal": "model goal", "tasks": list(tasks)})


GOOD_PLAN = plan_json(task("write-docs", "build"), task("build"))
NO_QUESTIONS = json.dumps({"questions": []})


class Env:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.tmp = tmp_path
        self.ws = tmp_path / "ws"
        self.config = tmp_path / "automa.toml"
        self.write_config()
        self.provider = FakeProvider()
        self.factory_calls: list[AutomaConfig] = []
        monkeypatch.chdir(tmp_path)  # no stray .env / automa.toml from the repo
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)

        def factory(config: AutomaConfig) -> FakeProvider:
            self.factory_calls.append(config)
            return self.provider

        monkeypatch.setattr(cli, "make_provider", factory)

    def write_config(
        self, *, project: float = 5.0, daily: float = 2.0, per_call: float = 0.25, extra: str = ""
    ) -> None:
        self.config.write_text(
            TOML.format(ws=self.ws.as_posix(), project=project, daily=daily, per_call=per_call)
            + extra,
            encoding="utf-8",
        )

    def run(self, *args: str, input: str | None = None) -> Any:
        return CliRunner().invoke(cli.app, [*args, "--config", str(self.config)], input=input)

    @property
    def store(self) -> ProjectStore:
        return ProjectStore.for_workspace(self.ws)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


# -- safety: the real provider is never reachable in tests -----------------------


def test_default_factory_requires_api_key() -> None:
    from automa.config import ConfigError

    with pytest.raises(ConfigError):
        provider_from_config(AutomaConfig())


def test_new_without_key_uses_real_factory_and_fails_before_creating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = Env(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "make_provider", provider_from_config)
    r = e.run("new", "goal", "--no-ask")
    assert r.exit_code == cli.EXIT_CONFIG
    assert "AUTOMA_API_KEY" in r.output
    assert e.store.list_ids() == []


# -- new ------------------------------------------------------------------------


def test_new_without_questions_writes_plan(env: Env) -> None:
    env.provider.push(NO_QUESTIONS, GOOD_PLAN)
    r = env.run("new", "  Build a site  ", "--id", "site")
    assert r.exit_code == 0, r.output
    assert "project: site" in r.output
    assert "plan v1: 2 task(s)" in r.output
    # topological order: build before write-docs
    assert r.output.index("build: Task build") < r.output.index("write-docs: Task write-docs")
    assert "(after: build)" in r.output
    assert "cost: $" in r.output
    plan = env.store.load_plan("site")
    assert plan.goal == "Build a site"
    assert env.store.load("site").status is ProjectStatus.PLANNING
    assert not env.store.has_clarifications("site")
    assert len(env.provider.requests) == 2


def test_new_generates_id(env: Env) -> None:
    env.provider.push(NO_QUESTIONS, GOOD_PLAN)
    r = env.run("new", "goal")
    assert r.exit_code == 0, r.output
    ids = env.store.list_ids()
    assert len(ids) == 1 and ids[0].startswith("p-")
    assert f"project: {ids[0]}" in r.output


def test_new_asks_questions_and_saves_answers(env: Env) -> None:
    env.provider.push(json.dumps({"questions": ["Which language?", "Deadline?"]}), GOOD_PLAN)
    r = env.run("new", "goal", "--id", "p1", input="Persian\n\n")
    assert r.exit_code == 0, r.output
    assert "Which language?" in r.output
    items = env.store.load_clarifications("p1")
    assert [(c.question, c.answer) for c in items] == [
        ("Which language?", "Persian"),
        ("Deadline?", ""),
    ]
    # the answers reach the planning prompt
    plan_prompt = env.provider.requests[1].messages[-1].content or ""
    assert "Q: Which language?" in plan_prompt and "A: Persian" in plan_prompt
    assert "A: (no answer)" in plan_prompt
    events = [e.type for e in env.store.events("p1").read()]
    assert "clarifications.saved" in events and "plan.saved" in events


def test_new_no_ask_skips_clarification_call(env: Env) -> None:
    env.provider.push(GOOD_PLAN)
    r = env.run("new", "goal", "--id", "p1", "--no-ask")
    assert r.exit_code == 0, r.output
    assert len(env.provider.requests) == 1
    assert env.store.has_plan("p1")


def test_new_aborted_prompt_leaves_project_without_plan(env: Env) -> None:
    env.provider.push(json.dumps({"questions": ["Q?"]}), GOOD_PLAN)
    r = env.run("new", "goal", "--id", "p1", input="")  # EOF at the prompt
    assert r.exit_code != 0
    assert not env.store.has_plan("p1")
    assert not env.store.has_clarifications("p1")
    assert len(env.provider.requests) == 1


@pytest.mark.parametrize("goal", ["", "   "])
def test_new_rejects_empty_goal(env: Env, goal: str) -> None:
    r = env.run("new", goal)
    assert r.exit_code == cli.EXIT_USER
    assert env.factory_calls == []
    assert not env.ws.exists()


@pytest.mark.parametrize("bad", ["../x", "Upper", "a/b", "-x", "x" * 65])
def test_new_rejects_unsafe_id(env: Env, bad: str) -> None:
    r = env.run("new", "goal", "--id", bad, "--no-ask")
    assert r.exit_code == cli.EXIT_USER
    assert "invalid project id" in r.output
    assert env.provider.requests == []
    assert not (env.tmp / "x").exists()


def test_new_existing_id_is_refused(env: Env) -> None:
    env.store.create("old goal", project_id="p1")
    r = env.run("new", "goal", "--id", "p1", "--no-ask")
    assert r.exit_code == cli.EXIT_USER
    assert "already exists" in r.output
    assert env.store.load("p1").goal == "old goal"
    assert env.provider.requests == []


def test_new_missing_tier_fails_before_creating(env: Env) -> None:
    env.config.write_text(f'workspace = "{env.ws.as_posix()}"\n', encoding="utf-8")
    r = env.run("new", "goal", "--no-ask")
    assert r.exit_code == cli.EXIT_CONFIG
    assert "tier" in r.output
    assert env.factory_calls == []
    assert env.store.list_ids() == []


def test_new_invalid_config(env: Env) -> None:
    env.config.write_text("not = [valid", encoding="utf-8")
    r = env.run("new", "goal")
    assert r.exit_code == cli.EXIT_CONFIG


def test_new_rejects_secret_in_config(env: Env) -> None:
    env.write_config(extra='\napi_key = "sk-secret-123"\n')
    r = env.run("new", "goal")
    assert r.exit_code == cli.EXIT_CONFIG
    assert "sk-secret-123" not in r.output


def test_new_planning_failed(env: Env) -> None:
    env.provider.push("not json", "still not", "{}")
    r = env.run("new", "goal", "--id", "p1", "--no-ask")
    assert r.exit_code == cli.EXIT_PLANNING
    assert "after 3 attempt(s)" in r.output
    assert env.store.load("p1").status is ProjectStatus.FAILED
    assert not env.store.has_plan("p1")


def test_new_clarify_planning_failed(env: Env) -> None:
    env.provider.push("x", "y", "z")
    r = env.run("new", "goal", "--id", "p1")
    assert r.exit_code == cli.EXIT_PLANNING
    assert env.store.load("p1").status is ProjectStatus.FAILED


def test_new_provider_error(env: Env) -> None:
    env.provider.push(ProviderError("HTTP 502", status=502, retryable=True))
    r = env.run("new", "goal", "--id", "p1", "--no-ask")
    assert r.exit_code == cli.EXIT_PROVIDER
    assert "HTTP 502" in r.output
    assert len(env.provider.requests) == 1  # never retried
    assert env.store.load("p1").status is ProjectStatus.NEW
    assert "cost: $0.0000" in r.output


def test_new_per_call_cap_blocks_before_provider(env: Env) -> None:
    # 8000 plan tokens * $2/Mtok = $0.016 > per-call cap: refused before any call.
    env.write_config(project=1.0, daily=1.0, per_call=0.001)
    r = env.run("new", "goal", "--id", "p1", "--no-ask")
    assert r.exit_code == cli.EXIT_BUDGET
    assert "per_call cap" in r.output
    assert env.provider.requests == []
    assert env.store.load("p1").status is ProjectStatus.PAUSED


def test_new_budget_stop_during_clarify_pauses(env: Env) -> None:
    env.write_config(project=0.001, daily=0.001, per_call=0.001)
    r = env.run("new", "goal", "--id", "p1")
    assert r.exit_code == cli.EXIT_BUDGET
    assert env.provider.requests == []
    assert env.store.load("p1").status is ProjectStatus.PAUSED
    assert "paused" in r.output


def test_new_project_cap_latches_across_runs(env: Env) -> None:
    # A daily cap reached by one project stops the next `new` too (shared ledger).
    env.write_config(project=0.02, daily=0.02, per_call=0.02)
    big = "x" * 40_000  # ~10k output tokens -> ~$0.02 actual cost, fills the cap
    env.provider.push(big, big, big)
    r1 = env.run("new", "goal", "--id", "p1", "--no-ask")
    assert r1.exit_code in (cli.EXIT_BUDGET, cli.EXIT_PLANNING)
    before = len(env.provider.requests)
    env.provider.push(GOOD_PLAN)
    r2 = env.run("new", "goal", "--id", "p2", "--no-ask")
    assert r2.exit_code == cli.EXIT_BUDGET
    assert len(env.provider.requests) == before
    assert env.store.load("p2").status is ProjectStatus.PAUSED


def test_new_records_cost_in_ledger(env: Env) -> None:
    env.provider.push(NO_QUESTIONS, GOOD_PLAN)
    env.run("new", "goal", "--id", "p1")
    from automa.budget import CostLedger

    with CostLedger.for_workspace(env.ws) as ledger:
        assert len(ledger.entries(project_id="p1")) == 2
        assert ledger.spent_nusd(project_id="p1") > 0


# -- status ---------------------------------------------------------------------


def test_status_empty_workspace_creates_nothing(env: Env) -> None:
    r = env.run("status")
    assert r.exit_code == 0
    assert "no projects yet" in r.output
    assert not env.ws.exists()
    assert env.factory_calls == []


def test_status_needs_no_key_and_makes_no_call(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    env.provider.push(NO_QUESTIONS, GOOD_PLAN)
    env.run("new", "goal", "--id", "p1")

    def boom(config: AutomaConfig) -> FakeProvider:
        raise AssertionError("status must not build a provider")

    monkeypatch.setattr(cli, "make_provider", boom)
    assert env.run("status").exit_code == 0
    assert env.run("status", "p1").exit_code == 0


def test_status_lists_projects(env: Env) -> None:
    env.provider.push(NO_QUESTIONS, GOOD_PLAN)
    env.run("new", "A goal that is fairly long " * 5, "--id", "p1")
    env.store.create("second", project_id="p2")
    r = env.run("status")
    assert r.exit_code == 0
    lines = r.output.strip().splitlines()
    assert lines[0].startswith("p1  planning")
    assert lines[1].startswith("p2  new")
    assert "$0.0000" in lines[1]
    assert lines[0].endswith("…")


def test_status_one_project_text(env: Env) -> None:
    env.provider.push(json.dumps({"questions": ["Which language?"]}), GOOD_PLAN)
    env.run("new", "goal", "--id", "p1", input="Go\n")
    r = env.run("status", "p1")
    assert r.exit_code == 0, r.output
    out = r.output
    assert "project: p1" in out and "goal: goal" in out and "status: planning" in out
    assert "Q: Which language?" in out and "A: Go" in out
    assert "plan v1: 2 task(s)" in out


def test_status_project_without_plan(env: Env) -> None:
    env.store.create("goal", project_id="p1")
    r = env.run("status", "p1")
    assert r.exit_code == 0
    assert "plan: none" in r.output
    assert "cost: $0.0000" in r.output
    assert not (env.ws / "ledger.sqlite3").exists()


def test_status_json(env: Env) -> None:
    env.provider.push(NO_QUESTIONS, GOOD_PLAN)
    env.run("new", "هدف فارسی", "--id", "p1")
    r = env.run("status", "p1", "--json")
    data = json.loads(r.output)
    assert data["id"] == "p1" and data["goal"] == "هدف فارسی"
    assert data["status"] == "planning"
    assert data["cost_usd"] > 0
    assert [t["id"] for t in data["plan"]["tasks"]] == ["write-docs", "build"]
    rows = json.loads(env.run("status", "--json").output)
    assert [row["id"] for row in rows] == ["p1"]
    assert rows[0]["cost_usd"] == data["cost_usd"]


def test_status_unknown_project(env: Env) -> None:
    r = env.run("status", "nope")
    assert r.exit_code == cli.EXIT_USER
    assert "no such project" in r.output


@pytest.mark.parametrize("bad", ["../etc", "A", "a/b"])
def test_status_rejects_unsafe_id(env: Env, bad: str) -> None:
    r = env.run("status", bad)
    assert r.exit_code == cli.EXIT_USER


def test_status_corrupt_project_is_an_error(env: Env) -> None:
    env.store.create("goal", project_id="p1")
    (env.ws / "projects" / "p1" / "project.json").write_text("{broken", encoding="utf-8")
    assert env.run("status").exit_code == cli.EXIT_USER
    assert env.run("status", "p1").exit_code == cli.EXIT_USER


def test_status_corrupt_plan_is_an_error(env: Env) -> None:
    env.store.create("goal", project_id="p1")
    (env.ws / "projects" / "p1" / "plan.json").write_text("[]", encoding="utf-8")
    assert env.run("status", "p1").exit_code == cli.EXIT_USER


def test_no_args_shows_help() -> None:
    r = CliRunner().invoke(cli.app, [])
    assert "new" in r.output and "status" in r.output


def test_cli_provider_only_reaches_a_gateway() -> None:
    """Structural: in cli.py the provider object is only ever passed to LLMGateway."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(cli))
    uses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "provider" and isinstance(node.ctx, ast.Load)
    ]
    assert len(uses) == 1
    gateway_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LLMGateway"
        and any(isinstance(a, ast.Name) and a.id == "provider" for a in node.args)
        and any(k.arg == "guard" for k in node.keywords)
    ]
    assert len(gateway_calls) == 1
    assert ".complete(" not in inspect.getsource(cli)
