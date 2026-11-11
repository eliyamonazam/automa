"""Tests for ``automa workflow save``, ``automa workflows`` and the suggestions in
``automa new`` (M6, D39).

Local only: the provider factory is replaced by a ``FakeProvider``. Saving, listing
and suggesting never need an API key and never call a model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.config import AutomaConfig
from automa.llm import FakeProvider
from automa.llm.factory import provider_from_config
from automa.project import ProjectStatus, ProjectStore
from automa.workflows import WorkflowStore
from test_cli_run import PID, task_dict, write
from test_workflows import Clock, done_project

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

NO_QUESTIONS = json.dumps({"questions": []})
PLAN = json.dumps(
    {
        "goal": "g",
        "tasks": [
            {
                "id": "t1",
                "title": "Do it",
                "role": "writer",
                "acceptance_criteria": [{"description": "done"}],
            }
        ],
    }
)


class Env:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.ws = tmp_path / "ws"
        self.config = tmp_path / "automa.toml"
        self.config.write_text(TOML.format(ws=self.ws.as_posix()), encoding="utf-8")
        self.provider = FakeProvider()
        self.factory_calls = 0
        monkeypatch.chdir(tmp_path)
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)

        def factory(config: AutomaConfig) -> FakeProvider:
            self.factory_calls += 1
            return self.provider

        monkeypatch.setattr(cli, "make_provider", factory)

    @property
    def store(self) -> ProjectStore:
        return ProjectStore.for_workspace(self.ws)

    @property
    def flows(self) -> WorkflowStore:
        return WorkflowStore.for_workspace(self.ws, clock=Clock())

    def cli(self, *args: str, input: str | None = None) -> Any:
        return CliRunner().invoke(cli.app, [*args, "--config", str(self.config)], input=input)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


def _tree(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(p.relative_to(root)) for p in root.rglob("*")}


# -- workflow save -------------------------------------------------------------------


def test_save_done_project(env: Env) -> None:
    done_project(env.store, "site", "Build a static website for a bakery", attempts_b=2)
    r = env.cli("workflow", "save", "site")
    assert r.exit_code == cli.EXIT_OK, r.output
    ids = env.flows.list_ids()
    assert len(ids) == 1
    assert f"workflow: {ids[0]}" in r.output
    assert "2 task(s)  3 attempt(s)" in r.output
    assert "from site" in r.output
    assert "plan v1: 2 task(s)" in r.output and "(after: a)" in r.output
    assert "note:" in r.output  # the user is told the plan text is copied
    assert env.factory_calls == 0 and env.provider.requests == []
    events = [e.type for e in env.store.events("site").read()]
    assert "workflow.saved" in events


def test_save_json(env: Env) -> None:
    done_project(env.store, "site", "Build a site")
    r = env.cli("workflow", "save", "site", "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    data = json.loads(r.stdout)
    assert data["id"] == env.flows.list_ids()[0]
    assert data["source_project"] == "site"
    assert data["plan"]["version"] == 1 and len(data["plan"]["tasks"]) == 2
    assert data["stats"]["tasks"] == 2 and data["stats"]["cost_nusd"] == 150
    assert data["stats"]["cost_usd"] == pytest.approx(150 / 1e9)


def test_save_needs_no_api_key(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "make_provider", provider_from_config)  # real factory, no key
    done_project(env.store, "site", "Build a site")
    r = env.cli("workflow", "save", "site")
    assert r.exit_code == cli.EXIT_OK, r.output


def test_save_twice_is_refused(env: Env) -> None:
    done_project(env.store, "site", "Build a site")
    assert env.cli("workflow", "save", "site").exit_code == cli.EXIT_OK
    r = env.cli("workflow", "save", "site")
    assert r.exit_code == cli.EXIT_USER
    assert "already saved" in r.output
    assert len(env.flows.list_ids()) == 1


@pytest.mark.parametrize(
    "status", [ProjectStatus.PLANNING, ProjectStatus.RUNNING, ProjectStatus.PAUSED]
)
def test_save_refuses_unfinished_project(env: Env, status: ProjectStatus) -> None:
    done_project(env.store, "site", "Build a site")
    env.store.set_status("site", status)
    r = env.cli("workflow", "save", "site")
    assert r.exit_code == cli.EXIT_USER
    assert "only a finished (done) project" in r.output
    assert not (env.ws / "workflows").exists()


def test_save_refuses_unknown_or_unsafe_project(env: Env) -> None:
    r = env.cli("workflow", "save", "nope")
    assert r.exit_code == cli.EXIT_USER
    r = env.cli("workflow", "save", "../x")
    assert r.exit_code == cli.EXIT_USER
    assert not (env.ws / "workflows").exists()


def test_save_project_without_plan(env: Env) -> None:
    env.store.create("goal", project_id="empty")
    env.store.set_status("empty", ProjectStatus.DONE)
    r = env.cli("workflow", "save", "empty")
    assert r.exit_code == cli.EXIT_USER
    assert env.flows.list_ids() == []


def test_save_refuses_symlinked_workflows_dir(env: Env, tmp_path: Path) -> None:
    done_project(env.store, "site", "Build a site")
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, env.ws / "workflows")
    r = env.cli("workflow", "save", "site")
    assert r.exit_code == cli.EXIT_USER
    assert "symlink" in r.output
    assert list(outside.iterdir()) == []


def test_workflow_group_without_command_shows_help(env: Env) -> None:
    r = CliRunner().invoke(cli.app, ["workflow"])
    assert "save" in r.output


# -- workflows (list) ----------------------------------------------------------------


def test_list_empty_is_read_only(env: Env) -> None:
    r = env.cli("workflows")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "no workflows yet" in r.output
    assert not env.ws.exists()  # listing creates nothing
    r = env.cli("workflows", "--json")
    assert r.exit_code == cli.EXIT_OK
    assert json.loads(r.stdout) == []
    assert not env.ws.exists()


def test_list_newest_first(env: Env) -> None:
    flows = env.flows
    done_project(env.store, "one", "Build a static website")
    done_project(env.store, "two", "Write a Persian poem about rain")
    first = flows.save_from_project(env.store, "one")
    second = flows.save_from_project(env.store, "two")
    r = env.cli("workflows")
    assert r.exit_code == cli.EXIT_OK, r.output
    lines = r.output.strip().splitlines()
    assert lines[0].startswith(second.id) and lines[1].startswith(first.id)
    assert "from two" in lines[0] and "Persian poem" in lines[0]
    r = env.cli("workflows", "--json")
    rows = json.loads(r.stdout)
    assert [row["id"] for row in rows] == [second.id, first.id]
    assert rows[0]["goal"] == "Write a Persian poem about rain"
    assert "cost_usd" in rows[0]["stats"]
    assert env.factory_calls == 0


def test_list_fails_closed_on_corrupt_template(env: Env) -> None:
    done_project(env.store, "one", "Build a site")
    t = env.flows.save_from_project(env.store, "one")
    path = env.flows.path(t.id)
    os.chmod(path, 0o644)
    path.write_text("{not json", encoding="utf-8")
    r = env.cli("workflows")
    assert r.exit_code == cli.EXIT_USER
    assert "unreadable workflow file" in r.output


def test_list_escapes_control_characters_in_goal(env: Env) -> None:
    done_project(env.store, "one", "Build a site\x1b[2J now")
    env.flows.save_from_project(env.store, "one")
    r = env.cli("workflows")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "\x1b" not in r.output and "\\x1b" in r.output


# -- suggestions in `automa new` -----------------------------------------------------


def test_new_shows_similar_workflows_without_reusing_them(env: Env) -> None:
    done_project(env.store, "old", "Build a static website for a bakery")
    t = env.flows.save_from_project(env.store, "old")
    env.provider.push(NO_QUESTIONS, PLAN)
    r = env.cli("new", "Build a static website for a flower shop", "--id", "fresh")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "similar workflows (suggestion only" in r.output
    assert t.id in r.output and "static, website" in r.output
    assert r.output.index("project: fresh") < r.output.index("similar workflows")
    # the plan still comes from the Executive; the template is not used
    assert len(env.provider.requests) == 2
    assert [task.id for task in env.store.load_plan("fresh").tasks] == ["t1"]
    assert env.flows.list_ids() == [t.id]


def test_new_without_similar_workflows_prints_nothing_extra(env: Env) -> None:
    done_project(env.store, "old", "Write a Persian poem about rain")
    env.flows.save_from_project(env.store, "old")
    env.provider.push(NO_QUESTIONS, PLAN)
    r = env.cli("new", "Build a static website", "--id", "fresh")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "similar workflows" not in r.output


def test_new_without_workflows_dir_creates_none(env: Env) -> None:
    env.provider.push(NO_QUESTIONS, PLAN)
    r = env.cli("new", "Build a site", "--id", "fresh")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "similar workflows" not in r.output
    assert not (env.ws / "workflows").exists()


def test_new_persian_goal_suggestion(env: Env) -> None:
    done_project(env.store, "old", "ساخت وب‌سایت فروشگاهی برای نانوایی")
    t = env.flows.save_from_project(env.store, "old")
    env.provider.push(NO_QUESTIONS, PLAN)
    r = env.cli("new", "يك وب‌سایت فروشگاهی برای گل‌فروشی بساز", "--id", "fresh")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert t.id in r.output


def test_new_with_corrupt_workflow_warns_and_continues(env: Env) -> None:
    done_project(env.store, "old", "Build a static website")
    t = env.flows.save_from_project(env.store, "old")
    path = env.flows.path(t.id)
    os.chmod(path, 0o644)
    path.write_text("[]", encoding="utf-8")
    env.provider.push(NO_QUESTIONS, PLAN)
    r = env.cli("new", "Build a static website", "--id", "fresh")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "warning: workflow suggestions unavailable" in r.stderr
    assert env.store.has_plan("fresh")


def test_new_config_error_happens_before_suggestions(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    done_project(env.store, "old", "Build a static website")
    env.flows.save_from_project(env.store, "old")
    monkeypatch.setattr(cli, "make_provider", provider_from_config)  # no key
    r = env.cli("new", "Build a static website", "--id", "fresh")
    assert r.exit_code == cli.EXIT_CONFIG
    assert "similar workflows" not in r.output
    assert "fresh" not in env.store.list_ids()


# -- end to end: new → run → save → suggested on the next new ------------------------


def test_run_then_save_then_suggest(env: Env) -> None:
    env.store.create("Write a short note about rivers", project_id=PID)
    from automa.plan import Plan

    env.store.save_plan(
        PID,
        Plan.model_validate(
            {
                "goal": "Write a short note about rivers",
                "tasks": [task_dict("b", "a"), task_dict("a")],
            }
        ),
    )
    env.provider.push(*write("a"), *write("b"))
    r = env.cli("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert env.store.load(PID).status is ProjectStatus.DONE
    calls = len(env.provider.requests)

    r = env.cli("workflow", "save", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    wid = env.flows.list_ids()[0]
    assert "2 task(s)  2 attempt(s)" in r.output
    assert len(env.provider.requests) == calls  # saving never calls a model

    before = _tree(env.ws / "workflows")
    env.provider.push(NO_QUESTIONS, PLAN)
    r = env.cli("new", "Write a short note about mountains", "--id", "next")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert wid in r.output
    assert _tree(env.ws / "workflows") == before  # suggesting writes nothing


# -- automa new --workflow (D40) ------------------------------------------------------


def _saved(env: Env, goal: str = "Build a static website for a bakery") -> str:
    done_project(env.store, "old", goal)
    return env.flows.save_from_project(env.store, "old").id


def test_new_from_workflow_uses_template_plan_without_model(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    wid = _saved(env)
    template_bytes = env.flows.path(wid).read_bytes()
    monkeypatch.setattr(cli, "make_provider", provider_from_config)  # no key: never built
    r = env.cli("new", "Build a static website for a flower shop", "--id", "fresh",
                "--workflow", wid)  # fmt: skip
    assert r.exit_code == cli.EXIT_OK, r.output
    assert f"workflow: {wid} (from old" in r.output
    assert "note:" in r.output and "cost: $0.0000 (no model call)" in r.output
    assert "plan v1: 2 task(s)" in r.output and "status: planning" in r.output
    assert "similar workflows" not in r.output  # no suggestions when one is chosen
    plan = env.store.load_plan("fresh")
    assert plan.goal == "Build a static website for a flower shop" and plan.version == 1
    assert [t.id for t in plan.tasks] == ["a", "b"]
    assert env.store.load("fresh").status is ProjectStatus.PLANNING
    assert env.provider.requests == []
    assert env.flows.path(wid).read_bytes() == template_bytes
    assert "workflow.used" in [e.type for e in env.store.events("fresh").read()]
    assert not (env.ws / cli.LEDGER_FILENAME).exists()  # no ledger: nothing was spent


def test_new_from_unknown_workflow_creates_nothing(env: Env) -> None:
    _saved(env)
    before = env.store.list_ids()
    for wid in ("wf-nope-000000", "../evil", "WF-UPPER"):
        r = env.cli("new", "Build a site", "--id", "fresh", "--workflow", wid)
        assert r.exit_code == cli.EXIT_USER, r.output
        assert "nothing was created" in r.stderr
    assert env.store.list_ids() == before
    assert env.factory_calls == 0 and env.provider.requests == []


def test_new_from_corrupt_workflow_creates_nothing(env: Env) -> None:
    wid = _saved(env)
    path = env.flows.path(wid)
    os.chmod(path, 0o644)
    path.write_text(json.dumps({"id": wid}), encoding="utf-8")
    r = env.cli("new", "Build a site", "--id", "fresh", "--workflow", wid)
    assert r.exit_code == cli.EXIT_USER, r.output
    assert "fresh" not in env.store.list_ids()


def test_new_from_workflow_with_unconfigured_tier(env: Env, tmp_path: Path) -> None:
    wid = _saved(env)
    no_medium = TOML.replace('medium = "fake-medium"\n', "")
    env.config.write_text(no_medium.format(ws=env.ws.as_posix()), encoding="utf-8")
    r = env.cli("new", "Build a site", "--id", "fresh", "--workflow", wid)
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert "medium" in r.stderr and "nothing was created" in r.stderr
    assert "fresh" not in env.store.list_ids()


def test_new_from_workflow_with_unknown_role(env: Env) -> None:
    wid = _saved(env)
    path = env.flows.path(wid)
    data = json.loads(path.read_text(encoding="utf-8"))
    for task in data["plan"]["tasks"]:
        task["role"] = "ghost"
    os.chmod(path, 0o644)
    path.write_text(json.dumps(data), encoding="utf-8")
    r = env.cli("new", "Build a site", "--id", "fresh", "--workflow", wid)
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert "unknown role(s): ghost" in r.stderr
    assert "fresh" not in env.store.list_ids()


def test_new_from_workflow_existing_project_id(env: Env) -> None:
    wid = _saved(env)
    r = env.cli("new", "Build a site", "--id", "old", "--workflow", wid)
    assert r.exit_code == cli.EXIT_USER, r.output
    assert env.store.load_plan("old").goal == "Build a static website for a bakery"


def test_new_from_workflow_escapes_template_goal(env: Env) -> None:
    wid = _saved(env, "Build a site \x1b[31mred")
    r = env.cli("new", "Build a site", "--id", "fresh", "--workflow", wid)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "\x1b" not in r.output and "\\x1b[31mred" in r.output


def test_new_from_workflow_persian_goal(env: Env) -> None:
    wid = _saved(env, "ساخت وب‌سایت فروشگاهی برای نانوایی")
    r = env.cli("new", "يك وب‌سایت فروشگاهی برای گل‌فروشی بساز", "--id", "fresh", "--workflow", wid)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert env.store.load_plan("fresh").goal == "يك وب‌سایت فروشگاهی برای گل‌فروشی بساز"


def test_new_from_workflow_then_run_to_done(env: Env) -> None:
    """End to end: run → save → new --workflow → run again with the copied plan."""
    from automa.plan import Plan

    env.store.create("Write a short note about rivers", project_id=PID)
    env.store.save_plan(
        PID,
        Plan.model_validate(
            {
                "goal": "Write a short note about rivers",
                "tasks": [task_dict("a"), task_dict("b", "a")],
            }
        ),
    )
    env.provider.push(*write("a"), *write("b"))
    assert env.cli("run", PID).exit_code == cli.EXIT_OK
    assert env.cli("workflow", "save", PID).exit_code == cli.EXIT_OK
    wid = env.flows.list_ids()[0]
    calls = len(env.provider.requests)

    r = env.cli("new", "Write a short note about mountains", "--id", "next", "--workflow", wid)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert len(env.provider.requests) == calls  # no Executive call

    env.provider.push(*write("a"), *write("b"))
    r = env.cli("run", "next")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert env.store.load("next").status is ProjectStatus.DONE
    assert (env.store.project_dir("next") / "files" / "b.txt").read_text() == "ok"


# -- user-defined and edited templates in existing commands (D41) ---------------------

USER_PLAN = {
    "tasks": [
        {
            "id": "t1",
            "title": "Do it",
            "role": "writer",
            "acceptance_criteria": [{"description": "ok"}],
        }
    ]
}


def _user_flows(env: Env) -> tuple[str, str]:
    from automa.config import Tier

    kw: dict[str, Any] = {"role_tiers": {"writer": Tier.MEDIUM}, "configured_tiers": set(Tier)}
    flows = env.flows
    parent = flows.create("Build a static website for a bakery", USER_PLAN, **kw)
    child = flows.revise(parent.id, goal="Build a static website for a florist", **kw)
    return parent.id, child.id


def test_list_shows_user_and_edited_templates(env: Env) -> None:
    parent, child = _user_flows(env)
    r = env.cli("workflows")
    assert r.exit_code == cli.EXIT_OK, r.output
    lines = r.output.strip().splitlines()
    assert lines[0].startswith(child) and f"user, edited from {parent}" in lines[0]
    assert lines[1].startswith(parent) and "user-defined" in lines[1]
    assert "not run yet" in lines[0] and "1 task(s)" in lines[0]
    rows = json.loads(env.cli("workflows", "--json").stdout)
    assert rows[0]["origin"] == "user" and rows[0]["stats"] is None
    assert rows[0]["parent"] == parent and rows[1]["parent"] is None
    assert env.factory_calls == 0


def test_new_from_user_template_and_suggestion(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    _, child = _user_flows(env)
    monkeypatch.setattr(cli, "make_provider", provider_from_config)  # no key: never built
    r = env.cli("new", "Build a website for a florist", "--id", "fresh", "--workflow", child)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert f"workflow: {child} (user, edited from" in r.output
    assert [t.id for t in env.store.load_plan("fresh").tasks] == ["t1"]
    assert env.provider.requests == []
