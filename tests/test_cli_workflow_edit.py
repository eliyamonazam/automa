"""Tests for ``automa workflow create``, ``workflow edit`` and ``workflow show`` (M6, D42).

Local only: the provider factory is replaced by a ``FakeProvider`` (or by the real
factory with no API key, to prove no provider is ever built). None of these commands
calls a model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from automa import cli
from automa.llm.factory import provider_from_config
from automa.project import ProjectStatus
from automa.workflows import WorkflowOrigin
from test_cli_run import task_dict, write
from test_cli_workflows import TOML, USER_PLAN, Env, _saved, _tree


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    e = Env(tmp_path, monkeypatch)
    e.tmp = tmp_path  # type: ignore[attr-defined]
    return e


def _plan_file(env: Env, data: Any, name: str = "plan.json") -> Path:
    path: Path = env.tmp / name  # type: ignore[attr-defined]
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    return path


TWO_TASKS = {"tasks": [task_dict("a"), task_dict("b", "a")]}


def _create(env: Env, goal: str = "Write a short note about rivers", plan: Any = None) -> str:
    r = env.cli(
        "workflow", "create", "--goal", goal, "--plan", str(_plan_file(env, plan or TWO_TASKS)),
        "--json",
    )  # fmt: skip
    assert r.exit_code == cli.EXIT_OK, r.output
    return str(json.loads(r.stdout)["id"])


# -- workflow create ----------------------------------------------------------------


def test_create_saves_user_template(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "make_provider", provider_from_config)  # no key: never built
    path = _plan_file(env, {**TWO_TASKS, "goal": "ignored", "version": 9})
    r = env.cli("workflow", "create", "--goal", "Write a note about rivers", "--plan", str(path))
    assert r.exit_code == cli.EXIT_OK, r.output
    ids = env.flows.list_ids()
    assert len(ids) == 1
    t = env.flows.load(ids[0])
    assert t.origin is WorkflowOrigin.USER and t.parent is None and t.stats is None
    assert t.goal == "Write a note about rivers" and t.plan.goal == t.goal
    assert t.plan.version == 1 and t.plan.task_ids == ["a", "b"]
    assert f"workflow: {t.id}" in r.output and "user-defined" in r.output
    assert "not run yet" in r.output and "plan v1: 2 task(s)" in r.output
    assert f"--workflow {t.id}" in r.output
    assert "Permission Policy" in r.output  # the user is told checks are still gated
    assert env.provider.requests == []
    assert not (env.ws / cli.LEDGER_FILENAME).exists()
    assert env.store.list_ids() == []  # no project is created


def test_create_json(env: Env) -> None:
    r = env.cli(
        "workflow", "create", "--goal", "g1", "--plan", str(_plan_file(env, USER_PLAN)), "--json"
    )
    assert r.exit_code == cli.EXIT_OK, r.output
    data = json.loads(r.stdout)
    assert data["origin"] == "user" and data["stats"] is None and data["parent"] is None
    assert data["goal"] == "g1" and data["plan"]["tasks"][0]["id"] == "t1"


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "{",
        "[]",
        json.dumps({"tasks": []}),
        json.dumps({"tasks": [{"id": "t1", "title": "x", "role": "writer"}]}),
        json.dumps({**USER_PLAN, "surprise": 1}),
        json.dumps({"tasks": [{**USER_PLAN["tasks"][0], "depends_on": ["t1"]}]}),
    ],
)
def test_create_invalid_plan_saves_nothing(env: Env, content: str) -> None:
    r = env.cli("workflow", "create", "--goal", "g", "--plan", str(_plan_file(env, content)))
    assert r.exit_code == cli.EXIT_USER, r.output
    assert "error:" in r.stderr
    assert _tree(env.ws / "workflows") == set()


def test_create_plan_file_problems(env: Env) -> None:
    tmp: Path = env.tmp  # type: ignore[attr-defined]
    missing = env.cli("workflow", "create", "--goal", "g", "--plan", str(tmp / "nope.json"))
    assert missing.exit_code == cli.EXIT_USER and "not found" in missing.stderr
    directory = env.cli("workflow", "create", "--goal", "g", "--plan", str(tmp))
    assert directory.exit_code == cli.EXIT_USER and "regular file" in directory.stderr
    binary = tmp / "bin.json"
    binary.write_bytes(b"\xff\xfe{}")
    r = env.cli("workflow", "create", "--goal", "g", "--plan", str(binary))
    assert r.exit_code == cli.EXIT_USER and "UTF-8" in r.stderr
    big = tmp / "big.json"
    big.write_text(" " * (cli.MAX_PLAN_FILE_BYTES + 1) + "{}", encoding="utf-8")
    r = env.cli("workflow", "create", "--goal", "g", "--plan", str(big))
    assert r.exit_code == cli.EXIT_USER and "larger than" in r.stderr
    bad = _plan_file(env, '{"tasks": [}', "bad.json")
    r = env.cli("workflow", "create", "--goal", "g", "--plan", str(bad))
    assert r.exit_code == cli.EXIT_USER and "line 1" in r.stderr
    assert _tree(env.ws / "workflows") == set()


def test_create_blank_goal(env: Env) -> None:
    r = env.cli("workflow", "create", "--goal", "   ", "--plan", str(_plan_file(env, USER_PLAN)))
    assert r.exit_code == cli.EXIT_USER and "goal" in r.stderr
    assert _tree(env.ws / "workflows") == set()


def test_create_unknown_role_is_config_error(env: Env) -> None:
    plan = {"tasks": [{**USER_PLAN["tasks"][0], "role": "ghost"}]}
    r = env.cli("workflow", "create", "--goal", "g", "--plan", str(_plan_file(env, plan)))
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert "unknown role(s): ghost" in r.stderr and "nothing was saved" in r.stderr
    assert _tree(env.ws / "workflows") == set()


def test_create_unconfigured_tier_is_config_error(env: Env) -> None:
    env.config.write_text(
        TOML.replace('medium = "fake-medium"\n', "").format(ws=env.ws.as_posix()),
        encoding="utf-8",
    )
    r = env.cli("workflow", "create", "--goal", "g", "--plan", str(_plan_file(env, USER_PLAN)))
    assert r.exit_code == cli.EXIT_CONFIG and "medium" in r.stderr
    assert _tree(env.ws / "workflows") == set()


def test_create_refuses_symlinked_workflows_dir(env: Env) -> None:
    outside: Path = env.tmp / "outside"  # type: ignore[attr-defined]
    outside.mkdir()
    env.ws.mkdir(parents=True)
    (env.ws / "workflows").symlink_to(outside, target_is_directory=True)
    r = env.cli("workflow", "create", "--goal", "g", "--plan", str(_plan_file(env, USER_PLAN)))
    assert r.exit_code == cli.EXIT_USER, r.output
    assert list(outside.iterdir()) == []


def test_create_escapes_control_characters_and_keeps_persian(env: Env) -> None:
    plan = {"tasks": [{**USER_PLAN["tasks"][0], "title": "نوشتن \x1b[31mred"}]}
    goal = "يك یادداشت کوتاه درباره رودخانه‌ها"
    r = env.cli("workflow", "create", "--goal", goal, "--plan", str(_plan_file(env, plan)))
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "\x1b" not in r.output and "\\x1b[31mred" in r.output
    assert env.flows.load(env.flows.list_ids()[0]).goal == goal


# -- workflow show --------------------------------------------------------------------


def test_show_user_template(env: Env) -> None:
    wid = _create(env)
    r = env.cli("workflow", "show", wid)
    assert r.exit_code == cli.EXIT_OK, r.output
    out = r.output
    assert f"workflow: {wid}" in out and "source: user-defined" in out
    assert "stats: not run yet" in out and "history:" not in out
    assert "goal: Write a short note about rivers" in out
    assert "plan v1: 2 task(s)" in out and "(after: a)" in out
    assert f"automa workflow edit {wid} --plan plan.json" in out
    assert env.factory_calls == 0 and env.provider.requests == []


def test_show_project_template_has_stats(env: Env) -> None:
    wid = _saved(env)
    r = env.cli("workflow", "show", wid)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "source: from old" in r.output and "stats: 2 attempt(s) for 2 task(s)" in r.output


def test_show_full_goal_not_preview(env: Env) -> None:
    goal = "Write " + "a very long goal " * 10
    wid = _create(env, goal=goal)
    r = env.cli("workflow", "show", wid)
    assert f"goal: {goal.strip()}" in r.output


def test_show_json_with_history(env: Env) -> None:
    parent = _create(env)
    r = env.cli("workflow", "edit", parent, "--goal", "Write a note about lakes", "--json")
    child = json.loads(r.stdout)["id"]
    data = json.loads(env.cli("workflow", "show", child, "--json").stdout)
    assert data["id"] == child and data["parent"] == parent
    assert data["history"] == [child, parent]
    assert data["stats"] is None and data["plan"]["goal"] == "Write a note about lakes"
    text = env.cli("workflow", "show", child).output
    assert f"history: {child} <- {parent}" in text
    assert f"source: user, edited from {parent}" in text


def test_show_plan_json_is_editable_round_trip(env: Env) -> None:
    wid = _create(env)
    r = env.cli("workflow", "show", wid, "--plan-json")
    assert r.exit_code == cli.EXIT_OK, r.output
    data = json.loads(r.stdout)
    assert "goal" not in data and "version" not in data
    assert [t["id"] for t in data["tasks"]] == ["a", "b"]
    # unchanged plan file → the edit is refused (nothing changed), nothing saved
    same = _plan_file(env, r.stdout, "same.json")
    r2 = env.cli("workflow", "edit", wid, "--plan", str(same))
    assert r2.exit_code == cli.EXIT_USER and "nothing changed" in r2.stderr
    assert env.flows.list_ids() == [wid]


def test_show_errors(env: Env) -> None:
    wid = _create(env)
    for bad in ("wf-nope-000000", "../evil", "WF-UPPER"):
        r = env.cli("workflow", "show", bad)
        assert r.exit_code == cli.EXIT_USER, r.output
    both = env.cli("workflow", "show", wid, "--json", "--plan-json")
    assert both.exit_code == cli.EXIT_USER and "not both" in both.stderr


def test_show_is_read_only(env: Env) -> None:
    r = env.cli("workflow", "show", "wf-nope-000000")
    assert r.exit_code == cli.EXIT_USER
    assert not env.ws.exists()


def test_show_corrupt_template_fails_closed(env: Env) -> None:
    wid = _create(env)
    path = env.flows.path(wid)
    os.chmod(path, 0o644)
    path.write_text("{broken", encoding="utf-8")
    r = env.cli("workflow", "show", wid)
    assert r.exit_code == cli.EXIT_USER, r.output


def test_show_missing_parent_warns(env: Env) -> None:
    parent = _create(env)
    child = json.loads(env.cli("workflow", "edit", parent, "--goal", "other", "--json").stdout)[
        "id"
    ]
    os.chmod(env.flows.path(parent), 0o644)
    env.flows.path(parent).unlink()
    r = env.cli("workflow", "show", child)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "history unavailable" in r.stderr and "missing parent" in r.stderr
    data = json.loads(env.cli("workflow", "show", child, "--json").stdout)
    assert data["history"] is None


def test_show_escapes_control_characters(env: Env) -> None:
    wid = _create(env, goal="Write about \x1b[2Jrivers")
    r = env.cli("workflow", "show", wid)
    assert "\x1b" not in r.output and "\\x1b[2Jrivers" in r.output


# -- workflow edit --------------------------------------------------------------------


def test_edit_goal_keeps_parent(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = _create(env)
    parent_bytes = env.flows.path(parent).read_bytes()
    monkeypatch.setattr(cli, "make_provider", provider_from_config)  # no key: never built
    r = env.cli("workflow", "edit", parent, "--goal", "Write a note about lakes")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert f"user, edited from {parent}" in r.output and "not run yet" in r.output
    child = next(i for i in env.flows.list_ids() if i != parent)
    t = env.flows.load(child)
    assert t.parent == parent and t.goal == "Write a note about lakes"
    assert t.plan.task_ids == ["a", "b"]
    assert env.flows.path(parent).read_bytes() == parent_bytes
    assert env.provider.requests == []


def test_edit_plan_from_show_output(env: Env) -> None:
    parent = _create(env)
    data = json.loads(env.cli("workflow", "show", parent, "--plan-json").stdout)
    data["tasks"][1]["title"] = "Review the note carefully"
    r = env.cli("workflow", "edit", parent, "--plan", str(_plan_file(env, data, "e.json")))
    assert r.exit_code == cli.EXIT_OK, r.output
    child = next(i for i in env.flows.list_ids() if i != parent)
    t = env.flows.load(child)
    assert t.goal == env.flows.load(parent).goal
    assert t.plan.tasks[1].title == "Review the note carefully"
    assert env.flows.load(parent).plan.tasks[1].title != "Review the note carefully"


def test_edit_project_template_drops_stats(env: Env) -> None:
    wid = _saved(env)
    r = env.cli("workflow", "edit", wid, "--goal", "Build a site for a florist", "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    data = json.loads(r.stdout)
    assert data["origin"] == "user" and data["stats"] is None and data["parent"] == wid
    assert data["source_project"] is None
    assert env.flows.load(wid).stats is not None


def test_edit_requires_a_change(env: Env) -> None:
    wid = _create(env)
    r = env.cli("workflow", "edit", wid)
    assert r.exit_code == cli.EXIT_USER and "--goal and/or --plan" in r.stderr
    same = env.cli("workflow", "edit", wid, "--goal", "Write a short note about rivers")
    assert same.exit_code == cli.EXIT_USER and "nothing changed" in same.stderr
    assert env.flows.list_ids() == [wid]


def test_edit_errors_save_nothing(env: Env) -> None:
    wid = _create(env)
    unknown = env.cli("workflow", "edit", "wf-nope-000000", "--goal", "x")
    assert unknown.exit_code == cli.EXIT_USER
    unsafe = env.cli("workflow", "edit", "../evil", "--goal", "x")
    assert unsafe.exit_code == cli.EXIT_USER
    bad_plan = _plan_file(env, {"tasks": []}, "bad.json")
    r = env.cli("workflow", "edit", wid, "--plan", str(bad_plan))
    assert r.exit_code == cli.EXIT_USER and "nothing was saved" in r.stderr
    ghost = _plan_file(env, {"tasks": [{**USER_PLAN["tasks"][0], "role": "ghost"}]}, "g.json")
    r = env.cli("workflow", "edit", wid, "--plan", str(ghost))
    assert r.exit_code == cli.EXIT_CONFIG and "ghost" in r.stderr
    missing = env.cli("workflow", "edit", wid, "--plan", "nope.json")
    assert missing.exit_code == cli.EXIT_USER
    blank = env.cli("workflow", "edit", wid, "--goal", "  ")
    assert blank.exit_code == cli.EXIT_USER
    assert env.flows.list_ids() == [wid]


def test_group_help_lists_new_commands(env: Env) -> None:
    from typer.testing import CliRunner

    r = CliRunner().invoke(cli.app, ["workflow", "--help"])
    assert r.exit_code == cli.EXIT_OK
    for name in ("create", "edit", "show", "save"):
        assert name in r.output


# -- end to end -----------------------------------------------------------------------


def test_create_edit_use_run_to_done(env: Env) -> None:
    """create → show --plan-json → edit → new --workflow → run to done; the only model
    calls are the FakeProvider's Worker/Verifier replies during ``run``."""
    wid = _create(env, plan={"tasks": [task_dict("a")]})
    data = json.loads(env.cli("workflow", "show", wid, "--plan-json").stdout)
    data["tasks"].append(task_dict("b", "a"))
    r = env.cli("workflow", "edit", wid, "--plan", str(_plan_file(env, data, "e.json")), "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    child = json.loads(r.stdout)["id"]
    assert env.provider.requests == []

    r = env.cli("new", "Write a short note about mountains", "--id", "next", "--workflow", child)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert f"workflow: {child} (user, edited from {wid}" in r.output
    assert env.provider.requests == []

    env.provider.push(*write("a"), *write("b"))
    r = env.cli("run", "next")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert env.store.load("next").status is ProjectStatus.DONE
    assert (env.store.project_dir("next") / "files" / "b.txt").read_text() == "ok"
