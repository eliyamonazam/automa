"""Tests for the delivery report (automa.delivery, D157) and the retry/deliver CLI.

FakeProvider only; no network, no real model calls.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from automa import cli
from automa.delivery import (
    DeliveryState,
    build_delivery,
    final_tasks,
    render_markdown,
    try_build_delivery,
)
from automa.plan import Plan, TaskStatus
from automa.project import ProjectStatus
from test_cli import Env
from test_orchestrator import PID, Rig, task_dict, write


def plan_with(*tasks: dict[str, Any], deliverable: dict[str, Any] | None = None) -> Plan:
    data: dict[str, Any] = {"goal": "test goal", "tasks": list(tasks)}
    if deliverable is not None:
        data["deliverable"] = deliverable
    return Plan.model_validate(data)


REVIEW_OK = '{"passed": true, "issues": []}'  # the final review's reply (D168)

DELIVERABLE = {
    "summary": "یک فایل متنی که می‌گه ok",
    "main_files": ["b.txt"],
    "how_to_use": "فایل b.txt رو باز کن.",
}


def test_ready_delivery_lists_product_files_and_checks(tmp_path: Path) -> None:
    plan = plan_with(task_dict("a"), task_dict("b", deps=["a"]), deliverable=DELIVERABLE)
    rig = Rig(tmp_path, [*write("a"), *write("b"), REVIEW_OK], plan)
    assert build_delivery(rig.store, PID).state is DeliveryState.NOT_STARTED  # type: ignore[union-attr]
    rig.run()
    d = build_delivery(rig.store, PID)
    assert d is not None
    assert d.state is DeliveryState.READY
    assert d.summary == DELIVERABLE["summary"] and not d.derived
    assert d.how_to_use == DELIVERABLE["how_to_use"]
    assert [(f.path, f.exists, f.size, f.task_id) for f in d.main_files] == [
        ("b.txt", True, 2, "b")
    ]
    assert d.passed_tasks == 2 and not d.problem_tasks and not d.missing_files
    assert all(c.passed is True for t in d.tasks for c in t.criteria)
    md = render_markdown(d, "fa")
    assert "✅ محصول آماده است." in md
    assert "`b.txt`" in md and "فایل b.txt رو باز کن." in md
    assert "## درخواست" in md  # goal differs from the summary
    en = render_markdown(d, "en")
    assert "The product is ready" in en and "What was checked" in en


def test_partial_delivery_explains_what_failed(tmp_path: Path) -> None:
    plan = plan_with(task_dict("a"), task_dict("b", deps=["a"]), deliverable=DELIVERABLE)
    rig = Rig(tmp_path, [*write("a", "bad"), *write("a", "bad")], plan)
    rig.run(max_attempts=2)
    d = build_delivery(rig.store, PID)
    assert d is not None
    assert d.state is DeliveryState.PARTIAL
    assert d.missing_files[0].path == "b.txt"
    problems = {t.id: t for t in d.problem_tasks}
    assert problems["a"].status is TaskStatus.FAILED and problems["a"].issues
    assert problems["a"].attempts == 2
    assert problems["b"].status is TaskStatus.BLOCKED
    assert d.tasks[0].criteria[0].passed is False
    assert d.tasks[1].criteria[0].passed is None
    md = render_markdown(d, "fa")
    assert "محصول کامل نشد" in md and "(ساخته نشد)" in md and "## مشکل‌ها" in md


def test_in_progress_and_no_plan(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [])
    assert build_delivery(rig.store, PID) is None
    plan = plan_with(task_dict("a"), task_dict("b"))
    rig.store.save_plan(PID, plan)
    rig.provider.push(*write("a"))
    rig.provider.push(RuntimeError("stop"))  # second task crashes the provider
    with pytest.raises(RuntimeError):
        rig.run()
    d = build_delivery(rig.store, PID)
    assert d is not None and d.state is DeliveryState.IN_PROGRESS


def test_derived_main_files_without_deliverable(tmp_path: Path) -> None:
    plan = plan_with(task_dict("a"), task_dict("b", deps=["a"]), task_dict("c"))
    assert final_tasks(plan) == ["b", "c"]
    rig = Rig(tmp_path, [*write("a"), *write("c"), *write("b")], plan)
    rig.run()
    d = build_delivery(rig.store, PID)
    assert d is not None and d.derived
    assert d.summary == "test goal"
    assert [f.path for f in d.main_files] == ["b.txt", "c.txt"]
    assert d.state is DeliveryState.READY
    assert "(برنامه شرح محصول نداشت" in render_markdown(d, "fa")
    assert "## درخواست" not in render_markdown(d, "fa")  # summary is the goal


def test_symlinked_main_file_is_not_a_product(tmp_path: Path) -> None:
    plan = plan_with(task_dict("a"), deliverable={"summary": "s", "main_files": ["a.txt"]})
    rig = Rig(tmp_path, [*write("a"), REVIEW_OK], plan)
    rig.run()
    files = rig.store.files_dir(PID)
    secret = tmp_path / "secret.txt"
    secret.write_text("x", encoding="utf-8")
    (files / "a.txt").unlink()
    os.symlink(secret, files / "a.txt")
    d = build_delivery(rig.store, PID)
    assert d is not None
    assert d.main_files[0].exists is False
    assert d.state is DeliveryState.PARTIAL  # done project, product file missing


def test_try_build_delivery_never_raises(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [])
    assert try_build_delivery(rig.store, "missing-project") is None
    rig.store.plan_file(PID).write_text("{broken", encoding="utf-8")
    assert try_build_delivery(rig.store, PID) is None


# --- CLI: retry and deliver ---------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


def _failed_project(env: Env) -> None:
    store = env.store
    store.create("make b", project_id="p1")
    store.save_plan("p1", plan_with(task_dict("b"), deliverable=DELIVERABLE))
    env.provider.push(*write("b", "bad"), *write("b", "bad"))
    r = env.run("run", "p1", "--max-attempts", "2", "--yes")
    assert r.exit_code == cli.EXIT_TASKS_FAILED, r.output
    assert "automa retry p1" in r.output
    assert "delivery: partial - main files: b.txt\n" in r.output  # made, not accepted


def test_cli_retry_then_run_delivers(env: Env) -> None:
    _failed_project(env)
    r = env.run("retry", "p1")
    assert r.exit_code == 0, r.output
    assert "1 task(s) get 2 more attempt(s) on the large tier: b" in r.output
    assert env.store.load("p1").status is ProjectStatus.PAUSED
    env.provider.push(*write("b", "ok"), REVIEW_OK)
    r = env.run("run", "p1", "--max-attempts", "2", "--yes")
    assert r.exit_code == 0, r.output
    assert "delivery: ready - main files: b.txt" in r.output
    r = env.run("deliver", "p1", "--lang", "en")
    assert r.exit_code == 0, r.output
    assert "The product is ready" in r.output and "`b.txt`" in r.output
    # a done project cannot be retried
    r = env.run("retry", "p1")
    assert r.exit_code == cli.EXIT_USER and "only a failed project" in r.output


def test_cli_retry_refusals(env: Env) -> None:
    r = env.run("retry", "nope")
    assert r.exit_code == cli.EXIT_USER
    _failed_project(env)
    r = env.run("retry", "p1", "--attempts", "0")
    assert r.exit_code == cli.EXIT_USER and "between 1 and" in r.output
    assert env.store.load("p1").status is ProjectStatus.FAILED


def test_cli_deliver_refusals(env: Env) -> None:
    env.store.create("goal", project_id="p2")
    r = env.run("deliver", "p2")
    assert r.exit_code == cli.EXIT_USER and "no plan yet" in r.output
    r = env.run("deliver", "p2", "--lang", "de")
    assert r.exit_code == cli.EXIT_USER
    r = env.run("deliver", "missing")
    assert r.exit_code == cli.EXIT_USER


def test_cli_web_preflight_grants_like_automa_retry(env: Env) -> None:
    _failed_project(env)
    pre = cli._WebRunPreflight(cli._load(None), None)
    assert pre.grant_retry("p1") == ["b"]
    assert env.store.load("p1").status is ProjectStatus.PAUSED
    grants = [e for e in env.store.events("p1").read() if e.type == "task.retry_granted"]
    assert grants[0].data["via"] == "web" and grants[0].data["tier"] == "large"
