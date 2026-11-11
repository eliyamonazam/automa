"""User-defined and edited workflow templates (``WorkflowStore.create``/``revise``, D41)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from automa.config import Tier
from automa.plan import Plan
from automa.project import ProjectStatus, ProjectStore
from automa.workflows import (
    WorkflowError,
    WorkflowIncompatible,
    WorkflowNotFound,
    WorkflowOrigin,
    WorkflowStats,
    WorkflowStore,
    WorkflowTemplate,
    check_compatible,
    user_plan,
)
from test_orchestrator import PID, Rig, task_dict, write
from test_workflows import ALL_TIERS, ROLE_TIERS, T0, Clock, done_project

KW: dict[str, Any] = {"role_tiers": ROLE_TIERS, "configured_tiers": ALL_TIERS}


def raw_plan(*tasks: dict[str, Any]) -> dict[str, Any]:
    return {
        "tasks": list(tasks)
        or [
            {
                "id": "a",
                "title": "write the page",
                "role": "writer",
                "acceptance_criteria": [{"description": "page exists"}],
            },
            {
                "id": "b",
                "title": "review it",
                "role": "writer",
                "depends_on": ["a"],
                "acceptance_criteria": [{"description": "reviewed"}],
            },
        ]
    }


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    return tmp_path / "ws"


@pytest.fixture
def projects(ws: Path) -> ProjectStore:
    return ProjectStore.for_workspace(ws)


@pytest.fixture
def flows(ws: Path) -> WorkflowStore:
    return WorkflowStore.for_workspace(ws, clock=Clock())


def _files(flows: WorkflowStore) -> list[str]:
    return sorted(p.name for p in flows.root.iterdir()) if flows.root.exists() else []


# --- user_plan / check_compatible -----------------------------------------------


def test_user_plan_overrides_goal_and_version() -> None:
    plan = user_plan({**raw_plan(), "goal": "other", "version": 7}, "Build a site")
    assert plan.goal == "Build a site" and plan.version == 1
    assert plan.task_ids == ["a", "b"]


@pytest.mark.parametrize(
    "data",
    [
        [],
        "nope",
        {"tasks": []},
        {"tasks": [{"id": "a", "title": "t", "role": "writer"}]},  # no criteria
        raw_plan(
            {
                "id": "a",
                "title": "t",
                "role": "writer",
                "depends_on": ["a"],
                "acceptance_criteria": [{"description": "x"}],
            }
        ),
        {**raw_plan(), "extra": 1},
        raw_plan(
            {
                "id": "a",
                "title": "t",
                "role": "writer",
                "acceptance_criteria": [
                    {"description": "x", "check": {"type": "file_exists", "path": "../out"}}
                ],
            }
        ),
    ],
)
def test_user_plan_rejects_invalid(data: Any) -> None:
    with pytest.raises(WorkflowError):
        user_plan(data, "g")


def test_user_plan_error_lists_problems() -> None:
    with pytest.raises(WorkflowError, match=r"tasks"):
        user_plan({"tasks": []}, "g")


def test_check_compatible_roles_and_tiers() -> None:
    plan = user_plan(raw_plan(), "g")
    check_compatible(plan, **KW)
    with pytest.raises(WorkflowIncompatible, match="unknown role"):
        check_compatible(plan, role_tiers={"architect": Tier.LARGE}, configured_tiers=ALL_TIERS)
    with pytest.raises(WorkflowIncompatible, match="medium"):
        check_compatible(plan, role_tiers=ROLE_TIERS, configured_tiers={Tier.LARGE})


# --- create ----------------------------------------------------------------------


def test_create_user_template(flows: WorkflowStore) -> None:
    t = flows.create("  Build a site  ", raw_plan(), **KW)
    assert t.origin is WorkflowOrigin.USER
    assert t.goal == "Build a site" and t.plan.goal == "Build a site" and t.plan.version == 1
    assert t.source_project is None and t.source_plan_version is None
    assert t.stats is None and t.parent is None and t.task_count == 2
    assert flows.load(t.id) == t
    assert flows.templates() == [t]
    on_disk = json.loads(flows.path(t.id).read_text(encoding="utf-8"))
    assert on_disk["origin"] == "user" and on_disk["stats"] is None


def test_create_accepts_plan_object(flows: WorkflowStore) -> None:
    plan = Plan.model_validate({**raw_plan(), "goal": "old", "version": 3})
    t = flows.create("new goal", plan, **KW)
    assert t.plan.goal == "new goal" and t.plan.version == 1


def test_create_writes_nothing_on_error(flows: WorkflowStore) -> None:
    with pytest.raises(WorkflowError):
        flows.create("g", {"tasks": []}, **KW)
    with pytest.raises(WorkflowError):
        flows.create("   ", raw_plan(), **KW)
    with pytest.raises(WorkflowIncompatible):
        flows.create(
            "g", raw_plan(), role_tiers={"architect": Tier.LARGE}, configured_tiers=ALL_TIERS
        )
    with pytest.raises(WorkflowIncompatible):
        flows.create("g", raw_plan(), role_tiers=ROLE_TIERS, configured_tiers=frozenset())
    with pytest.raises(WorkflowError):
        flows.create("x" * 100_000, raw_plan(), **KW)
    assert _files(flows) == []


def test_create_refuses_symlinked_directory(ws: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    ws.mkdir()
    os.symlink(outside, ws / "workflows")
    flows = WorkflowStore.for_workspace(ws, clock=Clock())
    with pytest.raises(WorkflowError, match="symlink"):
        flows.create("g", raw_plan(), **KW)
    assert list(outside.iterdir()) == []


def test_user_template_is_suggested(flows: WorkflowStore) -> None:
    t = flows.create("Build a weather site", raw_plan(), **KW)
    [match] = flows.suggest("weather site for my town")
    assert match.template.id == t.id


# --- revise ----------------------------------------------------------------------


def test_revise_creates_child_and_keeps_parent(flows: WorkflowStore) -> None:
    parent = flows.create("Build a site", raw_plan(), **KW)
    before = flows.path(parent.id).read_bytes()
    new_tasks = raw_plan()["tasks"][:1]
    child = flows.revise(parent.id, plan={"tasks": new_tasks}, **KW)
    assert child.id != parent.id and child.parent == parent.id
    assert child.origin is WorkflowOrigin.USER and child.goal == "Build a site"
    assert child.task_count == 1
    assert flows.path(parent.id).read_bytes() == before
    assert [t.id for t in flows.history(child.id)] == [child.id, parent.id]


def test_revise_goal_only(flows: WorkflowStore) -> None:
    parent = flows.create("Build a site", raw_plan(), **KW)
    child = flows.revise(parent.id, goal="Build a blog", **KW)
    assert child.goal == "Build a blog" and child.plan.goal == "Build a blog"
    assert child.plan.tasks == parent.plan.tasks


def test_revise_project_template_drops_proof(projects: ProjectStore, flows: WorkflowStore) -> None:
    done_project(projects, "old", "Build a site")
    proven = flows.save_from_project(projects, "old")
    assert proven.origin is WorkflowOrigin.PROJECT and proven.stats is not None
    child = flows.revise(proven.id, goal="Build a shop", **KW)
    assert child.origin is WorkflowOrigin.USER
    assert child.stats is None and child.source_project is None
    assert child.parent == proven.id
    # The project is still saved only once (the edit is not a project template).
    with pytest.raises(WorkflowError, match="already saved"):
        flows.save_from_project(projects, "old")


def test_revise_requires_a_change(flows: WorkflowStore) -> None:
    parent = flows.create("Build a site", raw_plan(), **KW)
    with pytest.raises(WorkflowError, match="nothing to change"):
        flows.revise(parent.id, **KW)
    with pytest.raises(WorkflowError, match="nothing changed"):
        flows.revise(parent.id, goal="Build a site", **KW)
    with pytest.raises(WorkflowError, match="nothing changed"):
        flows.revise(parent.id, plan=raw_plan(), **KW)
    assert _files(flows) == [f"{parent.id}.json"]


def test_revise_validates_and_writes_nothing_on_error(flows: WorkflowStore) -> None:
    parent = flows.create("Build a site", raw_plan(), **KW)
    with pytest.raises(WorkflowError):
        flows.revise(parent.id, plan={"tasks": []}, **KW)
    bad_role = raw_plan(
        {"id": "a", "title": "t", "role": "ghost", "acceptance_criteria": [{"description": "x"}]}
    )
    with pytest.raises(WorkflowIncompatible):
        flows.revise(parent.id, plan=bad_role, **KW)
    with pytest.raises(WorkflowNotFound):
        flows.revise("wf-missing", goal="x", **KW)
    with pytest.raises(WorkflowError):
        flows.revise("../evil", goal="x", **KW)
    assert _files(flows) == [f"{parent.id}.json"]


def test_revise_corrupt_parent_fails_closed(flows: WorkflowStore) -> None:
    parent = flows.create("Build a site", raw_plan(), **KW)
    flows.path(parent.id).write_text("{broken", encoding="utf-8")
    with pytest.raises(WorkflowError):
        flows.revise(parent.id, goal="x", **KW)


def test_history_fails_closed_on_missing_parent_and_cycle(flows: WorkflowStore) -> None:
    parent = flows.create("Build a site", raw_plan(), **KW)
    child = flows.revise(parent.id, goal="Build a blog", **KW)
    flows.path(parent.id).unlink()
    with pytest.raises(WorkflowError, match="missing parent"):
        flows.history(child.id)
    with pytest.raises(WorkflowNotFound):
        flows.history(parent.id)
    # Hand-made cycle: a -> b -> a.
    a = flows.create("goal a", raw_plan(), **KW)
    b = flows.revise(a.id, goal="goal b", **KW)
    data = json.loads(flows.path(a.id).read_text(encoding="utf-8"))
    data["parent"] = b.id
    flows.path(a.id).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(WorkflowError, match="cycle"):
        flows.history(b.id)


# --- model rules -----------------------------------------------------------------


def test_origin_field_rules() -> None:
    plan = user_plan(raw_plan(), "g")
    user = {"id": "wf-u1", "origin": "user", "goal": "g", "created_at": T0, "plan": plan}
    WorkflowTemplate(**user)
    stats = WorkflowStats(tasks=2, attempts=2, cost_nusd=0)
    for extra in (
        {"stats": stats},
        {"source_project": "p1"},
        {"source_plan_version": 1},
        {"goal": "other"},  # plan.goal differs
        {"parent": "wf-u1"},  # own parent
        {"parent": "bad"},
        {"plan": plan.model_copy(update={"version": 2})},
    ):
        with pytest.raises(ValidationError):
            WorkflowTemplate(**{**user, **extra})
    project = {
        "id": "wf-p1",
        "goal": "g",
        "source_project": "p1",
        "source_plan_version": 1,
        "created_at": T0,
        "plan": plan,
        "stats": stats,
    }
    assert WorkflowTemplate(**project).origin is WorkflowOrigin.PROJECT  # default (old files)
    for extra in (
        {"stats": None},
        {"source_project": None},
        {"source_plan_version": None},
        {"parent": "wf-p0"},
        {"stats": WorkflowStats(tasks=1, attempts=1, cost_nusd=0)},
    ):
        with pytest.raises(ValidationError):
            WorkflowTemplate(**{**project, **extra})


def test_old_project_template_file_still_loads(
    projects: ProjectStore, flows: WorkflowStore
) -> None:
    done_project(projects, "old", "Build a site")
    t = flows.save_from_project(projects, "old")
    data = json.loads(flows.path(t.id).read_text(encoding="utf-8"))
    for key in ("origin", "parent"):
        data.pop(key)
    flows.path(t.id).write_text(json.dumps(data), encoding="utf-8")
    assert flows.load(t.id) == t


def test_hand_written_user_file_with_stats_is_rejected(flows: WorkflowStore) -> None:
    t = flows.create("Build a site", raw_plan(), **KW)
    data = json.loads(flows.path(t.id).read_text(encoding="utf-8"))
    data["stats"] = {"tasks": 2, "attempts": 1, "cost_nusd": 0}
    flows.path(t.id).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(WorkflowError, match="invalid"):
        flows.load(t.id)


# --- end to end ------------------------------------------------------------------


def test_user_template_used_and_run_to_done(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [*write("a"), *write("b")])
    flows = WorkflowStore.for_workspace(tmp_path / "ws", clock=Clock())
    t = flows.create(
        "test goal",
        {"tasks": [task_dict("a"), task_dict("b", deps=["a"])]},
        role_tiers={"writer": Tier.MEDIUM},
        configured_tiers=ALL_TIERS,
    )
    flows.use(rig.store, PID, t.id, role_tiers={"writer": Tier.MEDIUM}, configured_tiers=ALL_TIERS)
    assert rig.store.load_plan(PID).task_ids == ["a", "b"]
    assert rig.run().status is ProjectStatus.DONE
