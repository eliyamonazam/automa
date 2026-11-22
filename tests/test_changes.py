"""Change requests after delivery (automa.changes, Executive.plan_change, D169).

FakeProvider behind the real gateway and guard; no network, no real calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from automa.agents.executive import parse_change
from automa.changes import ChangeError, can_change, load_changes, request_change
from automa.plan import Plan, PlanError, TaskStatus
from automa.project import ProjectStatus
from test_executive import Rig as ExecRig
from test_orchestrator import PID, Rig, write

REVIEW_OK = '{"passed": true, "issues": []}'


def task(tid: str, *deps: str, path: str | None = None) -> dict[str, Any]:
    path = path or f"{tid}.txt"
    return {
        "id": tid,
        "title": f"Write {tid}",
        "role": "writer",
        "depends_on": list(deps),
        "acceptance_criteria": [
            {
                "description": f"{path} says ok",
                "check": {"type": "file_contains", "path": path, "text": "ok"},
            }
        ],
    }


def base_plan() -> Plan:
    return Plan.model_validate(
        {
            "goal": "make b",
            "tasks": [task("a"), task("b", "a")],
            "deliverable": {"summary": "b file", "main_files": ["b.txt"]},
        }
    )


def change_json(*tasks: dict[str, Any], **deliverable: Any) -> str:
    data: dict[str, Any] = {"tasks": list(tasks)}
    if deliverable:
        data["deliverable"] = deliverable
    return json.dumps(data)


def test_parse_change_merges_and_validates() -> None:
    plan = base_plan()
    new = parse_change(change_json(task("c", "b")), plan, {"writer"})
    assert new.version == 2 and [t.id for t in new.tasks] == ["a", "b", "c"]
    assert new.deliverable == plan.deliverable  # kept when not given
    new2 = parse_change(
        change_json(task("c", path="c.txt"), summary="b and c", main_files=["b.txt", "c.txt"]),
        plan,
        None,
    )
    assert new2.deliverable is not None and new2.deliverable.main_files == ["b.txt", "c.txt"]
    for bad, match in [
        (change_json(task("a")), "already used"),
        (json.dumps({"tasks": []}), "no task"),
        ("not json", "not valid JSON"),
        (change_json(task("c", "zzz")), "failed validation"),
        (change_json({**task("c"), "role": "hacker"}), "unknown role"),
        (change_json(task("c"), summary="s", main_files=["nope.txt"]), "failed validation"),
    ]:
        with pytest.raises(PlanError, match=match):
            parse_change(bad, plan, {"writer"})


def done_rig(tmp_path: Path) -> Rig:
    rig = Rig(tmp_path, [*write("a"), *write("b"), REVIEW_OK], base_plan())
    assert rig.run().status is ProjectStatus.DONE
    return rig


def test_change_adds_tasks_keeps_finished_ones_and_runs(tmp_path: Path) -> None:
    rig = done_rig(tmp_path)
    assert can_change(rig.store, PID)
    ex = ExecRig(tmp_path / "ex", [change_json(task("c", "b"))])
    new = request_change(
        ex.agent(known_roles=["writer"]),
        rig.store,
        PID,
        "  also add c  ",
        [("logo.png", b"\x89PNG")],
        via="test",
    )
    assert new.version == 2
    assert rig.project_status() is ProjectStatus.PAUSED
    assert rig.state("a").status is TaskStatus.PASSED and rig.state("b").status is TaskStatus.PASSED
    assert rig.state("c").status is TaskStatus.PENDING
    records = load_changes(rig.store.project_dir(PID))
    assert len(records) == 1 and records[0].text == "also add c"
    assert records[0].new_tasks == ("c",) and records[0].files == ("inputs/change-1/logo.png",)
    assert (
        rig.store.files_dir(PID) / "inputs" / "change-1" / "logo.png"
    ).read_bytes() == b"\x89PNG"
    user = ex.provider.requests[0].messages[1].content or ""
    assert "also add c" in user and "inputs/change-1/logo.png" in user and '"id": "a"' in user
    assert "NEW tasks only" in (ex.provider.requests[0].messages[0].content or "")
    # the run does only c, then the final review sees the change request
    rig.provider.push(*write("c"), REVIEW_OK)
    assert rig.run().status is ProjectStatus.DONE
    review = rig.provider.requests[-1].messages[1].content or ""
    assert "Later change request from the user: also add c" in review


def test_change_refused_while_not_finished_and_empty(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [], base_plan())
    ex = ExecRig(tmp_path / "ex", [change_json(task("c"))])
    with pytest.raises(ChangeError, match="done or failed"):
        request_change(ex.agent(), rig.store, PID, "x")
    rig2 = done_rig(tmp_path / "d")
    with pytest.raises(ChangeError, match="write what"):
        request_change(ex.agent(), rig2.store, PID, "   ")
    with pytest.raises(ChangeError, match="longer"):
        request_change(ex.agent(), rig2.store, PID, "x" * 5_001)
    assert ex.provider.requests == []


def test_change_planning_failure_keeps_status(tmp_path: Path) -> None:
    from automa.agents import PlanningFailed

    rig = done_rig(tmp_path)
    ex = ExecRig(tmp_path / "ex", ["bad", "bad", "bad"])
    with pytest.raises(PlanningFailed):
        request_change(ex.agent(), rig.store, PID, "add c")
    assert rig.project_status() is ProjectStatus.DONE
    assert rig.store.load_plan(PID).version == 1
    assert load_changes(rig.store.project_dir(PID)) == []


def test_fix_round_allowed_again_after_a_change(tmp_path: Path) -> None:
    from automa.final_review import FIX_TASK_PREFIX, just_fixed

    plan = base_plan()
    assert not just_fixed(plan)
    fixed = Plan.model_validate(
        plan.model_dump()
        | {"version": 2, "tasks": [*plan.model_dump()["tasks"], task(FIX_TASK_PREFIX)]}
    )
    assert just_fixed(fixed)
    changed = parse_change(change_json(task("c")), fixed, None)
    assert not just_fixed(changed)
