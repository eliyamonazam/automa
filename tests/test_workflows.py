"""Tests for workflow templates (``automa.workflows``): save successful plans, suggest them."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from automa.config import Tier
from automa.plan import (
    AcceptanceCriterion,
    Issue,
    Plan,
    ResultStatus,
    Task,
    TaskResult,
    TaskState,
    TaskStatus,
    Verdict,
    VerdictOutcome,
)
from automa.project import ProjectStatus, ProjectStore
from automa.workflows import (
    MAX_SUGGESTIONS,
    WorkflowError,
    WorkflowNotFound,
    WorkflowStats,
    WorkflowStore,
    WorkflowTemplate,
    goal_terms,
    similarity,
    validate_workflow_id,
)
from test_orchestrator import PID, Rig, make_plan, task_dict, write

T0 = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def _task(tid: str, deps: list[str] | None = None) -> Task:
    return Task(
        id=tid,
        title=f"task {tid}",
        role="writer",
        depends_on=deps or [],
        acceptance_criteria=[AcceptanceCriterion(description="it is good")],
    )


def _plan(goal: str, version: int = 1) -> Plan:
    return Plan(version=version, goal=goal, tasks=[_task("a"), _task("b", ["a"])])


def _result(tid: str, attempt: int, cost: int) -> TaskResult:
    return TaskResult(
        task_id=tid,
        attempt=attempt,
        status=ResultStatus.COMPLETED,
        summary="did it",
        tier=Tier.MEDIUM,
        model="fake-medium",
        cost_nusd=cost,
        started_at=T0,
        finished_at=T0,
    )


def _verdict(tid: str, attempt: int, passed: bool) -> Verdict:
    if passed:
        return Verdict(task_id=tid, attempt=attempt, outcome=VerdictOutcome.PASS, created_at=T0)
    return Verdict(
        task_id=tid,
        attempt=attempt,
        outcome=VerdictOutcome.FAIL,
        issues=[Issue(description="not yet")],
        created_at=T0,
    )


def _attempt(store: ProjectStore, pid: str, tid: str, attempt: int, passed: bool, cost: int):
    store.save_task_result(pid, _result(tid, attempt, cost))
    store.save_verdict(pid, _verdict(tid, attempt, passed))


def _pass(store: ProjectStore, pid: str, tid: str, attempts: int = 1, cost: int = 100) -> None:
    for n in range(1, attempts + 1):
        _attempt(store, pid, tid, n, passed=n == attempts, cost=cost)
    current = store.load_task_state(pid, tid)
    state = TaskState(
        task_id=tid,
        status=TaskStatus.PASSED,
        attempts=attempts,
        plan_version=store.load_plan(pid).version,
        updated_at=T0,
        revision=current.revision if current else 0,
    )
    store.save_task_state(pid, state)


def done_project(store: ProjectStore, pid: str, goal: str, *, attempts_b: int = 1) -> None:
    store.create(goal, project_id=pid)
    store.save_plan(pid, _plan(goal))
    _pass(store, pid, "a", cost=100)
    _pass(store, pid, "b", attempts=attempts_b, cost=50)
    store.set_status(pid, ProjectStatus.DONE)


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    return tmp_path / "ws"


@pytest.fixture
def projects(ws: Path) -> ProjectStore:
    return ProjectStore.for_workspace(ws)


@pytest.fixture
def flows(ws: Path) -> WorkflowStore:
    return WorkflowStore.for_workspace(ws, clock=Clock())


# --- similarity ------------------------------------------------------------------


def test_goal_terms_normalizes_and_drops_stopwords() -> None:
    assert goal_terms("Write a CLI tool for the Weather API, 2026!") == {
        "cli",
        "tool",
        "weather",
        "api",
    }
    # Persian: stopwords dropped, Arabic yeh/kaf normalized, ZWNJ splits words
    assert goal_terms("یك اسكريپت برای گزارش‌ها بنویس") == {"اسکریپت", "گزارش", "ها"}
    assert goal_terms("") == frozenset() and goal_terms("the and of 42 _ x") == frozenset()


def test_similarity_is_symmetric_and_bounded() -> None:
    a, b = "weather CLI tool in python", "python weather dashboard"
    s1, shared = similarity(a, b)
    s2, _ = similarity(b, a)
    assert s1 == s2 and 0 < s1 < 1 and shared == {"python", "weather"}
    assert similarity(a, a)[0] == 1.0
    assert similarity(a, "")[0] == 0.0
    assert similarity("the of", "the of")[0] == 0.0  # only stopwords: no match


# --- saving ----------------------------------------------------------------------


def test_save_from_done_project(projects: ProjectStore, flows: WorkflowStore) -> None:
    done_project(projects, "p1", "build a weather CLI", attempts_b=2)
    t = flows.save_from_project(projects, "p1")
    assert t.id.startswith("wf-") and validate_workflow_id(t.id) == t.id
    assert t.goal == "build a weather CLI" and t.source_project == "p1"
    assert t.plan == projects.load_plan("p1")
    assert t.stats == WorkflowStats(tasks=2, attempts=3, cost_nusd=200)
    assert flows.load(t.id) == t
    assert flows.templates() == [t] and flows.list_ids() == [t.id]
    assert flows.find_by_project("p1") == t and flows.find_by_project("p2") is None
    events = [e for e in projects.events("p1").read() if e.type == "workflow.saved"]
    assert [e.data for e in events] == [{"workflow_id": t.id, "tasks": 2}]
    raw = json.loads(flows.path(t.id).read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1 and raw["plan"]["version"] == 1


def test_replanned_project_saves_current_plan_as_version_1(
    projects: ProjectStore, flows: WorkflowStore
) -> None:
    projects.create("goal x", project_id="p1")
    projects.save_plan("p1", _plan("goal x"))
    projects.save_plan("p1", Plan(version=3, goal="goal x", tasks=[_task("c")]))
    _pass(projects, "p1", "c")
    projects.set_status("p1", ProjectStatus.DONE)
    t = flows.save_from_project(projects, "p1")
    assert t.source_plan_version == 3 and t.plan.version == 1
    assert [task.id for task in t.plan.tasks] == ["c"]


@pytest.mark.parametrize(
    "status",
    [s for s in ProjectStatus if s is not ProjectStatus.DONE],
)
def test_only_done_projects_are_saved(
    projects: ProjectStore, flows: WorkflowStore, status: ProjectStatus
) -> None:
    done_project(projects, "p1", "goal")
    projects.set_status("p1", status)
    with pytest.raises(WorkflowError, match="only a finished"):
        flows.save_from_project(projects, "p1")
    assert flows.list_ids() == [] and not flows.root.exists()


def test_done_project_with_unpassed_task_is_refused(
    projects: ProjectStore, flows: WorkflowStore
) -> None:
    projects.create("goal", project_id="p1")
    projects.save_plan("p1", _plan("goal"))
    _pass(projects, "p1", "a")
    _attempt(projects, "p1", "b", 1, passed=False, cost=1)
    projects.set_status("p1", ProjectStatus.DONE)  # inconsistent on purpose
    with pytest.raises(WorkflowError, match="did not pass: b"):
        flows.save_from_project(projects, "p1")
    assert flows.list_ids() == []


def test_project_without_plan_or_unknown(projects: ProjectStore, flows: WorkflowStore) -> None:
    projects.create("goal", project_id="p1")
    projects.set_status("p1", ProjectStatus.DONE)
    with pytest.raises(Exception, match="no plan"):
        flows.save_from_project(projects, "p1")
    with pytest.raises(Exception, match="no such project"):
        flows.save_from_project(projects, "nope")
    with pytest.raises(Exception, match="invalid project id"):
        flows.save_from_project(projects, "../x")
    assert flows.list_ids() == []


def test_project_saved_at_most_once(projects: ProjectStore, flows: WorkflowStore) -> None:
    done_project(projects, "p1", "goal one")
    first = flows.save_from_project(projects, "p1")
    with pytest.raises(WorkflowError, match="already saved"):
        flows.save_from_project(projects, "p1")
    assert flows.templates() == [first]


def test_existing_file_is_never_overwritten(projects: ProjectStore, ws: Path) -> None:
    done_project(projects, "p1", "goal one")
    done_project(projects, "p2", "goal two")
    fixed = datetime(2026, 1, 1, tzinfo=UTC)
    flows = WorkflowStore.for_workspace(ws, clock=lambda: fixed)
    import automa.workflows as wf

    original = wf.secrets.token_hex
    wf.secrets.token_hex = lambda n: "abcdef"  # type: ignore[assignment]
    try:
        first = flows.save_from_project(projects, "p1")
        with pytest.raises(WorkflowError, match="already exists"):
            flows.save_from_project(projects, "p2")
    finally:
        wf.secrets.token_hex = original  # type: ignore[assignment]
    assert flows.load(first.id).source_project == "p1"


# --- reading (fail closed) ---------------------------------------------------------


def test_empty_store_and_unknown_ids(flows: WorkflowStore) -> None:
    assert flows.list_ids() == [] and flows.templates() == [] and flows.suggest("x") == []
    with pytest.raises(WorkflowNotFound):
        flows.load("wf-missing")
    for bad in ("../x", "wf-", "WF-ABC", "wf-a/b", "p1", "", "wf-" + "a" * 70):
        with pytest.raises(WorkflowError, match="invalid workflow id"):
            flows.load(bad)


def test_corrupt_or_tampered_files_fail_closed(
    projects: ProjectStore, flows: WorkflowStore
) -> None:
    done_project(projects, "p1", "goal")
    t = flows.save_from_project(projects, "p1")
    path = flows.path(t.id)
    good = path.read_text(encoding="utf-8")

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(WorkflowError, match="unreadable"):
        flows.templates()

    raw = json.loads(good)
    raw["extra"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(WorkflowError, match="invalid workflow"):
        flows.load(t.id)

    raw = json.loads(good)
    raw["id"] = "wf-other"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(WorkflowError, match="does not match"):
        flows.load(t.id)

    raw = json.loads(good)
    raw["plan"]["tasks"][0]["depends_on"] = ["b"]  # cycle a <-> b
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(WorkflowError, match="invalid workflow"):
        flows.load(t.id)

    raw = json.loads(good)
    raw["source_project"] = "../evil"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(WorkflowError, match="invalid workflow"):
        flows.load(t.id)


def test_symlinks_are_refused(projects: ProjectStore, ws: Path, tmp_path: Path) -> None:
    done_project(projects, "p1", "goal")
    flows = WorkflowStore.for_workspace(ws, clock=Clock())
    t = flows.save_from_project(projects, "p1")
    target = tmp_path / "elsewhere.json"
    target.write_text(flows.path(t.id).read_text(encoding="utf-8"), encoding="utf-8")
    link = flows.root / "wf-linked.json"
    os.symlink(target, link)
    with pytest.raises(WorkflowError, match="symlink"):
        flows.load("wf-linked")

    outside = tmp_path / "outside"
    outside.mkdir()
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    os.symlink(outside, ws2 / "workflows")
    linked = WorkflowStore.for_workspace(ws2)
    with pytest.raises(WorkflowError, match="symlink"):
        linked.list_ids()
    projects2 = ProjectStore.for_workspace(ws2)
    done_project(projects2, "p1", "goal")
    with pytest.raises(WorkflowError, match="symlink"):
        linked.save_from_project(projects2, "p1")
    assert list(outside.iterdir()) == []


def test_template_model_rules() -> None:
    plan = _plan("g")
    base = {
        "id": "wf-x1",
        "goal": "g",
        "source_project": "p1",
        "source_plan_version": 1,
        "created_at": T0,
        "plan": plan,
        "stats": WorkflowStats(tasks=2, attempts=2, cost_nusd=0),
    }
    WorkflowTemplate(**base)
    for key, value in (
        ("schema_version", 2),
        ("created_at", datetime(2026, 1, 1)),
        ("goal", ""),
        ("source_plan_version", 0),
        ("id", "x"),
    ):
        with pytest.raises(ValidationError):
            WorkflowTemplate(**{**base, key: value})
    with pytest.raises(ValidationError):
        WorkflowStats(tasks=0, attempts=0, cost_nusd=0)
    with pytest.raises(ValidationError):
        WorkflowStats(tasks=1, attempts=-1, cost_nusd=0)


# --- suggesting ------------------------------------------------------------------


def test_suggest_ranks_by_similarity(projects: ProjectStore, flows: WorkflowStore) -> None:
    done_project(projects, "p1", "weather CLI tool in python")
    done_project(projects, "p2", "landing page for a bakery")
    done_project(projects, "p3", "python CLI tool for currency rates")
    t1 = flows.save_from_project(projects, "p1")
    flows.save_from_project(projects, "p2")
    t3 = flows.save_from_project(projects, "p3")

    matches = flows.suggest("a python CLI tool that shows the weather")
    assert [m.template.id for m in matches] == [t1.id, t3.id]
    assert matches[0].score > matches[1].score >= 0.2
    assert matches[0].shared == ("cli", "python", "tool", "weather")
    assert flows.suggest("python CLI tool for weather", limit=1)[0].template.id == t1.id
    assert flows.suggest("knitting patterns for winter") == []
    assert flows.suggest("") == []


def test_suggest_persian_goals(projects: ProjectStore, flows: WorkflowStore) -> None:
    done_project(projects, "p1", "یک اسکریپت پایتون برای گزارش فروش ماهانه بنویس")
    done_project(projects, "p2", "طراحی صفحه فرود برای نانوایی")
    t1 = flows.save_from_project(projects, "p1")
    flows.save_from_project(projects, "p2")
    [m] = flows.suggest("اسكريپت پايتون براي گزارش فروش هفتگی")  # Arabic yeh/kaf
    assert m.template.id == t1.id and {"اسکریپت", "پایتون", "گزارش", "فروش"} <= set(m.shared)


def test_suggest_ties_prefer_newer_and_threshold(
    projects: ProjectStore, flows: WorkflowStore
) -> None:
    done_project(projects, "p1", "weather tool")
    done_project(projects, "p2", "weather tool")
    old = flows.save_from_project(projects, "p1")
    new = flows.save_from_project(projects, "p2")
    assert [m.template.id for m in flows.suggest("weather tool")] == [new.id, old.id]
    assert flows.suggest("weather tool plus maps charts", min_score=0.5) == []
    assert len(flows.suggest("weather tool plus maps charts", min_score=0.4)) == 2


def test_suggest_argument_bounds(flows: WorkflowStore) -> None:
    for limit in (0, MAX_SUGGESTIONS + 1):
        with pytest.raises(WorkflowError, match="limit"):
            flows.suggest("x", limit=limit)
    for score in (0.0, -1.0, 1.5):
        with pytest.raises(WorkflowError, match="min_score"):
            flows.suggest("x", min_score=score)


def test_suggest_fails_closed_on_corrupt_template(
    projects: ProjectStore, flows: WorkflowStore
) -> None:
    done_project(projects, "p1", "weather tool")
    t = flows.save_from_project(projects, "p1")
    flows.path(t.id).write_text("[]", encoding="utf-8")
    with pytest.raises(WorkflowError):
        flows.suggest("weather tool")


def test_naive_clock_rejected(projects: ProjectStore, ws: Path) -> None:
    done_project(projects, "p1", "goal")
    flows = WorkflowStore.for_workspace(ws, clock=lambda: datetime(2026, 1, 1))
    with pytest.raises(WorkflowError, match="naive"):
        flows.save_from_project(projects, "p1")


# --- end to end with the Orchestrator (FakeProvider, no real API) -------------------


def test_orchestrated_project_saved_and_suggested(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path, [*write("a"), *write("b")], make_plan(task_dict("b", deps=["a"]), task_dict("a"))
    )
    assert rig.run().status is ProjectStatus.DONE
    flows = WorkflowStore.for_workspace(tmp_path / "ws", clock=Clock())
    calls_before = len(rig.provider.requests)
    t = flows.save_from_project(rig.store, PID)
    assert t.stats.tasks == 2 and t.stats.attempts == 2 and t.stats.cost_nusd > 0
    assert [m.template.id for m in flows.suggest("test goal again")] == [t.id]
    assert len(rig.provider.requests) == calls_before  # saving/suggesting never calls a model


def test_failed_orchestrated_project_cannot_be_saved(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [*write("a", "nope")] * 5, make_plan(task_dict("a")))
    rig.run(max_attempts=1)
    assert rig.project_status() is not ProjectStatus.DONE
    flows = WorkflowStore.for_workspace(tmp_path / "ws")
    with pytest.raises(WorkflowError):
        flows.save_from_project(rig.store, PID)
    assert flows.list_ids() == []
