"""Tests for ProjectStore (automa.project). Local files only; FakeProvider, no network."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from automa.budget import BudgetGuard, CostLedger
from automa.config import AutomaConfig, Tier
from automa.events import Event
from automa.llm import CallContext, ChatResponse, FakeProvider, LLMGateway, Message, Usage
from automa.project import (
    SCHEMA_VERSION,
    Project,
    ProjectConflict,
    ProjectEventLog,
    ProjectExists,
    ProjectNotFound,
    ProjectStatus,
    ProjectStore,
    ProjectStoreError,
    validate_project_id,
)
from automa.project import store as store_mod

T0 = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)


class FixedClock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(T0)


@pytest.fixture
def store(tmp_path: Path, clock: FixedClock) -> ProjectStore:
    return ProjectStore.for_workspace(tmp_path / "ws", clock=clock)


# --- ids --------------------------------------------------------------------


@pytest.mark.parametrize("good", ["a", "p1", "p-20261109-120000-abcdef", "x" * 64, "0-0"])
def test_valid_project_ids(good: str) -> None:
    assert validate_project_id(good) == good


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "-a",
        "a-",
        "A",
        "a_b",
        "a b",
        "a/b",
        "..",
        ".",
        "../x",
        "a\\b",
        "x" * 65,
        "é",
        "a\n",
        "a\x00",
    ],
)
def test_invalid_project_ids_rejected(store: ProjectStore, bad: str) -> None:
    with pytest.raises(ProjectStoreError):
        validate_project_id(bad)
    with pytest.raises(ProjectStoreError):
        store.project_dir(bad)
    with pytest.raises(ProjectStoreError):
        store.create("goal", project_id=bad)
    with pytest.raises(ProjectStoreError):
        store.load(bad)


def test_validate_project_id_rejects_non_string() -> None:
    with pytest.raises(ProjectStoreError):
        validate_project_id(123)  # type: ignore[arg-type]


def test_new_id_format_and_uniqueness(store: ProjectStore) -> None:
    ids = {store.new_id() for _ in range(50)}
    assert len(ids) == 50
    for pid in ids:
        assert pid.startswith("p-20261109-120000-")
        validate_project_id(pid)


# --- create / load ----------------------------------------------------------


def test_for_workspace_root(tmp_path: Path) -> None:
    assert ProjectStore.for_workspace(tmp_path).root == tmp_path / "projects"


def test_create_builds_layout(store: ProjectStore) -> None:
    p = store.create("Build a CLI tool", project_id="demo")
    d = store.root / "demo"
    assert d.is_dir() and (d / "tasks").is_dir() and (d / "files").is_dir()
    assert (d / "project.json").is_file() and (d / "events.jsonl").is_file()
    assert not (d / "plan.json").exists()
    assert store.plan_file("demo") == d / "plan.json"
    assert store.tasks_dir("demo") == d / "tasks"
    assert store.files_dir("demo") == d / "files"
    assert p.id == "demo" and p.goal == "Build a CLI tool"
    assert p.status is ProjectStatus.NEW and p.revision == 0
    assert p.created_at == p.updated_at == T0
    assert p.schema_version == SCHEMA_VERSION


def test_create_generates_id(store: ProjectStore) -> None:
    p = store.create("goal")
    assert p.id.startswith("p-20261109-120000-")
    assert store.exists(p.id)


def test_create_round_trip_unicode_goal(store: ProjectStore) -> None:
    goal = "یک ابزار CLI ساده بساز 🚀"
    store.create(goal, project_id="fa")
    assert store.load("fa").goal == goal
    raw = (store.root / "fa" / "project.json").read_text(encoding="utf-8")
    assert goal in raw  # stored readable, not \u-escaped


def test_project_json_is_plain_json(store: ProjectStore) -> None:
    store.create("goal", project_id="p1")
    data = json.loads((store.root / "p1" / "project.json").read_text(encoding="utf-8"))
    assert data == {
        "schema_version": 1,
        "id": "p1",
        "goal": "goal",
        "status": "new",
        "created_at": "2026-11-09T12:00:00Z",
        "updated_at": "2026-11-09T12:00:00Z",
        "revision": 0,
    }


def test_create_logs_project_created_event(store: ProjectStore) -> None:
    store.create("my goal", project_id="p1")
    events = store.events("p1").read()
    assert [(e.type, e.project_id, e.data, e.ts) for e in events] == [
        ("project.created", "p1", {"goal": "my goal"}, T0)
    ]


def test_create_never_overwrites(store: ProjectStore) -> None:
    store.create("first", project_id="p1")
    with pytest.raises(ProjectExists):
        store.create("second", project_id="p1")
    assert store.load("p1").goal == "first"
    assert len(store.events("p1").read()) == 1


def test_create_refuses_existing_directory_without_project_file(store: ProjectStore) -> None:
    (store.root / "p1").mkdir(parents=True)
    (store.root / "p1" / "keep.txt").write_text("user data")
    with pytest.raises(ProjectExists):
        store.create("goal", project_id="p1")
    assert (store.root / "p1" / "keep.txt").read_text() == "user data"


@pytest.mark.parametrize("goal", ["", "   ", "\n\t", "x" * 10_001])
def test_create_rejects_bad_goal(store: ProjectStore, goal: str) -> None:
    with pytest.raises(ProjectStoreError):
        store.create(goal, project_id="p1")
    assert not (store.root / "p1").exists()


def test_concurrent_create_same_id_only_one_wins(store: ProjectStore) -> None:
    results: list[str] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        other = ProjectStore(store.root, clock=lambda: T0)  # separate store, same root
        try:
            other.create(f"goal {i}", project_id="race")
            outcome = "ok"
        except ProjectExists:
            outcome = "exists"
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 1 and results.count("exists") == 19
    assert len(store.events("race").read()) == 1


def test_load_missing_project(store: ProjectStore) -> None:
    with pytest.raises(ProjectNotFound):
        store.load("nope")
    with pytest.raises(ProjectNotFound):
        store.events("nope")
    assert not store.exists("nope")


def test_clock_must_be_timezone_aware(tmp_path: Path) -> None:
    s = ProjectStore(tmp_path, clock=lambda: datetime(2026, 11, 9, 12, 0))  # noqa: DTZ001
    with pytest.raises(ProjectStoreError):
        s.create("goal", project_id="p1")


def test_clock_normalized_to_utc(tmp_path: Path) -> None:
    tehran = timezone(timedelta(hours=3, minutes=30))
    s = ProjectStore(tmp_path, clock=lambda: datetime(2026, 11, 9, 15, 30, tzinfo=tehran))
    p = s.create("goal", project_id="p1")
    assert p.created_at == T0 and p.created_at.tzinfo is UTC


# --- corrupt / foreign files ------------------------------------------------


def _write_project_json(store: ProjectStore, pid: str, data: object) -> None:
    (store.root / pid).mkdir(parents=True, exist_ok=True)
    text = data if isinstance(data, str) else json.dumps(data)
    (store.root / pid / "project.json").write_text(text, encoding="utf-8")


def _valid(pid: str = "p1", **over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": 1,
        "id": pid,
        "goal": "g",
        "status": "new",
        "created_at": "2026-11-09T12:00:00Z",
        "updated_at": "2026-11-09T12:00:00Z",
        "revision": 0,
    }
    base.update(over)
    return base


def test_load_accepts_hand_written_valid_file(store: ProjectStore) -> None:
    _write_project_json(store, "p1", _valid())
    assert store.load("p1").goal == "g"


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "",
        "[]",
        json.dumps(_valid(schema_version=2)),
        json.dumps(_valid(schema_version=0)),
        json.dumps(_valid(status="exploded")),
        json.dumps(_valid(revision=-1)),
        json.dumps(_valid(goal="")),
        json.dumps(_valid(created_at="2026-11-09T12:00:00")),  # naive
        json.dumps(_valid(extra_field=1)),
        json.dumps({k: v for k, v in _valid().items() if k != "goal"}),
    ],
)
def test_load_rejects_corrupt_or_unknown_files(store: ProjectStore, content: str) -> None:
    _write_project_json(store, "p1", content)
    with pytest.raises(ProjectStoreError):
        store.load("p1")


def test_load_rejects_id_mismatch(store: ProjectStore) -> None:
    _write_project_json(store, "p1", _valid("p2"))
    with pytest.raises(ProjectStoreError, match="does not match"):
        store.load("p1")


def test_load_rejects_non_utf8(store: ProjectStore) -> None:
    (store.root / "p1").mkdir(parents=True)
    (store.root / "p1" / "project.json").write_bytes(b"\xff\xfe{}")
    with pytest.raises(ProjectStoreError):
        store.load("p1")


def test_symlinked_project_dir_rejected(store: ProjectStore, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "project.json").write_text(json.dumps(_valid("evil")))
    store.root.mkdir(parents=True)
    os.symlink(outside, store.root / "evil")
    with pytest.raises(ProjectStoreError, match="escapes"):
        store.load("evil")
    with pytest.raises(ProjectStoreError):
        store.files_dir("evil")
    assert "evil" not in store.list_ids()


def test_project_file_that_is_a_directory(store: ProjectStore) -> None:
    (store.root / "p1" / "project.json").mkdir(parents=True)
    assert not store.exists("p1")
    with pytest.raises(ProjectNotFound):
        store.load("p1")


# --- list -------------------------------------------------------------------


def test_list_empty_when_root_missing(store: ProjectStore) -> None:
    assert not store.root.exists()
    assert store.list_ids() == []
    assert store.list_projects() == []


def test_list_ignores_non_projects(store: ProjectStore, clock: FixedClock) -> None:
    store.create("b", project_id="b")
    clock.advance(1)
    store.create("a", project_id="a")
    (store.root / "empty-dir").mkdir()
    (store.root / "Bad_Name").mkdir()
    (store.root / "Bad_Name" / "project.json").write_text(json.dumps(_valid("x")))
    (store.root / "stray.txt").write_text("x")
    assert store.list_ids() == ["a", "b"]
    assert [p.id for p in store.list_projects()] == ["b", "a"]  # oldest first


def test_list_projects_raises_on_corrupt_project(store: ProjectStore) -> None:
    store.create("ok", project_id="ok")
    _write_project_json(store, "bad", "{")
    with pytest.raises(ProjectStoreError):
        store.list_projects()


# --- save / status ----------------------------------------------------------


def test_save_bumps_revision_and_updated_at(store: ProjectStore, clock: FixedClock) -> None:
    p = store.create("goal", project_id="p1")
    clock.advance(60)
    saved = store.save(p.model_copy(update={"status": ProjectStatus.PLANNING}))
    assert saved.revision == 1 and saved.status is ProjectStatus.PLANNING
    assert saved.updated_at == T0 + timedelta(seconds=60) and saved.created_at == T0
    assert store.load("p1") == saved


def test_save_detects_stale_revision(store: ProjectStore) -> None:
    p = store.create("goal", project_id="p1")
    store.save(p.model_copy(update={"status": ProjectStatus.RUNNING}))
    with pytest.raises(ProjectConflict):
        store.save(p.model_copy(update={"status": ProjectStatus.DONE}))
    assert store.load("p1").status is ProjectStatus.RUNNING


def test_save_rejects_forged_revision(store: ProjectStore) -> None:
    p = store.create("goal", project_id="p1")
    with pytest.raises(ProjectConflict):
        store.save(p.model_copy(update={"revision": 5}))


def test_save_missing_project(store: ProjectStore) -> None:
    p = Project(id="ghost", goal="g", created_at=T0, updated_at=T0)
    with pytest.raises(ProjectNotFound):
        store.save(p)
    assert not (store.root / "ghost").exists()


def test_save_cannot_change_created_at(store: ProjectStore) -> None:
    p = store.create("goal", project_id="p1")
    with pytest.raises(ProjectStoreError):
        store.save(p.model_copy(update={"created_at": T0 - timedelta(days=1)}))


def test_save_revalidates_bad_copy(store: ProjectStore) -> None:
    p = store.create("goal", project_id="p1")
    with pytest.raises(ProjectStoreError):
        store.save(p.model_copy(update={"goal": "   "}))
    assert store.load("p1").goal == "goal"


def test_updated_at_never_goes_backwards(store: ProjectStore, clock: FixedClock) -> None:
    p = store.create("goal", project_id="p1")
    clock.advance(-3600)  # system clock jumped back
    saved = store.save(p)
    assert saved.updated_at == T0


def test_save_is_atomic_on_write_failure(
    store: ProjectStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = store.create("goal", project_id="p1")
    before = (store.root / "p1" / "project.json").read_bytes()

    def boom(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store_mod.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        store.save(p.model_copy(update={"status": ProjectStatus.RUNNING}))
    assert (store.root / "p1" / "project.json").read_bytes() == before
    leftovers = [f.name for f in (store.root / "p1").iterdir() if f.name.endswith(".tmp")]
    assert leftovers == []


def test_set_status_logs_event(store: ProjectStore, clock: FixedClock) -> None:
    store.create("goal", project_id="p1")
    clock.advance(5)
    p = store.set_status("p1", ProjectStatus.RUNNING)
    assert p.status is ProjectStatus.RUNNING and p.revision == 1
    last = store.events("p1").read()[-1]
    assert last.type == "project.status"
    assert last.data == {"from": "new", "to": "running"}
    assert last.ts == T0 + timedelta(seconds=5)


def test_set_status_rejects_unknown_status(store: ProjectStore) -> None:
    store.create("goal", project_id="p1")
    with pytest.raises(ValueError):
        store.set_status("p1", "exploded")  # type: ignore[arg-type]
    assert store.load("p1").revision == 0


def test_concurrent_saves_lose_nothing_silently(store: ProjectStore) -> None:
    store.create("goal", project_id="p1")
    outcomes: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(10)

    def worker() -> None:
        p = store.load("p1")
        barrier.wait()
        try:
            store.save(p.model_copy(update={"status": ProjectStatus.RUNNING}))
            outcome = "ok"
        except ProjectConflict:
            outcome = "conflict"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("ok") == 1
    assert store.load("p1").revision == 1


def test_project_model_is_frozen() -> None:
    p = Project(id="p1", goal="g", created_at=T0, updated_at=T0)
    with pytest.raises(ValidationError):
        p.goal = "other"  # type: ignore[misc]


# --- events -----------------------------------------------------------------


def test_project_event_log_fills_and_enforces_project_id(store: ProjectStore) -> None:
    store.create("goal", project_id="p1")
    store.create("goal", project_id="p2")
    log = store.events("p1")
    assert isinstance(log, ProjectEventLog) and log.project_id == "p1"
    log.emit(Event(ts=T0, type="task.started"))
    with pytest.raises(ProjectStoreError):
        log.emit(Event(ts=T0, type="task.started", project_id="p2"))
    events = log.read()
    assert [(e.type, e.project_id) for e in events] == [
        ("project.created", "p1"),
        ("task.started", "p1"),
    ]
    assert [e.type for e in store.events("p2").read()] == ["project.created"]


def test_project_event_log_rejects_bad_id(tmp_path: Path) -> None:
    with pytest.raises(ProjectStoreError):
        ProjectEventLog(tmp_path / "events.jsonl", "../x")


def test_events_survive_reopen(tmp_path: Path, clock: FixedClock) -> None:
    s1 = ProjectStore.for_workspace(tmp_path, clock=clock)
    s1.create("goal", project_id="p1")
    s1.events("p1").emit(Event(ts=T0, type="task.done"))
    s2 = ProjectStore.for_workspace(tmp_path, clock=clock)
    assert [e.type for e in s2.events("p1").read()] == ["project.created", "task.done"]


# --- integration: BudgetGuard writes into the project's events.jsonl ---------


@pytest.fixture
def ledger(clock: FixedClock) -> Iterator[CostLedger]:
    with CostLedger(":memory:", clock=clock) as led:
        yield led


def test_budget_warnings_land_in_project_events(store: ProjectStore, ledger: CostLedger) -> None:
    config = AutomaConfig.model_validate(
        {
            "tiers": {"small": "m-small"},
            "prices": {"m-small": {"input_per_mtok": 1000.0, "output_per_mtok": 1000.0}},
            "limits": {"project_usd": 1.0, "daily_usd": 2.0, "per_call_usd": 0.5},
        }
    )
    project = store.create("goal", project_id="p1")
    usage = Usage(input_tokens=100, output_tokens=100)  # $0.2 per call
    fake = FakeProvider(
        [ChatResponse(model="m-small", message=Message.assistant("x"), usage=usage)] * 3
    )
    guard = BudgetGuard(config.limits, ledger, events=store.events(project.id))
    gw = LLMGateway(config, fake, guard=guard)
    ctx = CallContext(project_id=project.id, task_id="t1")
    for _ in range(3):
        gw.complete([Message.user("hi")], tier=Tier.SMALL, max_tokens=200, context=ctx)
    got = [(e.type, e.data.get("threshold_pct")) for e in store.events("p1").read()]
    assert got == [("project.created", None), ("budget.warning", 50)]
    assert ledger.project_cost("p1") == pytest.approx(0.6)
