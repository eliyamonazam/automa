"""Tests for ReviewStore (automa.reviews): storing review proposals and the user's decisions.

Local only: no network, no real model calls (one end-to-end test uses FakeProvider
behind a real LLMGateway + BudgetGuard). Only accepted proposals may become memory,
always with source = review.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from automa.agents import Proposal, ReviewAgent
from automa.agents.reviewer import Review
from automa.budget import BudgetGuard, CostLedger
from automa.config import AutomaConfig
from automa.llm import FakeProvider, LLMGateway
from automa.memory import MemoryScope, MemorySource, MemoryStore
from automa.memory.store import MAX_ENTRIES_PER_LAYER
from automa.project import ProjectNotFound, ProjectStatus, ProjectStore
from automa.reviews import (
    MAX_DECISION_FILE_BYTES,
    MAX_REVIEW_FILE_BYTES,
    MAX_REVIEWS_PER_PROJECT,
    REVIEWS_DIR,
    DecisionStatus,
    ProposalDecision,
    ReviewNotFound,
    ReviewStore,
    ReviewStoreError,
    StoredReview,
    validate_review_id,
)

T0 = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    return tmp_path / "ws"


@pytest.fixture
def projects(ws: Path, clock: Clock) -> ProjectStore:
    return ProjectStore.for_workspace(ws, clock=clock)


@pytest.fixture
def memory(ws: Path, clock: Clock) -> MemoryStore:
    return MemoryStore.for_workspace(ws, clock=clock)


@pytest.fixture
def store(projects: ProjectStore, memory: MemoryStore, clock: Clock) -> ReviewStore:
    return ReviewStore(projects, memory, clock=clock)


def finished(
    projects: ProjectStore, pid: str = "p1", status: ProjectStatus = ProjectStatus.DONE
) -> str:
    projects.create("build a CLI", project_id=pid)
    projects.set_status(pid, status)
    return pid


def review(
    pid: str = "p1",
    *proposals: tuple[str, str],
    status: ProjectStatus = ProjectStatus.DONE,
    dropped: int = 0,
) -> Review:
    if not proposals:
        proposals = (("project", "uses typer for the CLI"), ("lesson", "install pytest first"))
    return Review(
        project_id=pid,
        project_status=status,
        proposals=[Proposal(scope=s, text=t) for s, t in proposals],  # type: ignore[arg-type]
        dropped_duplicates=dropped,
        created_at=T0,
    )


def rdir(projects: ProjectStore, pid: str = "p1") -> Path:
    return projects.project_dir(pid) / REVIEWS_DIR


def events(projects: ProjectStore, pid: str = "p1") -> list[dict[str, Any]]:
    path = projects.project_dir(pid) / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def review_events(projects: ProjectStore, pid: str = "p1") -> list[dict[str, Any]]:
    return [e for e in events(projects, pid) if e["type"].startswith("review.")]


# -- save --------------------------------------------------------------------


def test_save_writes_one_file_and_no_memory(
    store: ReviewStore, projects: ProjectStore, memory: MemoryStore
) -> None:
    pid = finished(projects)
    stored = store.save(review(pid, dropped=2))
    validate_review_id(stored.id)
    assert stored.project_id == pid
    assert stored.project_status is ProjectStatus.DONE
    assert stored.dropped_duplicates == 2
    assert stored.created_at == T0
    assert [p.text for p in stored.proposals] == ["uses typer for the CLI", "install pytest first"]
    files = sorted(p.name for p in rdir(projects).iterdir())
    assert files == [f"{stored.id}.json"]
    assert not memory.root.exists()
    assert memory.context(pid).is_empty()
    ev = review_events(projects)
    assert [e["type"] for e in ev] == ["review.saved"]
    assert ev[0]["data"] == {"review_id": stored.id, "proposals": 2, "lessons": 1}


def test_save_roundtrip_and_list(store: ReviewStore, projects: ProjectStore) -> None:
    pid = finished(projects)
    a = store.save(review(pid))
    b = store.save(review(pid, ("lesson", "something else")))
    assert a.id != b.id
    assert store.get(pid, a.id) == a
    assert store.all(pid) == [a, b]
    assert store.latest(pid) == b
    assert store.pending(pid, a.id) == [1, 2]
    assert store.decisions(pid, a.id) == {}


def test_save_failed_project(store: ReviewStore, projects: ProjectStore) -> None:
    pid = finished(projects, status=ProjectStatus.FAILED)
    stored = store.save(review(pid, status=ProjectStatus.FAILED))
    assert stored.project_status is ProjectStatus.FAILED


def test_save_empty_review(store: ReviewStore, projects: ProjectStore) -> None:
    pid = finished(projects)
    r = Review(project_id=pid, project_status=ProjectStatus.DONE, proposals=[], created_at=T0)
    stored = store.save(r)
    assert stored.proposals == []
    assert store.pending(pid, stored.id) == []


def test_list_and_latest_without_reviews(store: ReviewStore, projects: ProjectStore) -> None:
    pid = finished(projects)
    assert store.all(pid) == []
    assert store.latest(pid) is None
    assert not rdir(projects).exists()  # reading creates nothing


@pytest.mark.parametrize(
    "status",
    [ProjectStatus.NEW, ProjectStatus.PLANNING, ProjectStatus.RUNNING, ProjectStatus.PAUSED],
)
def test_save_rejects_unfinished_project(
    store: ReviewStore, projects: ProjectStore, status: ProjectStatus
) -> None:
    projects.create("x", project_id="p1")
    if status is not ProjectStatus.NEW:
        projects.set_status("p1", status)
    with pytest.raises(ReviewStoreError, match="only reviews of done or failed"):
        store.save(review("p1"))
    assert not rdir(projects).exists()


def test_save_rejects_status_changed_since_review(
    store: ReviewStore, projects: ProjectStore
) -> None:
    pid = finished(projects, status=ProjectStatus.FAILED)
    with pytest.raises(ReviewStoreError, match="now failed"):
        store.save(review(pid, status=ProjectStatus.DONE))
    assert not rdir(projects).exists()


def test_save_rejects_missing_project(store: ReviewStore) -> None:
    with pytest.raises(ProjectNotFound):
        store.save(review("p1"))


def test_save_rejects_non_review(store: ReviewStore, projects: ProjectStore) -> None:
    finished(projects)
    with pytest.raises(TypeError):
        store.save({"project_id": "p1"})  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_save_revalidates_tampered_review(store: ReviewStore, projects: ProjectStore) -> None:
    pid = finished(projects)
    bad = review(pid).model_copy(update={"proposals": [{"scope": "user", "text": "x"}]})
    with pytest.raises(ReviewStoreError, match="invalid review"):
        store.save(bad)
    assert not rdir(projects).exists()


def test_save_rejects_duplicate_proposals(store: ReviewStore, projects: ProjectStore) -> None:
    pid = finished(projects)
    r = review(pid, ("lesson", "Install  pytest"), ("lesson", "install pytest"))
    with pytest.raises(ReviewStoreError, match="same note twice"):
        store.save(r)
    # Same text in different layers is fine.
    store.save(review(pid, ("lesson", "install pytest"), ("project", "install pytest")))


def test_save_rejects_naive_created_at(store: ReviewStore, projects: ProjectStore) -> None:
    pid = finished(projects)
    r = review(pid).model_copy(update={"created_at": datetime(2026, 1, 1)})
    with pytest.raises(ReviewStoreError, match="timezone-aware"):
        store.save(r)


def test_save_rejects_naive_clock(projects: ProjectStore, memory: MemoryStore) -> None:
    pid = finished(projects)
    s = ReviewStore(projects, memory, clock=lambda: datetime(2026, 1, 1))
    with pytest.raises(ReviewStoreError, match="naive"):
        s.save(review(pid))


def test_save_limit_per_project(
    store: ReviewStore, projects: ProjectStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import automa.reviews as mod

    monkeypatch.setattr(mod, "MAX_REVIEWS_PER_PROJECT", 2)
    pid = finished(projects)
    store.save(review(pid))
    store.save(review(pid))
    with pytest.raises(ReviewStoreError, match="already has 2 reviews"):
        store.save(review(pid))
    assert MAX_REVIEWS_PER_PROJECT == 50


def test_saved_file_is_never_overwritten(store: ReviewStore, projects: ProjectStore) -> None:
    pid = finished(projects)
    stored = store.save(review(pid))
    path = rdir(projects) / f"{stored.id}.json"
    before = path.read_bytes()
    store.accept(pid, stored.id, 1)
    store.reject(pid, stored.id, 2)
    assert path.read_bytes() == before


def test_reviews_of_projects_are_separate(store: ReviewStore, projects: ProjectStore) -> None:
    finished(projects, "p1")
    finished(projects, "p2")
    a = store.save(review("p1"))
    assert store.all("p2") == []
    with pytest.raises(ReviewNotFound):
        store.get("p2", a.id)


# -- accept / reject ---------------------------------------------------------


def test_accept_project_proposal_writes_project_memory(
    store: ReviewStore, projects: ProjectStore, memory: MemoryStore
) -> None:
    pid = finished(projects)
    stored = store.save(review(pid))
    d = store.accept(pid, stored.id, 1)
    assert d.status is DecisionStatus.ACCEPTED
    assert d.number == 1 and d.review_id == stored.id and d.project_id == pid
    assert not d.memory_existed
    entries = memory.entries(MemoryScope.PROJECT, pid)
    assert len(entries) == 1
    e = entries[0]
    assert e.id == d.memory_id
    assert e.text == "uses typer for the CLI"
    assert e.source is MemorySource.REVIEW
    assert e.project_id == pid
    assert memory.entries(MemoryScope.LESSON) == []
    assert memory.entries(MemoryScope.USER) == []
    assert store.pending(pid, stored.id) == [2]
    assert store.decisions(pid, stored.id) == {1: d}


def test_accept_lesson_writes_lessons_layer(
    store: ReviewStore, projects: ProjectStore, memory: MemoryStore
) -> None:
    pid = finished(projects)
    stored = store.save(review(pid))
    d = store.accept(pid, stored.id, 2)
    [e] = memory.entries(MemoryScope.LESSON)
    assert e.id == d.memory_id
    assert e.project_id is None
    assert e.source is MemorySource.REVIEW
    assert memory.entries(MemoryScope.PROJECT, pid) == []


def test_reject_writes_no_memory(
    store: ReviewStore, projects: ProjectStore, memory: MemoryStore
) -> None:
    pid = finished(projects)
    stored = store.save(review(pid))
    d = store.reject(pid, stored.id, 1)
    assert d.status is DecisionStatus.REJECTED
    assert d.memory_id is None
    assert not memory.root.exists()
    assert store.pending(pid, stored.id) == [2]


def test_decision_events(store: ReviewStore, projects: ProjectStore) -> None:
    pid = finished(projects)
    stored = store.save(review(pid))
    a = store.accept(pid, stored.id, 2)
    store.reject(pid, stored.id, 1)
    ev = review_events(projects)
    assert [e["type"] for e in ev] == [
        "review.saved",
        "review.proposal_accepted",
        "review.proposal_rejected",
    ]
    assert ev[1]["data"] == {
        "review_id": stored.id,
        "number": 2,
        "scope": "lesson",
        "memory_id": a.memory_id,
        "memory_existed": False,
    }
    assert ev[2]["data"] == {"review_id": stored.id, "number": 1, "scope": "project"}
    # Events never carry the (model-written) proposal text.
    assert "install pytest" not in json.dumps(ev)


@pytest.mark.parametrize("second", ["accept", "reject"])
@pytest.mark.parametrize("first", ["accept", "reject"])
def test_decision_only_once(
    store: ReviewStore, projects: ProjectStore, memory: MemoryStore, first: str, second: str
) -> None:
    pid = finished(projects)
    stored = store.save(review(pid))
    getattr(store, first)(pid, stored.id, 2)
    with pytest.raises(ReviewStoreError, match="already"):
        getattr(store, second)(pid, stored.id, 2)
    assert len(memory.entries(MemoryScope.LESSON)) == (1 if first == "accept" else 0)


@pytest.mark.parametrize("number", [0, -1, 3, 9, True, "1", 1.0])
def test_invalid_proposal_number(
    store: ReviewStore, projects: ProjectStore, memory: MemoryStore, number: Any
) -> None:
    pid = finished(projects)
    stored = store.save(review(pid))
    with pytest.raises(ReviewStoreError):
        store.accept(pid, stored.id, number)
    with pytest.raises(ReviewStoreError):
        store.reject(pid, stored.id, number)
    assert not memory.root.exists()
    assert store.pending(pid, stored.id) == [1, 2]


@pytest.mark.parametrize("rid", ["", "rev-1", "../x", "rev-20261113-120000-ABCDEF", "apr-x"])
def test_invalid_review_id(store: ReviewStore, projects: ProjectStore, rid: str) -> None:
    pid = finished(projects)
    with pytest.raises(ReviewStoreError, match="invalid review id"):
        store.get(pid, rid)
    with pytest.raises(ReviewStoreError, match="invalid review id"):
        store.accept(pid, rid, 1)


def test_unknown_review(store: ReviewStore, projects: ProjectStore) -> None:
    pid = finished(projects)
    with pytest.raises(ReviewNotFound):
        store.accept(pid, "rev-20261113-120000-abcdef", 1)
    with pytest.raises(ReviewNotFound):
        store.pending(pid, "rev-20261113-120000-abcdef")


def test_accept_after_project_deleted(store: ReviewStore, projects: ProjectStore) -> None:
    import shutil

    pid = finished(projects)
    stored = store.save(review(pid))
    shutil.rmtree(projects.project_dir(pid))
    with pytest.raises(ProjectNotFound):
        store.accept(pid, stored.id, 1)


def test_accept_is_allowed_after_project_rerun(
    store: ReviewStore, projects: ProjectStore, memory: MemoryStore
) -> None:
    # The decision is about the stored proposal; a later status change does not void it.
    pid = finished(projects, status=ProjectStatus.FAILED)
    stored = store.save(review(pid, status=ProjectStatus.FAILED))
    projects.set_status(pid, ProjectStatus.RUNNING)
    store.accept(pid, stored.id, 2)
    assert len(memory.entries(MemoryScope.LESSON)) == 1


def test_accept_existing_identical_memory_records_it(
    store: ReviewStore, projects: ProjectStore, memory: MemoryStore
) -> None:
    pid = finished(projects)
    old = memory.add(MemoryScope.LESSON, "Install   PYTEST first")
    stored = store.save(review(pid))
    d = store.accept(pid, stored.id, 2)
    assert d.memory_existed and d.memory_id == old.id
    assert memory.entries(MemoryScope.LESSON) == [old]  # no second entry


def test_accept_identical_project_memory_of_other_project_is_not_reused(
    store: ReviewStore, projects: ProjectStore, memory: MemoryStore
) -> None:
    pid = finished(projects, "p1")
    finished(projects, "p2")
    other = memory.add(MemoryScope.PROJECT, "uses typer for the CLI", project_id="p2")
    stored = store.save(review(pid))
    d = store.accept(pid, stored.id, 1)
    assert not d.memory_existed and d.memory_id != other.id
    assert len(memory.entries(MemoryScope.PROJECT, "p1")) == 1


def test_accept_same_lesson_from_two_reviews(
    store: ReviewStore, projects: ProjectStore, memory: MemoryStore
) -> None:
    pid = finished(projects)
    a = store.save(review(pid))
    b = store.save(review(pid))
    da = store.accept(pid, a.id, 2)
    db = store.accept(pid, b.id, 2)
    assert db.memory_existed and db.memory_id == da.memory_id
    assert len(memory.entries(MemoryScope.LESSON)) == 1


def test_accept_when_memory_full_records_nothing(
    store: ReviewStore,
    projects: ProjectStore,
    memory: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import automa.memory.store as mstore

    monkeypatch.setattr(mstore, "MAX_ENTRIES_PER_LAYER", 1)
    pid = finished(projects)
    memory.add(MemoryScope.LESSON, "an older lesson")
    stored = store.save(review(pid))
    with pytest.raises(ReviewStoreError, match="cannot write the accepted proposal"):
        store.accept(pid, stored.id, 2)
    assert store.pending(pid, stored.id) == [1, 2]
    assert MAX_ENTRIES_PER_LAYER == 200


def test_accept_with_broken_memory_records_nothing(
    store: ReviewStore, projects: ProjectStore, memory: MemoryStore
) -> None:
    pid = finished(projects)
    stored = store.save(review(pid))
    lessons = memory.root / "lessons"
    lessons.mkdir(parents=True)
    (lessons / "junk.txt").write_text("x")
    with pytest.raises(ReviewStoreError, match="cannot write"):
        store.accept(pid, stored.id, 2)
    assert store.pending(pid, stored.id) == [1, 2]
    # Rejecting does not touch memory, so it still works.
    store.reject(pid, stored.id, 2)


def test_failed_decision_write_removes_new_memory(
    store: ReviewStore,
    projects: ProjectStore,
    memory: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import automa.reviews as mod

    pid = finished(projects)
    stored = store.save(review(pid))

    def boom(path: Path, data: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(mod, "_write_json_exclusive", boom)
    with pytest.raises(OSError, match="disk full"):
        store.accept(pid, stored.id, 2)
    assert memory.entries(MemoryScope.LESSON) == []
    monkeypatch.undo()
    assert store.pending(pid, stored.id) == [1, 2]


def test_failed_decision_write_keeps_preexisting_memory(
    store: ReviewStore,
    projects: ProjectStore,
    memory: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import automa.reviews as mod

    pid = finished(projects)
    old = memory.add(MemoryScope.LESSON, "install pytest first")
    stored = store.save(review(pid))

    def boom(path: Path, data: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(mod, "_write_json_exclusive", boom)
    with pytest.raises(OSError):
        store.accept(pid, stored.id, 2)
    assert memory.entries(MemoryScope.LESSON) == [old]


def test_concurrent_decision_in_other_process_rolls_back(
    store: ReviewStore,
    projects: ProjectStore,
    memory: MemoryStore,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate another process deciding between our check and our write.
    import automa.reviews as mod

    pid = finished(projects)
    stored = store.save(review(pid))
    real = mod._write_json_exclusive

    def racing(path: Path, data: Any) -> None:
        other = ProposalDecision(
            review_id=stored.id,
            project_id=pid,
            number=2,
            status=DecisionStatus.REJECTED,
            decided_at=T0,
        )
        real(path, other.model_dump(mode="json"))
        real(path, data)  # ours now fails: the file exists

    monkeypatch.setattr(mod, "_write_json_exclusive", racing)
    with pytest.raises(ReviewStoreError, match="decided concurrently"):
        store.accept(pid, stored.id, 2)
    monkeypatch.undo()
    assert memory.entries(MemoryScope.LESSON) == []
    assert store.decisions(pid, stored.id)[2].status is DecisionStatus.REJECTED


def test_crash_after_memory_write_recovers_on_retry(
    store: ReviewStore, projects: ProjectStore, memory: MemoryStore
) -> None:
    # The process died after memory.add but before the decision file existed.
    pid = finished(projects)
    stored = store.save(review(pid))
    orphan = memory.add(MemoryScope.LESSON, "install pytest first", source=MemorySource.REVIEW)
    d = store.accept(pid, stored.id, 2)
    assert d.memory_existed and d.memory_id == orphan.id
    assert len(memory.entries(MemoryScope.LESSON)) == 1


def test_threads_decide_once(
    store: ReviewStore, projects: ProjectStore, memory: MemoryStore
) -> None:
    pid = finished(projects)
    stored = store.save(review(pid))
    results: list[str] = []

    def go() -> None:
        try:
            store.accept(pid, stored.id, 2)
            results.append("ok")
        except ReviewStoreError:
            results.append("err")

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 1
    assert len(memory.entries(MemoryScope.LESSON)) == 1


def test_accepted_memory_never_user_scope(
    store: ReviewStore, projects: ProjectStore, memory: MemoryStore
) -> None:
    pid = finished(projects)
    stored = store.save(review(pid))
    store.accept(pid, stored.id, 1)
    store.accept(pid, stored.id, 2)
    assert memory.entries(MemoryScope.USER) == []
    ctx = memory.context(pid)
    assert all(e.source is MemorySource.REVIEW for e in ctx.project + ctx.lessons)


def test_persian_text(store: ReviewStore, projects: ProjectStore, memory: MemoryStore) -> None:
    pid = finished(projects)
    text = "قبل از اجرای تست‌ها pytest را نصب کن"
    stored = store.save(review(pid, ("lesson", text)))
    raw = (rdir(projects) / f"{stored.id}.json").read_text(encoding="utf-8")
    assert text in raw
    store.accept(pid, stored.id, 1)
    assert memory.entries(MemoryScope.LESSON)[0].text == text


# -- fail closed on disk -----------------------------------------------------


def _saved(store: ReviewStore, projects: ProjectStore) -> tuple[str, StoredReview, Path]:
    pid = finished(projects)
    stored = store.save(review(pid))
    return pid, stored, rdir(projects) / f"{stored.id}.json"


def test_corrupt_review_json(store: ReviewStore, projects: ProjectStore) -> None:
    pid, stored, path = _saved(store, projects)
    path.write_text("{not json", encoding="utf-8")
    for call in (
        lambda: store.all(pid),
        lambda: store.get(pid, stored.id),
        lambda: store.accept(pid, stored.id, 1),
    ):
        with pytest.raises(ReviewStoreError, match="unreadable review"):
            call()


def test_invalid_review_content(store: ReviewStore, projects: ProjectStore) -> None:
    pid, stored, path = _saved(store, projects)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["proposals"][0]["scope"] = "user"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReviewStoreError, match="invalid review file"):
        store.all(pid)


@pytest.mark.parametrize(
    "field, value",
    [
        ("id", "rev-20200101-000000-aaaaaa"),
        ("project_id", "p2"),
        ("schema_version", 2),
        ("project_status", "running"),
        ("extra", 1),
    ],
)
def test_review_file_tampering(
    store: ReviewStore, projects: ProjectStore, field: str, value: Any
) -> None:
    pid, stored, path = _saved(store, projects)
    data = json.loads(path.read_text(encoding="utf-8"))
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReviewStoreError):
        store.get(pid, stored.id)


def test_oversized_review_file(store: ReviewStore, projects: ProjectStore) -> None:
    pid, stored, path = _saved(store, projects)
    path.write_bytes(b" " * (MAX_REVIEW_FILE_BYTES + 1))
    with pytest.raises(ReviewStoreError, match="larger than"):
        store.get(pid, stored.id)


def test_max_size_review_fits_limit(store: ReviewStore, projects: ProjectStore) -> None:
    pid = finished(projects)
    # Worst case: 8 proposals of 1000 chars, each a 4-byte UTF-8 character escaped by JSON.
    props = tuple(("lesson", f"{i}" + "𝔸" * 999) for i in range(8))
    stored = store.save(review(pid, *props))
    size = (rdir(projects) / f"{stored.id}.json").stat().st_size
    assert size < MAX_REVIEW_FILE_BYTES
    assert store.get(pid, stored.id) == stored


def test_review_file_symlink(store: ReviewStore, projects: ProjectStore, tmp_path: Path) -> None:
    pid, stored, path = _saved(store, projects)
    target = tmp_path / "elsewhere.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ReviewStoreError, match="cannot open"):
        store.get(pid, stored.id)


def test_reviews_dir_symlink(store: ReviewStore, projects: ProjectStore, tmp_path: Path) -> None:
    pid = finished(projects)
    outside = tmp_path / "outside"
    outside.mkdir()
    rdir(projects).symlink_to(outside)
    with pytest.raises(ReviewStoreError, match="symlink"):
        store.all(pid)
    with pytest.raises(ReviewStoreError, match="symlink"):
        store.save(review(pid))
    assert list(outside.iterdir()) == []


def test_reviews_path_is_a_file(store: ReviewStore, projects: ProjectStore) -> None:
    pid = finished(projects)
    rdir(projects).write_text("x")
    with pytest.raises(ReviewStoreError, match="not a directory"):
        store.all(pid)


def test_fifo_review_file_does_not_hang(store: ReviewStore, projects: ProjectStore) -> None:
    pid, stored, path = _saved(store, projects)
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ReviewStoreError, match="not a regular file"):
        store.get(pid, stored.id)


@pytest.mark.parametrize("name", ["notes.txt", "rev-1.json", "rev-20261113-120000-abcdef.d0.json"])
def test_unknown_file_fails_closed(store: ReviewStore, projects: ProjectStore, name: str) -> None:
    pid, stored, _ = _saved(store, projects)
    (rdir(projects) / name).write_text("{}")
    with pytest.raises(ReviewStoreError, match="unexpected entry"):
        store.all(pid)
    with pytest.raises(ReviewStoreError, match="unexpected entry"):
        store.accept(pid, stored.id, 1)


def test_hidden_temp_files_are_ignored(store: ReviewStore, projects: ProjectStore) -> None:
    pid, stored, _ = _saved(store, projects)
    (rdir(projects) / ".rev-x.json.tmp").write_text("partial")
    assert store.all(pid) == [stored]


def test_orphan_decision_fails_closed(store: ReviewStore, projects: ProjectStore) -> None:
    pid, stored, _ = _saved(store, projects)
    (rdir(projects) / "rev-20200101-000000-aaaaaa.d1.json").write_text("{}")
    with pytest.raises(ReviewStoreError, match="missing review"):
        store.all(pid)


def test_decision_for_missing_proposal_fails_closed(
    store: ReviewStore, projects: ProjectStore
) -> None:
    pid, stored, _ = _saved(store, projects)
    d = ProposalDecision(
        review_id=stored.id,
        project_id=pid,
        number=5,
        status=DecisionStatus.REJECTED,
        decided_at=T0,
    )
    (rdir(projects) / f"{stored.id}.d5.json").write_text(d.model_dump_json())
    with pytest.raises(ReviewStoreError, match="does not have"):
        store.decisions(pid, stored.id)
    with pytest.raises(ReviewStoreError, match="does not have"):
        store.all(pid)


@pytest.mark.parametrize(
    "update",
    [
        {"number": 1},
        {"project_id": "p2"},
        {"review_id": "rev-20200101-000000-aaaaaa"},
        {"status": "accepted"},  # accepted without memory_id
        {"memory_id": "mem-20261113-120000-abcdef"},  # rejected with memory_id
        {"memory_existed": True},
        {"decided_at": "2026-01-01T00:00:00"},
        {"schema_version": 2},
    ],
)
def test_decision_file_tampering(
    store: ReviewStore, projects: ProjectStore, update: dict[str, Any]
) -> None:
    pid, stored, _ = _saved(store, projects)
    store.reject(pid, stored.id, 2)
    path = rdir(projects) / f"{stored.id}.d2.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(update)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReviewStoreError):
        store.decisions(pid, stored.id)
    with pytest.raises(ReviewStoreError):
        store.accept(pid, stored.id, 1)


def test_oversized_decision_file(store: ReviewStore, projects: ProjectStore) -> None:
    pid, stored, _ = _saved(store, projects)
    (rdir(projects) / f"{stored.id}.d1.json").write_bytes(b" " * (MAX_DECISION_FILE_BYTES + 1))
    with pytest.raises(ReviewStoreError, match="larger than"):
        store.pending(pid, stored.id)


def test_decision_file_symlink(store: ReviewStore, projects: ProjectStore, tmp_path: Path) -> None:
    pid, stored, _ = _saved(store, projects)
    store.reject(pid, stored.id, 1)
    path = rdir(projects) / f"{stored.id}.d1.json"
    target = tmp_path / "d.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ReviewStoreError, match="cannot open"):
        store.decisions(pid, stored.id)


# -- models --------------------------------------------------------------------


def test_decision_model_rules() -> None:
    base: dict[str, Any] = {
        "review_id": "rev-20261113-120000-abcdef",
        "project_id": "p1",
        "number": 1,
        "decided_at": T0,
    }
    ProposalDecision(**base, status="rejected")
    ProposalDecision(**base, status="accepted", memory_id="mem-20261113-120000-abcdef")
    for bad in (
        {"status": "accepted"},
        {"status": "rejected", "memory_id": "mem-20261113-120000-abcdef"},
        {"status": "rejected", "memory_existed": True},
        {"status": "accepted", "memory_id": "mem-bad"},
        {"status": "maybe"},
    ):
        with pytest.raises(ValueError):
            ProposalDecision(**base, **bad)
    with pytest.raises(ValueError):
        ProposalDecision(**{**base, "number": 9}, status="rejected")
    with pytest.raises(ValueError):
        ProposalDecision(**{**base, "project_id": "../x"}, status="rejected")


def test_stored_review_model_rejects_unfinished_status() -> None:
    with pytest.raises(ValueError):
        StoredReview(
            id="rev-20261113-120000-abcdef",
            project_id="p1",
            project_status=ProjectStatus.RUNNING,
            created_at=T0,
            saved_at=T0,
        )


def test_for_workspace(ws: Path) -> None:
    s = ReviewStore.for_workspace(ws)
    p = ProjectStore.for_workspace(ws)
    finished(p)
    stored = s.save(review("p1"))
    s.accept("p1", stored.id, 2)
    assert len(MemoryStore.for_workspace(ws).entries(MemoryScope.LESSON)) == 1


# -- end to end with the reviewer ---------------------------------------------


def test_end_to_end_with_review_agent(
    tmp_path: Path, store: ReviewStore, projects: ProjectStore, memory: MemoryStore
) -> None:
    cfg = AutomaConfig.model_validate(
        {
            "tiers": {"small": "fake-small", "medium": "fake-medium", "large": "fake-large"},
            "prices": {
                m: {"input_per_mtok": 1.0, "output_per_mtok": 2.0}
                for m in ("fake-small", "fake-medium", "fake-large")
            },
        }
    )
    reply = json.dumps(
        {
            "proposals": [
                {"scope": "project", "text": "the CLI uses typer"},
                {"scope": "lesson", "text": "run ruff before pytest"},
                {"scope": "lesson", "text": "an already known lesson"},
            ]
        }
    )
    ledger = CostLedger(tmp_path / "ledger.sqlite3")
    gateway = LLMGateway(cfg, FakeProvider([reply]), guard=BudgetGuard(cfg.limits, ledger))
    pid = finished(projects)
    memory.add(MemoryScope.LESSON, "an already known lesson")
    r = ReviewAgent(gateway).review(projects, pid, existing=memory.context(pid))
    assert r.dropped_duplicates == 1
    stored = store.save(r)
    assert len(memory.entries(MemoryScope.LESSON)) == 1  # saving wrote no memory
    store.accept(pid, stored.id, 2)
    store.reject(pid, stored.id, 1)
    assert store.pending(pid, stored.id) == []
    lessons = [e.text for e in memory.entries(MemoryScope.LESSON)]
    assert lessons == ["an already known lesson", "run ruff before pytest"]
    assert memory.entries(MemoryScope.PROJECT, pid) == []
