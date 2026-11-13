"""Tests for the three-layer memory store (M7, D48). No model, no provider."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from automa.memory import (
    MAX_ENTRIES_PER_LAYER,
    MAX_MEMORY_FILE_BYTES,
    MAX_MEMORY_TEXT_CHARS,
    MemoryEntry,
    MemoryFull,
    MemoryNotFound,
    MemoryScope,
    MemorySource,
    MemoryStore,
    MemoryStoreError,
    validate_memory_id,
)


class Clock:
    def __init__(self) -> None:
        self.t = datetime(2026, 11, 13, 10, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.t += timedelta(seconds=1)
        return self.t


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore.for_workspace(tmp_path, clock=Clock())


def _entry(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "mem-20261113-100000-abcdef",
        "scope": "user",
        "text": "Prefer Persian reports.",
        "created_at": "2026-11-13T10:00:00Z",
    }
    base.update(kw)
    return base


# -- model ------------------------------------------------------------------


def test_entry_valid_and_frozen() -> None:
    e = MemoryEntry.model_validate(_entry())
    assert e.scope is MemoryScope.USER and e.source is MemorySource.USER
    with pytest.raises(ValidationError):
        e.text = "x"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kw",
    [
        {"text": ""},
        {"text": "   \n "},
        {"text": "a" * (MAX_MEMORY_TEXT_CHARS + 1)},
        {"text": "bad\x00text"},
        {"text": "bell\x07"},
        {"text": "esc\x1b[31m"},
        {"id": "mem-1"},
        {"id": "../evil"},
        {"scope": "global"},
        {"source": "model"},
        {"extra": 1},
        {"schema_version": 2},
        {"created_at": "2026-11-13T10:00:00"},
        {"project_id": "p1"},  # user memory with a project
        {"source": "review"},  # review cannot write user preferences
        {"scope": "project"},  # project memory without project_id
        {"scope": "project", "project_id": "../x"},
        {"scope": "project", "project_id": "Bad_ID"},
    ],
)
def test_entry_invalid(kw: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MemoryEntry.model_validate(_entry(**kw))


def test_entry_text_normalized() -> None:
    e = MemoryEntry.model_validate(_entry(text="  line one\r\nline two\t!  "))
    assert e.text == "line one\nline two\t!"


def test_entry_persian_text() -> None:
    e = MemoryEntry.model_validate(_entry(text="گزارش‌ها به فارسی باشه"))
    assert "فارسی" in e.text


def test_lesson_may_name_source_project() -> None:
    e = MemoryEntry.model_validate(_entry(scope="lesson", project_id="p1", source="review"))
    assert e.project_id == "p1"


def test_validate_memory_id() -> None:
    assert validate_memory_id("mem-20261113-100000-abcdef")
    for bad in ["", "mem-x", "wf-20261113-100000-abcdef", "mem-20261113-100000-ABCDEF"]:
        with pytest.raises(MemoryStoreError):
            validate_memory_id(bad)


# -- add / entries ------------------------------------------------------------


def test_empty_store_creates_nothing(store: MemoryStore) -> None:
    assert store.entries(MemoryScope.USER) == []
    assert store.entries(MemoryScope.LESSON) == []
    assert store.entries(MemoryScope.PROJECT, "p1") == []
    assert store.project_ids() == []
    assert store.context("p1").is_empty()
    assert not store.root.exists()


def test_add_three_layers(store: MemoryStore) -> None:
    u = store.add(MemoryScope.USER, "Reports in Persian")
    p = store.add(MemoryScope.PROJECT, "Uses FastAPI", project_id="p1")
    lesson = store.add(MemoryScope.LESSON, "Service X is slow", source=MemorySource.REVIEW)
    assert (store.root / "user" / f"{u.id}.json").is_file()
    assert (store.root / "projects" / "p1" / f"{p.id}.json").is_file()
    assert (store.root / "lessons" / f"{lesson.id}.json").is_file()
    assert store.entries(MemoryScope.USER) == [u]
    assert store.entries(MemoryScope.PROJECT, "p1") == [p]
    assert store.entries(MemoryScope.LESSON) == [lesson]
    assert store.project_ids() == ["p1"]


def test_context_layers(store: MemoryStore) -> None:
    u = store.add(MemoryScope.USER, "pref")
    p1 = store.add(MemoryScope.PROJECT, "p1 fact", project_id="p1")
    store.add(MemoryScope.PROJECT, "p2 fact", project_id="p2")
    lesson = store.add(MemoryScope.LESSON, "lesson")
    ctx = store.context("p1")
    assert ctx.user == (u,) and ctx.project == (p1,) and ctx.lessons == (lesson,)
    assert store.context().project == ()
    assert not ctx.is_empty()


def test_entries_oldest_first(store: MemoryStore) -> None:
    a = store.add(MemoryScope.USER, "first")
    b = store.add(MemoryScope.USER, "second")
    c = store.add(MemoryScope.USER, "third")
    assert store.entries(MemoryScope.USER) == [a, b, c]


def test_file_contents_roundtrip(store: MemoryStore) -> None:
    e = store.add(MemoryScope.USER, "گزارش فارسی")
    raw = json.loads((store.root / "user" / f"{e.id}.json").read_text(encoding="utf-8"))
    assert raw["text"] == "گزارش فارسی" and raw["scope"] == "user"
    assert MemoryEntry.model_validate(raw) == e


def test_add_duplicate_in_layer_rejected(store: MemoryStore) -> None:
    e = store.add(MemoryScope.USER, "Prefer short answers")
    with pytest.raises(MemoryStoreError, match=e.id):
        store.add(MemoryScope.USER, "  prefer   SHORT answers ")
    # the same text in another layer or project is fine
    store.add(MemoryScope.LESSON, "Prefer short answers")
    store.add(MemoryScope.PROJECT, "Prefer short answers", project_id="p1")
    store.add(MemoryScope.PROJECT, "Prefer short answers", project_id="p2")


def test_add_invalid_writes_nothing(store: MemoryStore) -> None:
    for text in ["", "x\x00", "a" * (MAX_MEMORY_TEXT_CHARS + 1)]:
        with pytest.raises(MemoryStoreError):
            store.add(MemoryScope.USER, text)
    with pytest.raises(MemoryStoreError):
        store.add(MemoryScope.USER, "x", project_id="p1")
    with pytest.raises(MemoryStoreError):
        store.add(MemoryScope.USER, "x", source=MemorySource.REVIEW)
    with pytest.raises(MemoryStoreError):
        store.add(MemoryScope.PROJECT, "x")
    with pytest.raises(MemoryStoreError):
        store.add(MemoryScope.PROJECT, "x", project_id="../etc")
    assert not store.root.exists()


def test_layer_full(store: MemoryStore) -> None:
    for i in range(MAX_ENTRIES_PER_LAYER):
        store.add(MemoryScope.LESSON, f"lesson {i}")
    with pytest.raises(MemoryFull):
        store.add(MemoryScope.LESSON, "one more")
    store.add(MemoryScope.USER, "other layers still work")
    assert len(store.entries(MemoryScope.LESSON)) == MAX_ENTRIES_PER_LAYER


def test_layer_dir_rules(store: MemoryStore) -> None:
    with pytest.raises(MemoryStoreError):
        store.layer_dir(MemoryScope.USER, "p1")
    with pytest.raises(MemoryStoreError):
        store.layer_dir(MemoryScope.PROJECT)
    with pytest.raises(ValueError):
        store.layer_dir("bogus")  # type: ignore[arg-type]


# -- find / forget ----------------------------------------------------------


def test_find_each_layer(store: MemoryStore) -> None:
    entries = [
        store.add(MemoryScope.USER, "u"),
        store.add(MemoryScope.LESSON, "l"),
        store.add(MemoryScope.PROJECT, "p", project_id="p2"),
    ]
    for e in entries:
        assert store.find(e.id) == e
    with pytest.raises(MemoryNotFound):
        store.find("mem-20991231-000000-000000")
    with pytest.raises(MemoryStoreError):
        store.find("../../x")


def test_forget(store: MemoryStore) -> None:
    keep = store.add(MemoryScope.USER, "keep")
    gone = store.add(MemoryScope.PROJECT, "gone", project_id="p1")
    assert store.forget(gone.id) == gone
    assert store.entries(MemoryScope.PROJECT, "p1") == []
    assert store.entries(MemoryScope.USER) == [keep]
    with pytest.raises(MemoryNotFound):
        store.forget(gone.id)
    # forgetting frees the duplicate key
    store.add(MemoryScope.PROJECT, "gone", project_id="p1")


# -- fail closed --------------------------------------------------------------


def _user_dir(store: MemoryStore) -> Path:
    store.add(MemoryScope.USER, "seed")
    return store.root / "user"


def test_corrupt_json_rejects_layer(store: MemoryStore) -> None:
    d = _user_dir(store)
    (d / "mem-20261113-100100-aaaaaa.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(MemoryStoreError, match="unreadable"):
        store.entries(MemoryScope.USER)
    with pytest.raises(MemoryStoreError):
        store.add(MemoryScope.USER, "new")
    with pytest.raises(MemoryStoreError):
        store.context()


def test_non_utf8_rejected(store: MemoryStore) -> None:
    d = _user_dir(store)
    (d / "mem-20261113-100100-aaaaaa.json").write_bytes(b"\xff\xfe")
    with pytest.raises(MemoryStoreError, match="unreadable"):
        store.entries(MemoryScope.USER)


def test_invalid_entry_rejected(store: MemoryStore) -> None:
    d = _user_dir(store)
    bad = _entry(id="mem-20261113-100100-aaaaaa", text="x\x00")
    (d / "mem-20261113-100100-aaaaaa.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(MemoryStoreError, match="invalid memory file"):
        store.entries(MemoryScope.USER)


def test_id_mismatch_rejected(store: MemoryStore) -> None:
    d = _user_dir(store)
    (d / "mem-20261113-100100-aaaaaa.json").write_text(json.dumps(_entry()), encoding="utf-8")
    with pytest.raises(MemoryStoreError, match="does not match"):
        store.entries(MemoryScope.USER)


def test_wrong_layer_rejected(store: MemoryStore) -> None:
    store.add(MemoryScope.LESSON, "seed")
    d = store.root / "lessons"
    entry = _entry(id="mem-20261113-100100-aaaaaa")  # a user entry in lessons/
    (d / "mem-20261113-100100-aaaaaa.json").write_text(json.dumps(entry), encoding="utf-8")
    with pytest.raises(MemoryStoreError, match="layer"):
        store.entries(MemoryScope.LESSON)


def test_wrong_project_rejected(store: MemoryStore) -> None:
    store.add(MemoryScope.PROJECT, "seed", project_id="p1")
    d = store.root / "projects" / "p1"
    entry = _entry(id="mem-20261113-100100-aaaaaa", scope="project", project_id="p2")
    (d / "mem-20261113-100100-aaaaaa.json").write_text(json.dumps(entry), encoding="utf-8")
    with pytest.raises(MemoryStoreError, match="wrong place"):
        store.entries(MemoryScope.PROJECT, "p1")


def test_unexpected_file_rejected(store: MemoryStore) -> None:
    d = _user_dir(store)
    (d / "notes.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(MemoryStoreError, match="unexpected"):
        store.entries(MemoryScope.USER)


def test_hidden_temp_files_ignored(store: MemoryStore) -> None:
    d = _user_dir(store)
    (d / ".mem-x.json.tmp").write_text("partial", encoding="utf-8")
    assert len(store.entries(MemoryScope.USER)) == 1


def test_oversized_file_rejected(store: MemoryStore) -> None:
    d = _user_dir(store)
    (d / "mem-20261113-100100-aaaaaa.json").write_bytes(b" " * (MAX_MEMORY_FILE_BYTES + 1))
    with pytest.raises(MemoryStoreError, match="larger"):
        store.entries(MemoryScope.USER)


def test_symlinked_file_rejected(store: MemoryStore, tmp_path: Path) -> None:
    d = _user_dir(store)
    target = tmp_path / "outside.json"
    target.write_text(json.dumps(_entry(id="mem-20261113-100100-aaaaaa")), encoding="utf-8")
    (d / "mem-20261113-100100-aaaaaa.json").symlink_to(target)
    with pytest.raises(MemoryStoreError):
        store.entries(MemoryScope.USER)
    with pytest.raises(MemoryStoreError):
        store.find("mem-20261113-100100-aaaaaa")


def test_fifo_rejected_without_hanging(store: MemoryStore) -> None:
    d = _user_dir(store)
    os.mkfifo(d / "mem-20261113-100100-aaaaaa.json")
    with pytest.raises(MemoryStoreError, match="regular"):
        store.entries(MemoryScope.USER)


@pytest.mark.parametrize("which", ["root", "user", "projects", "project"])
def test_symlinked_directory_rejected(tmp_path: Path, which: str) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    store = MemoryStore.for_workspace(ws, clock=Clock())
    if which == "root":
        (ws / "memory").symlink_to(outside, target_is_directory=True)
    elif which == "user":
        (ws / "memory").mkdir()
        (ws / "memory" / "user").symlink_to(outside, target_is_directory=True)
    elif which == "projects":
        (ws / "memory").mkdir()
        (ws / "memory" / "projects").symlink_to(outside, target_is_directory=True)
    else:
        (ws / "memory" / "projects").mkdir(parents=True)
        (ws / "memory" / "projects" / "p1").symlink_to(outside, target_is_directory=True)
    with pytest.raises(MemoryStoreError, match="symlink"):
        if which == "user" or which == "root":
            store.entries(MemoryScope.USER)
        else:
            store.entries(MemoryScope.PROJECT, "p1")
    with pytest.raises(MemoryStoreError):
        if which == "user" or which == "root":
            store.add(MemoryScope.USER, "x")
        else:
            store.add(MemoryScope.PROJECT, "x", project_id="p1")
    assert list(outside.iterdir()) == []


def test_layer_path_is_a_file(tmp_path: Path) -> None:
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "lessons").write_text("x", encoding="utf-8")
    store = MemoryStore.for_workspace(tmp_path)
    with pytest.raises(MemoryStoreError, match="not a directory"):
        store.entries(MemoryScope.LESSON)


def test_bad_project_dir_name_rejected(store: MemoryStore) -> None:
    (store.root / "projects" / "Bad_Name").mkdir(parents=True)
    with pytest.raises(MemoryStoreError):
        store.project_ids()


def test_files_never_overwritten(store: MemoryStore) -> None:
    class FixedClock:
        def __call__(self) -> datetime:
            return datetime(2026, 11, 13, 10, 0, 0, tzinfo=UTC)

    s = MemoryStore(store.root, clock=FixedClock())
    a = s.add(MemoryScope.USER, "a")
    b = s.add(MemoryScope.USER, "b")
    assert a.id != b.id  # random suffix
    assert {e.text for e in s.entries(MemoryScope.USER)} == {"a", "b"}


def test_package_has_no_model_imports() -> None:
    import automa.memory.models as models
    import automa.memory.store as store_mod

    for mod in (models, store_mod):
        src = Path(mod.__file__ or "").read_text(encoding="utf-8")
        assert "automa.llm" not in src and "automa.tools" not in src
