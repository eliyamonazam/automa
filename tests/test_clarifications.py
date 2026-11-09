"""Tests for ``Clarification`` models and ``ProjectStore`` clarifications persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from automa.agents import Clarification as AgentsClarification
from automa.plan import Clarification, Clarifications
from automa.plan.models import MAX_ANSWER_CHARS, MAX_CLARIFICATIONS, MAX_QUESTION_CHARS
from automa.project import ProjectConflict, ProjectNotFound, ProjectStore, ProjectStoreError


@pytest.fixture
def store(tmp_path: Path) -> ProjectStore:
    s = ProjectStore.for_workspace(tmp_path / "ws")
    s.create("goal", project_id="p1")
    return s


def test_agents_reexport_is_same_class() -> None:
    assert AgentsClarification is Clarification


def test_clarification_limits() -> None:
    Clarification(question="q" * MAX_QUESTION_CHARS, answer="a" * MAX_ANSWER_CHARS)
    with pytest.raises(ValidationError):
        Clarification(question="  ", answer="")
    with pytest.raises(ValidationError):
        Clarification(question="q" * (MAX_QUESTION_CHARS + 1), answer="")
    with pytest.raises(ValidationError):
        Clarification(question="q", answer="a" * (MAX_ANSWER_CHARS + 1))
    with pytest.raises(ValidationError):
        Clarification.model_validate({"question": "q", "answer": "", "extra": 1})


def test_clarifications_record_limits() -> None:
    with pytest.raises(ValidationError):
        Clarifications(schema_version=99)
    item = Clarification(question="q", answer="")
    with pytest.raises(ValidationError):
        Clarifications(items=[item] * (MAX_CLARIFICATIONS + 1))


def test_roundtrip_and_event(store: ProjectStore) -> None:
    assert store.load_clarifications("p1") == []
    assert not store.has_clarifications("p1")
    items = [
        Clarification(question=" زبان؟ ", answer=" فارسی "),
        Clarification(question="Deadline?", answer=""),
    ]
    saved = store.save_clarifications("p1", items)
    assert saved[0].question == "زبان؟" and saved[0].answer == "فارسی"
    assert store.load_clarifications("p1") == saved
    assert store.has_clarifications("p1")
    raw = store.clarifications_file("p1").read_text(encoding="utf-8")
    assert "فارسی" in raw  # human-readable, not escaped
    events = [e for e in store.events("p1").read() if e.type == "clarifications.saved"]
    assert len(events) == 1 and events[0].data == {"count": 2}


def test_never_overwritten(store: ProjectStore) -> None:
    store.save_clarifications("p1", [Clarification(question="q", answer="first")])
    with pytest.raises(ProjectConflict):
        store.save_clarifications("p1", [Clarification(question="q", answer="second")])
    assert store.load_clarifications("p1")[0].answer == "first"


def test_unvalidated_items_are_rejected(store: ProjectStore) -> None:
    bad = Clarification.model_construct(question="", answer="x")
    with pytest.raises(ProjectStoreError):
        store.save_clarifications("p1", [bad])
    assert not store.has_clarifications("p1")


def test_unknown_project(store: ProjectStore) -> None:
    with pytest.raises(ProjectNotFound):
        store.save_clarifications("nope", [])
    with pytest.raises(ProjectNotFound):
        store.load_clarifications("nope")
    with pytest.raises(ProjectStoreError):
        store.load_clarifications("../p1")


@pytest.mark.parametrize(
    "content",
    ["{broken", "[]", json.dumps({"items": [{"question": ""}]}), json.dumps({"x": 1})],
)
def test_corrupt_file_raises(store: ProjectStore, content: str) -> None:
    store.clarifications_file("p1").write_text(content, encoding="utf-8")
    with pytest.raises(ProjectStoreError):
        store.load_clarifications("p1")


def test_symlink_refused(store: ProjectStore, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({"items": []}), encoding="utf-8")
    store.clarifications_file("p1").symlink_to(target)
    with pytest.raises(ProjectStoreError):
        store.load_clarifications("p1")
    with pytest.raises(ProjectStoreError):
        store.save_clarifications("p1", [])
    assert json.loads(target.read_text(encoding="utf-8")) == {"items": []}
