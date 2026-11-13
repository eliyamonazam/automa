"""``automa memory list|add|forget`` and the memory line in ``automa status`` (D49).

No API key, no provider, no model call. ``list`` and ``status`` write no files.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.config import AutomaConfig
from automa.memory import MemoryScope, MemorySource, MemoryStore
from automa.project import ProjectStore

TOML = """\
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


def tree(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        os.path.relpath(os.path.join(d, f), root)
        for d, dirs, files in os.walk(root)
        for f in [*dirs, *files]
    }


class Env:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.ws = tmp_path / "ws"
        self.config = tmp_path / "automa.toml"
        self.config.write_text(TOML.format(ws=self.ws.as_posix()), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)

        def no_provider(config: AutomaConfig) -> Any:
            raise AssertionError("memory commands must not build a provider")

        monkeypatch.setattr(cli, "make_provider", no_provider)
        self._tick = 0

    def clock(self) -> datetime:
        self._tick += 1
        return datetime(2026, 11, 13, 12, 0, self._tick, tzinfo=UTC)

    @property
    def memory(self) -> MemoryStore:
        return MemoryStore.for_workspace(self.ws, clock=self.clock)

    def project(self, goal: str = "Write a haiku") -> str:
        return ProjectStore.for_workspace(self.ws).create(goal).id

    def run(self, *args: str, input: str | None = None) -> Any:
        return CliRunner().invoke(cli.app, [*args, "--config", str(self.config)], input=input)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


# -- list -------------------------------------------------------------------------


def test_list_empty_creates_nothing(env: Env) -> None:
    result = env.run("memory", "list")
    assert result.exit_code == 0, result.output
    assert "no memory yet" in result.output
    assert not env.ws.exists()


def test_list_json_empty(env: Env) -> None:
    result = env.run("memory", "list", "--json")
    assert result.exit_code == 0
    assert json.loads(result.output) == []
    assert not env.ws.exists()


def test_list_shows_all_layers_in_order_with_full_text(env: Env) -> None:
    pid = env.project()
    store = env.memory
    lesson = store.add(MemoryScope.LESSON, "Small steps work better.")
    user = store.add(MemoryScope.USER, "Answer in Persian.\nKeep it short.")
    proj = store.add(MemoryScope.PROJECT, "Haiku is 5-7-5.", project_id=pid)
    before = tree(env.ws)
    result = env.run("memory", "list")
    assert result.exit_code == 0, result.output
    out = result.output
    assert out.index(user.id) < out.index(proj.id) < out.index(lesson.id)
    assert "  Answer in Persian.\n  Keep it short.\n" in out
    assert f"{proj.id}  project  {pid}  user" in out
    assert tree(env.ws) == before


def test_list_json_fields(env: Env) -> None:
    entry = env.memory.add(MemoryScope.USER, "ترجیح: پاسخ کوتاه")
    rows = json.loads(env.run("memory", "list", "--json").output)
    assert rows == [entry.model_dump(mode="json")]
    assert set(rows[0]) == {
        "schema_version",
        "id",
        "scope",
        "text",
        "project_id",
        "source",
        "created_at",
    }
    assert rows[0]["text"] == "ترجیح: پاسخ کوتاه"


def test_list_filter_scope(env: Env) -> None:
    pid = env.project()
    store = env.memory
    store.add(MemoryScope.USER, "u")
    store.add(MemoryScope.LESSON, "l")
    p = store.add(MemoryScope.PROJECT, "p", project_id=pid)
    rows = json.loads(env.run("memory", "list", "--scope", "project", "--json").output)
    assert [r["id"] for r in rows] == [p.id]
    rows = json.loads(env.run("memory", "list", "--scope", "lesson", "--json").output)
    assert [r["text"] for r in rows] == ["l"]


def test_list_project_shows_what_applies_to_it(env: Env) -> None:
    a, b = env.project("Goal A"), env.project("Goal B")
    store = env.memory
    store.add(MemoryScope.USER, "u")
    store.add(MemoryScope.PROJECT, "only a", project_id=a)
    store.add(MemoryScope.PROJECT, "only b", project_id=b)
    store.add(MemoryScope.LESSON, "l")
    rows = json.loads(env.run("memory", "list", "--project", a, "--json").output)
    assert [r["text"] for r in rows] == ["u", "only a", "l"]
    rows = json.loads(
        env.run("memory", "list", "--project", b, "--scope", "project", "--json").output
    )
    assert [r["text"] for r in rows] == ["only b"]


def test_list_project_with_user_scope_is_an_error(env: Env) -> None:
    pid = env.project()
    result = env.run("memory", "list", "--project", pid, "--scope", "user")
    assert result.exit_code == cli.EXIT_USER
    assert "not tied to a project" in result.output


def test_list_invalid_project_id(env: Env) -> None:
    result = env.run("memory", "list", "--project", "../etc")
    assert result.exit_code == cli.EXIT_USER
    assert "invalid project id" in result.output


def test_list_invalid_scope(env: Env) -> None:
    result = env.run("memory", "list", "--scope", "secret")
    assert result.exit_code == 2  # typer usage error
    assert not env.ws.exists()


def test_list_broken_memory_fails_closed(env: Env) -> None:
    env.memory.add(MemoryScope.USER, "fine")
    bad = env.ws / "memory" / "user" / "mem-20261113-120000-abcdef.json"
    bad.write_text("{not json", encoding="utf-8")
    result = env.run("memory", "list")
    assert result.exit_code == cli.EXIT_USER
    assert "unreadable memory file" in result.output
    assert "fine" not in result.output


def test_list_escapes_control_characters_in_files(env: Env) -> None:
    """A review entry is model-written; the terminal must not interpret it. Entries
    with control characters are invalid anyway, so a forged file is rejected."""
    entry = env.memory.add(MemoryScope.LESSON, "ok")
    path = env.ws / "memory" / "lessons" / f"{entry.id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["text"] = "evil \x1b[2J"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = env.run("memory", "list")
    assert result.exit_code == cli.EXIT_USER
    assert "\x1b" not in result.output


def test_list_escapes_format_characters(env: Env) -> None:
    """Bidi overrides are valid in stored text but must not reorder the terminal."""
    entry = env.memory.add(MemoryScope.LESSON, "abc\u202edef", source=MemorySource.REVIEW)
    result = env.run("memory", "list")
    assert result.exit_code == 0, result.output
    assert "\u202e" not in result.output
    assert "abc\\u202edef" in result.output
    assert f"{entry.id}  lesson   -  review" in result.output


def test_list_keeps_persian_zwnj(env: Env) -> None:
    env.memory.add(MemoryScope.USER, "می‌خوام")
    result = env.run("memory", "list")
    assert "می‌خوام" in result.output


# -- add --------------------------------------------------------------------------


def test_add_user_default_scope(env: Env) -> None:
    result = env.run("memory", "add", "Prefer Python.")
    assert result.exit_code == 0, result.output
    assert result.output.startswith("remembered: mem-")
    entries = MemoryStore.for_workspace(env.ws).entries(MemoryScope.USER)
    assert [(e.text, e.source) for e in entries] == [("Prefer Python.", MemorySource.USER)]
    assert result.output.strip().endswith(entries[0].id)


def test_add_json(env: Env) -> None:
    result = env.run("memory", "add", "--scope", "lesson", "Test first.", "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["scope"] == "lesson" and data["source"] == "user"
    assert data["project_id"] is None


def test_add_project_memory(env: Env) -> None:
    pid = env.project()
    result = env.run("memory", "add", "--scope", "project", "--project", pid, "Uses uv.")
    assert result.exit_code == 0, result.output
    [entry] = MemoryStore.for_workspace(env.ws).entries(MemoryScope.PROJECT, pid)
    assert entry.text == "Uses uv." and entry.project_id == pid


def test_add_lesson_with_source_project(env: Env) -> None:
    pid = env.project()
    result = env.run("memory", "add", "--scope", "lesson", "--project", pid, "X was slow.")
    assert result.exit_code == 0, result.output
    [entry] = MemoryStore.for_workspace(env.ws).entries(MemoryScope.LESSON)
    assert entry.project_id == pid


def test_add_project_needs_project(env: Env) -> None:
    result = env.run("memory", "add", "--scope", "project", "x")
    assert result.exit_code == cli.EXIT_USER
    assert "--project" in result.output
    assert not env.ws.exists()


def test_add_user_with_project_is_an_error(env: Env) -> None:
    pid = env.project()
    result = env.run("memory", "add", "--project", pid, "x")
    assert result.exit_code == cli.EXIT_USER
    assert not (env.ws / "memory").exists()


def test_add_unknown_project_is_an_error(env: Env) -> None:
    result = env.run("memory", "add", "--scope", "project", "--project", "nope", "x")
    assert result.exit_code == cli.EXIT_USER
    assert not (env.ws / "memory").exists()


def test_add_invalid_project_id(env: Env) -> None:
    result = env.run("memory", "add", "--scope", "project", "--project", "../x", "x")
    assert result.exit_code == cli.EXIT_USER
    assert not (env.ws / "memory").exists()


@pytest.mark.parametrize("text", ["", "   ", "a\x00b", "bell\x07", "x" * 2001])
def test_add_invalid_text_writes_nothing(env: Env, text: str) -> None:
    result = env.run("memory", "add", text)
    assert result.exit_code == cli.EXIT_USER, result.output
    assert "invalid memory" in result.output
    assert not (env.ws / "memory").exists()
    assert "\x00" not in result.output and "\x07" not in result.output


def test_add_duplicate_is_an_error(env: Env) -> None:
    assert env.run("memory", "add", "Prefer  Python.").exit_code == 0
    result = env.run("memory", "add", "prefer python.")
    assert result.exit_code == cli.EXIT_USER
    assert "already exists" in result.output
    assert len(MemoryStore.for_workspace(env.ws).entries(MemoryScope.USER)) == 1


def test_add_full_layer(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("automa.memory.store.MAX_ENTRIES_PER_LAYER", 2)
    assert env.run("memory", "add", "one").exit_code == 0
    assert env.run("memory", "add", "two").exit_code == 0
    result = env.run("memory", "add", "three")
    assert result.exit_code == cli.EXIT_USER
    assert "full" in result.output


def test_add_into_broken_layer_fails_closed(env: Env) -> None:
    (env.ws / "memory" / "user").mkdir(parents=True)
    (env.ws / "memory" / "user" / "junk.txt").write_text("x", encoding="utf-8")
    result = env.run("memory", "add", "x")
    assert result.exit_code == cli.EXIT_USER
    assert "unexpected entry" in result.output
    assert tree(env.ws / "memory") == {"user", os.path.join("user", "junk.txt")}


def test_add_multiline_persian(env: Env) -> None:
    result = env.run("memory", "add", "خط اول\nخط دوم")
    assert result.exit_code == 0, result.output
    listed = env.run("memory", "list").output
    assert "  خط اول\n  خط دوم\n" in listed


def test_add_needs_no_api_key_and_ignores_it(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOMA_API_KEY", "sk-test-not-used")
    assert env.run("memory", "add", "x").exit_code == 0


# -- forget -----------------------------------------------------------------------


def test_forget_with_confirmation(env: Env) -> None:
    entry = env.memory.add(MemoryScope.USER, "Old preference.")
    result = env.run("memory", "forget", entry.id, input="y\n")
    assert result.exit_code == 0, result.output
    assert "Old preference." in result.output.split("Forget")[0]  # shown before asking
    assert f"forgotten: {entry.id}" in result.output
    assert MemoryStore.for_workspace(env.ws).entries(MemoryScope.USER) == []


def test_forget_declined_changes_nothing(env: Env) -> None:
    entry = env.memory.add(MemoryScope.USER, "Keep me.")
    result = env.run("memory", "forget", entry.id, input="n\n")
    assert result.exit_code == cli.EXIT_USER
    assert "nothing was changed" in result.output
    assert [e.id for e in MemoryStore.for_workspace(env.ws).entries(MemoryScope.USER)] == [entry.id]


def test_forget_default_is_no(env: Env) -> None:
    entry = env.memory.add(MemoryScope.USER, "Keep me.")
    result = env.run("memory", "forget", entry.id, input="\n")
    assert result.exit_code == cli.EXIT_USER
    assert [e.id for e in MemoryStore.for_workspace(env.ws).entries(MemoryScope.USER)] == [entry.id]


def test_forget_yes_project_entry(env: Env) -> None:
    pid = env.project()
    store = env.memory
    keep = store.add(MemoryScope.PROJECT, "keep", project_id=pid)
    gone = store.add(MemoryScope.PROJECT, "gone", project_id=pid)
    result = env.run("memory", "forget", gone.id, "--yes")
    assert result.exit_code == 0, result.output
    left = MemoryStore.for_workspace(env.ws).entries(MemoryScope.PROJECT, pid)
    assert [e.id for e in left] == [keep.id]


def test_forget_unknown(env: Env) -> None:
    result = env.run("memory", "forget", "mem-20261113-120000-abcdef", "--yes")
    assert result.exit_code == cli.EXIT_USER
    assert "no such memory" in result.output
    assert not env.ws.exists()


@pytest.mark.parametrize("bad", ["../x", "mem-1", "mem-20261113-120000-ABCDEF"])
def test_forget_invalid_id(env: Env, bad: str) -> None:
    result = env.run("memory", "forget", bad, "--yes")
    assert result.exit_code == cli.EXIT_USER
    assert "invalid memory id" in result.output


# -- status -----------------------------------------------------------------------


def test_status_all_memory_line(env: Env) -> None:
    a, b = env.project("A"), env.project("B")
    store = env.memory
    store.add(MemoryScope.USER, "u")
    store.add(MemoryScope.PROJECT, "pa", project_id=a)
    store.add(MemoryScope.PROJECT, "pb", project_id=b)
    store.add(MemoryScope.LESSON, "l1")
    store.add(MemoryScope.LESSON, "l2")
    before = tree(env.ws)
    result = env.run("status")
    assert result.exit_code == 0, result.output
    assert "memory: 1 user, 2 project (all projects), 2 lessons" in result.output
    assert tree(env.ws) == before


def test_status_one_memory_line_and_json(env: Env) -> None:
    a, b = env.project("A"), env.project("B")
    store = env.memory
    store.add(MemoryScope.PROJECT, "pa", project_id=a)
    store.add(MemoryScope.PROJECT, "pb", project_id=b)
    store.add(MemoryScope.PROJECT, "pb2", project_id=b)
    result = env.run("status", b)
    assert "memory: 0 user, 2 project (this project), 0 lessons" in result.output
    data = json.loads(env.run("status", a, "--json").output)
    assert data["memory"] == {"user": 0, "project": 1, "lessons": 0, "error": None}


def test_status_without_memory_creates_nothing(env: Env) -> None:
    pid = env.project()
    before = tree(env.ws)
    result = env.run("status", pid)
    assert result.exit_code == 0, result.output
    assert "memory: 0 user, 0 project (this project), 0 lessons" in result.output
    assert tree(env.ws) == before


def test_status_reports_broken_memory_without_failing(env: Env) -> None:
    pid = env.project()
    (env.ws / "memory" / "lessons").mkdir(parents=True)
    (env.ws / "memory" / "lessons" / "junk").write_text("x", encoding="utf-8")
    result = env.run("status", pid)
    assert result.exit_code == 0, result.output
    assert "memory: error:" in result.output
    data = json.loads(env.run("status", pid, "--json").output)
    assert data["memory"]["error"] and data["memory"]["lessons"] == 0
    assert env.run("status").exit_code == 0


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_status_symlinked_memory_dir_is_reported(env: Env, tmp_path: Path) -> None:
    env.project()
    outside = tmp_path / "outside"
    (outside / "user").mkdir(parents=True)
    os.symlink(outside, env.ws / "memory")
    result = env.run("status")
    assert result.exit_code == 0
    assert "memory: error:" in result.output and "symlink" in result.output
    assert env.run("memory", "list").exit_code == cli.EXIT_USER
    assert env.run("memory", "add", "x").exit_code == cli.EXIT_USER
    assert list((outside / "user").iterdir()) == []
