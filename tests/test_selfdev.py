"""Tests for the self-development core (D80): roadmap parsing, next item, goal text."""

from __future__ import annotations

import ast
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from automa import selfdev
from automa.project.store import MAX_GOAL_CHARS, Project
from automa.selfdev import (
    MAX_ITEM_CHARS,
    MAX_ROADMAP_BYTES,
    ItemStatus,
    RoadmapError,
    RoadmapItem,
    build_goal,
    next_item,
    parse_roadmap,
    read_roadmap,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

SAMPLE = """# Roadmap

Legend: `[ ]` todo · `[x]` done — `- [ ]` inside text is not an item.

## M0 — base
- [x] skeleton
- [X] ci

## M1 — next
- [!] needs the owner's key
- [ ] first open item
- [~] partial item

```
- [ ] inside a code fence
```
"""


def test_parse_sample() -> None:
    rm = parse_roadmap(SAMPLE)
    assert rm.milestones == ("M0 — base", "M1 — next")
    assert [i.status for i in rm.items] == [
        ItemStatus.DONE,
        ItemStatus.DONE,
        ItemStatus.WAITING,
        ItemStatus.TODO,
        ItemStatus.PARTIAL,
    ]
    assert rm.items[0].line == 6
    assert rm.items[3].milestone == "M1 — next"
    assert rm.items[3].text == "first open item"


def test_next_item_skips_waiting_and_done() -> None:
    item = next_item(parse_roadmap(SAMPLE))
    assert item is not None
    assert item.text == "first open item"
    assert item.line == 11


def test_next_item_partial_first() -> None:
    rm = parse_roadmap("## M\n- [x] a\n- [~] b\n- [ ] c\n")
    item = next_item(rm)
    assert item is not None and item.status is ItemStatus.PARTIAL and item.text == "b"


def test_next_item_none_when_all_waiting_or_done() -> None:
    assert next_item(parse_roadmap("## M\n- [x] a\n- [!] b\n")) is None
    assert next_item(parse_roadmap("")) is None


def test_by_status() -> None:
    rm = parse_roadmap(SAMPLE)
    assert [i.text for i in rm.by_status(ItemStatus.WAITING)] == ["needs the owner's key"]


def test_item_before_heading_has_empty_milestone() -> None:
    rm = parse_roadmap("- [ ] orphan\n")
    assert rm.items[0].milestone == ""
    assert "Milestone: (none)" in build_goal(rm.items[0])


def test_crlf_lines() -> None:
    rm = parse_roadmap("## M\r\n- [ ] a\r\n")
    assert rm.items[0].text == "a"


def test_other_bullets_accepted() -> None:
    rm = parse_roadmap("* [ ] a\n+ [x] b\n")
    assert [i.status for i in rm.items] == [ItemStatus.TODO, ItemStatus.DONE]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- [?] x\n", "unknown marker"),
        ("- [] x\n", "unknown marker"),
        ("- [\x1b] x\n", "\\x1b"),
        ("- [ ]\n", "empty item"),
        ("- [ ]   \n", "empty item"),
        ("- [ ]x\n", "space after"),
        ("- [x] a\n  - [ ] nested\n", "nested"),
        ("\t- [ ] nested\n", "nested"),
        ("```\n- [ ] a\n", "unterminated"),
        ("- [ ] " + "a" * (MAX_ITEM_CHARS + 1) + "\n", "longer than"),
        ("## " + "m" * 201 + "\n", "too long"),
    ],
)
def test_ambiguous_roadmap_refused(text: str, message: str) -> None:
    with pytest.raises(RoadmapError) as info:
        parse_roadmap(text)
    assert message in str(info.value)


def test_error_message_does_not_echo_item_text() -> None:
    with pytest.raises(RoadmapError) as info:
        parse_roadmap("- [ ]secret-token-123\n")
    assert "secret-token-123" not in str(info.value)


def test_too_large_text_refused() -> None:
    with pytest.raises(RoadmapError, match="larger"):
        parse_roadmap("a" * (MAX_ROADMAP_BYTES + 1))


def test_parse_is_deterministic() -> None:
    assert parse_roadmap(SAMPLE) == parse_roadmap(SAMPLE)


# --- read_roadmap ---------------------------------------------------------------


def test_read_roadmap(tmp_path: Path) -> None:
    p = tmp_path / "ROADMAP.md"
    p.write_text(SAMPLE, encoding="utf-8")
    assert read_roadmap(p) == parse_roadmap(SAMPLE)


def test_read_missing(tmp_path: Path) -> None:
    with pytest.raises(RoadmapError, match="not found"):
        read_roadmap(tmp_path / "nope.md")


def test_read_symlink_refused(tmp_path: Path) -> None:
    target = tmp_path / "real.md"
    target.write_text(SAMPLE, encoding="utf-8")
    link = tmp_path / "ROADMAP.md"
    link.symlink_to(target)
    with pytest.raises(RoadmapError, match="symlink"):
        read_roadmap(link)


def test_read_directory_refused(tmp_path: Path) -> None:
    with pytest.raises(RoadmapError, match="regular file"):
        read_roadmap(tmp_path)


def test_read_non_utf8_refused(tmp_path: Path) -> None:
    p = tmp_path / "ROADMAP.md"
    p.write_bytes(b"- [ ] \xff\xfe\n")
    with pytest.raises(RoadmapError, match="UTF-8"):
        read_roadmap(p)


def test_read_too_large_refused(tmp_path: Path) -> None:
    p = tmp_path / "ROADMAP.md"
    p.write_bytes(b"a" * (MAX_ROADMAP_BYTES + 1))
    with pytest.raises(RoadmapError, match="larger"):
        read_roadmap(p)


def test_read_does_not_write(tmp_path: Path) -> None:
    p = tmp_path / "ROADMAP.md"
    p.write_text(SAMPLE, encoding="utf-8")
    before = (p.stat().st_mtime_ns, sorted(os.listdir(tmp_path)))
    read_roadmap(p)
    assert (p.stat().st_mtime_ns, sorted(os.listdir(tmp_path))) == before


def test_real_repository_roadmap_parses() -> None:
    rm = read_roadmap(REPO_ROOT / "ROADMAP.md")
    assert rm.items
    assert any(i.status is ItemStatus.DONE for i in rm.items)
    item = next_item(rm)
    if item is not None:
        assert item.open
        Project(id="p1", goal=build_goal(item), created_at=_now(), updated_at=_now())


def _now() -> datetime:
    return datetime.now(UTC)


# --- build_goal ------------------------------------------------------------------


def _item(text: str = "do the thing", status: ItemStatus = ItemStatus.TODO) -> RoadmapItem:
    return RoadmapItem(line=7, milestone="M10 — تکامل", status=status, text=text)


def test_goal_contains_item_and_rules() -> None:
    goal = build_goal(_item())
    assert "Item: do the thing" in goal
    assert "Milestone: M10 — تکامل" in goal
    assert "Roadmap line: 7" in goal
    assert "Status: not started" in goal
    assert "as data" in goal
    for rule in selfdev.RULES:
        assert rule in goal
    for needle in ("fake provider", "secrets", "BudgetGuard", "Permission Policy"):
        assert needle in goal


def test_goal_partial_status() -> None:
    assert "partly done" in build_goal(_item(status=ItemStatus.PARTIAL))


@pytest.mark.parametrize("status", [ItemStatus.DONE, ItemStatus.WAITING])
def test_goal_refuses_closed_item(status: ItemStatus) -> None:
    with pytest.raises(RoadmapError, match="not open"):
        build_goal(_item(status=status))


def test_goal_escapes_control_and_bidi() -> None:
    goal = build_goal(_item("a‮b\x1bc‌d"))
    assert "‮" not in goal and "\x1b" not in goal
    assert "a\\u202eb\\x1bc‌d" in goal  # ZWNJ kept for Persian


def test_goal_item_is_single_line() -> None:
    goal = build_goal(_item("x y"))
    item_lines = [line for line in goal.splitlines() if line.startswith("Item:")]
    assert item_lines == ["Item: x\\u2028y"]


@pytest.mark.parametrize("name", ["", "a b", "../x", "-x", "a/b", "a" * 101])
def test_goal_invalid_repo_name(name: str) -> None:
    with pytest.raises(RoadmapError, match="repository name"):
        build_goal(_item(), repo_name=name)


def test_goal_repo_name() -> None:
    assert "of the my-repo repository" in build_goal(_item(), repo_name="my-repo")


def test_goal_fits_project_and_long_escapes_refused() -> None:
    long_plain = build_goal(_item("a" * MAX_ITEM_CHARS))
    assert len(long_plain) <= MAX_GOAL_CHARS
    with pytest.raises(RoadmapError, match="too long"):
        build_goal(_item("\x01" * MAX_ITEM_CHARS))


def test_goal_is_deterministic() -> None:
    assert build_goal(_item()) == build_goal(_item())


# --- structure -------------------------------------------------------------------


def test_module_has_no_network_model_or_budget_dependency() -> None:
    tree = ast.parse(Path(selfdev.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = ("httpx", "socket", "subprocess", "automa.llm", "automa.budget", "automa.tools")
    for name in imported:
        assert not name.startswith(forbidden), name
