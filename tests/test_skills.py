"""Tests for skills as ``SKILL.md`` files (``automa.skills``). Local only, no model calls."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from automa.skills import (
    MAX_SKILL_BODY_CHARS,
    MAX_SKILL_FILE_BYTES,
    SKILL_FILE,
    Skill,
    SkillError,
    SkillRegistry,
    parse_skill,
    split_frontmatter,
)

GOOD = """---
name: pdf-report
description: Write a short report with sources at the end.
---

# PDF report

Always cite sources.
"""


def _write(root: Path, name: str, text: str) -> Path:
    d = root / name
    d.mkdir()
    (d / SKILL_FILE).write_text(text, encoding="utf-8")
    return d


def _skill_text(name: str = "pdf-report", **extra: str) -> str:
    lines = ["---", f"name: {name}", "description: Does a thing."]
    lines += [f"{k}: {v}" for k, v in extra.items()]
    return "\n".join(lines) + "\n---\nBody text.\n"


# --- parsing -------------------------------------------------------------


def test_parse_good_skill() -> None:
    s = parse_skill(GOOD, expected_name="pdf-report")
    assert s.name == "pdf-report"
    assert s.description == "Write a short report with sources at the end."
    assert s.license is None
    assert s.body == "# PDF report\n\nAlways cite sources."


def test_parse_quoted_values_and_license() -> None:
    text = (
        "---\n"
        "name: 'notes'\n"
        'description: "Summaries: short, with a colon"\n'
        "license: 'It''s MIT'\n"
        "# a comment\n"
        "\n"
        "---\nDo it.\n"
    )
    s = parse_skill(text, expected_name="notes")
    assert s.description == "Summaries: short, with a colon"
    assert s.license == "It's MIT"


def test_parse_persian_text_crlf_and_bom() -> None:
    text = "﻿---\r\nname: farsi\r\ndescription: نوشتن گزارش فارسی\r\n---\r\nمتن دستورالعمل\r\n"
    s = parse_skill(text, expected_name="farsi")
    assert s.description == "نوشتن گزارش فارسی"
    assert s.body == "متن دستورالعمل"


def test_trailing_comment_on_plain_value_is_dropped() -> None:
    fields, _ = split_frontmatter("---\nname: a # comment\ndescription: x\n---\nb\n")
    assert fields["name"] == "a"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("name: a\n---\nbody", "must start with"),
        ("---\nname: a\ndescription: x\n", "not closed"),
        ("---\nname: a\ndescription: x\nversion: 1\n---\nb", "unknown field 'version'"),
        ("---\nname: a\nname: a\ndescription: x\n---\nb", "duplicate field"),
        ("---\nname: a\ndescription:\n---\nb", "empty value"),
        ("---\nname a\ndescription: x\n---\nb", "expected 'key: value'"),
        ("---\nname: a\ndescription: >\n  folded\n---\nb", "plain single-line"),
        ("---\nname: a\ndescription: x\n  more: y\n---\nb", "indented"),
        ("---\nname: a\ndescription: [x, y]\n---\nb", "plain single-line"),
        ('---\nname: a\ndescription: "x\n---\nb', "unterminated"),
        ("---\nname: a\ndescription: 'it's'\n---\nb", "unescaped quote"),
        ('---\nname: a\ndescription: "a\\nb"\n---\nb', "escapes"),
    ],
)
def test_frontmatter_errors(text: str, message: str) -> None:
    with pytest.raises(SkillError, match=message):
        parse_skill(text, expected_name="a")


@pytest.mark.parametrize("key", ["allowed-tools", "allowed_tools", "tools"])
def test_skill_cannot_grant_tools(key: str) -> None:
    text = f"---\nname: a\ndescription: x\n{key}: run_command\n---\nb"
    with pytest.raises(SkillError, match="never grants tools"):
        parse_skill(text, expected_name="a")


def test_name_must_match_directory() -> None:
    with pytest.raises(SkillError, match="does not match directory"):
        parse_skill(GOOD, expected_name="other")


@pytest.mark.parametrize("name", ["Bad", "-a", "a-", "a--b", "a_b", "a b", "x" * 65, "..", "a/b"])
def test_invalid_names(name: str) -> None:
    with pytest.raises(SkillError, match="invalid skill"):
        parse_skill(_skill_text(name), expected_name=name)


def test_blank_body_rejected() -> None:
    with pytest.raises(SkillError, match="body"):
        parse_skill("---\nname: a\ndescription: x\n---\n   \n\n", expected_name="a")


def test_too_long_body_and_description() -> None:
    long_body = "---\nname: a\ndescription: x\n---\n" + "x" * (MAX_SKILL_BODY_CHARS + 1)
    with pytest.raises(SkillError, match="body"):
        parse_skill(long_body, expected_name="a")
    long_desc = "---\nname: a\ndescription: " + "d" * 1025 + "\n---\nb"
    with pytest.raises(SkillError, match="description"):
        parse_skill(long_desc, expected_name="a")


def test_model_is_strict_and_frozen() -> None:
    with pytest.raises(ValidationError):
        Skill(name="a", description="x", body="b", tools=["run_command"])  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Skill(name="a", description="two\nlines", body="b")
    s = Skill(name="a", description="x", body="b")
    with pytest.raises(ValidationError):
        s.name = "c"  # type: ignore[misc]


# --- registry ------------------------------------------------------------


def test_registry_loads_sorted_and_ignores_plain_and_hidden(tmp_path: Path) -> None:
    _write(tmp_path, "zeta", _skill_text("zeta"))
    _write(tmp_path, "alpha", _skill_text("alpha"))
    (tmp_path / "README.md").write_text("not a skill", encoding="utf-8")
    hidden = tmp_path / ".draft"
    hidden.mkdir()
    reg = SkillRegistry.from_dir(tmp_path)
    assert [s.name for s in reg] == ["alpha", "zeta"]
    assert len(reg) == 2
    assert "alpha" in reg and "beta" not in reg
    assert reg.names() == frozenset({"alpha", "zeta"})
    assert reg.get("zeta").body == "Body text."
    with pytest.raises(SkillError, match="unknown skill"):
        reg.get("beta")


def test_empty_dir_is_empty_registry(tmp_path: Path) -> None:
    assert len(SkillRegistry.from_dir(tmp_path)) == 0


def test_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(SkillError, match="not found"):
        SkillRegistry.from_dir(tmp_path / "nope")
    assert len(SkillRegistry.from_optional_dir(tmp_path / "nope")) == 0


def test_symlinked_skills_dir_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(SkillError, match="not found"):
        SkillRegistry.from_dir(link)
    with pytest.raises(SkillError, match="not found"):
        SkillRegistry.from_optional_dir(link)
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "gone")
    with pytest.raises(SkillError):
        SkillRegistry.from_optional_dir(dangling)


def test_symlinked_skill_dir_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    _write(outside, "evil", _skill_text("evil"))
    root = tmp_path / "skills"
    root.mkdir()
    (root / "evil").symlink_to(outside / "evil")
    with pytest.raises(SkillError, match="symlinks are not allowed"):
        SkillRegistry.from_dir(root)


def test_symlinked_skill_file_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    target.write_text(_skill_text("a"), encoding="utf-8")
    root = tmp_path / "skills"
    root.mkdir()
    (root / "a").mkdir()
    (root / "a" / SKILL_FILE).symlink_to(target)
    with pytest.raises(SkillError, match="cannot open"):
        SkillRegistry.from_dir(root)


def test_missing_skill_file(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(SkillError, match="missing SKILL.md"):
        SkillRegistry.from_dir(tmp_path)


def test_skill_file_is_directory(tmp_path: Path) -> None:
    (tmp_path / "a" / SKILL_FILE).mkdir(parents=True)
    with pytest.raises(SkillError, match="not a regular file"):
        SkillRegistry.from_dir(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_fifo_skill_file_rejected_without_hanging(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    os.mkfifo(tmp_path / "a" / SKILL_FILE)
    with pytest.raises(SkillError, match="not a regular file"):
        SkillRegistry.from_dir(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_special_top_level_entry_rejected(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(SkillError, match="not a directory"):
        SkillRegistry.from_dir(tmp_path)


def test_oversized_skill_file(tmp_path: Path) -> None:
    _write(tmp_path, "a", _skill_text("a") + "x" * MAX_SKILL_FILE_BYTES)
    with pytest.raises(SkillError, match="larger than"):
        SkillRegistry.from_dir(tmp_path)


def test_non_utf8_skill_file(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / SKILL_FILE).write_bytes(b"---\nname: a\ndescription: \xff\n---\nb")
    with pytest.raises(SkillError, match="UTF-8"):
        SkillRegistry.from_dir(tmp_path)


def test_one_bad_skill_rejects_whole_registry(tmp_path: Path) -> None:
    _write(tmp_path, "good", _skill_text("good"))
    _write(tmp_path, "bad", _skill_text("other"))
    with pytest.raises(SkillError, match="bad/SKILL.md"):
        SkillRegistry.from_dir(tmp_path)


def test_registry_rejects_duplicates_and_non_skills() -> None:
    s = Skill(name="a", description="x", body="b")
    with pytest.raises(SkillError, match="duplicate"):
        SkillRegistry([s, s])
    with pytest.raises(TypeError):
        SkillRegistry(["a"])  # type: ignore[list-item]
