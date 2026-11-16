"""``automa self next``: read-only view of the next roadmap item (D81).

No model, no config, no API key: the command only reads a roadmap file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from automa import cli

REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()

SAMPLE = """# Roadmap

## M1 — first
- [x] done one
- [!] waiting for a key
- [~] half done
- [ ] todo later

## M2 — second
- [ ] another
"""


def write(tmp_path: Path, text: str, name: str = "ROADMAP.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def run(*args: str) -> object:
    return runner.invoke(cli.app, ["self", "next", *args])


def listing(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


def test_text_output_shows_next_waiting_and_goal(tmp_path: Path) -> None:
    result = run("--roadmap", str(write(tmp_path, SAMPLE)))
    assert result.exit_code == cli.EXIT_OK
    out = result.stdout
    assert "items: todo 2, partial 1, done 1, waiting 1 (5 total)" in out
    assert "next: line 6 [~] M1 — first" in out
    assert "  half done" in out
    assert "waiting for the owner ([!]): 1" in out
    assert "line 5: waiting for a key" in out
    assert "goal:" in out
    assert "Status: partly done; continue it" in out
    assert "Item: half done" in out
    assert "Rules:" in out


def test_waiting_item_is_never_chosen(tmp_path: Path) -> None:
    text = "## M\n- [!] needs the owner\n- [ ] the real next\n"
    result = run("--roadmap", str(write(tmp_path, text)))
    assert result.exit_code == cli.EXIT_OK
    assert "next: line 3 [ ] M" in result.stdout
    assert "Item: the real next" in result.stdout
    assert "Item: needs the owner" not in result.stdout


def test_json_output(tmp_path: Path) -> None:
    result = run("--roadmap", str(write(tmp_path, SAMPLE)), "--json")
    assert result.exit_code == cli.EXIT_OK
    data = json.loads(result.stdout)
    assert data["next"] == {
        "line": 6,
        "milestone": "M1 — first",
        "status": "partial",
        "text": "half done",
    }
    assert data["waiting"] == [
        {"line": 5, "milestone": "M1 — first", "status": "waiting", "text": "waiting for a key"}
    ]
    assert data["counts"] == {"todo": 2, "partial": 1, "done": 1, "waiting": 1}
    assert data["goal"].startswith("Implement one item of the ROADMAP.md")
    assert "Roadmap line: 6" in data["goal"]


def test_nothing_open_exits_9_and_lists_waiting(tmp_path: Path) -> None:
    text = "## M\n- [x] done\n- [!] key needed\n"
    result = run("--roadmap", str(write(tmp_path, text)))
    assert result.exit_code == cli.EXIT_SELF_NOTHING_OPEN == 9
    assert "no open item left" in result.stdout
    assert "line 3: key needed" in result.stdout
    assert "goal:" not in result.stdout


def test_nothing_open_json(tmp_path: Path) -> None:
    text = "## M\n- [x] done\n- [!] key needed\n"
    result = run("--roadmap", str(write(tmp_path, text)), "--json")
    assert result.exit_code == cli.EXIT_SELF_NOTHING_OPEN
    data = json.loads(result.stdout)
    assert data["next"] is None
    assert data["goal"] is None
    assert [w["line"] for w in data["waiting"]] == [3]


def test_empty_roadmap_has_nothing_open(tmp_path: Path) -> None:
    result = run("--roadmap", str(write(tmp_path, "# nothing here\n")))
    assert result.exit_code == cli.EXIT_SELF_NOTHING_OPEN
    assert "items: todo 0, partial 0, done 0, waiting 0 (0 total)" in result.stdout


@pytest.mark.parametrize(
    "text",
    [
        "## M\n- [?] unknown marker\n",
        "## M\n- [ ] ok\n  - [ ] nested\n",
        "## M\n- [ ]\n",
        "## M\n```\n- [ ] in an open fence\n",
    ],
)
def test_broken_roadmap_exits_10_without_guessing(tmp_path: Path, text: str) -> None:
    result = run("--roadmap", str(write(tmp_path, text)))
    assert result.exit_code == cli.EXIT_ROADMAP_INVALID == 10
    assert "error: roadmap:" in result.stderr
    assert result.stdout == ""


def test_broken_roadmap_json_also_exits_10(tmp_path: Path) -> None:
    result = run("--roadmap", str(write(tmp_path, "## M\n- [?] x\n")), "--json")
    assert result.exit_code == cli.EXIT_ROADMAP_INVALID
    assert result.stdout == ""


def test_missing_roadmap_exits_10(tmp_path: Path) -> None:
    result = run("--roadmap", str(tmp_path / "nope.md"))
    assert result.exit_code == cli.EXIT_ROADMAP_INVALID
    assert "roadmap not found" in result.stderr


def test_symlink_roadmap_is_refused(tmp_path: Path) -> None:
    real = write(tmp_path, SAMPLE, "real.md")
    link = tmp_path / "ROADMAP.md"
    os.symlink(real, link)
    result = run("--roadmap", str(link))
    assert result.exit_code == cli.EXIT_ROADMAP_INVALID
    assert "symlink" in result.stderr


def test_non_utf8_roadmap_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ROADMAP.md"
    path.write_bytes(b"## M\n- [ ] \xff\xfe\n")
    result = run("--roadmap", str(path))
    assert result.exit_code == cli.EXIT_ROADMAP_INVALID
    assert "UTF-8" in result.stderr


def test_control_and_bidi_characters_are_escaped(tmp_path: Path) -> None:
    text = "## M\x1b[31m\n- [ ] evil‮ text\x07 here\n- [!] wait‭\n"
    for args in ((), ("--json",)):
        result = run("--roadmap", str(write(tmp_path, text)), *args)
        assert result.exit_code == cli.EXIT_OK
        assert "\x1b" not in result.stdout
        assert "‮" not in result.stdout
        assert "‭" not in result.stdout
        assert "\x07" not in result.stdout
    assert json.loads(result.stdout)["next"]["text"] == "evil\\u202e text\\x07 here"
    text_out = run("--roadmap", str(write(tmp_path, text))).stdout
    assert "evil\\u202e text\\x07 here" in text_out


def test_persian_joiner_is_kept(tmp_path: Path) -> None:
    result = run("--roadmap", str(write(tmp_path, "## M\n- [ ] نیم‌فاصله\n")))
    assert result.exit_code == cli.EXIT_OK
    assert "نیم‌فاصله" in result.stdout


def test_needs_no_config_or_key_and_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(tmp_path, SAMPLE)
    monkeypatch.chdir(tmp_path)
    for var in list(os.environ):
        if var.startswith("AUTOMA") or var.endswith("API_KEY"):
            monkeypatch.delenv(var, raising=False)

    def boom(*_a: object, **_k: object) -> object:
        raise AssertionError("self next must not build a provider")

    monkeypatch.setattr(cli, "make_provider", boom)
    before = listing(tmp_path)
    mtime = path.stat().st_mtime_ns
    result = run()  # default ./ROADMAP.md
    assert result.exit_code == cli.EXIT_OK
    assert "next: line 6" in result.stdout
    assert listing(tmp_path) == before
    assert path.stat().st_mtime_ns == mtime
    assert path.read_text(encoding="utf-8") == SAMPLE


def test_default_roadmap_missing_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = run()
    assert result.exit_code == cli.EXIT_ROADMAP_INVALID
    assert listing(tmp_path) == []


def test_long_waiting_item_is_previewed(tmp_path: Path) -> None:
    long_text = "w" * 500
    result = run("--roadmap", str(write(tmp_path, f"## M\n- [!] {long_text}\n- [ ] x\n")))
    assert result.exit_code == cli.EXIT_OK
    assert long_text not in result.stdout
    assert "…" in result.stdout
    json_result = run("--roadmap", str(write(tmp_path, f"## M\n- [!] {long_text}\n")), "--json")
    assert json.loads(json_result.stdout)["waiting"][0]["text"] == long_text


def test_real_repository_roadmap() -> None:
    result = run("--roadmap", str(REPO / "ROADMAP.md"), "--json")
    assert result.exit_code in (cli.EXIT_OK, cli.EXIT_SELF_NOTHING_OPEN)
    data = json.loads(result.stdout)
    assert sum(data["counts"].values()) > 0
    if data["next"] is not None:
        assert data["next"]["status"] in ("todo", "partial")
    assert all(w["status"] == "waiting" for w in data["waiting"])


def test_exit_codes_are_distinct() -> None:
    codes = [
        cli.EXIT_OK,
        cli.EXIT_USER,
        cli.EXIT_CONFIG,
        cli.EXIT_BUDGET,
        cli.EXIT_PLANNING,
        cli.EXIT_PROVIDER,
        cli.EXIT_TASKS_FAILED,
        cli.EXIT_PAUSED,
        cli.EXIT_BENCH_FAILED,
        cli.EXIT_SELF_NOTHING_OPEN,
        cli.EXIT_ROADMAP_INVALID,
        cli.EXIT_INTERRUPTED,
    ]
    assert len(set(codes)) == len(codes)
