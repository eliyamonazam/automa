"""``automa update`` against real local git repositories, no network and no pip (D170)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from automa import update as upd
from automa.cli import EXIT_USER, app
from automa.tools.command import CommandResult, run_update_command
from automa.update import UpdateStatus, report_lines, run_update

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

_ENV_ARGS = ["-c", "user.name=Test", "-c", "user.email=test@example.invalid"]


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(  # noqa: S603 - test helper
        ["git", *_ENV_ARGS, *args],  # noqa: S607
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _commit(repo: Path, name: str, content: str, message: str) -> str:
    (repo / name).write_text(content, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


class Setup:
    def __init__(self, tmp: Path) -> None:
        self.origin = tmp / "origin.git"
        self.upstream = tmp / "upstream"
        self.local = tmp / "local"
        _git(tmp, "init", "-q", "--bare", "-b", "main", str(self.origin))
        _git(tmp, "clone", "-q", str(self.origin), str(self.upstream))
        _git(self.upstream, "checkout", "-q", "-B", "main")
        _commit(self.upstream, "pyproject.toml", "[project]\nname='automa'\n", "M1: start")
        _commit(self.upstream, "README.md", "hello\n", "M1: readme")
        _git(self.upstream, "push", "-q", "origin", "main")
        _git(tmp, "clone", "-q", "-b", "main", str(self.origin), str(self.local))

    def publish(self, name: str, content: str, message: str) -> str:
        sha = _commit(self.upstream, name, content, message)
        _git(self.upstream, "push", "-q", "origin", "main")
        return sha

    def head(self) -> str:
        return _git(self.local, "rev-parse", "HEAD")


class Recorder:
    """Real git, fake pip; records every command."""

    def __init__(self, pip_code: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.pip_code = pip_code

    def __call__(self, cmd: Sequence[str], cwd: Path, timeout: float) -> CommandResult:
        cmd = list(cmd)
        self.calls.append(cmd)
        if cmd[:3] == [sys.executable, "-m", "pip"]:
            return CommandResult(self.pip_code, "", "pip boom")
        return upd.default_runner(cmd, cwd, timeout)

    @property
    def pip_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[:1] == [sys.executable]]

    @property
    def git_subcommands(self) -> list[str]:
        return [c[1] for c in self.calls if c[:1] == ["git"]]


@pytest.fixture()
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path)


def test_up_to_date_changes_nothing_and_no_pip(setup: Setup) -> None:
    rec = Recorder()
    before = setup.head()
    result = run_update(setup.local, runner=rec)
    assert result.status is UpdateStatus.UP_TO_DATE
    assert result.ok
    assert result.old == result.new == before == setup.head()
    assert rec.pip_calls == []
    assert report_lines(result)[0].startswith(f"Already up to date: {before[:7]}")


def test_fast_forward_lists_new_commits_without_reinstall(setup: Setup) -> None:
    old = setup.head()
    setup.publish("a.txt", "a", "M16: first change")
    new = setup.publish("b.txt", "b", "M16: second change")
    rec = Recorder()
    result = run_update(setup.local, runner=rec)
    assert result.status is UpdateStatus.UPDATED
    assert (result.old, result.new) == (old, new)
    assert setup.head() == new
    assert result.commits == 2
    assert result.changes == ("M16: second change", "M16: first change")
    assert not result.reinstalled
    assert rec.pip_calls == []
    lines = report_lines(result)
    assert lines[0].startswith(f"Updated: {old[:7]} (")
    assert "  - M16: first change" in lines
    assert "No reinstall needed (pyproject.toml unchanged)." in lines
    assert all(line for line in lines)


def test_pyproject_change_reinstalls_with_the_same_python(setup: Setup) -> None:
    setup.publish("pyproject.toml", "[project]\nname='automa'\nversion='2'\n", "deps")
    rec = Recorder()
    result = run_update(setup.local, runner=rec)
    assert result.status is UpdateStatus.UPDATED
    assert result.reinstalled
    assert rec.pip_calls == [
        [sys.executable, "-m", "pip", "install", "--quiet", "-e", str(setup.local.resolve())]
    ]
    assert "Dependencies reinstalled (pyproject.toml changed)." in report_lines(result)


def test_pip_failure_is_reported_but_code_stays_updated(setup: Setup) -> None:
    new = setup.publish("pyproject.toml", "changed\n", "deps")
    result = run_update(setup.local, runner=Recorder(pip_code=1))
    assert result.status is UpdateStatus.INSTALL_FAILED
    assert not result.ok
    assert setup.head() == new
    assert "pip boom" in result.detail
    assert "automa update --reinstall" in report_lines(result)[-1]


def test_reinstall_flag_forces_and_suppresses_pip(setup: Setup) -> None:
    rec = Recorder()
    assert run_update(setup.local, runner=rec, reinstall=True).reinstalled
    assert len(rec.pip_calls) == 1
    setup.publish("pyproject.toml", "changed\n", "deps")
    rec2 = Recorder()
    result = run_update(setup.local, runner=rec2, reinstall=False)
    assert result.status is UpdateStatus.UPDATED and not result.reinstalled
    assert rec2.pip_calls == []


def test_local_edits_block_and_are_kept(setup: Setup) -> None:
    setup.publish("a.txt", "a", "upstream change")
    before = setup.head()
    (setup.local / "README.md").write_text("my edit\n", encoding="utf-8")
    rec = Recorder()
    result = run_update(setup.local, runner=rec)
    assert result.status is UpdateStatus.LOCAL_CHANGES
    assert result.dirty_files == ("README.md",)
    assert setup.head() == before
    assert (setup.local / "README.md").read_text(encoding="utf-8") == "my edit\n"
    assert "fetch" not in rec.git_subcommands
    text = "\n".join(report_lines(result))
    assert "  - README.md" in text and "stash" in text


def test_staged_edits_also_block(setup: Setup) -> None:
    (setup.local / "new.txt").write_text("x", encoding="utf-8")
    _git(setup.local, "add", "new.txt")
    result = run_update(setup.local, runner=Recorder())
    assert result.status is UpdateStatus.LOCAL_CHANGES
    assert result.dirty_files == ("new.txt",)


def test_untracked_files_do_not_block(setup: Setup) -> None:
    new = setup.publish("a.txt", "a", "upstream change")
    (setup.local / ".env").write_text("SECRET=x\n", encoding="utf-8")
    result = run_update(setup.local, runner=Recorder())
    assert result.status is UpdateStatus.UPDATED
    assert setup.head() == new
    assert (setup.local / ".env").read_text(encoding="utf-8") == "SECRET=x\n"


def test_other_branch_blocks(setup: Setup) -> None:
    _git(setup.local, "checkout", "-q", "-b", "experiment")
    result = run_update(setup.local, runner=Recorder())
    assert result.status is UpdateStatus.WRONG_BRANCH
    assert result.branch == "experiment"
    assert _git(setup.local, "rev-parse", "--abbrev-ref", "HEAD") == "experiment"
    assert "switch main" in "\n".join(report_lines(result))


def test_detached_head_blocks(setup: Setup) -> None:
    _git(setup.local, "checkout", "-q", "--detach")
    result = run_update(setup.local, runner=Recorder())
    assert result.status is UpdateStatus.WRONG_BRANCH
    assert result.branch == ""
    assert "detached HEAD" in report_lines(result)[0]


def test_local_commits_block_without_rewriting(setup: Setup) -> None:
    setup.publish("a.txt", "a", "upstream change")
    mine = _commit(setup.local, "mine.txt", "m", "my local commit")
    result = run_update(setup.local, runner=Recorder())
    assert result.status is UpdateStatus.DIVERGED
    assert result.commits == 1
    assert setup.head() == mine
    assert "never rewrites history" in "\n".join(report_lines(result))


def test_local_commit_ahead_without_upstream_change_is_diverged(setup: Setup) -> None:
    mine = _commit(setup.local, "mine.txt", "m", "my local commit")
    result = run_update(setup.local, runner=Recorder())
    assert result.status is UpdateStatus.DIVERGED
    assert setup.head() == mine


def test_fetch_failure_changes_nothing(setup: Setup, tmp_path: Path) -> None:
    _git(setup.local, "remote", "set-url", "origin", str(tmp_path / "missing.git"))
    before = setup.head()
    result = run_update(setup.local, runner=Recorder())
    assert result.status is UpdateStatus.FETCH_FAILED
    assert setup.head() == before
    assert result.detail
    assert "could not download" in report_lines(result)[0]


def test_not_a_clone(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    result = run_update(plain, runner=Recorder())
    assert result.status is UpdateStatus.NOT_A_CLONE
    assert "--repo" in "\n".join(report_lines(result))


def test_subdirectory_of_a_clone_is_refused(setup: Setup) -> None:
    sub = setup.local / "sub"
    sub.mkdir()
    assert run_update(sub, runner=Recorder()).status is UpdateStatus.NOT_A_CLONE


def test_commit_subjects_are_sanitized_and_capped(setup: Setup) -> None:
    for i in range(upd.MAX_CHANGES + 3):
        setup.publish(f"f{i}.txt", str(i), f"change {i}‮ evil\u0007")
    result = run_update(setup.local, runner=Recorder())
    assert len(result.changes) == upd.MAX_CHANGES
    assert result.more_changes == 3
    for line in result.changes:
        assert "‮" not in line and "\u0007" not in line
    assert "  ... and 3 more (see CHANGELOG.md)" in report_lines(result)


def test_only_safe_git_commands_are_used(setup: Setup) -> None:
    setup.publish("a.txt", "a", "upstream change")
    rec = Recorder()
    run_update(setup.local, runner=rec)
    allowed = {"rev-parse", "symbolic-ref", "status", "fetch", "log", "merge-base"}
    allowed |= {"rev-list", "diff", "merge"}
    assert set(rec.git_subcommands) <= allowed
    merges = [c for c in rec.calls if c[:2] == ["git", "merge"]]
    assert merges == [["git", "merge", "--ff-only", "--quiet", "origin/main"]]
    flat = " ".join(" ".join(c) for c in rec.calls)
    for forbidden in ("--force", "reset", "stash", "rebase", "clean", "push"):
        assert forbidden not in flat


def test_source_never_mentions_destructive_git_commands() -> None:
    source = Path(upd.__file__).read_text(encoding="utf-8")
    code = source.split("def report_lines", 1)[0]
    for forbidden in ('"reset"', '"--force"', '"stash"', '"rebase"', '"clean"', '"push"'):
        assert forbidden not in code


def test_installed_repo_is_this_checkout() -> None:
    root = Path(__file__).resolve().parent.parent
    if (root / ".git").exists():
        assert upd.installed_repo() == root


def test_report_lines_are_ascii_for_the_terminal(setup: Setup) -> None:
    statuses = [
        upd.UpdateResult(status=s, repo=setup.local, old="a" * 40, new="b" * 40)
        for s in UpdateStatus
    ]
    for result in statuses:
        lines = report_lines(result)
        assert lines and all(line for line in lines)
        "\n".join(lines).encode("ascii")


# --- CLI ------------------------------------------------------------------------


def _cli(*args: str) -> object:
    return CliRunner().invoke(app, ["update", *args])


def test_cli_update_ok(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    setup.publish("a.txt", "a", "M16: cli change")
    monkeypatch.setattr(upd.run_update, "__kwdefaults__", {"runner": Recorder(), "reinstall": None})
    result = _cli("--repo", str(setup.local))
    assert result.exit_code == 0, result.output
    assert "Updated:" in result.output
    assert "M16: cli change" in result.output


def test_cli_update_blocked_exits_user(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upd.run_update, "__kwdefaults__", {"runner": Recorder(), "reinstall": None})
    (setup.local / "README.md").write_text("edit\n", encoding="utf-8")
    result = _cli("--repo", str(setup.local))
    assert result.exit_code == EXIT_USER
    assert "local edits" in result.output


def test_cli_no_reinstall_flag(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder()
    monkeypatch.setattr(upd.run_update, "__kwdefaults__", {"runner": rec, "reinstall": None})
    setup.publish("pyproject.toml", "changed\n", "deps")
    result = _cli("--repo", str(setup.local), "--no-reinstall")
    assert result.exit_code == 0, result.output
    assert rec.pip_calls == []


# --- the allow-list in tools/command.py ----------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["git", "reset", "--hard", "origin/main"],
        ["git", "push", "origin", "main"],
        ["git", "stash"],
        ["git", "merge", "origin/main"],
        ["git", "merge", "--ff-only", "--quiet", "origin/other"],
        ["git", "fetch", "--force", "origin", "main"],
        ["git", "fetch", "-f", "origin", "main"],
        ["git"],
        ["sh", "-c", "git pull"],
        [sys.executable, "-m", "pip", "install", "-e", "/elsewhere"],
        [sys.executable, "-m", "pip", "uninstall", "automa"],
        ["python3", "-m", "pip", "install", "--quiet", "-e", "."],
    ],
)
def test_update_command_allow_list_refuses(argv: list[str], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        run_update_command(argv, tmp_path, 5)


def test_update_command_runs_allowed_git(setup: Setup) -> None:
    res = run_update_command(["git", "rev-parse", "HEAD"], setup.local, 30)
    assert res.returncode == 0 and res.stdout.strip() == setup.head()


def test_update_command_missing_program_is_127(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(setup.local / "nothing-here"))
    res = run_update_command(["git", "status"], setup.local, 30)
    assert res.returncode == 127
