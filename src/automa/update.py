"""Safe self-update of an Automa git clone (``automa update``, D170).

The flow is deliberately conservative: it only ever fast-forwards the local
``main`` branch to ``origin/main``. It never forces, resets, stashes, rebases or
deletes anything. When the clone is on another branch, has local edits to tracked
files, or has commits that are not on ``origin/main``, it stops and says exactly what
is in the way and what the user can do.

``pip install -e`` runs again only when ``pyproject.toml`` changed (new dependencies
or entry points); with an editable install every other change is live right after
the merge. All external commands go through an injectable runner so tests use local
temporary git repositories and never touch the network or pip.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from automa.tools.command import CommandResult, run_update_command

BRANCH = "main"
REMOTE = "origin"
MAX_CHANGES = 12
_GIT_TIMEOUT_S = 120.0
_PIP_TIMEOUT_S = 900.0

Runner = Callable[[Sequence[str], Path, float], CommandResult]


class UpdateStatus(StrEnum):
    UPDATED = "updated"
    UP_TO_DATE = "up_to_date"
    NOT_A_CLONE = "not_a_clone"
    WRONG_BRANCH = "wrong_branch"
    LOCAL_CHANGES = "local_changes"
    DIVERGED = "diverged"
    FETCH_FAILED = "fetch_failed"
    MERGE_FAILED = "merge_failed"
    INSTALL_FAILED = "install_failed"


#: Statuses that leave the clone exactly as it was (nothing merged).
BLOCKED = frozenset(
    {
        UpdateStatus.NOT_A_CLONE,
        UpdateStatus.WRONG_BRANCH,
        UpdateStatus.LOCAL_CHANGES,
        UpdateStatus.DIVERGED,
        UpdateStatus.FETCH_FAILED,
        UpdateStatus.MERGE_FAILED,
    }
)


class UpdateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: UpdateStatus
    repo: Path | None = None
    old: str = ""
    new: str = ""
    old_date: str = ""
    new_date: str = ""
    commits: int = 0
    changes: tuple[str, ...] = ()
    more_changes: int = 0
    reinstalled: bool = False
    branch: str = ""
    dirty_files: tuple[str, ...] = ()
    detail: str = Field(default="", max_length=2000)

    @property
    def ok(self) -> bool:
        return self.status in (UpdateStatus.UPDATED, UpdateStatus.UP_TO_DATE)


def default_runner(cmd: Sequence[str], cwd: Path, timeout: float) -> CommandResult:
    """Runs through the allow-list in ``automa.tools.command``, the only module
    that starts processes (a missing program is exit code 127)."""
    return run_update_command(cmd, cwd, timeout)


def installed_repo() -> Path | None:
    """The clone this Automa runs from (editable install), or ``None``."""
    root = Path(__file__).resolve().parents[2]
    if (root / ".git").exists() and (root / "pyproject.toml").is_file():
        return root
    return None


def pip_command(repo: Path) -> list[str]:
    return [sys.executable, "-m", "pip", "install", "--quiet", "-e", str(repo)]


def _clip(text: str, limit: int = 600) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _safe_line(text: str) -> str:
    """One printable line: control and bidi characters are dropped."""
    out = []
    for ch in text:
        code = ord(ch)
        if code < 32 or code == 127 or 0x202A <= code <= 0x202E or 0x2066 <= code <= 0x2069:
            continue
        out.append(ch)
    return _clip("".join(out), 200)


class _Git:
    def __init__(self, repo: Path, runner: Runner) -> None:
        self.repo = repo
        self.runner = runner

    def run(self, *args: str) -> CommandResult:
        return self.runner(["git", *args], self.repo, _GIT_TIMEOUT_S)

    def out(self, *args: str) -> str | None:
        proc = self.run(*args)
        return proc.stdout.strip() if proc.returncode == 0 else None


def run_update(
    repo: Path | None = None,
    *,
    runner: Runner = default_runner,
    reinstall: bool | None = None,
) -> UpdateResult:
    """Fast-forward ``repo`` to ``origin/main`` and reinstall when needed.

    ``reinstall``: ``None`` = only when ``pyproject.toml`` changed, ``True`` = always
    after a successful check (even when already up to date), ``False`` = never.
    """
    if repo is None:
        repo = installed_repo()
    if repo is None:
        return UpdateResult(status=UpdateStatus.NOT_A_CLONE)
    repo = repo.resolve()
    git = _Git(repo, runner)
    top = git.out("rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != repo:
        return UpdateResult(status=UpdateStatus.NOT_A_CLONE, repo=repo)

    branch = git.out("symbolic-ref", "--quiet", "--short", "HEAD") or ""
    if branch != BRANCH:
        return UpdateResult(status=UpdateStatus.WRONG_BRANCH, repo=repo, branch=branch)

    status = git.run("status", "--porcelain", "--untracked-files=no")
    if status.returncode != 0:
        return UpdateResult(status=UpdateStatus.NOT_A_CLONE, repo=repo, branch=branch)
    dirty = status.stdout.rstrip("\n")  # keep the leading status column
    if dirty.strip():
        files = tuple(_safe_line(line[3:]) for line in dirty.splitlines() if line.strip())
        return UpdateResult(
            status=UpdateStatus.LOCAL_CHANGES, repo=repo, branch=branch, dirty_files=files[:20]
        )

    old = git.out("rev-parse", "HEAD") or ""
    fetch = git.run("fetch", "--quiet", REMOTE, BRANCH)
    if fetch.returncode != 0:
        return UpdateResult(
            status=UpdateStatus.FETCH_FAILED,
            repo=repo,
            branch=branch,
            old=old,
            detail=_clip(fetch.stderr or fetch.stdout),
        )
    remote_ref = f"{REMOTE}/{BRANCH}"
    target = git.out("rev-parse", remote_ref) or ""
    old_date = git.out("log", "-1", "--format=%cs", old) or ""
    if target == old:
        reinstalled = False
        if reinstall:
            failed = _pip(repo, runner)
            if failed is not None:
                return failed.model_copy(update={"old": old, "new": old, "branch": branch})
            reinstalled = True
        return UpdateResult(
            status=UpdateStatus.UP_TO_DATE,
            repo=repo,
            branch=branch,
            old=old,
            new=old,
            old_date=old_date,
            new_date=old_date,
            reinstalled=reinstalled,
        )

    ancestor = git.run("merge-base", "--is-ancestor", "HEAD", remote_ref)
    if ancestor.returncode != 0:
        ahead = git.out("rev-list", "--count", f"{remote_ref}..HEAD") or "?"
        return UpdateResult(
            status=UpdateStatus.DIVERGED,
            repo=repo,
            branch=branch,
            old=old,
            new=target,
            commits=int(ahead) if ahead.isdigit() else 0,
        )

    subjects = git.out("log", "--format=%s", "--no-merges", f"{old}..{target}") or ""
    lines = [_safe_line(s) for s in subjects.splitlines() if s.strip()]
    count_out = git.out("rev-list", "--count", f"{old}..{target}") or "0"
    pyproject_changed = bool(git.out("diff", "--name-only", old, target, "--", "pyproject.toml"))

    merge = git.run("merge", "--ff-only", "--quiet", remote_ref)
    if merge.returncode != 0:
        return UpdateResult(
            status=UpdateStatus.MERGE_FAILED,
            repo=repo,
            branch=branch,
            old=old,
            new=target,
            detail=_clip(merge.stderr or merge.stdout),
        )
    new = git.out("rev-parse", "HEAD") or target
    new_date = git.out("log", "-1", "--format=%cs", new) or ""
    base = UpdateResult(
        status=UpdateStatus.UPDATED,
        repo=repo,
        branch=branch,
        old=old,
        new=new,
        old_date=old_date,
        new_date=new_date,
        commits=int(count_out) if count_out.isdigit() else len(lines),
        changes=tuple(lines[:MAX_CHANGES]),
        more_changes=max(0, len(lines) - MAX_CHANGES),
    )
    want_pip = pyproject_changed if reinstall is None else reinstall
    if not want_pip:
        return base
    failed = _pip(repo, runner)
    if failed is not None:
        return base.model_copy(update={"status": failed.status, "detail": failed.detail})
    return base.model_copy(update={"reinstalled": True})


def _pip(repo: Path, runner: Runner) -> UpdateResult | None:
    proc = runner(pip_command(repo), repo, _PIP_TIMEOUT_S)
    if proc.returncode != 0:
        return UpdateResult(
            status=UpdateStatus.INSTALL_FAILED,
            repo=repo,
            detail=_clip((proc.stderr or proc.stdout)[-1500:]),
        )
    return None


def report_lines(result: UpdateResult) -> list[str]:
    """Plain English lines for the terminal (Persian is shown reversed in the
    Windows/WSL console, so the terminal report stays English; D170)."""
    return [line for line in _report_lines(result) if line]


def _report_lines(result: UpdateResult) -> list[str]:
    short_old = result.old[:7]
    short_new = result.new[:7]
    s = result.status
    if s is UpdateStatus.UPDATED:
        lines = [
            f"Updated: {short_old} ({result.old_date}) -> {short_new} ({result.new_date}),"
            f" {result.commits} new commit(s).",
        ]
        if result.changes:
            lines.append("What is new:")
            lines.extend(f"  - {c}" for c in result.changes)
            if result.more_changes:
                lines.append(f"  ... and {result.more_changes} more (see CHANGELOG.md)")
        lines.append(
            "Dependencies reinstalled (pyproject.toml changed)."
            if result.reinstalled
            else "No reinstall needed (pyproject.toml unchanged)."
        )
        lines.append("If Automa web is open, close it and open it again to use the new version.")
        return lines
    if s is UpdateStatus.UP_TO_DATE:
        line = f"Already up to date: {short_old} ({result.old_date})."
        return [line, "Dependencies reinstalled."] if result.reinstalled else [line]
    if s is UpdateStatus.NOT_A_CLONE:
        where = f" ({result.repo})" if result.repo else ""
        return [
            f"Not updated: this Automa does not run from a git clone{where}.",
            "Run it from the clone (e.g. ~/automa/.venv/bin/automa update) or pass --repo.",
        ]
    if s is UpdateStatus.WRONG_BRANCH:
        name = result.branch or "a detached HEAD"
        return [
            f"Not updated: the clone is on {name}, not '{BRANCH}'. Nothing was changed.",
            f"To go back: git -C {result.repo} switch {BRANCH}   (then run automa update again)",
        ]
    if s is UpdateStatus.LOCAL_CHANGES:
        lines = ["Not updated: these files have local edits. Nothing was changed:"]
        lines.extend(f"  - {f}" for f in result.dirty_files)
        r = result.repo
        lines.append(f"Keep them: git -C {r} stash   (later: git -C {r} stash pop)")
        lines.append(f"Or throw them away: git -C {r} restore <file>")
        lines.append("Then run automa update again.")
        return lines
    if s is UpdateStatus.DIVERGED:
        return [
            f"Not updated: the clone has {result.commits} local commit(s) that are not on"
            f" {REMOTE}/{BRANCH}. Nothing was changed.",
            "Automa never rewrites history; ask for help or move your commits to a branch first.",
        ]
    if s is UpdateStatus.FETCH_FAILED:
        return [
            f"Not updated: could not download from '{REMOTE}' (internet or GitHub login).",
            f"Git said: {result.detail}" if result.detail else "",
            "Check the connection, or that 'git fetch' works in this folder, and try again.",
        ]
    if s is UpdateStatus.MERGE_FAILED:
        return [
            "Not updated: git refused the fast-forward (nothing was changed).",
            f"Git said: {result.detail}" if result.detail else "",
        ]
    return [
        f"The code is now at {short_new}, but 'pip install -e .' failed.",
        f"pip said: {result.detail}" if result.detail else "",
        "Try again: automa update --reinstall",
    ]
