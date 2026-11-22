"""Does a newer Automa wait on ``origin/main``? (web UI banner, D178).

The web UI shows a small "a new version is ready" bar when the clone Automa runs
from is behind ``origin/main``. Checking never changes the clone: it runs at most one
``git fetch --quiet origin main`` a day (download only; nothing about the user or
their projects is sent), in a background thread, through the same allow-list as
``automa update`` (``run_update_command``, D170). Everything else is local git
(``rev-parse``, ``merge-base``, ``rev-list``, ``diff``). Any failure — no network, no
git, not a clone, another branch, local commits — means "no banner", silently.

"What is new" comes from the ``## `` headings that ``origin/main`` adds to
``CHANGELOG.md`` (Persian titles written by each run), not from commit subjects.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from automa.update import BRANCH, REMOTE, Runner, default_runner, installed_repo

#: How often the remote is fetched at most (the roadmap's "at most once a day").
FETCH_INTERVAL = timedelta(hours=24)
#: How often the (local, cheap) comparison is redone, e.g. after ``automa update``.
LOCAL_INTERVAL = timedelta(minutes=10)
MAX_TITLES = 8
STATE_FILENAME = "update-check.json"
_FETCH_TIMEOUT_S = 60.0
_LOCAL_TIMEOUT_S = 20.0
_TITLE_LIMIT = 160

_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}\s*[—–-]\s*")
_MILESTONE = re.compile(r"^M\d+[a-z]?\s*:\s*", re.IGNORECASE)
_DECISION_SUFFIX = re.compile(r"\s*\(D\d+[^()]*\)\s*$")


class UpdateNotice(BaseModel):
    """A newer version is waiting: how many commits and what is new."""

    model_config = ConfigDict(frozen=True)

    commits: int
    titles: tuple[str, ...] = ()
    more: int = 0


def _clean(text: str) -> str:
    """One printable line: control and bidi-override characters dropped."""
    out = []
    for ch in text:
        code = ord(ch)
        if code < 32 or code == 127 or 0x202A <= code <= 0x202E or 0x2066 <= code <= 0x2069:
            continue
        out.append(ch)
    line = " ".join("".join(out).split())
    return line if len(line) <= _TITLE_LIMIT else line[: _TITLE_LIMIT - 1] + "…"


def friendly_title(heading: str) -> str:
    """``## 2026-11-22 — M16: بازطراحی ظاهر (D177)`` → ``بازطراحی ظاهر``."""
    title = heading.strip().removeprefix("##").strip()
    title = _DATE_PREFIX.sub("", title)
    title = _MILESTONE.sub("", title)
    title = _DECISION_SUFFIX.sub("", title)
    return _clean(title)


def changelog_titles(diff: str) -> list[str]:
    """The friendly titles of the ``## `` headings a unified diff adds (in order)."""
    titles: list[str] = []
    for line in diff.splitlines():
        if not line.startswith("+## "):
            continue
        title = friendly_title(line[1:])
        if title and title not in titles:
            titles.append(title)
    return titles


def _git(repo: Path, runner: Runner, *args: str, timeout: float = _LOCAL_TIMEOUT_S) -> str | None:
    try:
        proc = runner(["git", *args], repo, timeout)
    except (OSError, ValueError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def fetch_remote(repo: Path, runner: Runner = default_runner) -> bool:
    """``git fetch --quiet origin main`` (download only); ``False`` on any failure."""
    out = _git(repo, runner, "fetch", "--quiet", REMOTE, BRANCH, timeout=_FETCH_TIMEOUT_S)
    return out is not None


def pending_update(repo: Path, runner: Runner = default_runner) -> UpdateNotice | None:
    """What ``automa update`` would bring in now, from local refs only (no network).

    ``None`` unless the clone is on ``main`` and ``origin/main`` is strictly ahead
    of it (a fast-forward): another branch or local commits are ``automa update``'s
    to explain, not a banner's."""
    if _git(repo, runner, "symbolic-ref", "--quiet", "--short", "HEAD") != BRANCH:
        return None
    head = _git(repo, runner, "rev-parse", "HEAD")
    target = _git(repo, runner, "rev-parse", "--verify", "--quiet", f"{REMOTE}/{BRANCH}")
    if not head or not target or head == target:
        return None
    if _git(repo, runner, "merge-base", "--is-ancestor", "HEAD", f"{REMOTE}/{BRANCH}") is None:
        return None
    count = _git(repo, runner, "rev-list", "--count", f"HEAD..{REMOTE}/{BRANCH}") or ""
    if not count.isdigit() or int(count) == 0:
        return None
    diff = _git(
        repo,
        runner,
        "diff",
        "--no-color",
        "--no-ext-diff",
        "-U0",
        "HEAD",
        f"{REMOTE}/{BRANCH}",
        "--",
        "CHANGELOG.md",
    )
    titles = changelog_titles(diff or "")
    return UpdateNotice(
        commits=int(count),
        titles=tuple(titles[:MAX_TITLES]),
        more=max(0, len(titles) - MAX_TITLES),
    )


def _start_thread(job: Callable[[], None]) -> None:
    threading.Thread(target=job, name="automa-update-check", daemon=True).start()


class UpdateWatcher:
    """Keeps the web UI's "new version" notice without ever blocking a request.

    :meth:`notice` returns the last known result at once and, when due, starts one
    background job: a fetch if the last attempt (kept in ``state_path`` so restarts
    do not fetch again) is a day old or more, then the local comparison."""

    def __init__(
        self,
        repo: Path,
        state_path: Path,
        *,
        runner: Runner = default_runner,
        clock: Callable[[], datetime] | None = None,
        start: Callable[[Callable[[], None]], None] = _start_thread,
    ) -> None:
        self._repo = repo
        self._state_path = state_path
        self._runner = runner
        self._clock = clock or (lambda: datetime.now(UTC))
        self._start = start
        self._lock = threading.Lock()
        self._busy = False
        self._notice: UpdateNotice | None = None
        self._local_at: datetime | None = None
        self._attempt_at: datetime | None = None  # also when the state file is unwritable

    def notice(self) -> UpdateNotice | None:
        now = self._clock()
        with self._lock:
            start = not self._busy and (
                self._local_at is None
                or now - self._local_at >= LOCAL_INTERVAL
                or self._fetch_due(now)
            )
            if start:
                self._busy = True
        if start:
            try:
                self._start(self._job)
            except Exception:  # noqa: BLE001 - no thread, no banner
                with self._lock:
                    self._busy = False
        with self._lock:
            return self._notice

    def _job(self) -> None:
        try:
            now = self._clock()
            if self._fetch_due(now):
                self._attempt_at = now
                self._write_attempt(now)  # attempt, not success: at most once a day
                fetch_remote(self._repo, self._runner)
            notice = pending_update(self._repo, self._runner)
        except Exception:  # noqa: BLE001 - checking must never break the page
            notice = None
        with self._lock:
            self._notice = notice
            self._local_at = self._clock()
            self._busy = False

    def _last_attempt(self) -> datetime | None:
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            value = datetime.fromisoformat(str(raw["last_fetch"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return value if value.tzinfo is not None else None

    def _fetch_due(self, now: datetime) -> bool:
        last = self._last_attempt()
        mem = self._attempt_at
        if mem is not None and (last is None or mem > last):
            last = mem
        # A time in the future (clock moved back) also counts as due.
        return last is None or last > now or now - last >= FETCH_INTERVAL

    def _write_attempt(self, now: datetime) -> None:
        tmp = self._state_path.with_name(self._state_path.name + ".tmp")
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({"last_fetch": now.isoformat()}), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError:
            pass


def watcher_for(workspace: Path) -> UpdateWatcher | None:
    """The watcher of the clone this Automa runs from, or ``None`` (not a clone)."""
    repo = installed_repo()
    if repo is None:
        return None
    return UpdateWatcher(repo, workspace / STATE_FILENAME)
