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

After an update (D179) the first web page shows, once, "you are on the new version"
with the same kind of list: the commit the web UI last ran is kept in the workspace
(``seen_head``) and compared with the commit this process started from.
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
#: How long the "you are on the new version" bar stays after it was first shown.
WELCOME_WINDOW = timedelta(minutes=10)
_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

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
    return _notice_between(repo, runner, "HEAD", f"{REMOTE}/{BRANCH}")


def _notice_between(repo: Path, runner: Runner, old: str, new: str) -> UpdateNotice | None:
    """Commits ``old..new`` and the CHANGELOG titles they add; ``None`` when none."""
    count = _git(repo, runner, "rev-list", "--count", f"{old}..{new}") or ""
    if not count.isdigit() or int(count) == 0:
        return None
    diff = _git(
        repo, runner, "diff", "--no-color", "--no-ext-diff", "-U0", old, new, "--", "CHANGELOG.md"
    )
    titles = changelog_titles(diff or "")
    return UpdateNotice(
        commits=int(count),
        titles=tuple(titles[:MAX_TITLES]),
        more=max(0, len(titles) - MAX_TITLES),
    )


def just_updated(
    repo: Path, seen: str, head: str, runner: Runner = default_runner
) -> UpdateNotice | None:
    """What came in between the commit the web UI last ran (``seen``) and ``head``.

    ``None`` unless both are full commit ids (the state file is never trusted as a
    git argument) and ``head`` is strictly ahead of ``seen``: another branch or an
    older commit is not "an update"."""
    if not _SHA.match(seen) or not _SHA.match(head) or seen == head:
        return None
    if _git(repo, runner, "merge-base", "--is-ancestor", seen, head) is None:
        return None
    return _notice_between(repo, runner, seen, head)


def _start_thread(job: Callable[[], None]) -> None:
    threading.Thread(target=job, name="automa-update-check", daemon=True).start()


class UpdateWatcher:
    """Keeps the web UI's "new version" notice without ever blocking a request.

    :meth:`notice` returns the last known result at once and, when due, starts one
    background job: a fetch if the last attempt (kept in ``state_path`` so restarts
    do not fetch again) is a day old or more, then the local comparison.

    The first job also compares the commit this process started from with the one
    the web UI last ran (``seen_head`` in the same file): if it moved forward,
    :meth:`welcome` returns what came in until :data:`WELCOME_WINDOW` after
    :meth:`welcome_shown` is first called, and ``seen_head`` is updated at once so
    a restart does not show it again (D179)."""

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
        self._state_lock = threading.Lock()
        self._busy = False
        self._notice: UpdateNotice | None = None
        self._local_at: datetime | None = None
        self._attempt_at: datetime | None = None  # also when the state file is unwritable
        self._boot_checked = False
        self._boot_head: str | None = None
        self._welcome: UpdateNotice | None = None
        self._welcome_until: datetime | None = None

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

    def welcome(self) -> UpdateNotice | None:
        """ "You are on the new version" (what came in), or ``None``; never blocks."""
        now = self._clock()
        with self._lock:
            if self._welcome is None:
                return None
            if self._welcome_until is not None and now >= self._welcome_until:
                self._welcome = None
                return None
            return self._welcome

    def welcome_shown(self) -> None:
        """The bar was put on a page: start its window and remember this commit."""
        with self._lock:
            if self._welcome is None or self._welcome_until is not None:
                return
            self._welcome_until = self._clock() + WELCOME_WINDOW
            head = self._boot_head
        if head is not None:
            self._write_state(seen_head=head)

    def _check_boot(self) -> None:
        """Once per process: did the commit we started from move since last time?"""
        head = _git(self._repo, self._runner, "rev-parse", "HEAD")
        if not head or not _SHA.match(head):
            return
        self._boot_head = head
        seen = self._read_state().get("seen_head")
        notice = None
        if isinstance(seen, str) and seen != head:
            notice = just_updated(self._repo, seen, head, self._runner)
        if notice is None:
            if seen != head:
                # First run, another branch or an older commit: remember, say nothing.
                self._write_state(seen_head=head)
            return
        with self._lock:
            self._welcome = notice

    def _job(self) -> None:
        if not self._boot_checked:
            self._boot_checked = True
            try:
                self._check_boot()
            except Exception:  # noqa: BLE001, S110 - a welcome must never break the page
                pass
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

    def _read_state(self) -> dict[str, object]:
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _last_attempt(self) -> datetime | None:
        try:
            value = datetime.fromisoformat(str(self._read_state()["last_fetch"]))
        except (ValueError, KeyError, TypeError):
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
        self._write_state(last_fetch=now.isoformat())

    def _write_state(self, **values: str) -> None:
        """Merge ``values`` into the state file (atomic); unwritable is fine."""
        with self._state_lock:
            state = self._read_state()
            state.update(values)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            try:
                self._state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(state), encoding="utf-8")
                os.replace(tmp, self._state_path)
            except OSError:
                pass


def watcher_for(workspace: Path) -> UpdateWatcher | None:
    """The watcher of the clone this Automa runs from, or ``None`` (not a clone)."""
    repo = installed_repo()
    if repo is None:
        return None
    return UpdateWatcher(repo, workspace / STATE_FILENAME)
