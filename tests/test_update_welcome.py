"""The web UI's one-time "Automa is updated to the new version" bar (D179).

Real local git repositories with a bare repository as ``origin``; no network, no
model. The background thread is replaced by a synchronous starter.
"""

from __future__ import annotations

import json
import shutil
from datetime import timedelta
from pathlib import Path

import pytest

from automa import updatecheck as uc
from automa.updatecheck import UpdateNotice, UpdateWatcher, just_updated
from automa.webui import server as ws
from automa.webui.render import TopBar, page, top_bar
from test_update import Setup, _commit, _git
from test_updatecheck import HOST, PORT, T0, Clock, Recorder, _changelog, _sync

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

CHANGELOG_OLD = "# گزارش تغییرات\n\n## 2026-11-20 — M15: کار قدیمی (D150)\n- جزئیات\n"


@pytest.fixture
def repo(tmp_path: Path) -> Setup:
    s = Setup(tmp_path)
    s.publish("CHANGELOG.md", CHANGELOG_OLD, "M15: old work")
    _git(s.local, "pull", "-q", "--ff-only")
    return s


def _state(tmp: Path) -> Path:
    return tmp / "ws" / uc.STATE_FILENAME


def _read(tmp: Path) -> dict[str, object]:
    return json.loads(_state(tmp).read_text(encoding="utf-8"))


def _watcher(s: Setup, tmp: Path, clock: Clock, rec: Recorder | None = None) -> UpdateWatcher:
    return UpdateWatcher(s.local, _state(tmp), runner=rec or Recorder(), clock=clock, start=_sync)


def _update_clone(s: Setup) -> None:
    """What ``automa update`` does to the clone (a fast-forward)."""
    _git(s.local, "pull", "-q", "--ff-only")


def _boot(s: Setup, tmp: Path, clock: Clock) -> UpdateWatcher:
    """A fresh process: the first page starts the background job."""
    w = _watcher(s, tmp, clock)
    w.notice()
    return w


# --- just_updated ---------------------------------------------------------------


def test_just_updated_lists_titles_between_two_commits(repo: Setup) -> None:
    old = repo.head()
    repo.publish(
        "CHANGELOG.md",
        _changelog("2026-11-23 — M16: دومی (D2)", "2026-11-23 — M16: اولی (D1)"),
        "M16: two",
    )
    repo.publish("README.md", "x\n", "M16: readme")
    _update_clone(repo)
    notice = just_updated(repo.local, old, repo.head(), Recorder())
    assert notice == UpdateNotice(commits=2, titles=("دومی", "اولی"))


@pytest.mark.parametrize(
    "seen",
    ["", "HEAD", "--output=/tmp/x", "abc", "g" * 40, "A" * 40, "0" * 39],
)
def test_just_updated_never_passes_untrusted_text_to_git(repo: Setup, seen: str) -> None:
    rec = Recorder()
    assert just_updated(repo.local, seen, repo.head(), rec) is None
    assert rec.calls == []


def test_just_updated_needs_head_strictly_ahead(repo: Setup) -> None:
    head = repo.head()
    assert just_updated(repo.local, head, head, Recorder()) is None
    old = head
    repo.publish("README.md", "x\n", "M16: tiny")
    _update_clone(repo)
    new = repo.head()
    assert just_updated(repo.local, new, old, Recorder()) is None  # went back
    assert just_updated(repo.local, "f" * 40, new, Recorder()) is None  # unknown commit


def test_just_updated_on_diverged_history_is_none(repo: Setup) -> None:
    base = repo.head()
    side = _commit(repo.local, "a.txt", "a\n", "side")
    _git(repo.local, "reset", "-q", "--hard", base)
    other = _commit(repo.local, "b.txt", "b\n", "other")
    assert just_updated(repo.local, side, other, Recorder()) is None


# --- the watcher --------------------------------------------------------------------


def test_first_run_remembers_the_commit_and_says_nothing(repo: Setup, tmp_path: Path) -> None:
    w = _boot(repo, tmp_path, Clock())
    assert w.welcome() is None
    assert _read(tmp_path)["seen_head"] == repo.head()
    assert "last_fetch" in _read(tmp_path)  # the fetch time is kept alongside


def test_same_commit_after_restart_says_nothing(repo: Setup, tmp_path: Path) -> None:
    clock = Clock()
    _boot(repo, tmp_path, clock)
    assert _boot(repo, tmp_path, clock).welcome() is None


def test_after_update_the_next_start_welcomes_once(repo: Setup, tmp_path: Path) -> None:
    clock = Clock()
    _boot(repo, tmp_path, clock)
    old = repo.head()
    repo.publish("CHANGELOG.md", _changelog("2026-11-23 — M16: پیام به‌روز شدی (D179)"), "M16")
    _update_clone(repo)

    w = _boot(repo, tmp_path, clock)
    notice = w.welcome()
    assert notice == UpdateNotice(commits=1, titles=("پیام به‌روز شدی",))
    assert _read(tmp_path)["seen_head"] == old  # not yet shown on a page

    w.welcome_shown()
    assert _read(tmp_path)["seen_head"] == repo.head()
    clock.now = T0 + uc.WELCOME_WINDOW - timedelta(seconds=1)
    assert w.welcome() == notice  # still there while the user looks around
    w.welcome_shown()  # does not extend the window
    clock.now = T0 + uc.WELCOME_WINDOW
    assert w.welcome() is None
    clock.now = T0 + uc.WELCOME_WINDOW + timedelta(hours=1)
    assert w.welcome() is None

    assert _boot(repo, tmp_path, clock).welcome() is None  # a restart: not again


def test_not_shown_yet_means_shown_after_restart(repo: Setup, tmp_path: Path) -> None:
    clock = Clock()
    _boot(repo, tmp_path, clock)
    repo.publish("README.md", "x\n", "M16: tiny")
    _update_clone(repo)
    _boot(repo, tmp_path, clock)  # only self-refreshing pages were opened
    clock.now = T0 + timedelta(days=2)
    assert _boot(repo, tmp_path, clock).welcome() == UpdateNotice(commits=1)


def test_update_while_running_waits_for_a_restart(repo: Setup, tmp_path: Path) -> None:
    """The running server still has the old code: no welcome until it is reopened."""
    clock = Clock()
    w = _boot(repo, tmp_path, clock)
    repo.publish("README.md", "x\n", "M16: tiny")
    _update_clone(repo)
    clock.now = T0 + uc.LOCAL_INTERVAL
    w.notice()
    assert w.welcome() is None
    assert _boot(repo, tmp_path, clock).welcome() == UpdateNotice(commits=1)


def test_switching_back_to_an_older_commit_is_quiet(repo: Setup, tmp_path: Path) -> None:
    clock = Clock()
    old = repo.head()
    repo.publish("README.md", "x\n", "M16: tiny")
    _update_clone(repo)
    _boot(repo, tmp_path, clock)
    _git(repo.local, "reset", "-q", "--hard", old)
    assert _boot(repo, tmp_path, clock).welcome() is None
    assert _read(tmp_path)["seen_head"] == old


@pytest.mark.parametrize("seen", ["--help", 5, None, "x" * 40])
def test_bad_seen_head_is_replaced_quietly(repo: Setup, tmp_path: Path, seen: object) -> None:
    _state(tmp_path).parent.mkdir()
    _state(tmp_path).write_text(json.dumps({"seen_head": seen}), encoding="utf-8")
    rec = Recorder()
    w = _watcher(repo, tmp_path, Clock(), rec)
    w.notice()
    assert w.welcome() is None
    assert _read(tmp_path)["seen_head"] == repo.head()
    assert all(not arg.startswith("--help") for c in rec.calls for arg in c)


def test_boot_check_runs_once_per_process(repo: Setup, tmp_path: Path) -> None:
    rec, clock = Recorder(), Clock()
    w = _watcher(repo, tmp_path, clock, rec)
    w.notice()
    heads = sum(1 for c in rec.calls if c[1:3] == ["rev-parse", "HEAD"])
    clock.now = T0 + uc.LOCAL_INTERVAL
    w.notice()
    assert sum(1 for c in rec.calls if c[1:3] == ["rev-parse", "HEAD"]) == heads + 1  # pending
    assert heads == 2  # boot + the first pending check


def test_unwritable_state_never_welcomes(repo: Setup, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory", encoding="utf-8")
    w = UpdateWatcher(
        repo.local, blocker / uc.STATE_FILENAME, runner=Recorder(), clock=Clock(), start=_sync
    )
    w.notice()
    assert w.welcome() is None
    w.welcome_shown()  # nothing to do, no error


def test_not_a_repository_is_silent(tmp_path: Path) -> None:
    w = UpdateWatcher(tmp_path, _state(tmp_path), runner=Recorder(), clock=Clock(), start=_sync)
    w.notice()
    assert w.welcome() is None
    assert not _state(tmp_path).exists() or "seen_head" not in _read(tmp_path)


def test_existing_fetch_time_is_kept(repo: Setup, tmp_path: Path) -> None:
    _state(tmp_path).parent.mkdir()
    stamp = (T0 - timedelta(hours=1)).isoformat()
    _state(tmp_path).write_text(json.dumps({"last_fetch": stamp}), encoding="utf-8")
    rec = Recorder()
    _watcher(repo, tmp_path, Clock(), rec).notice()
    assert rec.fetches == 0
    assert _read(tmp_path) == {"last_fetch": stamp, "seen_head": repo.head()}


# --- the page -----------------------------------------------------------------------

MARK = 'class="updatebar welcomebar"'
WELCOME = UpdateNotice(commits=5, titles=("پیام به‌روز شدی", "<i>y</i>"), more=1)


def test_welcome_bar_in_persian_calls_back_once_per_page() -> None:
    shown: list[int] = []
    with top_bar(TopBar(welcome=WELCOME, welcome_shown=lambda: shown.append(1))):
        doc = page("t", "<p>body</p>", lang="fa")
    assert 'class="updatebar welcomebar"' in doc
    assert "Automa به نسخهٔ تازه به‌روز شد" in doc
    assert "<details open>" in doc
    assert "پیام به‌روز شدی" in doc
    assert "<i>y</i>" not in doc and "&lt;i&gt;y&lt;/i&gt;" in doc
    assert "و ۱ مورد دیگه" in doc
    assert "automa update" not in doc  # nothing to do
    assert doc.index(MARK) < doc.index('<main class="wrap">')
    assert shown == [1]


def test_welcome_bar_in_english_without_titles() -> None:
    with top_bar(TopBar(welcome=UpdateNotice(commits=2))):
        doc = page("t", "", lang="en")
    assert "Automa is updated to the new version" in doc
    assert "2 small improvements" in doc


def test_welcome_not_on_refreshing_pages_and_not_counted() -> None:
    shown: list[int] = []
    with top_bar(TopBar(welcome=WELCOME, welcome_shown=lambda: shown.append(1))):
        doc = page("t", "", lang="fa", refresh=3)
    assert MARK not in doc
    assert shown == []


def test_welcome_then_ready_bar_together() -> None:
    with top_bar(TopBar(welcome=WELCOME, update=UpdateNotice(commits=1))):
        doc = page("t", "", lang="fa")
    assert doc.index(MARK) < doc.index("نسخهٔ تازهٔ Automa آماده‌ست")


def test_broken_callback_does_not_break_the_page() -> None:
    def boom() -> None:
        raise RuntimeError("boom")

    with top_bar(TopBar(welcome=WELCOME, welcome_shown=boom)):
        assert MARK in page("t", "", lang="fa")


class FakeWatcher:
    def __init__(self, welcome: UpdateNotice | None = None, boom: bool = False) -> None:
        self._welcome = welcome
        self.boom = boom
        self.shown = 0

    def notice(self) -> UpdateNotice | None:
        return None

    def welcome(self) -> UpdateNotice | None:
        if self.boom:
            raise RuntimeError("boom")
        return self._welcome

    def welcome_shown(self) -> None:
        self.shown += 1


def _app(root: Path, watcher: object) -> ws.WebApp:
    return ws.WebApp(
        root,
        root / "ledger.sqlite3",
        ws.allowed_hosts("127.0.0.1", PORT),
        language="fa",
        updates=watcher,  # type: ignore[arg-type]
    )


def test_server_shows_welcome_and_reports_it(tmp_path: Path) -> None:
    watcher = FakeWatcher(WELCOME)
    body = _app(tmp_path, watcher).handle("GET", "/", HOST).body.decode("utf-8")
    assert MARK in body
    assert watcher.shown == 1


def test_server_with_broken_or_old_watcher(tmp_path: Path) -> None:
    class OnlyNotice:
        def notice(self) -> UpdateNotice | None:
            return None

    for watcher in (FakeWatcher(), FakeWatcher(WELCOME, boom=True), OnlyNotice()):
        r = _app(tmp_path, watcher).handle("GET", "/", HOST)
        assert r.status == 200
        assert MARK not in r.body.decode("utf-8")


def test_real_watcher_end_to_end(repo: Setup, tmp_path: Path) -> None:
    clock = Clock()
    _boot(repo, tmp_path, clock)
    repo.publish("CHANGELOG.md", _changelog("2026-11-23 — M16: چیز تازه (D179)"), "M16")
    _update_clone(repo)
    w = _watcher(repo, tmp_path, clock)
    app = _app(tmp_path, w)
    app.handle("GET", "/", HOST)  # starts the job (synchronously here)
    body = app.handle("GET", "/", HOST).body.decode("utf-8")
    assert MARK in body and "چیز تازه" in body
    assert _read(tmp_path)["seen_head"] == repo.head()
    clock.now = T0 + uc.WELCOME_WINDOW
    assert MARK not in app.handle("GET", "/", HOST).body.decode("utf-8")
