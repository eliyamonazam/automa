"""The web UI's "new version is ready" bar (D178).

Real local git repositories with a bare repository as ``origin``; no network, no
model. The background thread is replaced by a synchronous starter.
"""

from __future__ import annotations

import json
import shutil
import threading
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from automa import update as upd
from automa import updatecheck as uc
from automa.config import DisplayConfig
from automa.tools.command import CommandResult
from automa.updatecheck import (
    UpdateNotice,
    UpdateWatcher,
    changelog_titles,
    friendly_title,
    pending_update,
)
from automa.webui import server as ws
from automa.webui.render import TopBar, page, top_bar
from test_update import Setup, _commit, _git

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

PORT = 8765
HOST = f"127.0.0.1:{PORT}"
T0 = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)

CHANGELOG_OLD = "# گزارش تغییرات\n\n## 2026-11-20 — M15: کار قدیمی (D150)\n- جزئیات\n"


def _changelog(*headings: str) -> str:
    new = "".join(f"## {h}\n- جزئیات\n\n" for h in headings)
    return CHANGELOG_OLD.replace("\n\n## 2026-11-20", "\n\n" + new + "## 2026-11-20", 1)


class Recorder:
    """Real git through the allow-list; records every command."""

    def __init__(self, fail_fetch: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail_fetch = fail_fetch

    def __call__(self, cmd: Sequence[str], cwd: Path, timeout: float) -> CommandResult:
        cmd = list(cmd)
        self.calls.append(cmd)
        if self.fail_fetch and cmd[1:2] == ["fetch"]:
            return CommandResult(128, "", "fatal: unable to access")
        return upd.default_runner(cmd, cwd, timeout)

    @property
    def fetches(self) -> int:
        return sum(1 for c in self.calls if c[1:2] == ["fetch"])


class Clock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _sync(job: Callable[[], None]) -> None:
    job()


@pytest.fixture
def repo(tmp_path: Path) -> Setup:
    s = Setup(tmp_path)
    s.publish("CHANGELOG.md", CHANGELOG_OLD, "M15: old work")
    _git(s.local, "pull", "-q", "--ff-only")
    return s


def _watcher(s: Setup, tmp: Path, runner: Recorder, clock: Clock) -> UpdateWatcher:
    return UpdateWatcher(
        s.local, tmp / "ws" / uc.STATE_FILENAME, runner=runner, clock=clock, start=_sync
    )


# --- titles -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("heading", "title"),
    [
        ("## 2026-11-22 — M16: بازطراحی ظاهر، بخش ۳ب (D177)", "بازطراحی ظاهر، بخش ۳ب"),
        ("## 2026-11-22 — M16: دستیار دقیق (D155–D158)", "دستیار دقیق"),
        ("## 2026-11-21 - M9: Web UI", "Web UI"),
        ("## بدون تاریخ", "بدون تاریخ"),
        ("## 2026-11-22 — M16: a‮b\x07c", "abc"),
    ],
)
def test_friendly_title(heading: str, title: str) -> None:
    assert friendly_title(heading) == title


def test_friendly_title_is_clipped() -> None:
    assert len(friendly_title("## " + "x" * 500)) == 160


def test_changelog_titles_reads_added_headings_only() -> None:
    diff = (
        "diff --git a/CHANGELOG.md b/CHANGELOG.md\n"
        "--- a/CHANGELOG.md\n+++ b/CHANGELOG.md\n"
        "@@ -2,0 +3,4 @@\n"
        "+## 2026-11-22 — M16: دوم (D2)\n+- detail\n"
        "+## 2026-11-21 — M16: اول (D1)\n"
        "-## 2026-11-20 — M15: removed\n"
        "+## 2026-11-21 — M16: اول (D1)\n"
        " ## context line\n"
    )
    assert changelog_titles(diff) == ["دوم", "اول"]


# --- pending_update (local refs only) -------------------------------------------


def test_nothing_pending_when_up_to_date(repo: Setup) -> None:
    assert pending_update(repo.local, Recorder()) is None


def test_pending_after_fetch_lists_new_changelog_titles(repo: Setup) -> None:
    repo.publish(
        "CHANGELOG.md",
        _changelog("2026-11-22 — M16: دومی (D2)", "2026-11-22 — M16: اولی (D1)"),
        "M16: two",
    )
    repo.publish("README.md", "more\n", "M16: readme")
    assert pending_update(repo.local, Recorder()) is None  # not fetched yet
    assert uc.fetch_remote(repo.local, Recorder())
    head = repo.head()
    notice = pending_update(repo.local, Recorder())
    assert notice == UpdateNotice(commits=2, titles=("دومی", "اولی"), more=0)
    assert repo.head() == head  # checking never moves the clone


def test_many_titles_are_capped(repo: Setup) -> None:
    heads = [f"2026-11-22 — M16: مورد {i} (D{i})" for i in range(uc.MAX_TITLES + 3)]
    repo.publish("CHANGELOG.md", _changelog(*heads), "M16: many")
    uc.fetch_remote(repo.local, Recorder())
    notice = pending_update(repo.local, Recorder())
    assert notice is not None
    assert len(notice.titles) == uc.MAX_TITLES and notice.more == 3
    assert notice.titles[0] == "مورد 0"


def test_pending_without_changelog_change_has_no_titles(repo: Setup) -> None:
    repo.publish("README.md", "x\n", "M16: tiny")
    uc.fetch_remote(repo.local, Recorder())
    assert pending_update(repo.local, Recorder()) == UpdateNotice(commits=1)


def test_no_banner_on_another_branch(repo: Setup) -> None:
    repo.publish("README.md", "x\n", "M16: tiny")
    uc.fetch_remote(repo.local, Recorder())
    _git(repo.local, "switch", "-q", "-c", "mine")
    assert pending_update(repo.local, Recorder()) is None


def test_no_banner_with_local_commits(repo: Setup) -> None:
    repo.publish("README.md", "x\n", "M16: tiny")
    uc.fetch_remote(repo.local, Recorder())
    _commit(repo.local, "mine.txt", "mine\n", "local work")
    assert pending_update(repo.local, Recorder()) is None


def test_not_a_repository_is_silent(tmp_path: Path) -> None:
    assert pending_update(tmp_path, Recorder()) is None
    assert uc.fetch_remote(tmp_path, Recorder()) is False


def test_runner_errors_are_silent(tmp_path: Path) -> None:
    def broken(cmd: Sequence[str], cwd: Path, timeout: float) -> CommandResult:
        raise ValueError("command not allowed")

    assert pending_update(tmp_path, broken) is None


def test_only_read_commands_and_one_fetch(repo: Setup, tmp_path: Path) -> None:
    repo.publish("CHANGELOG.md", _changelog("2026-11-22 — M16: تازه (D3)"), "M16: new")
    rec = Recorder()
    _watcher(repo, tmp_path, rec, Clock()).notice()
    subs = {c[1] for c in rec.calls}
    assert subs <= {"symbolic-ref", "rev-parse", "merge-base", "rev-list", "diff", "fetch"}
    assert rec.fetches == 1
    fetch = next(c for c in rec.calls if c[1] == "fetch")
    assert fetch == ["git", "fetch", "--quiet", "origin", "main"]


# --- the watcher ------------------------------------------------------------------


def test_watcher_fetches_at_most_once_a_day(repo: Setup, tmp_path: Path) -> None:
    rec, clock = Recorder(), Clock()
    w = _watcher(repo, tmp_path, rec, clock)
    assert w.notice() is None and rec.fetches == 1
    repo.publish("CHANGELOG.md", _changelog("2026-11-22 — M16: تازه (D3)"), "M16: new")
    clock.now = T0 + timedelta(hours=23)
    assert w.notice() is None  # local recheck only: origin/main not fetched again
    assert rec.fetches == 1
    clock.now = T0 + timedelta(hours=24)
    notice = w.notice()
    assert rec.fetches == 2
    assert notice is not None and notice.titles == ("تازه",)


def test_local_recheck_is_throttled(repo: Setup, tmp_path: Path) -> None:
    rec, clock = Recorder(), Clock()
    w = _watcher(repo, tmp_path, rec, clock)
    w.notice()
    n = len(rec.calls)
    clock.now = T0 + timedelta(minutes=5)
    w.notice()
    assert len(rec.calls) == n
    clock.now = T0 + uc.LOCAL_INTERVAL
    w.notice()
    assert len(rec.calls) > n and rec.fetches == 1


def test_banner_goes_away_after_update(repo: Setup, tmp_path: Path) -> None:
    repo.publish("README.md", "x\n", "M16: tiny")
    rec, clock = Recorder(), Clock()
    w = _watcher(repo, tmp_path, rec, clock)
    assert w.notice() == UpdateNotice(commits=1)
    _git(repo.local, "merge", "-q", "--ff-only", "origin/main")
    clock.now = T0 + uc.LOCAL_INTERVAL
    assert w.notice() is None


def test_last_fetch_survives_restart(repo: Setup, tmp_path: Path) -> None:
    rec, clock = Recorder(), Clock()
    _watcher(repo, tmp_path, rec, clock).notice()
    state = json.loads((tmp_path / "ws" / uc.STATE_FILENAME).read_text(encoding="utf-8"))
    assert datetime.fromisoformat(state["last_fetch"]) == T0
    clock.now = T0 + timedelta(hours=2)
    _watcher(repo, tmp_path, rec, clock).notice()
    assert rec.fetches == 1


def test_failed_fetch_is_silent_and_not_retried_soon(repo: Setup, tmp_path: Path) -> None:
    rec, clock = Recorder(fail_fetch=True), Clock()
    w = _watcher(repo, tmp_path, rec, clock)
    assert w.notice() is None
    clock.now = T0 + timedelta(hours=1)
    w.notice()
    assert rec.fetches == 1


def test_unwritable_state_still_fetches_once(repo: Setup, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory", encoding="utf-8")
    rec, clock = Recorder(), Clock()
    w = UpdateWatcher(repo.local, blocker / uc.STATE_FILENAME, runner=rec, clock=clock, start=_sync)
    w.notice()
    clock.now = T0 + uc.LOCAL_INTERVAL
    w.notice()
    assert rec.fetches == 1


@pytest.mark.parametrize("content", ["{", "[]", '{"last_fetch": "nope"}', '{"last_fetch": 5}'])
def test_corrupt_state_means_due(repo: Setup, tmp_path: Path, content: str) -> None:
    state = tmp_path / "ws" / uc.STATE_FILENAME
    state.parent.mkdir()
    state.write_text(content, encoding="utf-8")
    rec = Recorder()
    _watcher(repo, tmp_path, rec, Clock()).notice()
    assert rec.fetches == 1


def test_future_state_time_counts_as_due(repo: Setup, tmp_path: Path) -> None:
    state = tmp_path / "ws" / uc.STATE_FILENAME
    state.parent.mkdir()
    state.write_text(json.dumps({"last_fetch": (T0 + timedelta(days=30)).isoformat()}))
    rec = Recorder()
    _watcher(repo, tmp_path, rec, Clock()).notice()
    assert rec.fetches == 1


def test_notice_never_blocks_on_a_running_job(repo: Setup, tmp_path: Path) -> None:
    started: list[Callable[[], None]] = []
    w = UpdateWatcher(
        repo.local,
        tmp_path / uc.STATE_FILENAME,
        runner=Recorder(),
        clock=Clock(),
        start=started.append,
    )
    assert w.notice() is None
    assert w.notice() is None
    assert len(started) == 1  # one job at a time
    started[0]()
    assert len(started) == 1


def test_failed_thread_start_is_silent(repo: Setup, tmp_path: Path) -> None:
    def refuse(job: Callable[[], None]) -> None:
        raise RuntimeError("can't start new thread")

    w = UpdateWatcher(repo.local, tmp_path / "s.json", runner=Recorder(), start=refuse)
    assert w.notice() is None
    assert w.notice() is None


def test_watcher_for_a_non_clone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(uc, "installed_repo", lambda: None)
    assert uc.watcher_for(tmp_path) is None
    monkeypatch.setattr(uc, "installed_repo", lambda: tmp_path)
    w = uc.watcher_for(tmp_path / "ws")
    assert isinstance(w, UpdateWatcher)


def test_config_default_and_off() -> None:
    assert DisplayConfig().update_check is True
    assert DisplayConfig(update_check=False).update_check is False


# --- the page ---------------------------------------------------------------------

NOTICE = UpdateNotice(commits=3, titles=("بازطراحی ظاهر", "<b>x</b>"), more=2)


def test_bar_in_persian() -> None:
    with top_bar(TopBar(update=NOTICE)):
        doc = page("t", "<p>body</p>", lang="fa")
    assert 'class="updatebar"' in doc
    assert "نسخهٔ تازهٔ Automa آماده‌ست" in doc
    assert "چه چیزهایی تازه‌ست" in doc
    assert "بازطراحی ظاهر" in doc
    assert "<b>x</b>" not in doc and "&lt;b&gt;x&lt;/b&gt;" in doc
    assert "و ۲ مورد دیگه" in doc
    assert "به‌روزرسانی Automa" in doc
    assert '<code dir="ltr">automa update</code>' in doc
    assert doc.index("updatebar") < doc.index('<main class="wrap">')


def test_bar_in_english_without_titles() -> None:
    with top_bar(TopBar(update=UpdateNotice(commits=4))):
        doc = page("t", "<p>body</p>", lang="en")
    assert "A new version of Automa is ready" in doc
    assert "4 small improvements" in doc
    assert "Update Automa" in doc


def test_no_bar_without_notice_or_on_refreshing_pages() -> None:
    with top_bar(TopBar()):
        assert 'class="updatebar"' not in page("t", "", lang="fa")
    with top_bar(TopBar(update=NOTICE)):
        assert 'class="updatebar"' not in page("t", "", lang="fa", refresh=5)


class FakeWatcher:
    def __init__(self, notice: UpdateNotice | None = None, boom: bool = False) -> None:
        self._notice = notice
        self.boom = boom
        self.calls = 0

    def notice(self) -> UpdateNotice | None:
        self.calls += 1
        if self.boom:
            raise RuntimeError("boom")
        return self._notice


def _app(root: Path, watcher: FakeWatcher | None) -> ws.WebApp:
    return ws.WebApp(
        root,
        root / "ledger.sqlite3",
        ws.allowed_hosts("127.0.0.1", PORT),
        language="fa",
        updates=watcher,  # type: ignore[arg-type]
    )


def test_server_shows_bar_on_every_page(tmp_path: Path) -> None:
    watcher = FakeWatcher(NOTICE)
    app = _app(tmp_path, watcher)
    for target in ("/", "/nowhere"):
        body = app.handle("GET", target, HOST).body.decode("utf-8")
        assert 'class="updatebar"' in body
    assert watcher.calls == 2


def test_server_without_watcher_or_with_broken_watcher(tmp_path: Path) -> None:
    for watcher in (None, FakeWatcher(), FakeWatcher(NOTICE, boom=True)):
        r = _app(tmp_path, watcher).handle("GET", "/", HOST)
        assert r.status == 200
        assert 'class="updatebar"' not in r.body.decode("utf-8")


def test_static_assets_do_not_trigger_a_check(tmp_path: Path) -> None:
    watcher = FakeWatcher(NOTICE)
    app = _app(tmp_path, watcher)
    app.handle("GET", "/static/vazirmatn.woff2", HOST)
    assert watcher.calls == 0


def test_create_server_passes_the_watcher(tmp_path: Path) -> None:
    watcher = FakeWatcher(NOTICE)
    server = ws.create_server(
        tmp_path,
        tmp_path / "ledger.sqlite3",
        port=0,
        language="fa",
        updates=watcher,  # type: ignore[arg-type]
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(ws.server_url(server), timeout=10) as r:  # noqa: S310
            body = r.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
    assert 'class="updatebar"' in body
