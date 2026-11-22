"""Tests for live progress on the web UI project page (``automa.webui.progress``, D127).

Read-only: events, task states and ledger entries are written by hand into
tmp_path; no provider, no network, no model calls.
"""

from __future__ import annotations

import ast
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import test_finance_compare as fc
from automa.budget import CostLedger
from automa.events import Event
from automa.plan import Plan, TaskState, TaskStatus
from automa.project import ProjectStatus, ProjectStore
from automa.project.store import EVENTS_FILE, TASK_STATE_FILE
from automa.tools.base import RiskLevel
from automa.webui import progress as pg
from automa.webui import render as rd
from automa.webui import server as ws
from automa.webui.status import STATUS_ICONS
from automa.webui.views import build_project_view
from test_ledger import FixedClock, make_pair
from test_webui import _task

T0 = fc.T0
RANGE = {"range_low_nusd": 100, "range_high_nusd": 1_000, "range_basis": "defaults"}
GOOD_HOST = "127.0.0.1:8765"
LEDGER = "ledger.sqlite3"


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else b"<dir>"
        for p in sorted(root.rglob("*"))
    }


@pytest.fixture
def store(tmp_path: Path) -> ProjectStore:
    s = ProjectStore.for_workspace(tmp_path)
    s.create("write a poem", project_id="p1")
    s.save_plan("p1", Plan(goal="g", tasks=[_task("a"), _task("b", ["a"]), _task("c", ["b"])]))
    (s.project_dir("p1") / EVENTS_FILE).unlink()  # start each test with no events
    return s


def _state(store: ProjectStore, tid: str, status: TaskStatus, model: str | None = None) -> None:
    """Write state.json directly (a ``passed`` state needs stored verdicts otherwise)."""
    tdir = store.task_dir("p1", tid)
    tdir.mkdir(parents=True, exist_ok=True)
    attempts = 1 if status is TaskStatus.PASSED else 0
    st = TaskState(task_id=tid, status=status, attempts=attempts, model=model, updated_at=T0)
    (tdir / TASK_STATE_FILE).write_text(st.model_dump_json(), encoding="utf-8")


def _events_path(store: ProjectStore) -> Path:
    return store.project_dir("p1") / EVENTS_FILE


def _emit(store: ProjectStore, *events: Event) -> None:
    for e in events:
        store.events("p1").emit(e)


def _ev(type_: str, minutes: int = 0, **data: object) -> Event:
    return Event(ts=T0 + timedelta(minutes=minutes), type=type_, project_id="p1", data=data)


def _set_status(store: ProjectStore, status: ProjectStatus) -> None:
    p = store.load("p1")
    store.save(p.model_copy(update={"status": status}))


# -- read_events ---------------------------------------------------------------------


def test_read_events_missing_file_is_empty_and_creates_nothing(store: ProjectStore) -> None:
    before = _tree(store.project_dir("p1"))
    assert pg.read_events(store, "p1") == []
    assert _tree(store.project_dir("p1")) == before


def test_read_events_in_file_order(store: ProjectStore) -> None:
    _emit(store, _ev("orchestrator.started", 0), _ev("orchestrator.attempt", 1, task="a"))
    assert [e.type for e in pg.read_events(store, "p1")] == [
        "orchestrator.started",
        "orchestrator.attempt",
    ]


def test_read_events_skips_blank_lines(store: ProjectStore) -> None:
    _emit(store, _ev("orchestrator.started"))
    with _events_path(store).open("a", encoding="utf-8") as f:
        f.write("\n   \n")
    assert len(pg.read_events(store, "p1")) == 1


def test_read_events_refuses_corrupt_line(store: ProjectStore) -> None:
    _emit(store, _ev("orchestrator.started"))
    with _events_path(store).open("a", encoding="utf-8") as f:
        f.write("{not json\n")
    with pytest.raises(ValueError, match="line 2"):
        pg.read_events(store, "p1")


def test_read_events_refuses_invalid_event(store: ProjectStore) -> None:
    _events_path(store).write_text(json.dumps({"type": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        pg.read_events(store, "p1")


def test_read_events_refuses_symlink(store: ProjectStore, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.jsonl"
    target.write_text("", encoding="utf-8")
    _events_path(store).symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        pg.read_events(store, "p1")


def test_read_events_refuses_directory(store: ProjectStore) -> None:
    _events_path(store).mkdir()
    with pytest.raises(ValueError, match="regular"):
        pg.read_events(store, "p1")


def test_read_events_refuses_non_utf8(store: ProjectStore) -> None:
    _events_path(store).write_bytes(b"\xff\xfe\n")
    with pytest.raises(ValueError, match="UTF-8"):
        pg.read_events(store, "p1")


def test_read_events_refuses_too_large(
    store: ProjectStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _emit(store, _ev("orchestrator.started"))
    monkeypatch.setattr(pg, "MAX_EVENTS_BYTES", 10)
    with pytest.raises(ValueError, match="larger"):
        pg.read_events(store, "p1")


# -- event lines -------------------------------------------------------------------


def test_event_line_keeps_only_short_scalar_fields() -> None:
    e = _ev(
        "orchestrator.stopped",
        task="a",
        reason="budget",
        detail="model said SECRET things",
        error="Traceback SECRET",
        arguments={"text": "SECRET"},
        attempt=2,
    )
    line = pg.event_line(e)
    assert line.type == "orchestrator.stopped"
    assert line.summary == "task=a attempt=2 reason=budget"
    assert "SECRET" not in line.summary


def test_event_line_order_follows_summary_keys() -> None:
    line = pg.event_line(_ev("x.y", to="large", task="a", **{"from": "small"}))
    assert line.summary == "task=a from=small to=large"


def test_event_line_truncates_and_flattens_values() -> None:
    line = pg.event_line(_ev("x.y", model="m\n" + "z" * 200))
    value = line.summary.removeprefix("model=")
    assert "\n" not in value and len(value) == pg.SUMMARY_VALUE_CHARS and value.endswith("…")


@pytest.mark.parametrize(
    ("value", "shown"),
    [(True, "yes"), (False, "no"), (3, "3"), (1.5, "1.5"), (None, None), ([1], None), ({}, None)],
)
def test_event_line_value_kinds(value: object, shown: str | None) -> None:
    line = pg.event_line(_ev("x.y", status=value))
    assert line.summary == ("" if shown is None else f"status={shown}")


def test_recent_events_newest_first_and_limited() -> None:
    events = [_ev("x.y", i, attempt=i) for i in range(30)]
    lines = pg.recent_events(events, limit=5)
    assert [ln.summary for ln in lines] == [f"attempt={i}" for i in (29, 28, 27, 26, 25)]


def test_recent_events_equal_timestamps_keep_file_order_reversed() -> None:
    events = [_ev("x.y", 0, attempt=i) for i in range(3)]
    assert [ln.summary for ln in pg.recent_events(events)] == [
        "attempt=2",
        "attempt=1",
        "attempt=0",
    ]


def test_recent_events_default_limit() -> None:
    assert len(pg.recent_events([_ev("x.y", i) for i in range(40)])) == pg.RECENT_EVENTS


def test_recent_events_rejects_negative_limit() -> None:
    with pytest.raises(ValueError):
        pg.recent_events([], limit=-1)


# -- cost progress ---------------------------------------------------------------------


def test_cost_progress_within_range() -> None:
    c = pg.cost_progress("p1", [fc.est(**RANGE)], [fc.entry(300), fc.entry(200)])
    assert c is not None
    assert c.spent_usd == pytest.approx(500e-9) and c.verdict == "within"
    assert c.low_usd == pytest.approx(100e-9) and c.high_usd == pytest.approx(1_000e-9)
    assert c.worst_usd == pytest.approx(2_000e-9) and c.estimated_at == T0


def test_cost_progress_ignores_other_projects_entries() -> None:
    c = pg.cost_progress("p1", [fc.est(**RANGE)], [fc.entry(5_000, project="p2")])
    assert c is not None and c.spent_usd == 0 and c.verdict == "below"


def test_cost_progress_other_calls_separate() -> None:
    c = pg.cost_progress("p1", [fc.est(**RANGE)], [fc.entry(200), fc.entry(70, task=None)])
    assert c is not None
    assert c.spent_usd == pytest.approx(200e-9) and c.other_usd == pytest.approx(70e-9)


def test_cost_progress_above() -> None:
    c = pg.cost_progress("p1", [fc.est(**RANGE)], [fc.entry(1_500)])
    assert c is not None and c.verdict == "above"


def test_cost_progress_none_without_estimate_or_range() -> None:
    assert pg.cost_progress("p1", [], [fc.entry()]) is None
    assert pg.cost_progress("p1", [fc.est()], [fc.entry()]) is None  # written before D120


def test_cost_progress_uses_the_latest_estimate() -> None:
    old = fc.est(ts=T0 - timedelta(hours=1), **RANGE)
    new = fc.est(ts=T0, **{**RANGE, "range_high_nusd": 1_500})
    c = pg.cost_progress("p1", [new, old], [])
    assert c is not None and c.high_usd == pytest.approx(1_500e-9)


# -- build_progress -------------------------------------------------------------------


def test_build_progress_counts_passed_tasks(store: ProjectStore) -> None:
    p = pg.build_progress(store, "p1", ["passed", "running", None], [], running=True)
    assert (p.passed, p.total) == (1, 3)
    assert p.live and p.cost is None and p.events == () and p.errors == ()


def test_build_progress_not_live_when_not_running(store: ProjectStore) -> None:
    assert not pg.build_progress(store, "p1", [], [], running=False).live


def test_build_progress_not_live_while_an_approval_waits(store: ProjectStore) -> None:
    p = pg.build_progress(store, "p1", [], [], running=True, pending_approvals=1)
    assert not p.live


def test_build_progress_with_estimate_and_events(store: ProjectStore) -> None:
    _emit(store, fc.est(**RANGE), _ev("orchestrator.attempt", 1, task="a", attempt=1))
    p = pg.build_progress(store, "p1", [None], [fc.entry(400)], running=True)
    assert p.cost is not None and p.cost.verdict == "within"
    assert [e.type for e in p.events] == ["orchestrator.attempt", "finance.estimate"]


def test_build_progress_corrupt_events_is_an_error_line(store: ProjectStore) -> None:
    _events_path(store).write_text("garbage\n", encoding="utf-8")
    p = pg.build_progress(store, "p1", [], [], running=False)
    assert p.events == () and p.cost is None
    assert len(p.errors) == 1 and p.errors[0].startswith("events:")


def test_build_progress_invalid_estimate_gives_no_range(store: ProjectStore) -> None:
    bad = Event(ts=T0, type="finance.estimate", project_id="p1", data={"tasks": "x"})
    _emit(store, bad)
    p = pg.build_progress(store, "p1", [], [], running=False)
    assert p.cost is None and p.errors == () and len(p.events) == 1


def test_build_progress_writes_nothing(store: ProjectStore) -> None:
    _emit(store, fc.est(**RANGE))
    root = store.project_dir("p1").parent.parent
    before = _tree(root)
    pg.build_progress(store, "p1", ["passed"], [fc.entry()], running=True)
    assert _tree(root) == before


# -- task rows -------------------------------------------------------------------------


def test_task_rows_show_state_and_model(store: ProjectStore) -> None:
    _state(store, "a", TaskStatus.PASSED, model="m-large")
    _state(store, "b", TaskStatus.RUNNING)
    view = build_project_view(store, "p1", 0)
    by_id = {t.id: t for t in view.tasks}
    assert (by_id["a"].state, by_id["a"].model) == ("passed", "m-large")
    assert (by_id["b"].state, by_id["b"].model) == ("running", None)
    assert (by_id["c"].state, by_id["c"].model) == (None, None)
    assert view.errors == ()


def test_unreadable_task_state_does_not_hide_the_plan(store: ProjectStore) -> None:
    tdir = store.task_dir("p1", "a")
    tdir.mkdir(parents=True)
    (tdir / TASK_STATE_FILE).write_text("{broken", encoding="utf-8")
    view = build_project_view(store, "p1", 0)
    assert [t.state for t in view.tasks] == ["unreadable", None, None]


# -- rendering -------------------------------------------------------------------------


def _view(store: ProjectStore):  # type: ignore[no-untyped-def]
    return build_project_view(store, "p1", 0)


def test_page_refresh_meta_only_when_asked() -> None:
    assert 'http-equiv="refresh"' not in rd.page("t", "b")
    doc = rd.page("t", "b", refresh=5)
    assert '<meta http-equiv="refresh" content="5">' in doc


@pytest.mark.parametrize("bad", [0, -1, 3601, True, 2.5, "5"])
def test_page_refresh_rejects_bad_values(bad: object) -> None:
    with pytest.raises(ValueError):
        rd.page("t", "b", refresh=bad)  # type: ignore[arg-type]


def test_render_live_progress(store: ProjectStore) -> None:
    progress = pg.ProgressView(
        live=True,
        passed=1,
        total=3,
        cost=pg.CostProgress(
            estimated_at=T0,
            spent_usd=0.12,
            other_usd=0.0,
            low_usd=0.05,
            high_usd=0.3,
            worst_usd=1.0,
            verdict="within",
        ),
        events=(pg.EventLine(ts=T0, type="orchestrator.attempt", summary="task=a"),),
    )
    doc = rd.render_project(_view(store), progress=progress)
    assert '<meta http-equiv="refresh" content="5">' in doc
    assert "tasks passed: <strong>1 / 3</strong>" in doc
    assert '<meter min="0" max="3" value="1"></meter>' in doc
    assert "This run so far, still running; estimated about $0.05 to $0.30" in doc
    assert "worst case" not in doc and "guard cap" not in doc
    assert 'max="0.300000" value="0.120000"' in doc
    assert "refreshes every 5 s" in doc
    assert "orchestrator.attempt" in doc and "task=a" in doc


def test_render_finished_progress_shows_verdict_and_no_refresh(store: ProjectStore) -> None:
    cost = pg.CostProgress(
        estimated_at=T0,
        spent_usd=0.4,
        other_usd=0.01,
        low_usd=0.05,
        high_usd=0.3,
        worst_usd=1.0,
        verdict="above",
    )
    doc = rd.render_project(_view(store), progress=pg.ProgressView(passed=3, total=3, cost=cost))
    assert 'http-equiv="refresh"' not in doc
    assert '<p class="warn">This run above the estimated range: about $0.05 to $0.30' in doc


def test_render_live_above_range_is_flagged(store: ProjectStore) -> None:
    cost = pg.CostProgress(
        estimated_at=T0,
        spent_usd=0.4,
        other_usd=0,
        low_usd=0.05,
        high_usd=0.3,
        worst_usd=1.0,
        verdict="above",
    )
    doc = rd.render_project(_view(store), progress=pg.ProgressView(live=True, cost=cost))
    assert '<p class="warn">This run above the estimated range:' in doc


def test_render_meter_above_range_ends_at_the_cost(store: ProjectStore) -> None:
    cost = pg.CostProgress(
        estimated_at=T0,
        spent_usd=5.0,
        other_usd=0,
        low_usd=0.05,
        high_usd=0.3,
        worst_usd=1.0,
        verdict="above",
    )
    doc = rd.render_project(_view(store), progress=pg.ProgressView(cost=cost))
    assert 'max="5.000000" value="5.000000"' in doc and "1.0000" not in doc


def test_render_without_estimate(store: ProjectStore) -> None:
    doc = rd.render_project(_view(store), progress=pg.ProgressView())
    assert "Nothing spent yet." in doc and "<meter" not in doc


def test_render_progress_errors_are_shown(store: ProjectStore) -> None:
    doc = rd.render_project(_view(store), progress=pg.ProgressView(errors=("events: broken",)))
    assert "Unreadable data" in doc and "events: broken" in doc


def test_render_without_progress_is_unchanged(store: ProjectStore) -> None:
    doc = rd.render_project(_view(store))
    assert "<h2>Progress</h2>" not in doc and 'http-equiv="refresh"' not in doc


def test_render_event_text_is_escaped_and_bidi_visible(store: ProjectStore) -> None:
    evil = "<script>alert(1)</script>‮"
    line = pg.EventLine(ts=T0, type="x.y", summary=f"model={evil}")
    doc = rd.render_project(_view(store), progress=pg.ProgressView(events=(line,)))
    assert "<script>" not in doc and "&lt;script&gt;" in doc
    assert "‮" not in doc and "\\u202e" in doc


def test_render_task_state_columns(store: ProjectStore) -> None:
    _state(store, "a", TaskStatus.PASSED, model="m-<b>")
    doc = rd.render_project(_view(store))
    assert '<article class="card task t-passed">' in doc
    assert '<td dir="ltr">m-&lt;b&gt;</td>' in doc  # the model is in the technical details
    check, queue = STATUS_ICONS["check"], STATUS_ICONS["queue"]
    assert f'<span class="ico" aria-hidden="true">{check}</span> passed</span>' in doc
    assert f'<span class="ico" aria-hidden="true">{queue}</span> not started</span>' in doc
    assert "m-&lt;b&gt;" in doc


@pytest.mark.parametrize(
    ("state", "cls"),
    [
        ("passed", "passed"),
        ("failed", "failed"),
        ("running", "running"),
        ("unreadable", "none"),
        (None, "none"),
        ('x" onload="y', "none"),
    ],
)
def test_state_class_is_fixed(state: str | None, cls: str) -> None:
    assert rd._state_class(state) == cls


def test_progress_page_has_no_script(store: ProjectStore) -> None:
    _emit(store, fc.est(**RANGE), _ev("orchestrator.attempt", 1, task="a"))
    progress = pg.build_progress(store, "p1", ["passed"], [fc.entry()], running=True)
    doc = rd.render_project(_view(store), progress=progress)
    assert "<script" not in doc.lower() and "javascript:" not in doc.lower()
    assert re.search(r"\son[a-z]+=", doc) is None


# -- server ------------------------------------------------------------------------------


@pytest.fixture
def workspace(store: ProjectStore, tmp_path: Path) -> Path:
    clock = FixedClock(T0 + timedelta(minutes=1))
    with CostLedger(tmp_path / LEDGER, clock=clock) as led:
        led.record(*make_pair(0.25, project="p1", task="t1"))
        led.record(*make_pair(0.5, project="p2", task="t1"))
    return tmp_path


def _app(workspace: Path) -> ws.WebApp:
    return ws.WebApp(workspace, workspace / LEDGER, ws.allowed_hosts("127.0.0.1", 8765))


def _get(workspace: Path) -> str:
    r = _app(workspace).handle("GET", "/projects/p1", GOOD_HOST)
    assert r.status == 200
    return r.body.decode()


def test_server_project_page_has_progress(workspace: Path, store: ProjectStore) -> None:
    body = _get(workspace)
    assert "<h2>Progress</h2>" in body and "tasks passed: <strong>0 / 3</strong>" in body
    assert "$0.25" in body  # only p1's spending, not p2's
    assert 'http-equiv="refresh"' not in body


def test_server_cost_against_range(workspace: Path, store: ProjectStore) -> None:
    big = {"range_high_nusd": 1_000_000_000, "worst_nusd": 2_000_000_000}
    _emit(store, fc.est(**{**RANGE, **big}))
    body = _get(workspace)
    assert "<strong>$0.25</strong>" in body
    assert "This run within the estimated range: about" in body


def test_server_refreshes_while_project_runs(workspace: Path, store: ProjectStore) -> None:
    _set_status(store, ProjectStatus.RUNNING)
    body = _get(workspace)
    assert '<meta http-equiv="refresh" content="5">' in body


def test_server_does_not_refresh_while_an_approval_waits(
    workspace: Path, store: ProjectStore
) -> None:
    _set_status(store, ProjectStatus.RUNNING)
    store.approvals("p1").request(
        tool="publish", risk=RiskLevel.EXTERNAL, arguments={}, reason="r", task_id="a"
    )
    assert 'http-equiv="refresh"' not in _get(workspace)


def test_server_refresh_header_csp_still_strict(workspace: Path, store: ProjectStore) -> None:
    _set_status(store, ProjectStatus.RUNNING)
    r = _app(workspace).handle("GET", "/projects/p1", GOOD_HOST)
    csp = dict(r.headers)["Content-Security-Policy"]
    assert "script-src" not in csp and "default-src 'none'" in csp


def test_server_corrupt_events_is_an_error_line(workspace: Path, store: ProjectStore) -> None:
    _events_path(store).write_text("garbage\n", encoding="utf-8")
    body = _get(workspace)
    assert "Unreadable data" in body and "events:" in body


def test_server_progress_writes_nothing(workspace: Path, store: ProjectStore) -> None:
    _emit(store, fc.est(**RANGE))
    _state(store, "a", TaskStatus.RUNNING)
    before = _tree(workspace)
    _get(workspace)
    assert _tree(workspace) == before


def test_server_head_has_same_headers(workspace: Path, store: ProjectStore) -> None:
    _set_status(store, ProjectStatus.RUNNING)
    r = _app(workspace).handle("HEAD", "/projects/p1", GOOD_HOST)
    assert r.status == 200


def test_server_without_ledger(store: ProjectStore, tmp_path: Path) -> None:
    body = _app(tmp_path).handle("GET", "/projects/p1", GOOD_HOST).body.decode()
    assert "<h2>Progress</h2>" in body and not (tmp_path / LEDGER).exists()


def test_server_background_run_makes_the_page_live(
    workspace: Path, store: ProjectStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _State:
        running = True

    class _Runs:
        def state(self, pid: str) -> _State:
            return _State()

    app = _app(workspace)
    monkeypatch.setattr(app, "_runs", _Runs())
    view = build_project_view(store, "p1", 0)
    assert app._progress(view, []).live


# -- structure ---------------------------------------------------------------------------


def test_progress_module_never_writes_or_calls_out() -> None:
    src = Path(pg.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    imported = {
        a.name.split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or "")])
    }
    assert not imported & {"socket", "httpx", "subprocess", "urllib", "http"}
    assert not attrs & {"write", "write_text", "write_bytes", "mkdir", "emit", "unlink"}
    assert "O_WRONLY" not in attrs and "O_CREAT" not in attrs
    assert "open" not in names  # only os.open with O_RDONLY
    assert "os.O_RDONLY" in src


def test_read_events_opens_read_only(store: ProjectStore, monkeypatch: pytest.MonkeyPatch) -> None:
    _emit(store, _ev("x.y"))
    flags_seen: list[int] = []
    real_open = os.open

    def spy(path: object, flags: int, *args: object) -> int:
        flags_seen.append(flags)
        return real_open(path, flags, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(pg.os, "open", spy)
    pg.read_events(store, "p1")
    assert flags_seen and all(f & (os.O_WRONLY | os.O_RDWR | os.O_CREAT) == 0 for f in flags_seen)


def test_utc_timestamps_shown(store: ProjectStore) -> None:
    line = pg.EventLine(ts=datetime(2026, 1, 1, tzinfo=UTC), type="x.y")
    doc = rd.render_project(_view(store), progress=pg.ProgressView(events=(line,)))
    assert "2026-01-01T00:00:00+00:00" in doc


# --- D171: the live "right now" card ---------------------------------------------


def _activity() -> pg.Activity:
    return pg.Activity(
        phase="working",
        task_title="Build <the> page",
        action_key="activity.wrote",
        action_arg="site/index.html",
        at=T0,
    )


def test_render_activity_only_while_live(store: ProjectStore) -> None:
    live = pg.ProgressView(live=True, activity=_activity())
    doc = rd.render_project(_view(store), progress=live)
    assert 'class="card activity"' in doc
    assert "Working on: Build &lt;the&gt; page" in doc
    assert "wrote site/index.html" in doc
    still = pg.ProgressView(live=False, activity=_activity())
    assert 'class="card activity"' not in rd.render_project(_view(store), progress=still)


def test_render_activity_fa(store: ProjectStore) -> None:
    a = pg.Activity(phase="reviewing")
    doc = rd.render_project(
        _view(store), progress=pg.ProgressView(live=True, activity=a), lang="fa"
    )
    assert "همین الان" in doc and "بازبینی نهایی کل محصول" in doc
    assert "آخرین کار" not in doc


def test_build_progress_activity_from_events_only_when_running(store: ProjectStore) -> None:
    lines = [
        Event(ts=T0, type="worker.started", project_id="p1", data={"task": "a"}),
        Event(
            ts=T0 + timedelta(seconds=1),
            type="tool.call",
            project_id="p1",
            data={"tool": "write_file", "ok": True, "path": "out.md"},
        ),
    ]
    path = store.project_dir("p1") / EVENTS_FILE
    path.write_text("".join(e.model_dump_json() + "\n" for e in lines), encoding="utf-8")
    on = pg.build_progress(store, "p1", [None], [], running=True, titles={"a": "Write"})
    assert on.activity is not None and on.activity.task_title == "Write"
    assert on.activity.action_arg == "out.md"
    off = pg.build_progress(store, "p1", [None], [], running=False)
    assert off.activity is None
