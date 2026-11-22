"""Tests for the task brake's "continue?" in the web UI (D154).

Unit tests use hand-written events and fake child processes (``test_webui_runs``);
the end-to-end test runs the real ``automa run`` stack with a scripted
``FakeProvider`` (no network, no real model calls): the brake stops a task, the web
page asks, "Continue" records ``brake.continued`` and resumes the same command.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from automa import cli
from automa.budget.brake import (
    BRAKE_CONTINUED,
    BRAKE_DECLINED,
    BRAKE_STOP,
    RUN_KEY,
    BrakeError,
    continue_task,
    decline_task,
    declined_brake,
    pending_brake,
)
from automa.budget.ledger import usd_to_nusd
from automa.events import Event, MemoryEventSink
from automa.project import ProjectStatus, ProjectStore
from automa.webui import runs as wr
from automa.webui import server as ws
from automa.webui import stops as sp
from test_e2e import E2E, PID, build_attempt, design_attempt, docs_attempt, judgment
from test_task_brake import _brake_config, costly, small_profiles
from test_webui_runs import HOST, LEDGER, PORT, TOKEN, Env, FakePreflight, FakeProcess, _settings

T0 = datetime(2026, 11, 20, 20, 0, tzinfo=UTC)
SPENT = usd_to_nusd(0.8)
LIMIT = usd_to_nusd(0.5)


def _ev(type_: str, minutes: int, **data: object) -> Event:
    return Event(ts=T0 + timedelta(minutes=minutes), type=type_, project_id="p1", data=data)


def _brake_events(task: str = "build") -> list[Event]:
    return [
        _ev("orchestrator.started", 0, plan_version=1),
        _ev(BRAKE_STOP, 1, task=task, spent_nusd=SPENT, limit_nusd=LIMIT, step_nusd=LIMIT),
        _ev("orchestrator.paused", 2, reason="task_brake", task=task),
    ]


# -- core: decline ------------------------------------------------------------------


def test_decline_needs_a_pending_stop() -> None:
    sink = MemoryEventSink()
    with pytest.raises(BrakeError):
        decline_task([], sink, "p1", "build", via="web")
    assert sink.events == []


def test_decline_allows_nothing_and_keeps_the_stop_pending() -> None:
    events = _brake_events()
    sink = MemoryEventSink()
    event = decline_task(events, sink, "p1", "build", via="web", now=T0)
    assert event.type == BRAKE_DECLINED and event.data == {
        "task": "build",
        "via": "web",
        "spent_nusd": SPENT,
    }
    events.append(event)
    assert pending_brake(events, "build") is not None  # a later continue is possible
    assert declined_brake(events, "build")
    # A continue after the decline is still accepted and ends the decline.
    events.append(continue_task(events, sink, "p1", "build", via="web"))
    assert pending_brake(events, "build") is None and not declined_brake(events, "build")
    with pytest.raises(BrakeError):
        decline_task(events, sink, "p1", "build", via="web")


def test_a_new_stop_after_a_decline_is_not_declined() -> None:
    events = [*_brake_events(), _ev(BRAKE_DECLINED, 3, task="build")]
    events.append(_ev(BRAKE_CONTINUED, 4, task="build"))
    events.append(_ev(BRAKE_STOP, 5, task="build", spent_nusd=SPENT * 2, limit_nusd=LIMIT * 2))
    assert pending_brake(events, "build") is not None and not declined_brake(events, "build")


def test_other_tasks_decline_is_ignored() -> None:
    events = [*_brake_events(), _ev(BRAKE_DECLINED, 3, task="other")]
    assert not declined_brake(events, "build")


# -- latest_stop / describe_stop -----------------------------------------------------


def test_latest_stop_carries_the_brake_numbers() -> None:
    stop = sp.latest_stop(_brake_events())
    assert stop is not None and stop.kind == "task_brake"
    assert (stop.task, stop.spent_nusd, stop.limit_nusd) == ("build", SPENT, LIMIT)
    assert stop.awaiting and not stop.declined
    line = sp.describe_stop(stop, lang="fa", tz=UTC, now=T0)
    assert "ترمز خرج" in line.text and line.resume is False


def test_continued_brake_can_resume_with_a_plain_start() -> None:
    events = [*_brake_events(), _ev(BRAKE_CONTINUED, 3, task="build")]
    stop = sp.latest_stop(events)
    assert stop is not None and not stop.awaiting
    line = sp.describe_stop(stop, lang="fa", tz=UTC, now=T0)
    assert line.resume is True and "اجازه داده شده" in line.text


def test_declined_brake_sentence() -> None:
    events = [*_brake_events(), _ev(BRAKE_DECLINED, 3, task="build")]
    stop = sp.latest_stop(events)
    assert stop is not None and stop.declined and stop.awaiting
    line = sp.describe_stop(stop, lang="fa", tz=UTC, now=T0)
    assert "به انتخاب تو" in line.text and line.resume is False


@pytest.mark.parametrize("spent", [None, -1, "0.8", True])
def test_bad_recorded_numbers_are_dropped(spent: object) -> None:
    events = _brake_events()
    events[1] = _ev(BRAKE_STOP, 1, task="build", spent_nusd=spent, limit_nusd=LIMIT)
    stop = sp.latest_stop(events)
    assert stop is not None and stop.spent_nusd is None and stop.limit_nusd == LIMIT


# -- web: page and form --------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    e = Env(tmp_path)
    yield e
    for _, _, proc in e.spawned:
        if proc.code is None:
            proc.finish(0)


def _braked(env: Env, task: str = "build") -> str:
    pid = env.project(status=ProjectStatus.PAUSED)
    sink = env.store.events(pid)
    for e in _brake_events(task):
        sink.emit(e.model_copy(update={"project_id": pid}))
    return pid


def _fa_app(env: Env) -> ws.WebApp:
    return ws.WebApp(
        env.root,
        env.root / LEDGER,
        ws.allowed_hosts("127.0.0.1", PORT),
        actions=_settings(),
        runs=env.background,
        language="fa",
        toman_per_usd=323_000.0,
    )


def _page(app: ws.WebApp, pid: str) -> str:
    r = app.handle("GET", f"/projects/{pid}", HOST)
    assert r.status == 200
    return r.body.decode("utf-8")


def _brake(env: Env, pid: str, **fields: str) -> ws.Response:
    return env.post(f"/projects/{pid}/brake", {"csrf": TOKEN, "task": "build", **fields})


def _types(env: Env, pid: str) -> list[object]:
    return [e["type"] for e in env.events(pid)]


def test_page_asks_in_persian_with_toman_and_two_buttons(env: Env) -> None:
    pid = _braked(env)
    body = _page(_fa_app(env), pid)
    assert "ادامه بدم؟" in body and "خیلی بیشتر از پیش‌بینی" in body
    assert "تومان" in body  # 0.8 USD at 323,000 toman
    assert "ادامه (یک بار دیگه همین‌قدر)" in body and ">توقف</button>" in body
    assert f'action="/projects/{pid}/brake"' in body
    assert body.count('name="decision"') == 2
    assert f'name="csrf" value="{TOKEN}"' in body


def test_page_uses_the_task_title(env: Env) -> None:
    pid = _braked(env)
    assert "<bdi>Build</bdi>" in _page(_fa_app(env), pid)


def test_question_carries_the_cost_box_when_needed(env: Env) -> None:
    env.preflight.confirm = True
    pid = _braked(env)
    body = _page(_fa_app(env), pid)
    card = body[body.index('class="card brake"') :]
    assert 'name="confirm"' in card.split("</form>")[0]


def test_read_only_server_shows_no_question(tmp_path: Path) -> None:
    env = Env(tmp_path, actions=False, runs=False)
    pid = _braked(env)
    body = env.get(f"/projects/{pid}").body.decode("utf-8")
    assert "/brake" not in body and "automa brake continue" in body


def test_no_question_after_continue(env: Env) -> None:
    pid = _braked(env)
    env.store.events(pid).emit(
        Event(ts=T0, type=BRAKE_CONTINUED, project_id=pid, data={"task": "build"})
    )
    body = _page(_fa_app(env), pid)
    assert "/brake" not in body and f'action="/projects/{pid}/start"' in body


def test_post_continue_records_and_starts(env: Env) -> None:
    pid = _braked(env)
    r = _brake(env, pid, decision="continue")
    assert r.status == 303, r.body
    assert len(env.spawned) == 1 and env.preflight.commands == [(pid, False)]
    continued = [e for e in env.events(pid) if e["type"] == BRAKE_CONTINUED]
    assert len(continued) == 1 and continued[0]["data"]["via"] == "web"  # type: ignore[index]
    assert continued[0]["data"]["steps"] == 2  # type: ignore[index]
    # The second press (after the run ended) has nothing to continue.
    env.spawned[0][2].finish(7)
    env.wait_event(pid, wr.FINISHED_EVENT)
    r = _brake(env, pid, decision="continue")
    assert r.status == 409 and len(env.spawned) == 1


def test_continue_needs_the_cost_box_and_records_nothing_without_it(env: Env) -> None:
    env.preflight.confirm = True
    pid = _braked(env)
    r = _brake(env, pid, decision="continue")
    assert r.status == 409 and env.spawned == []
    assert BRAKE_CONTINUED not in _types(env, pid)
    r = _brake(env, pid, decision="continue", confirm="yes")
    assert r.status == 303 and env.preflight.commands == [(pid, True)]


def test_post_stop_records_a_decline_and_starts_nothing(env: Env) -> None:
    pid = _braked(env)
    r = _brake(env, pid, decision="stop")
    assert r.status == 303 and env.spawned == []
    types = _types(env, pid)
    assert BRAKE_DECLINED in types and BRAKE_CONTINUED not in types
    assert env.store.load(pid).status is ProjectStatus.PAUSED
    body = _page(_fa_app(env), pid)
    assert "به انتخاب تو متوقف موند" in body
    assert "ادامه بدم؟" not in body and ">توقف</button>" not in body
    assert "ادامه (یک بار دیگه همین‌قدر)" in body  # changing one's mind is possible
    r = _brake(env, pid, decision="continue")
    assert r.status == 303 and len(env.spawned) == 1


def test_unknown_task_is_refused(env: Env) -> None:
    pid = _braked(env)
    r = env.post(f"/projects/{pid}/brake", {"csrf": TOKEN, "task": "nope", "decision": "continue"})
    assert r.status == 409 and env.spawned == []
    assert BRAKE_CONTINUED not in _types(env, pid)


@pytest.mark.parametrize(
    "fields",
    [
        {"decision": "maybe"},
        {"decision": ""},
        {"task": "", "decision": "continue"},
        {"task": "x" * 201, "decision": "continue"},
        {"decision": "stop", "confirm": "yes"},
        {"decision": "continue", "confirm": "no"},
    ],
)
def test_malformed_form_is_refused(env: Env, fields: dict[str, str]) -> None:
    pid = _braked(env)
    r = env.post(f"/projects/{pid}/brake", {"csrf": TOKEN, "task": "build", **fields})
    assert r.status == 400 and env.spawned == []
    assert BRAKE_CONTINUED not in _types(env, pid) and BRAKE_DECLINED not in _types(env, pid)


@pytest.mark.parametrize(
    ("fields", "headers"),
    [
        ({"csrf": "x" * 43}, {}),
        ({}, {}),
        ({"csrf": TOKEN}, {"Origin": "http://evil.example"}),
        ({"csrf": TOKEN}, {"Content-Type": "text/plain"}),
        ({"csrf": TOKEN, "extra": "1"}, {}),
    ],
)
def test_csrf_origin_and_fields_checked(
    env: Env, fields: dict[str, str], headers: dict[str, str]
) -> None:
    pid = _braked(env)
    form = {"task": "build", "decision": "continue", **fields}
    r = env.post(f"/projects/{pid}/brake", form, headers=headers)
    assert r.status in (400, 403, 415) and env.spawned == []
    assert BRAKE_CONTINUED not in _types(env, pid)


def test_get_brake_path_is_not_an_action(env: Env) -> None:
    pid = _braked(env)
    assert env.get(f"/projects/{pid}/brake").status == 404


def test_query_string_and_unknown_project(env: Env) -> None:
    pid = _braked(env)
    r = env.post(f"/projects/{pid}/brake?x=1", {"csrf": TOKEN, "task": "build", "decision": "stop"})
    assert r.status == 404
    r = env.post("/projects/p-nope/brake", {"csrf": TOKEN, "task": "build", "decision": "stop"})
    assert r.status == 404


def test_refused_while_running(env: Env) -> None:
    pid = _braked(env)
    assert env.start(pid).status == 303
    for decision in ("continue", "stop"):
        r = _brake(env, pid, decision=decision)
        assert r.status == 409
    types = _types(env, pid)
    assert BRAKE_CONTINUED not in types and BRAKE_DECLINED not in types


def test_server_without_runs_has_no_brake_endpoint(tmp_path: Path) -> None:
    env = Env(tmp_path, runs=False)
    pid = _braked(env)
    r = env.post(f"/projects/{pid}/brake", {"csrf": TOKEN, "task": "build", "decision": "stop"})
    assert r.status == 404


# -- end to end: automa run brakes, the web page continues -----------------------------


class _CliSpawn:
    """Runs the child's ``automa run`` in-process (same CLI, same FakeProvider)."""

    def __init__(self, e2e: E2E) -> None:
        self.e2e = e2e
        self.codes: list[int] = []

    def __call__(self, argv: Sequence[str], log_path: Path) -> FakeProcess:
        r = self.e2e.cli(*argv[1:])
        log_path.write_text(r.output, encoding="utf-8")
        self.codes.append(r.exit_code)
        proc = FakeProcess()
        proc.finish(r.exit_code)
        return proc


class _Preflight(FakePreflight):
    def command(self, project_id: str, *, yes: bool) -> list[str]:
        super().command(project_id, yes=yes)
        return ["automa", "run", project_id, "--yes"]


def test_e2e_web_continue_after_a_real_brake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small_profiles(monkeypatch)
    e2e = E2E(tmp_path, monkeypatch)
    e2e.new_project()
    _brake_config(e2e)
    e2e.script(costly(("write_file", {"path": "notes.md", "content": "x"})), *design_attempt())
    r = e2e.cli("run", PID, "--yes")
    assert r.exit_code == cli.EXIT_PAUSED, r.output
    requests = len(e2e.provider.requests)

    store = ProjectStore.for_workspace(e2e.ws)
    spawn = _CliSpawn(e2e)
    runs = wr.BackgroundRuns(store, _Preflight(), spawn=spawn)
    app = ws.WebApp(
        e2e.ws,
        e2e.ws / cli.LEDGER_FILENAME,
        ws.allowed_hosts("127.0.0.1", PORT),
        actions=_settings(),
        runs=runs,
        language="fa",
        toman_per_usd=323_000.0,
    )
    body = _page(app, PID)
    assert "ادامه بدم؟" in body and "<bdi>" in body

    def post(fields: dict[str, str]) -> ws.Response:
        env = Env.__new__(Env)
        env.app = app
        return env.post(f"/projects/{PID}/brake", {"csrf": TOKEN, "task": "design", **fields})

    # "Stop": nothing runs, no model call.
    assert post({"decision": "stop"}).status == 303
    assert spawn.codes == [] and len(e2e.provider.requests) == requests

    # "Continue": one more step, the same `automa run`, and the project completes.
    e2e.script(*build_attempt(), judgment(True), *docs_attempt())
    assert post({"decision": "continue"}).status == 303
    assert spawn.codes == [cli.EXIT_OK]
    assert e2e.status_json()["status"] == ProjectStatus.DONE.value
    continued = [e for e in store.events(PID).read() if e.type == BRAKE_CONTINUED]
    assert len(continued) == 1 and continued[0].data["via"] == "web"


# -- the whole-run brake (D172) -------------------------------------------------------


def _run_braked(env: Env) -> str:
    pid = env.project(status=ProjectStatus.PAUSED)
    sink = env.store.events(pid)
    for e in (
        _ev("orchestrator.started", 0, plan_version=1),
        _ev(BRAKE_STOP, 1, task=RUN_KEY, scope="run", spent_nusd=SPENT, limit_nusd=LIMIT),
        _ev("orchestrator.paused", 2, reason="run_brake", task="build"),
    ):
        sink.emit(e.model_copy(update={"project_id": pid}))
    return pid


def test_run_brake_page_asks_about_the_run(env: Env) -> None:
    pid = _run_braked(env)
    body = _page(_fa_app(env), pid)
    assert "این اجرا" in body and "ادامه بدم؟" in body and "تومان" in body
    assert f'name="task" value="{RUN_KEY}"' in body
    assert ">توقف</button>" in body and "ترمز تازه" in body
    assert "<bdi>Build</bdi>" not in body.split('class="card brake"')[1].split("</section>")[0]


def test_run_brake_continue_and_stop_from_the_web(env: Env) -> None:
    pid = _run_braked(env)
    r = env.post(f"/projects/{pid}/brake", {"csrf": TOKEN, "task": RUN_KEY, "decision": "stop"})
    assert r.status == 303 and env.spawned == []
    assert "به انتخاب تو متوقف موند: این اجرا" in _page(_fa_app(env), pid)
    r = env.post(f"/projects/{pid}/brake", {"csrf": TOKEN, "task": RUN_KEY, "decision": "continue"})
    assert r.status == 303 and len(env.spawned) == 1
    continued = [e for e in env.events(pid) if e["type"] == BRAKE_CONTINUED]
    assert [c["data"]["task"] for c in continued] == [RUN_KEY]  # type: ignore[index]
