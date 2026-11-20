"""Tests for the plain reason of a pause and the next step (``automa.webui.stops``, D151).

Unit tests use hand-written events; the end-to-end tests run the real ``automa run``
stack (BudgetGuard, Orchestrator, …) with a scripted ``FakeProvider`` and read the
recorded events back. No network, no real model calls.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from automa import cli
from automa.events import Event
from automa.l10n import resolve_timezone
from automa.llm import ProviderError
from automa.plan import Plan
from automa.project import ProjectStatus, ProjectStore
from automa.webui import render as rd
from automa.webui import stops as sp
from automa.webui.journey import build_journey
from automa.webui.progress import ProgressView, build_progress
from automa.webui.views import build_project_view
from test_e2e import E2E, PID, TOML, design_attempt
from test_webui import _task

T0 = datetime(2026, 11, 20, 18, 0, tzinfo=UTC)
TEHRAN = resolve_timezone("Asia/Tehran")


def _ev(type_: str, minutes: int = 0, **data: object) -> Event:
    return Event(ts=T0 + timedelta(minutes=minutes), type=type_, project_id="p1", data=data)


def _budget(scope: str, key: str | None, minutes: int = 1) -> Event:
    return _ev("budget.stop", minutes, scope=scope, key=key, limit_usd=1.0, spent_usd=1.0)


def _paused(reason: str, minutes: int = 2, **data: object) -> Event:
    return _ev("orchestrator.paused", minutes, reason=reason, task="a", **data)


STARTED = _ev("orchestrator.started", 0, plan_version=1)


# -- latest_stop ------------------------------------------------------------------


def test_no_events_no_stop() -> None:
    assert sp.latest_stop([]) is None


def test_started_without_pause_is_no_stop() -> None:
    assert sp.latest_stop([STARTED]) is None


def test_finished_run_is_no_stop() -> None:
    assert sp.latest_stop([STARTED, _ev("orchestrator.finished", 3, status="done")]) is None


def test_an_old_pause_before_a_new_start_is_ignored() -> None:
    events = [STARTED, _paused("user"), _ev("orchestrator.started", 5, plan_version=1)]
    assert sp.latest_stop(events) is None


def test_daily_cap_with_reset_at_next_utc_midnight() -> None:
    stop = sp.latest_stop([STARTED, _budget("daily", "2026-11-20"), _paused("budget")])
    assert stop is not None and stop.kind == "daily_cap"
    assert stop.reset_at == datetime(2026, 11, 21, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize("key", [None, "not-a-date", 7])
def test_daily_cap_bad_key_uses_the_stop_day(key: object) -> None:
    event = _ev("budget.stop", 1, scope="daily", key=key)
    stop = sp.latest_stop([STARTED, event, _paused("budget")])
    assert stop is not None and stop.reset_at == datetime(2026, 11, 21, tzinfo=UTC)


@pytest.mark.parametrize(
    ("scope", "kind"),
    [("project", "project_cap"), ("per_call", "per_call_cap"), ("weird", "budget")],
)
def test_budget_scopes(scope: str, kind: str) -> None:
    stop = sp.latest_stop([STARTED, _budget(scope, "p1"), _paused("budget")])
    assert stop is not None and stop.kind == kind and stop.reset_at is None


def test_budget_pause_without_budget_event_is_generic() -> None:
    stop = sp.latest_stop([STARTED, _paused("budget")])
    assert stop is not None and stop.kind == "budget"


def test_budget_stop_without_pause_event() -> None:
    # e.g. refused before the orchestrator could record a pause
    stop = sp.latest_stop([_budget("project", "p1")])
    assert stop is not None and stop.kind == "project_cap"


def test_latest_budget_event_wins() -> None:
    events = [STARTED, _budget("daily", "2026-11-20", 1), _budget("project", "p1", 2)]
    stop = sp.latest_stop([*events, _paused("budget", 3)])
    assert stop is not None and stop.kind == "project_cap"


@pytest.mark.parametrize(
    ("detail", "kind"),
    [
        ("provider returned HTTP 402: payment required", "balance"),
        ("provider returned HTTP 400: Insufficient balance", "balance"),
        ("provider returned HTTP 429: insufficient_quota", "balance"),
        ("provider returned HTTP 400: موجودی کافی نیست", "balance"),
        ("provider returned HTTP 401: invalid", "key"),
        ("provider returned HTTP 403", "key"),
        ("provider returned HTTP 400: Invalid API key", "key"),
        ("could not connect: ConnectError", "network"),
        ("request timed out", "network"),
        ("network error: ReadError", "network"),
        ("provider returned HTTP 500", "provider"),
        ("", "provider"),
    ],
)
def test_provider_errors_are_classified(detail: str, kind: str) -> None:
    stop = sp.latest_stop([STARTED, _paused("provider", detail=detail)])
    assert stop is not None and stop.kind == kind


def test_provider_detail_not_a_string_is_generic() -> None:
    stop = sp.latest_stop([STARTED, _paused("provider", detail=402)])
    assert stop is not None and stop.kind == "provider"


@pytest.mark.parametrize(
    ("reason", "kind"),
    [
        ("approval", "approval"),
        ("user", "user"),
        ("verification", "verification"),
        ("tool_layer", "tool_layer"),
        ("config", "config"),
        ("something_new", "other"),
    ],
)
def test_other_reasons(reason: str, kind: str) -> None:
    stop = sp.latest_stop([STARTED, _paused(reason)])
    assert stop is not None and stop.kind == kind


def test_stop_info_holds_no_recorded_text() -> None:
    leaked = "sk-SECRET provider returned HTTP 402"
    stop = sp.latest_stop([STARTED, _paused("provider", detail=leaked)])
    assert stop is not None and "SECRET" not in stop.model_dump_json()


# -- describe_stop ----------------------------------------------------------------


def _daily() -> sp.StopInfo:
    return sp.StopInfo(kind="daily_cap", at=T0, reset_at=datetime(2026, 11, 21, 0, 0, tzinfo=UTC))


def test_daily_wait_says_the_local_reset_time_tomorrow() -> None:
    # 18:00 UTC = 21:30 Tehran; reset 00:00 UTC = 03:30 Tehran on the next day
    line = sp.describe_stop(_daily(), lang="fa", tz=TEHRAN, now=T0)
    assert (
        line.text
        == "متوقف شد: سقف خرج روزانه (daily_usd) پر شده؛ از فردا ساعت ۳:۳۰ می‌شه ادامه داد."
    )
    assert line.tone == "bad" and not line.resume


def test_daily_wait_today_after_local_midnight() -> None:
    now = datetime(2026, 11, 20, 22, 0, tzinfo=UTC)  # 01:30 Tehran on the 21st
    line = sp.describe_stop(_daily(), lang="fa", tz=TEHRAN, now=now)
    assert "از امروز ساعت ۳:۳۰" in line.text and not line.resume
    en = sp.describe_stop(_daily(), lang="en", tz=TEHRAN, now=now)
    assert en.text.endswith("continue from today at 3:30.")


def test_daily_wait_far_away_is_a_date() -> None:
    now = datetime(2026, 11, 18, 10, 0, tzinfo=UTC)
    line = sp.describe_stop(_daily(), lang="en", tz=UTC, now=now)
    assert "from 21 Nov at 00:00" in line.text


def test_daily_ready_after_reset() -> None:
    now = datetime(2026, 11, 21, 0, 0, tzinfo=UTC)
    line = sp.describe_stop(_daily(), lang="fa", tz=TEHRAN, now=now)
    assert "الان می‌شه ادامه داد" in line.text
    assert line.tone == "info" and line.resume


def test_daily_without_reset_is_generic_budget() -> None:
    stop = sp.StopInfo(kind="daily_cap", at=T0)
    line = sp.describe_stop(stop, lang="en", tz=UTC, now=T0)
    assert line.text == sp.st("stop.budget", "en") and not line.resume


@pytest.mark.parametrize(
    ("kind", "resume", "tone", "fa_part"),
    [
        ("project_cap", False, "bad", "project_usd"),
        ("per_call_cap", False, "bad", "per_call_usd"),
        ("budget", False, "bad", "نگهبان هزینه"),
        ("balance", True, "bad", "حساب رو شارژ کن"),
        ("key", False, "bad", ".env"),
        ("network", True, "bad", "وصل نشد"),
        ("provider", True, "bad", "درگاه مدل خطا داد"),
        ("approval", True, "info", "منتظر تأیید تو"),
        ("user", True, "info", "به درخواست تو"),
        ("verification", True, "bad", "دوباره امتحان"),
        ("tool_layer", True, "bad", "لایهٔ ابزارها"),
        ("config", False, "bad", "automa.toml"),
        ("task_brake", False, "bad", "ترمز خرج"),
        ("other", False, "bad", "گزارش رو ببین"),
    ],
)
def test_each_kind_has_a_sentence(kind: str, resume: bool, tone: str, fa_part: str) -> None:
    stop = sp.StopInfo(kind=kind, at=T0)  # type: ignore[arg-type]
    line = sp.describe_stop(stop, lang="fa", tz=TEHRAN, now=T0)
    assert fa_part in line.text and line.resume is resume and line.tone == tone
    assert sp.describe_stop(stop, lang="en", tz=TEHRAN, now=T0).text.isascii()


def test_every_kind_has_a_message() -> None:
    kinds = sp.StopKind.__args__  # type: ignore[attr-defined]
    for kind in kinds:
        if kind != "daily_cap":
            assert f"stop.{kind}" in sp.MESSAGES


def test_st_rejects_bad_language_and_parameters() -> None:
    with pytest.raises(ValueError):
        sp.st("stop.user", "de")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sp.st("stop.daily_wait", "fa")


def test_module_is_pure() -> None:
    tree = ast.parse(Path(sp.__file__).read_text(encoding="utf-8"))
    names = {
        a.name.split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in n.names
    } | {
        n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert not names & {"os", "subprocess", "socket", "httpx", "pathlib", "shutil"}


# -- journey and page -------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> ProjectStore:
    s = ProjectStore.for_workspace(tmp_path)
    s.create("write a poem", project_id="p1")
    s.save_plan("p1", Plan(goal="g", tasks=[_task("a"), _task("b", ["a"])]))
    p = s.load("p1")
    s.save(p.model_copy(update={"status": ProjectStatus.PAUSED}))
    return s


def _emit(store: ProjectStore, *events: Event) -> None:
    for e in events:
        store.events("p1").emit(e)


def _page(
    store: ProjectStore,
    *,
    lang: str = "fa",
    now: datetime = T0,
    run: rd.RunPanel | None = None,
    forms: rd.ActionForms | None = None,
) -> str:
    view = build_project_view(store, "p1", 0)
    progress = build_progress(store, "p1", [t.state for t in view.tasks], [], running=False)
    return rd.render_project(
        view,
        forms,
        run=run,
        progress=progress,
        lang=lang,
        tz=TEHRAN,
        now=now,  # type: ignore[arg-type]
    )


FORMS = rd.ActionForms(csrf_token="tok-123")  # noqa: S106 - test token
STARTABLE = rd.RunPanel(can_start=True, needs_confirm=False, state="ended", exit_code=3)


def test_build_progress_carries_the_stop(store: ProjectStore) -> None:
    _emit(store, STARTED, _budget("daily", "2026-11-20"), _paused("budget"))
    view = build_project_view(store, "p1", 0)
    progress = build_progress(store, "p1", [t.state for t in view.tasks], [], running=False)
    assert progress.stop is not None and progress.stop.kind == "daily_cap"


def test_page_daily_wait_has_time_and_no_continue(store: ProjectStore) -> None:
    _emit(store, STARTED, _budget("daily", "2026-11-20"), _paused("budget"))
    page = _page(store, run=STARTABLE, forms=FORMS)
    assert "سقف خرج روزانه (daily_usd) پر شده؛ از فردا ساعت ۳:۳۰ می‌شه ادامه داد" in page
    assert "خرج پروژه به سقفش رسید" not in page  # not the project-cap sentence
    assert ">ادامه</button>" not in page


def test_page_daily_ready_offers_continue(store: ProjectStore) -> None:
    _emit(store, STARTED, _budget("daily", "2026-11-20"), _paused("budget"))
    page = _page(store, now=datetime(2026, 11, 21, 1, tzinfo=UTC), run=STARTABLE, forms=FORMS)
    assert "سقف دوباره باز شده" in page
    assert page.count('action="/projects/p1/start"') == 2  # Continue + the run panel
    assert '<button type="submit">ادامه</button>' in page
    assert page.count('value="tok-123"') >= 2


def test_continue_keeps_the_cost_confirmation(store: ProjectStore) -> None:
    _emit(store, STARTED, _paused("user"))
    run = rd.RunPanel(can_start=True, needs_confirm=True)
    page = _page(store, run=run, forms=FORMS)
    assert page.count('name="confirm" value="yes" required') == 2


def test_no_continue_without_forms_or_start(store: ProjectStore) -> None:
    _emit(store, STARTED, _paused("user"))
    assert "به درخواست تو متوقف شد" in _page(store)
    assert ">ادامه</button>" not in _page(store)  # read-only page
    cannot = rd.RunPanel(can_start=False, needs_confirm=False)
    assert ">ادامه</button>" not in _page(store, run=cannot, forms=FORMS)


def test_page_balance_in_english(store: ProjectStore) -> None:
    _emit(store, STARTED, _paused("provider", detail="provider returned HTTP 402"))
    page = _page(store, lang="en", run=STARTABLE, forms=FORMS)
    assert "out of credit. Top it up, then continue." in page
    assert '<button type="submit">Continue</button>' in page
    assert "HTTP 402" not in page.split("<details")[0]  # error text not in the sentence


def test_pending_approval_comes_first(store: ProjectStore) -> None:
    _emit(store, STARTED, _paused("approval"))
    view = build_project_view(store, "p1", 0)
    pending = view.model_copy(update={"pending": (object(),)})
    j = build_journey(pending, lang="fa", stop_now=("x", "info"))
    assert "تأیید تو رو می‌خواد" in j.now


def test_stop_is_ignored_while_running_or_not_paused(store: ProjectStore) -> None:
    _emit(store, STARTED, _paused("user"))
    view = build_project_view(store, "p1", 0)
    assert build_journey(view, running=True, stop_now=("x", "bad")).now != "x"
    progress = ProgressView(stop=sp.StopInfo(kind="user", at=T0))
    done = view.model_copy(update={"status": "running"})
    page = rd.render_project(done, progress=progress, lang="fa", tz=TEHRAN, now=T0)
    assert "به درخواست تو متوقف شد" not in page


def test_without_events_old_exit_code_sentence_stays(store: ProjectStore) -> None:
    page = _page(store, lang="en", run=STARTABLE, forms=FORMS)
    assert "reached its spending cap" in page
    assert ">Continue</button>" not in page


def test_fixed_offset_timezone() -> None:
    line = sp.describe_stop(_daily(), lang="en", tz=timezone(timedelta(hours=1)), now=T0)
    assert "tomorrow at 1:00" in line.text


# -- end to end with the real stack and FakeProvider -------------------------------


def _write_config(e2e: E2E, *, daily: float, project: float = 5.0, per_call: float = 0.25) -> None:
    text = TOML.format(ws=e2e.ws.as_posix(), project=project, per_call=per_call)
    e2e.config.write_text(text.replace("daily_usd = 5.0", f"daily_usd = {daily}"), "utf-8")


@pytest.fixture
def e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> E2E:
    return E2E(tmp_path, monkeypatch)


def _recorded_stop(e2e: E2E) -> sp.StopInfo | None:
    return sp.latest_stop(e2e.store.events(PID).read())


def test_e2e_project_cap_stop(e2e: E2E) -> None:
    e2e.new_project()
    cap = e2e.ledger_spent_nusd() / 1e9 + 0.037
    e2e.write_config(project=cap)
    e2e.script(*design_attempt())
    r = e2e.cli("run", PID, "--yes")
    assert r.exit_code == cli.EXIT_BUDGET, r.output
    stop = _recorded_stop(e2e)
    assert stop is not None and stop.kind == "project_cap"


def test_e2e_daily_cap_stop(e2e: E2E) -> None:
    e2e.new_project()
    spent = e2e.ledger_spent_nusd() / 1e9
    daily = spent + 0.037
    _write_config(e2e, daily=daily, per_call=min(0.25, daily))
    e2e.script(*design_attempt())
    r = e2e.cli("run", PID, "--yes")
    assert r.exit_code == cli.EXIT_BUDGET, r.output
    stop = _recorded_stop(e2e)
    assert stop is not None and stop.kind == "daily_cap"
    assert stop.reset_at is not None and stop.reset_at.hour == 0
    assert stop.reset_at > stop.at


def test_e2e_gateway_out_of_credit(e2e: E2E) -> None:
    e2e.new_project()
    e2e.script(ProviderError("provider returned HTTP 402: insufficient balance", status=402))
    r = e2e.cli("run", PID, "--yes")
    assert r.exit_code == cli.EXIT_PROVIDER, r.output
    stop = _recorded_stop(e2e)
    assert stop is not None and stop.kind == "balance"
    assert e2e.store.load(PID).status is ProjectStatus.PAUSED


def test_e2e_resume_clears_the_stop(e2e: E2E) -> None:
    e2e.new_project()
    e2e.script(ProviderError("could not connect: ConnectError"))
    e2e.cli("run", PID, "--yes")
    stop = _recorded_stop(e2e)
    assert stop is not None and stop.kind == "network"
    e2e.script(*design_attempt())  # the next run starts, then runs out of script
    e2e.cli("run", PID, "--yes")
    events = e2e.store.events(PID).read()
    assert sum(1 for e in events if e.type == "orchestrator.started") == 2
    again = _recorded_stop(e2e)
    assert again is not None and again.at > stop.at
