"""Tests for the Persian project page (D144).

Pure rendering plus the in-process ``WebApp``; no model, no network.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from automa.project import ProjectStore
from automa.webui import i18n
from automa.webui import render as rd
from automa.webui import runs as wr
from automa.webui.progress import CostProgress, EventLine, ProgressView, UnusualSpending
from automa.webui.views import ApprovalView, ProjectView, TaskRow
from test_webui_i18n import LATIN_WORD, NOW, RATE, TEHRAN, _app, _strip_data

FORMS = rd.ActionForms("tok")
# Recorded values that stay as written (ids, tool, role, model, risk, event types).
RECORDED = (
    "fetch_url",
    "developer",
    "external",
    "gpt-5.5",
    "task.passed",
    "project.created",
    "confirm_above_usd",
)


def _view(*, pending: bool = True, plan: bool = True) -> ProjectView:
    tasks = (
        TaskRow(
            order=1,
            id="t1",
            title="ساختن صفحه",
            role="developer",
            attempts=2,
            state="passed",
            model="gpt-5.5",
            last_result="passed",
        ),
        TaskRow(order=2, id="t2", title="نوشتن راهنما", role="developer", depends_on=("t1",)),
    )
    approvals = (
        ApprovalView(
            id="a1",
            task_id="t1",
            tool="fetch_url",
            risk="external",
            reason="خواندن مستندات",
            created_at=NOW - timedelta(minutes=3),
            arguments={"url": "https://example.com/"},
        ),
    )
    return ProjectView(
        id="p-1",
        goal="یک ابزار تبدیل ارقام",
        status="running",
        created_at=NOW - timedelta(hours=1),
        updated_at=NOW - timedelta(minutes=5),
        cost_usd=0.2,
        planning_usd=0.05,
        plan_version=1 if plan else None,
        tasks=tasks if plan else (),
        pending=approvals if pending else (),
    )


def _progress(live: bool = False) -> ProgressView:
    return ProgressView(
        live=live,
        passed=1,
        total=2,
        cost=CostProgress(
            estimated_at=NOW - timedelta(hours=1),
            spent_usd=0.15,
            other_usd=0.0,
            low_usd=0.1,
            high_usd=0.3,
            worst_usd=1.0,
            verdict="within",
        ),
        unusual=UnusualSpending(count=2, largest_usd=0.06, ratio=5.5),
        events=(EventLine(ts=NOW - timedelta(minutes=1), type="task.passed", summary="t1"),),
    )


def _run(**kw: object) -> rd.RunPanel:
    data: dict[str, object] = {
        "can_start": False,
        "needs_confirm": False,
        "can_stop": True,
        "state": "running in the background",
        "running": True,
        "started_at": NOW - timedelta(minutes=2),
        "log_name": "run-1.log",
        "tail": ("step 1",),
    }
    data.update(kw)
    return rd.RunPanel(**data)  # type: ignore[arg-type]


def _render(lang: str = "fa", **kw: object) -> str:
    args: dict[str, object] = {
        "progress": _progress(),
        "run": _run(),
        "rate": RATE,
        "task_costs": {"t1": 0.1},
        "cost_range": "$0.10–$0.30",
        "questions_waiting": True,
        "tz": TEHRAN,
        "now": NOW,
    }
    args.update(kw)
    view = args.pop("view", _view())
    forms = args.pop("forms", FORMS)
    return rd.render_project(view, forms, lang=lang, **args)  # type: ignore[arg-type]


def _interface(doc: str) -> str:
    doc = re.sub(r"<(pre|code)\b.*?</(pre|code)>", "", doc, flags=re.S)
    doc = _strip_data(doc)
    for value in RECORDED:
        doc = doc.replace(value, "")
    return doc


# -- no English left ---------------------------------------------------------------


def test_persian_project_page_has_no_english_interface_text() -> None:
    doc = _render()
    assert LATIN_WORD.findall(_interface(doc)) == []


@pytest.mark.parametrize(
    "run",
    [
        _run(),
        _run(stopping=True),
        _run(running=False, exit_code=0, can_stop=False, can_start=True, needs_confirm=True),
        _run(running=False, exit_code=3, can_stop=False),
        _run(running=False, exit_code=99, can_stop=False),
        _run(running=False, exit_code=-9, can_stop=False),
        rd.RunPanel(can_start=False, needs_confirm=False),
        rd.RunPanel(can_start=True, needs_confirm=True),
    ],
)
def test_persian_run_panel_variants_have_no_english(run: rd.RunPanel) -> None:
    doc = _render(run=run, progress=_progress(live=True))
    assert LATIN_WORD.findall(_interface(doc)) == []


def test_persian_read_only_and_empty_pages_have_no_english() -> None:
    for view in (_view(), _view(pending=False, plan=False)):
        doc = _render(view=view, forms=None, run=None, progress=None, cost_range=None)
        assert LATIN_WORD.findall(_interface(doc)) == []


def test_every_exit_code_has_a_persian_meaning_equal_to_the_english_one() -> None:
    for code, meaning in wr.EXIT_MEANINGS.items():
        assert i18n.exit_label(code, "en") == meaning
        assert (
            LATIN_WORD.findall(i18n.exit_label(code, "fa").replace("confirm_above_usd", "")) == []
        )
    assert i18n.exit_label(-9, "en") == wr.exit_meaning(-9)
    assert i18n.exit_label(42, "en") == wr.exit_meaning(42)


def test_english_run_state_matches_the_recorded_summary() -> None:
    started = NOW - timedelta(minutes=2)
    for exit_code, stopping in ((None, False), (None, True), (3, False), (-2, False)):
        state = wr.RunState(started, Path("x.log"), False, exit_code, stop_requested=stopping)
        panel = _run(state=state.summary, running=state.running, exit_code=exit_code)
        panel = rd.RunPanel(**{**panel.__dict__, "stopping": stopping})
        assert rd._run_state(panel, "fa") != state.summary
        # Rebuilt from the same fields, the English table says what runs.py says.
        if exit_code is None:
            key = "run.state.stopping" if stopping else "run.state.running"
            assert i18n.t(key, "en") == state.summary
        else:
            meaning = i18n.exit_label(exit_code, "en")
            assert i18n.t("run.state.ended", "en", code=exit_code, meaning=meaning) == (
                state.summary
            )


# -- what the Persian page says ----------------------------------------------------


def test_persian_page_is_rtl_with_persian_digits_and_relative_times() -> None:
    doc = _render()
    assert '<html lang="fa" dir="rtl">' in doc
    assert "ساخته‌شده امروز ساعت ۲۰:۴۰ · آخرین تغییر ۵ دقیقه پیش" in doc
    assert "مرحله‌های تموم‌شده:" in doc and "۱ / ۲" in doc
    assert "برنامه (نسخهٔ ۱، ۲ مرحله)" in doc
    assert "درخواست‌های منتظر تأیید (۱)" in doc
    assert "تلاش‌ها: ۲" in doc
    assert "#۱" in doc
    assert "۵٫۵ برابر معمول" in doc
    assert "در پس‌زمینه در حال کاره" in doc
    assert "(شروع: ۲ دقیقه پیش)" in doc
    assert "2026-" not in _interface(doc)


def test_persian_task_states() -> None:
    assert i18n.task_state_label(None, "fa") == "شروع نشده"
    assert i18n.task_state_label("passed", "fa") == "تموم شد"
    assert i18n.task_state_label("weird", "fa") == "weird"
    for state in ("pending", "running", "verifying", "passed", "failed", "blocked"):
        assert i18n.task_state_label(state, "en") == state
    assert i18n.task_state_label(None, "en") == "not started"


def test_persian_exit_meanings() -> None:
    assert i18n.exit_label(3, "fa").startswith("سقف هزینه پر شد")
    assert i18n.exit_label(-9, "fa") == "با سیگنال ۹ بسته شد"
    assert i18n.exit_label(99, "fa") == "متوقف شد (گزارش رو ببین)"
    doc = _render(run=_run(running=False, exit_code=3, can_stop=False))
    assert "تموم شد (کد ۳): سقف هزینه پر شد" in doc


def test_persian_buttons_and_forms_keep_their_safety_fields() -> None:
    doc = _render(
        run=_run(running=False, exit_code=0, can_stop=False, can_start=True, needs_confirm=True)
    )
    assert ">تأیید</button>" in doc and ">رد</button>" in doc
    assert ">شروع</button>" in doc
    assert 'name="csrf" value="tok"' in doc
    assert doc.count('name="confirm" value="yes" required') == 2
    assert "فقط یک بار و دقیقاً با همین‌ها اجرا بشه" in doc
    assert '<pre dir="ltr">' in doc  # full arguments stay left-to-right


def test_persian_recorded_data_is_escaped_and_untranslated() -> None:
    view = _view().model_copy(
        update={
            "tasks": (TaskRow(order=1, id="t1", title="<b>x</b>", role="dev<", model="m&"),),
        }
    )
    doc = _render(view=view)
    assert "<b>x</b>" not in doc and "&lt;b&gt;x&lt;/b&gt;" in doc
    # Role and model are recorded data in the technical details (D177), escaped.
    assert "<td>dev&lt;</td>" in doc and '<td dir="ltr">m&amp;</td>' in doc
    assert "جزئیات فنی" in doc and "شناسه" in doc


def test_persian_unusual_line_singular() -> None:
    line = rd._unusual_line(UnusualSpending(count=1, largest_usd=0.06, ratio=None), RATE, "fa")
    assert "یک مرحله" in line and "برابر" not in line


def test_english_page_without_a_clock_keeps_iso_times() -> None:
    doc = _render(lang="en", tz=None, now=None)
    assert "created 2026-11-19T17:10:00+00:00" in doc
    assert "Pending approvals (1)" in doc and "Plan v1 (2 tasks)" in doc


def test_english_page_with_a_clock_is_relative() -> None:
    doc = _render(lang="en")
    assert "created today at 20:40 · updated 5 minutes ago" in doc
    assert "2026-11-19T17:10" not in doc


# -- the server --------------------------------------------------------------------


def test_server_renders_the_project_page_in_persian(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("یک ابزار تبدیل ارقام", project_id="p1")
    app = _app(tmp_path, language="fa", clock=lambda: datetime.now(UTC))
    resp = app.handle("GET", "/projects/p1", "127.0.0.1:8765")
    doc = resp.body.decode()
    assert resp.status == 200
    assert '<html lang="fa" dir="rtl">' in doc
    assert "هدف:" in doc
    assert "درخواست‌های منتظر تأیید" not in doc  # the section is shown only when needed
    assert "همین الان" in doc
    assert LATIN_WORD.findall(_interface(doc)) == []
