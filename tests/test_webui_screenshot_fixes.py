"""Tests for the first fixes from the owner's screenshots (M13, D131).

CRLF from a Windows browser form is stored and shown as line breaks (never as a
visible ``\\x0d``), the web page shows no theoretical "worst case" and no terminal
commands, and the raw run log is folded into "Technical details". Model calls only
through ``FakeProvider``; no network.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from automa import cli
from automa.project import ProjectStore
from automa.webui import actions as wa
from automa.webui import progress as pg
from automa.webui import render as rd
from automa.webui.views import build_project_view
from test_webui_ideas import HOST, PLAN, TOKEN, RealEnv

FORM_HEADERS = {"content-type": wa.FORM_CONTENT_TYPE}
T0 = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)


# --- form input: CRLF -> LF -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a%0D%0Ab", "a\nb"),
        ("a%0Db", "a\nb"),
        ("a%0D%0D%0Ab", "a\n\nb"),
        ("a%0A%0D%0Ab", "a\n\nb"),
        ("%0D%0A", "\n"),
        ("plain", "plain"),
    ],
)
def test_parse_form_normalizes_line_breaks(raw: str, expected: str) -> None:
    form = wa.parse_form(FORM_HEADERS, f"goal={raw}".encode(), fields=frozenset({"goal"}))
    assert form == {"goal": expected}


def test_parse_form_keeps_other_control_characters() -> None:
    form = wa.parse_form(FORM_HEADERS, b"goal=a%07b%09c", fields=frozenset({"goal"}))
    assert form["goal"] == "a\x07b\tc"


def test_parse_form_size_limit_still_counts_the_raw_body() -> None:
    body = b"goal=" + b"%0D%0A" * 20
    with pytest.raises(wa.ActionRefused) as e:
        wa.parse_form(FORM_HEADERS, body, fields=frozenset({"goal"}), max_bytes=len(body) - 1)
    assert e.value.status == 413


def test_new_idea_goal_and_answers_are_stored_with_lf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = RealEnv(tmp_path, monkeypatch)
    e.provider.push(json.dumps({"questions": ["Which language?"]}), PLAN)
    goal = "Build a site\r\nwith two pages\r\n\r\nin Persian"
    r = e.post("/new", {"csrf": TOKEN, "goal": goal, "ask": "yes"})
    assert r.status == 303, r.body
    (pid,) = e.store.list_ids()
    assert e.store.load(pid).goal == "Build a site\nwith two pages\n\nin Persian"
    r = e.post(f"/projects/{pid}/clarify", {"csrf": TOKEN, "a0": "فارسی\r\nو انگلیسی"})
    assert r.status == 303, r.body
    assert e.store.load_clarifications(pid)[0].answer == "فارسی\nو انگلیسی"
    body = e.app.handle("GET", f"/projects/{pid}", HOST).body.decode()
    assert "\\x0d" not in body and "\r" not in body
    assert "Build a site\nwith two pages" in body


# --- display: CRLF already stored ---------------------------------------------------


def test_text_shows_crlf_as_line_breaks() -> None:
    assert rd.text("a\r\nb\rc") == "a\nb\nc"
    assert "x0d" not in rd.text("line 1\r\nline 2")


def test_text_still_shows_other_control_and_bidi_characters() -> None:
    assert rd.text("a\x07b") == "a\\x07b"
    assert rd.text("a‮b") == "a\\u202eb"
    assert rd.text("<b>\r\n") == "&lt;b&gt;\n"


def test_normalize_newlines() -> None:
    assert rd.normalize_newlines("\r\n\r\r\n\n") == "\n\n\n\n"
    assert rd.normalize_newlines("no breaks") == "no breaks"


def test_old_project_with_crlf_goal_renders_cleanly(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path / "ws")
    store.create("یک صفحه HTML\r\nبا دو بخش", project_id="p-crlf")
    html = rd.render_project(build_project_view(store, "p-crlf", 0))
    assert "\\x0d" not in html
    assert '<p class="goal" dir="auto">یک صفحه HTML\nبا دو بخش</p>' in html


def test_goal_keeps_its_line_breaks_on_screen() -> None:
    assert "white-space:pre-line" in rd.page("t", "b")


# --- no worst case, no terminal commands --------------------------------------------


def _cost(spent: float, high: float = 0.3, worst: float = 336.0) -> pg.CostProgress:
    return pg.CostProgress(
        estimated_at=T0,
        spent_usd=spent,
        other_usd=0.0,
        low_usd=0.05,
        high_usd=high,
        worst_usd=worst,
        verdict="within",
    )


@pytest.mark.parametrize("live", [True, False])
def test_progress_never_shows_the_worst_case(tmp_path: Path, live: bool) -> None:
    store = ProjectStore.for_workspace(tmp_path / "ws")
    store.create("g", project_id="p1")
    progress = pg.ProgressView(live=live, passed=1, total=2, cost=_cost(0.12))
    html = rd.render_project(build_project_view(store, "p1", 0), progress=progress)
    assert "worst" not in html.lower() and "336" not in html and "guard cap" not in html
    assert 'max="0.300000" value="0.120000"' in html


def test_range_meter_grows_with_a_cost_above_the_range() -> None:
    meter = rd._range_meter(_cost(0.9))
    assert 'max="0.900000" value="0.900000"' in meter and 'high="0.300000"' in meter


def _panel(**kw: Any) -> rd.RunPanel:
    base = rd.RunPanel(can_start=True, needs_confirm=True, can_stop=True, state="running")
    return replace(base, **kw)


def test_run_panel_has_no_terminal_commands(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path / "ws")
    store.create("g", project_id="p1")
    forms = rd.ActionForms(csrf_token=TOKEN)
    html = rd.render_project(build_project_view(store, "p1", 0), forms, run=_panel())
    assert "automa run" not in html and "<code>automa" not in html
    assert "Start run" in html and "Stop" in html


def test_run_log_is_folded_into_technical_details(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path / "ws")
    store.create("g", project_id="p1")
    forms = rd.ActionForms(csrf_token=TOKEN)
    tail = ("cost range: about $0.05; start with: automa run p1", "done")
    run = _panel(log_name="run-1.log", tail=tail, running=False, can_stop=False)
    html = rd.render_project(build_project_view(store, "p1", 0), forms, run=run)
    m = re.search(r'<details class="tech">(.*?)</details>', html, re.S)
    assert m is not None
    assert "Technical details" in m.group(1)
    assert "web-runs/run-1.log" in m.group(1) and "start with: automa run p1" in m.group(1)
    outside = html.replace(m.group(0), "")
    assert "automa run" not in outside and "web-runs/" not in outside


def test_no_technical_details_without_a_log(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path / "ws")
    store.create("g", project_id="p1")
    forms = rd.ActionForms(csrf_token=TOKEN)
    html = rd.render_project(build_project_view(store, "p1", 0), forms, run=_panel(state=None))
    assert "(run log)" not in html  # only the page's own technical details (D177)
    assert html.count("<details") == 1


# --- the web cost range line --------------------------------------------------------


def _planned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kw: Any) -> tuple[RealEnv, str]:
    e = RealEnv(tmp_path, monkeypatch, **kw)
    e.provider.push(PLAN)
    r = e.post("/new", {"csrf": TOKEN, "goal": "Build a site"})
    assert r.status == 303, r.body
    (pid,) = e.store.list_ids()
    return e, pid


def test_web_cost_range_shows_the_real_project_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    e, pid = _planned(tmp_path, monkeypatch, project=3.0)
    line = cli._web_cost_range(e.config, e.store, pid)
    assert re.search(r"about \$\d+\.\d\d to \$\d+\.\d\d", line), line
    assert "project safety cap $3.00: spending never goes above it" in line
    assert "worst" not in line and "automa" not in line


def test_web_cost_range_with_toman(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    e, pid = _planned(tmp_path, monkeypatch, project=3.0)
    config = e.config.model_copy(
        update={"display": e.config.display.model_copy(update={"toman_per_usd": 323000.0})}
    )
    line = cli._web_cost_range(config, e.store, pid)
    assert "project safety cap about 969 thousand toman ($3.00)" in line
    assert re.search(r"about [\d.,]+ (to [\d.,]+ )?(thousand |million )?toman", line), line


def test_web_cost_range_says_confirmation_is_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    e, pid = _planned(tmp_path, monkeypatch)
    config = e.config.model_copy(
        update={"limits": e.config.limits.model_copy(update={"confirm_above_usd": 0.0})}
    )
    line = cli._web_cost_range(config, e.store, pid)
    assert "you confirm before it starts (above $0.00)" in line
    assert "automa run" not in line


def test_web_cost_range_unavailable_has_no_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    e, pid = _planned(tmp_path, monkeypatch)

    def boom(*_: Any, **__: Any) -> Any:
        raise ValueError("bad\nestimate")

    monkeypatch.setattr(cli, "estimate_plan", boom)
    line = cli._web_cost_range(e.config, e.store, pid)
    assert line.startswith("unavailable (bad estimate); project safety cap $")
    assert "automa" not in line


def test_cli_new_line_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The terminal keeps its full line (worst case and the next command)."""
    e, pid = _planned(tmp_path, monkeypatch)
    line = cli._new_cost_range(e.config, e.store, pid)
    assert "guard cap (worst case)" in line and f"start with: automa run {pid}" in line


def test_web_page_cost_range_uses_the_web_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    e, pid = _planned(tmp_path, monkeypatch)
    body = e.app.handle("GET", f"/projects/{pid}", HOST).body.decode()
    assert "Cost range:" in body and "project safety cap" in body
    assert "worst case" not in body and "automa run" not in body


# --- notices --------------------------------------------------------------------------


def test_decision_notices_have_no_commands() -> None:
    import inspect

    from automa.webui import runs, server

    src = inspect.getsource(server.WebApp._notice) + inspect.getsource(runs)
    assert "on the next: automa run" not in src
    assert "use automa run --force" not in src
