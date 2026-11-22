"""Per-task ``needs`` in the plan and the warning before "Start" (M16, D164)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from automa import feasibility as fz
from automa.agents.executive import PLAN_SYSTEM_PROMPT
from automa.agents.worker import task_message
from automa.cli import _plan_lines
from automa.plan import Plan
from automa.plan.models import TASK_NEEDS, Task, parse_plan
from automa.webui.i18n import t
from automa.webui.render import ActionForms, RunPanel, _run_panel
from test_cli import NO_QUESTIONS, plan_json
from test_cli import Env as CliEnv
from test_webui_runs import Env as RunsEnv

EVIL = "<script>alert(1)</script>"


def _task(tid: str, needs: list[str] | None = None, title: str | None = None) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "id": tid,
        "title": title or f"Task {tid}",
        "role": "developer",
        "acceptance_criteria": [{"description": "works"}],
    }
    if needs is not None:
        raw["needs"] = needs
    return raw


def _plan(*tasks: dict[str, Any]) -> Plan:
    return Plan.model_validate({"goal": "g", "tasks": list(tasks)})


# --- model ------------------------------------------------------------------------


def test_needs_default_to_empty_and_old_plans_still_load() -> None:
    plan = _plan(_task("a"))
    assert plan.tasks[0].needs == []
    assert fz.plan_warnings(plan) == ()


def test_needs_are_deduplicated_in_canonical_order() -> None:
    task = Task.model_validate(_task("a", ["gui", "live_web_data", "gui", "login"]))
    assert task.needs == ["live_web_data", "login", "gui"]


@pytest.mark.parametrize("bad", [["internet"], ["LIVE_WEB_DATA"], [""], "live_web_data", [1]])
def test_unknown_needs_are_rejected(bad: object) -> None:
    with pytest.raises(ValidationError):
        Task.model_validate({**_task("a"), "needs": bad})


def test_invalid_needs_give_model_readable_feedback() -> None:
    with pytest.raises(Exception) as e:
        parse_plan(json.dumps({"goal": "g", "tasks": [_task("a", ["internet"])]}))
    assert "live_web_data" in str(getattr(e.value, "details", e.value))


def test_needs_round_trip_through_json() -> None:
    plan = _plan(_task("a", ["paid_api"]))
    again = Plan.model_validate_json(plan.model_dump_json())
    assert again == plan and again.tasks[0].needs == ["paid_api"]


# --- warnings ---------------------------------------------------------------------


def test_plan_warnings_group_by_kind_in_plan_order() -> None:
    plan = _plan(
        _task("a", ["gui"], "A"),
        _task("b", ["live_web_data", "gui"], "B"),
        _task("c", [], "C"),
        _task("d", ["live_web_data"], "D"),
    )
    assert fz.plan_warnings(plan) == (
        fz.FeasibilityWarning("live_web_data", ("B", "D")),
        fz.FeasibilityWarning("gui", ("A", "B")),
    )


def test_every_need_kind_has_texts() -> None:
    for need in TASK_NEEDS:
        assert fz.WARNING_TEXT[need]
        for lang in ("fa", "en"):
            assert t(f"run.feas.{need}", lang)
            assert t(f"run.feas.{need}.alt", lang)


def test_live_web_warning_offers_a_file_instead() -> None:
    assert "CSV" in fz.WARNING_TEXT["live_web_data"]
    assert "CSV" in t("run.feas.live_web_data.alt", "fa")


# --- worker -----------------------------------------------------------------------


def test_task_without_needs_keeps_its_message() -> None:
    task = Task.model_validate(_task("a"))
    assert fz.worker_note(task) == ""
    assert "What this task depends on" not in task_message(task)


def test_live_web_task_is_told_never_to_invent() -> None:
    task = Task.model_validate(_task("a", ["live_web_data"]))
    msg = task_message(task)
    assert "## What this task depends on (from the plan)" in msg
    assert "do not invent" in msg and "ask_user" in msg
    assert fz.NO_WEB_NOTE not in msg


def test_other_needs_forbid_downloading_data() -> None:
    task = Task.model_validate(_task("a", ["login", "gui"]))
    note = fz.worker_note(task)
    assert "never ask for or store a password" in note and "desktop program" in note
    assert fz.NO_WEB_NOTE in note


def test_note_comes_after_criteria_and_before_feedback() -> None:
    task = Task.model_validate(_task("a", ["paid_api"]))
    msg = task_message(task, attachments="## Files the user attached\nx")
    assert (
        msg.index("## Acceptance criteria")
        < msg.index("## What this task")
        < msg.index("## Files the user attached")
    )


def test_developer_role_limits_downloads_and_inline_code() -> None:
    text = (Path(__file__).parents[1] / "src/automa/roles/builtin/developer.toml").read_text(
        encoding="utf-8"
    )
    assert "live web data" in text and "write_file first" in text


# --- executive prompt -------------------------------------------------------------


def test_plan_prompt_explains_needs() -> None:
    for need in TASK_NEEDS:
        assert f'"{need}"' in PLAN_SYSTEM_PROMPT
    assert '"needs": []' in PLAN_SYSTEM_PROMPT


# --- CLI --------------------------------------------------------------------------


def test_plan_lines_without_needs_are_unchanged() -> None:
    assert _plan_lines(_plan(_task("a"))) == ["plan v1: 1 task(s)", "  1. [developer] a: Task a"]


def test_plan_lines_show_needs_and_warnings() -> None:
    lines = _plan_lines(_plan(_task("a", ["live_web_data"], "Fetch prices"), _task("b")))
    assert lines[1].endswith("[needs: live_web_data]")
    assert lines[-1].startswith("warning: Fetch prices: needs live data from websites")


def test_cli_new_prints_the_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = CliEnv(tmp_path, monkeypatch)
    env.provider.push(NO_QUESTIONS, plan_json(_task("fetch", ["live_web_data"], "Fetch")))
    r = env.run("new", "stock scanner", "--id", "p1")
    assert r.exit_code == 0, r.output
    assert "[needs: live_web_data]" in r.output and "Excel/CSV" in r.output
    assert env.store.load_plan("p1").tasks[0].needs == ["live_web_data"]


# --- web --------------------------------------------------------------------------


def _panel(warnings: tuple[fz.FeasibilityWarning, ...], can_start: bool = True) -> str:
    run = RunPanel(can_start=can_start, needs_confirm=False, feasibility=warnings)
    return _run_panel("p1", run, ActionForms("t" * 43), "fa")


def test_panel_shows_persian_warning_and_escapes_titles() -> None:
    html = _panel((fz.FeasibilityWarning("live_web_data", (EVIL, "B")),))
    assert 'class="missing-needs feas"' in html
    assert t("run.feas.heading", "fa") in html and "CSV" in html
    assert EVIL not in html and "&lt;script&gt;" in html
    assert "، B" in html
    assert html.index("feas") < html.index('action="/projects/p1/start"')


def test_panel_without_warnings_or_start_has_none() -> None:
    assert "feas" not in _panel(())
    assert "feas" not in _panel((fz.FeasibilityWarning("gui", ("A",)),), can_start=False)


def test_run_panel_default_has_no_feasibility() -> None:
    assert RunPanel(can_start=True, needs_confirm=False).feasibility == ()


def test_project_page_warns_before_start(tmp_path: Path) -> None:
    env = RunsEnv(tmp_path)
    pid = env.project(plan=False)
    env.store.save_plan(pid, _plan(_task("fetch", ["live_web_data", "login"], "گرفتن قیمت‌ها")))
    body = env.get(f"/projects/{pid}").body.decode()
    assert 'class="missing-needs feas"' in body
    assert "گرفتن قیمت‌ها" in body and t("run.feas.login", "en") in body
    assert f'action="/projects/{pid}/start"' in body  # starting is still allowed


def test_project_page_without_needs_has_no_warning(tmp_path: Path) -> None:
    env = RunsEnv(tmp_path)
    pid = env.project()
    assert "missing-needs feas" not in env.get(f"/projects/{pid}").body.decode()


def test_broken_plan_never_breaks_the_page(tmp_path: Path) -> None:
    env = RunsEnv(tmp_path)
    pid = env.project()
    env.store.plan_file(pid).write_text("{broken", encoding="utf-8")
    r = env.get(f"/projects/{pid}")
    assert r.status == 200 and "missing-needs feas" not in r.body.decode()
