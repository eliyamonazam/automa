"""Tests for questions to the user (automa.userinput, tools/ask.py, D159).

The real orchestrator, worker and executor with a scripted FakeProvider (Rig);
no network, no real model calls.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from automa import cli
from automa.orchestrator import StopReason
from automa.plan import TaskStatus
from automa.project import ProjectStatus
from automa.tools.ask import ASK_TOOL, AskUser, UserInputRequired
from automa.userinput import (
    MAX_ANSWER_FILES,
    MAX_REQUESTS_PER_TASK,
    RequestStatus,
    UserRequestError,
    UserRequestNotFound,
    UserRequestStore,
    question_key,
    safe_file_name,
)
from test_delivery import Env as CliEnv
from test_orchestrator import PID, Rig, calls, make_plan, task_dict, write

T0 = datetime(2026, 11, 22, 9, 0, tzinfo=UTC)


def ask(question: str = "Please upload the sales sheet", kind: str = "file") -> object:
    return calls((ASK_TOOL, {"question": question, "why": "to build the chart", "kind": kind}))


def requests_of(rig: Rig) -> UserRequestStore:
    return rig.store.user_requests(PID)


# --- the store --------------------------------------------------------------------


def test_store_request_answer_decline(tmp_path: Path) -> None:
    ticks = iter(range(100))
    store = UserRequestStore(
        tmp_path / "req",
        "p1",
        tmp_path / "files",
        clock=lambda: T0.replace(minute=next(ticks)),
    )
    r1 = store.request("a", "Upload the file?", why="needed", kind="file")
    assert store.request("a", "  upload THE file? ") == r1  # same pending question
    assert store.pending() == [r1] and store.find("a", "upload the file?") == r1
    done = store.answer(
        r1.id, "here\r\nit is", [("../../evil.csv", b"1,2"), ("evil.csv", b"3")], via="test"
    )
    assert done.status is RequestStatus.ANSWERED and done.answer == "here\nit is"
    assert done.files == (f"inputs/{r1.id}/evil.csv", f"inputs/{r1.id}/evil-2.csv")
    assert (tmp_path / "files" / "inputs" / r1.id / "evil.csv").read_bytes() == b"1,2"
    assert not (tmp_path / "evil.csv").exists()
    with pytest.raises(UserRequestError, match="already answered"):
        store.answer(r1.id, "again", via="test")
    with pytest.raises(UserRequestError, match="already answered"):
        store.decline(r1.id, via="test")
    r2 = store.request("a", "Which colour?")
    assert r2.id != r1.id
    declined = store.decline(r2.id, "no idea", via="test")
    assert declined.status is RequestStatus.DECLINED
    assert "cannot provide this" in declined.as_text() and "never invent" in declined.as_text()
    assert [r.id for r in store.for_task("a")] == [r1.id, r2.id]
    assert store.pending() == []


def test_store_refusals(tmp_path: Path) -> None:
    store = UserRequestStore(tmp_path / "req", "p1", tmp_path / "files")
    r = store.request("a", "q")
    with pytest.raises(UserRequestError, match="needs text or"):
        store.answer(r.id, "   ", via="t")
    with pytest.raises(UserRequestError, match="at most"):
        store.answer(r.id, "x", [("f", b"")] * (MAX_ANSWER_FILES + 1), via="t")
    with pytest.raises(UserRequestNotFound):
        store.get("req-nothex")
    with pytest.raises(UserRequestNotFound):
        store.get("req-000000000000")
    with pytest.raises(UserRequestError, match="longer"):
        store.answer(r.id, "x" * 20_001, via="t")
    assert store.get(r.id).status is RequestStatus.PENDING


def test_store_rejects_symlinked_directory(tmp_path: Path) -> None:
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", tmp_path / "req")
    store = UserRequestStore(tmp_path / "req", "p1", tmp_path / "files")
    with pytest.raises(UserRequestError, match="symlink"):
        store.request("a", "q")
    with pytest.raises(UserRequestError, match="symlink"):
        store.list()


def test_store_refuses_upload_through_symlinked_inputs(tmp_path: Path) -> None:
    files = tmp_path / "files"
    files.mkdir()
    (tmp_path / "out").mkdir()
    os.symlink(tmp_path / "out", files / "inputs")
    store = UserRequestStore(tmp_path / "req", "p1", files)
    r = store.request("a", "q", kind="file")
    with pytest.raises(UserRequestError, match="could not save"):
        store.answer(r.id, "", [("x.txt", b"x")], via="t")
    assert list((tmp_path / "out").iterdir()) == []
    assert store.get(r.id).status is RequestStatus.PENDING


def test_helpers() -> None:
    assert question_key("  Hello\tWORLD ") == "hello world"
    assert safe_file_name("../a/b/c.txt") == "c.txt"
    assert safe_file_name("..") == "file"
    assert safe_file_name("a<>|.csv") == "a_.csv"
    assert safe_file_name("گزارش فروش.xlsx") == "گزارش فروش.xlsx"
    assert len(safe_file_name("x" * 300 + ".txt")) <= 120


# --- the tool ----------------------------------------------------------------------


def test_ask_tool_flow(tmp_path: Path) -> None:
    store = UserRequestStore(tmp_path / "req", "p1", tmp_path / "files")
    tool = AskUser(store, "a", max_requests=2)
    args = tool.args_model.model_validate({"question": "q1"})
    with pytest.raises(UserInputRequired) as e:
        tool.ask(args)
    rid = e.value.request_id
    with pytest.raises(UserInputRequired):  # still pending: same request
        tool.ask(args)
    assert len(store.list()) == 1
    store.answer(rid, "yes", via="t")
    assert "A: yes" in tool.ask(args)
    assert [r.id for r in tool.answered()] == [rid]
    with pytest.raises(UserInputRequired):
        tool.ask(tool.args_model.model_validate({"question": "q2"}))
    from automa.tools.base import ToolError

    with pytest.raises(ToolError, match="question limit"):
        tool.ask(tool.args_model.model_validate({"question": "q3"}))
    with pytest.raises(TypeError):
        AskUser("nope", "a")  # type: ignore[arg-type]


# --- with the orchestrator ------------------------------------------------------------


def test_question_pauses_answer_resumes_and_reaches_the_worker(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [ask()], make_plan(task_dict("a")))
    report = rig.run()
    assert report.status is ProjectStatus.PAUSED
    assert report.reason is StopReason.USER_INPUT
    assert report.detail == "Please upload the sales sheet"
    pending = requests_of(rig).pending()
    assert len(pending) == 1 and report.request_id == pending[0].id
    assert pending[0].kind == "file" and pending[0].why == "to build the chart"
    assert rig.store.attempt_count(PID, "a") == 0  # nothing stored for the attempt
    assert rig.state("a").status is TaskStatus.PENDING
    paused = [e for e in rig.store.events(PID).read() if e.type == "orchestrator.paused"]
    assert paused[-1].data["request"] == report.request_id

    # running again before an answer spends nothing
    n = len(rig.provider.requests)
    again = rig.run()
    assert again.reason is StopReason.USER_INPUT and len(rig.provider.requests) == n

    requests_of(rig).answer(pending[0].id, "sheet attached", [("sales.csv", b"q1,10")], via="test")
    rig.provider.push(*write("a"))
    done = rig.run()
    assert done.status is ProjectStatus.DONE
    first = rig.provider.requests[n]
    # D166: the same attempt continues; the answer is the ask_user call's result.
    assert [m.role.value for m in first.messages[:3]] == ["system", "user", "assistant"]
    assert first.messages[2].tool_calls[0].name == ASK_TOOL
    answer = first.messages[-1].content or ""
    assert first.messages[-1].role.value == "tool"
    assert "sheet attached" in answer and f"inputs/{pending[0].id}/sales.csv" in answer
    assert ASK_TOOL in {t.name for t in first.tools}
    assert "Never invent data" in (first.messages[0].content or "")
    assert not (rig.store.project_dir(PID) / "suspended" / "a.json").exists()
    resumed = [e for e in rig.store.events(PID).read() if e.type == "orchestrator.resumed_attempt"]
    assert resumed and resumed[0].data["request"] == pending[0].id
    assert rig.store.attempt_count(PID, "a") == 1  # one attempt, not two


def test_same_question_after_answer_is_answered_without_pause(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [ask("Which city?", "text")], make_plan(task_dict("a")))
    rig.run()
    rid = requests_of(rig).pending()[0].id
    requests_of(rig).answer(rid, "Tehran", via="test")
    rig.provider.push(ask("which  city?", "text"), *write("a"))
    assert rig.run().status is ProjectStatus.DONE
    assert len(requests_of(rig).list()) == 1  # not asked twice


def test_declined_question_continues_without_it(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [ask("Your API key?", "text")], make_plan(task_dict("a")))
    rig.run()
    rid = requests_of(rig).pending()[0].id
    requests_of(rig).decline(rid, "I have none", via="test")
    rig.provider.push(ask("Your API key?", "text"), *write("a"))
    assert rig.run().status is ProjectStatus.DONE
    texts = [m.content or "" for r in rig.provider.requests for m in r.messages]
    assert any("cannot provide this" in t and "I have none" in t for t in texts)


def test_invalid_ask_arguments_are_a_tool_error(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        [calls((ASK_TOOL, {"question": ""})), *write("a")],
        make_plan(task_dict("a")),
    )
    assert rig.run().status is ProjectStatus.DONE
    assert requests_of(rig).list() == []


def test_question_limit_per_task(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [], make_plan(task_dict("a")))
    store = requests_of(rig)
    for i in range(MAX_REQUESTS_PER_TASK):
        store.answer(store.request("a", f"q{i}").id, "x", via="t")
    rig.provider.push(ask("one more?", "text"), *write("a"))
    assert rig.run().status is ProjectStatus.DONE
    assert len(store.list()) == MAX_REQUESTS_PER_TASK


# --- CLI ----------------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CliEnv:
    return CliEnv(tmp_path, monkeypatch)


def test_cli_question_answer_with_file_and_run(env: CliEnv, tmp_path: Path) -> None:
    env.store.create("chart", project_id="p1")
    env.store.save_plan("p1", make_plan(task_dict("a")))
    env.provider.push(ask())
    r = env.run("run", "p1", "--yes")
    assert r.exit_code == cli.EXIT_NEEDS_INPUT, r.output
    rid = env.store.user_requests("p1").pending()[0].id
    assert f"automa needs something from you ({rid}): Please upload the sales sheet" in r.output
    assert f"automa answer p1 {rid}" in r.output and f"automa decline p1 {rid}" in r.output
    r = env.run("requests", "p1")
    assert r.exit_code == 0 and rid in r.output and "to build the chart" in r.output
    upload = tmp_path / "sales.csv"
    upload.write_text("q1,10", encoding="utf-8")
    r = env.run("answer", "p1", rid, "--text", "here", "--file", str(upload))
    assert r.exit_code == 0, r.output
    assert f"inputs/{rid}/sales.csv" in r.output
    assert env.run("requests", "p1").output.strip() == "no open questions"
    assert "answered" in env.run("requests", "p1", "--all").output
    env.provider.push(*write("a"))
    r = env.run("run", "p1", "--yes")
    assert r.exit_code == 0, r.output


def test_cli_decline_and_refusals(env: CliEnv, tmp_path: Path) -> None:
    env.store.create("g", project_id="p1")
    rid = env.store.user_requests("p1").request("a", "q").id
    assert env.run("answer", "p1", rid).exit_code == cli.EXIT_USER  # no text, no file
    assert env.run("answer", "p1", rid, "--file", str(tmp_path / "nope")).exit_code == cli.EXIT_USER
    assert env.run("answer", "p1", "req-000000000000", "--text", "x").exit_code == cli.EXIT_USER
    assert env.run("answer", "nope", rid, "--text", "x").exit_code == cli.EXIT_USER
    r = env.run("decline", "p1", rid, "--reason", "cannot")
    assert r.exit_code == 0 and "declined" in r.output
    assert env.run("decline", "p1", rid).exit_code == cli.EXIT_USER
    assert env.run("requests", "nope").exit_code == cli.EXIT_USER


# --- continuing the same attempt (D166) ---------------------------------------------


def test_calls_after_the_question_in_the_same_reply_run_after_the_answer(tmp_path: Path) -> None:
    both = calls(
        (ASK_TOOL, {"question": "Name?", "kind": "text"}),
        ("write_file", {"path": "a.txt", "content": "ok"}),
    )
    rig = Rig(tmp_path, [both], make_plan(task_dict("a")))
    rig.run()
    assert not (rig.store.files_dir(PID) / "a.txt").exists()  # not run before the answer
    rid = requests_of(rig).pending()[0].id
    requests_of(rig).answer(rid, "Ali", via="test")
    rig.provider.push("done")
    assert rig.run().status is ProjectStatus.DONE
    assert (rig.store.files_dir(PID) / "a.txt").read_text() == "ok"
    last = (
        rig.provider.requests[-2] if len(rig.provider.requests) > 2 else rig.provider.requests[-1]
    )
    tool_texts = [m.content or "" for m in last.messages if m.role.value == "tool"]
    assert any("Ali" in t for t in tool_texts) and any("wrote" in t for t in tool_texts)


def test_broken_suspended_file_falls_back_to_a_fresh_attempt(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [ask("Which city?", "text")], make_plan(task_dict("a")))
    rig.run()
    path = rig.store.project_dir(PID) / "suspended" / "a.json"
    assert path.is_file()
    path.write_text("{broken", encoding="utf-8")
    rid = requests_of(rig).pending()[0].id
    requests_of(rig).answer(rid, "Tehran", via="test")
    rig.provider.push(*write("a"))
    n = len(rig.provider.requests)
    assert rig.run().status is ProjectStatus.DONE
    user = rig.provider.requests[n].messages[1].content or ""
    assert "Answers from the user" in user and "Tehran" in user  # restarted with answers
    assert not path.exists()


def test_take_suspended_checks_task_and_attempt(tmp_path: Path) -> None:
    from automa.config import Tier
    from automa.llm.types import Message, ToolCall
    from automa.suspend import SuspendedAttempt, has_suspended, save_suspended, take_suspended

    s = SuspendedAttempt(
        task_id="a",
        attempt=2,
        request_id="req-000000000001",
        messages=(Message.system("s"), Message.user("u")),
        calls_left=(ToolCall(id="c", name=ASK_TOOL, arguments="{}"),),
        intact_left=(True,),
        steps=3,
        cost_nusd=5,
        tier=Tier.MEDIUM,
        model="m",
        started_at=T0,
    )
    save_suspended(tmp_path, s)
    assert has_suspended(tmp_path, "a")
    assert take_suspended(tmp_path, "a", 1) is None  # other attempt: dropped
    assert not has_suspended(tmp_path, "a")
    save_suspended(tmp_path, s)
    assert take_suspended(tmp_path, "a", 2) == s
    assert take_suspended(tmp_path, "a", 2) is None  # used once
    with pytest.raises(ValueError):
        take_suspended(tmp_path, "../x", 1)
