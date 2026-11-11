"""Tests for the Approval Gate CLI: ``automa approvals``, ``approve``, ``reject``, and
pending approvals in ``automa status`` / ``automa run``.

Local only: the provider factory is replaced by a ``FakeProvider``; the real
``LLMGateway`` + ``BudgetGuard``, ``Orchestrator``, agents, tool executor,
``ApprovalStore`` and ``ProjectStore`` are used. No network, no real model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.config import AutomaConfig
from automa.llm import ChatResponse, FakeProvider, FinishReason, Message, ToolCall, Usage
from automa.llm import Role as MsgRole
from automa.plan import Plan, TaskStatus
from automa.project import ProjectStatus, ProjectStore
from automa.roles import Role, RoleRegistry
from automa.tools import Tool, ToolArgs, Workspace, default_tools
from automa.tools.approvals import ApprovalRequest, ApprovalStatus, ApprovalStore
from automa.tools.base import RiskLevel

PID = "proj-1"

TOML = """
workspace = "{ws}"
[tiers]
small = "fake-small"
medium = "fake-medium"
large = "fake-large"
[prices."fake-small"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[prices."fake-medium"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[prices."fake-large"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[limits]
project_usd = 5.0
daily_usd = 2.0
per_call_usd = 0.25
"""


class _PubArgs(ToolArgs):
    text: str = ""


class _Publish(Tool):
    name: ClassVar[str] = "publish"
    description: ClassVar[str] = "Publish to the internet."
    risk: ClassVar[RiskLevel] = RiskLevel.EXTERNAL
    args_model: ClassVar[type[ToolArgs]] = _PubArgs
    runs: ClassVar[list[str]] = []

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        assert isinstance(args, _PubArgs)
        _Publish.runs.append(args.text)
        return "published"


def _pub_tools() -> list[Tool]:
    return [*default_tools(), _Publish()]


class _PubRegistry:
    """Stands in for ``RoleRegistry`` in the CLI: one role that may ``publish``."""

    @staticmethod
    def builtin(*, tools: Any) -> RoleRegistry:
        publisher = Role(
            name="publisher",
            description="Writes and publishes.",
            tools=("write_file", "publish"),
            max_risk=RiskLevel.WRITE,
            instructions="Publish the text, then write the file.",
        )
        return RoleRegistry([publisher], tools=_pub_tools())


def calls(*items: tuple[str, dict[str, Any]]) -> ChatResponse:
    tcs = tuple(
        ToolCall(id=f"x{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(items)
    )
    return ChatResponse(
        model="fake",
        message=Message(role=MsgRole.ASSISTANT, content=None, tool_calls=tcs),
        usage=Usage(input_tokens=100, output_tokens=20),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def publish(text: str = "hello world") -> ChatResponse:
    return calls(("publish", {"text": text}))


PASS_ALL = json.dumps({"criteria": []})


class Env:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.ws = tmp_path / "ws"
        self.config = tmp_path / "automa.toml"
        self.config.write_text(TOML.format(ws=self.ws.as_posix()), encoding="utf-8")
        self.provider = FakeProvider()
        self.factory_calls = 0
        monkeypatch.chdir(tmp_path)
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)

        def factory(config: AutomaConfig) -> FakeProvider:
            self.factory_calls += 1
            return self.provider

        monkeypatch.setattr(cli, "make_provider", factory)
        _Publish.runs = []

    @property
    def store(self) -> ProjectStore:
        return ProjectStore.for_workspace(self.ws)

    @property
    def approvals(self) -> ApprovalStore:
        return self.store.approvals(PID)

    def project(self, pid: str = PID) -> None:
        self.store.create("test goal", project_id=pid)

    def request(self, text: str = "hello world", *, pid: str = PID, task: str = "a") -> Any:
        return self.store.approvals(pid).request(
            tool="publish",
            risk=RiskLevel.EXTERNAL,
            arguments={"text": text},
            reason="external tool needs approval",
            task_id=task,
        )

    def cli(self, *args: str, input: str | None = None) -> Any:
        return CliRunner().invoke(cli.app, [*args, "--config", str(self.config)], input=input)

    def tree(self) -> dict[str, bytes]:
        return {
            str(p.relative_to(self.ws)): p.read_bytes() for p in self.ws.rglob("*") if p.is_file()
        }


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


# -- automa approvals ----------------------------------------------------------------


def test_approvals_lists_pending_with_exact_arguments(env: Env) -> None:
    env.project()
    req = env.request("release notes v1")
    r = env.cli("approvals", PID)
    assert r.exit_code == 0, r.output
    assert req.id in r.output
    assert "[pending]" in r.output
    assert "tool: publish (risk: external)" in r.output
    assert "task: a" in r.output
    assert '"text": "release notes v1"' in r.output
    assert f"automa approve {PID} {req.id}" in r.output
    assert f"automa reject {PID} {req.id}" in r.output
    assert env.factory_calls == 0


def test_approvals_is_read_only_and_needs_no_key(env: Env) -> None:
    env.project()
    env.request()
    before = env.tree()
    assert env.cli("approvals").exit_code == 0
    assert env.cli("approvals", PID, "--all", "--json").exit_code == 0
    assert env.tree() == before
    assert env.factory_calls == 0


def test_approvals_without_any(env: Env) -> None:
    r = env.cli("approvals")
    assert r.exit_code == 0
    assert "no pending approval requests" in r.output
    env.project()
    r = env.cli("approvals", PID, "--all")
    assert "no approval requests" in r.output
    assert not (env.ws / "projects" / PID / "approvals").exists()


def test_approvals_across_projects(env: Env) -> None:
    env.project()
    env.project("other")
    a = env.request("one")
    b = env.request("two", pid="other")
    r = env.cli("approvals")
    assert r.exit_code == 0
    assert a.id in r.output and b.id in r.output
    assert f"automa approve other {b.id}" in r.output


def test_approvals_hides_decided_unless_all(env: Env) -> None:
    env.project()
    done = env.request("old")
    env.approvals.decide(done.id, approve=False, note="no")
    pending = env.request("new")
    r = env.cli("approvals", PID)
    assert pending.id in r.output and done.id not in r.output
    r = env.cli("approvals", PID, "--all")
    assert done.id in r.output and "[rejected]" in r.output and "note: no" in r.output


def test_approvals_shows_used_decisions(env: Env) -> None:
    env.project()
    req = env.request()
    env.approvals.decide(req.id, approve=True)
    env.approvals.consume(req.id)
    r = env.cli("approvals", PID, "--all")
    assert "[approved, used]" in r.output
    rows = json.loads(env.cli("approvals", PID, "--all", "--json").output)
    assert rows[0]["used"] is True and rows[0]["status"] == "approved"


def test_approvals_json(env: Env) -> None:
    env.project()
    req = env.request("x")
    r = env.cli("approvals", PID, "--json")
    assert r.exit_code == 0
    rows = json.loads(r.output)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == req.id
    assert row["arguments"] == {"text": "x"}
    assert row["status"] == "pending" and row["used"] is False
    assert row["fingerprint"] == req.fingerprint
    assert Path(row["path"]).is_file()


def test_approvals_truncates_long_arguments_but_points_to_the_file(env: Env) -> None:
    env.project()
    req = env.request("y" * (cli.ARGUMENTS_PREVIEW_CHARS * 2))
    r = env.cli("approvals", PID)
    assert "truncated" in r.output
    assert f"{req.id}.json" in r.output
    assert len(r.output) < cli.ARGUMENTS_PREVIEW_CHARS * 2


@pytest.mark.parametrize("bad", ["../x", "A B"])
def test_approvals_unknown_or_unsafe_project(env: Env, bad: str) -> None:
    assert env.cli("approvals", "nope").exit_code == cli.EXIT_USER
    assert env.cli("approvals", bad).exit_code == cli.EXIT_USER


def test_approvals_corrupt_file_fails_closed(env: Env) -> None:
    env.project()
    req = env.request()
    (env.approvals.directory / f"{req.id}.json").write_text("{", encoding="utf-8")
    r = env.cli("approvals", PID)
    assert r.exit_code == cli.EXIT_USER
    assert "unreadable approval file" in r.output
    assert env.cli("approvals").exit_code == cli.EXIT_USER


@pytest.mark.parametrize(
    "evil",
    [
        "\x1b[2K\x1b[1Ahidden",  # ANSI: erase line, cursor up
        "safe‮evil",  # right-to-left override
        "a\rb",  # carriage return overwrites the line
        "\x9b31m",  # C1 control sequence introducer
    ],
)
def test_control_characters_are_escaped(env: Env, evil: str) -> None:
    env.project()
    env.request(evil)
    for out in (env.cli("approvals", PID).output, env.cli("status", PID).output):
        for ch in ("\x1b", "‮", "\r", "\x9b"):
            assert ch not in out


def test_safe_keeps_persian_text() -> None:
    text = "می‌خواهم منتشر کنم\n\tok"  # contains a zero-width non-joiner
    assert cli._safe(text) == text
    assert cli._safe("\x1b[0m") == "\\x1b[0m"
    assert cli._safe("‮") == "\\u202e"


# -- automa approve / reject ---------------------------------------------------------


def test_approve_shows_request_and_asks(env: Env) -> None:
    env.project()
    req = env.request("ship it")
    r = env.cli("approve", PID, req.id, input="y\n")
    assert r.exit_code == 0, r.output
    assert '"text": "ship it"' in r.output
    assert "exactly these arguments" in r.output
    assert f"{req.id}: approved" in r.output
    decided = env.approvals.get(req.id)
    assert decided.status is ApprovalStatus.APPROVED
    assert not env.approvals.is_used(req.id)
    assert env.factory_calls == 0


@pytest.mark.parametrize("answer", ["n\n", "\n", ""])
def test_approve_declined_records_nothing(env: Env, answer: str) -> None:
    env.project()
    req = env.request()
    before = env.tree()
    r = env.cli("approve", PID, req.id, input=answer)
    assert r.exit_code == cli.EXIT_USER
    assert env.approvals.get(req.id).status is ApprovalStatus.PENDING
    assert env.tree() == before


def test_approve_yes_with_note(env: Env) -> None:
    env.project()
    req = env.request()
    r = env.cli("approve", PID, req.id, "--yes", "--note", "  fine once  ")
    assert r.exit_code == 0, r.output
    decided = env.approvals.get(req.id)
    assert decided.status is ApprovalStatus.APPROVED and decided.note == "fine once"
    events = [e for e in env.store.events(PID).read() if e.type == "approval.decided"]
    assert len(events) == 1
    assert events[0].data == {
        "approval": req.id,
        "task": "a",
        "tool": "publish",
        "status": "approved",
        "note": True,
    }


def test_reject_needs_no_confirmation(env: Env) -> None:
    env.project()
    req = env.request()
    r = env.cli("reject", PID, req.id, "--note", "not public yet")
    assert r.exit_code == 0, r.output
    decided = env.approvals.get(req.id)
    assert decided.status is ApprovalStatus.REJECTED and decided.note == "not public yet"
    assert "rejected" in r.output


def test_blank_note_is_no_note(env: Env) -> None:
    env.project()
    req = env.request()
    assert env.cli("reject", PID, req.id, "--note", "   ").exit_code == 0
    assert env.approvals.get(req.id).note is None


def test_decision_is_made_only_once(env: Env) -> None:
    env.project()
    req = env.request()
    assert env.cli("approve", PID, req.id, "-y").exit_code == 0
    for args in (("approve", PID, req.id, "-y"), ("reject", PID, req.id)):
        r = env.cli(*args)
        assert r.exit_code == cli.EXIT_USER
        assert "already approved" in r.output
    assert env.approvals.get(req.id).status is ApprovalStatus.APPROVED


def test_too_long_note_is_refused(env: Env) -> None:
    env.project()
    req = env.request()
    r = env.cli("reject", PID, req.id, "--note", "x" * 2001)
    assert r.exit_code == cli.EXIT_USER
    assert env.approvals.get(req.id).status is ApprovalStatus.PENDING


@pytest.mark.parametrize("bad", ["apr-000000000000", "apr-xyz", "../../etc", ""])
def test_unknown_or_invalid_request_id(env: Env, bad: str) -> None:
    env.project()
    env.request()
    for cmd in ("approve", "reject"):
        assert env.cli(cmd, PID, bad, "--note", "n").exit_code == cli.EXIT_USER


def test_request_of_another_project_cannot_be_decided_here(env: Env) -> None:
    env.project()
    env.project("other")
    req = env.request(pid="other")
    assert env.cli("approve", PID, req.id, "-y").exit_code == cli.EXIT_USER
    assert env.cli("reject", PID, req.id).exit_code == cli.EXIT_USER
    assert env.store.approvals("other").get(req.id).status is ApprovalStatus.PENDING


def test_approve_unknown_project(env: Env) -> None:
    assert env.cli("approve", "nope", "apr-000000000000", "-y").exit_code == cli.EXIT_USER


def test_tampered_request_cannot_be_approved(env: Env) -> None:
    env.project()
    req = env.request("harmless")
    path = env.approvals.directory / f"{req.id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["arguments"] = {"text": "something else"}  # fingerprint no longer matches
    path.write_text(json.dumps(data), encoding="utf-8")
    r = env.cli("approve", PID, req.id, "-y")
    assert r.exit_code == cli.EXIT_USER
    assert "approved" not in path.read_text(encoding="utf-8")


# -- automa status --------------------------------------------------------------------


def test_status_shows_pending_approvals(env: Env) -> None:
    env.project()
    req = env.request()
    r = env.cli("status", PID)
    assert r.exit_code == 0
    assert "pending approvals: 1" in r.output and req.id in r.output
    assert f"automa approvals {PID}" in r.output
    data = json.loads(env.cli("status", PID, "--json").output)
    assert [a["id"] for a in data["pending_approvals"]] == [req.id]
    listing = env.cli("status")
    assert "[1 approval(s) pending]" in listing.output
    rows = json.loads(env.cli("status", "--json").output)
    assert rows[0]["pending_approvals"] == 1


def test_status_without_approvals(env: Env) -> None:
    env.project()
    r = env.cli("status", PID)
    assert "pending approvals" not in r.output
    assert json.loads(env.cli("status", PID, "--json").output)["pending_approvals"] == []
    assert not (env.ws / "projects" / PID / "approvals").exists()


def test_status_corrupt_approval_fails_closed(env: Env) -> None:
    env.project()
    req = env.request()
    (env.approvals.directory / f"{req.id}.json").write_text("{}", encoding="utf-8")
    assert env.cli("status", PID).exit_code == cli.EXIT_USER


# -- end to end through ``automa run`` ------------------------------------------------


@pytest.fixture
def pub(env: Env, monkeypatch: pytest.MonkeyPatch) -> Env:
    monkeypatch.setattr(cli, "RoleRegistry", _PubRegistry)
    monkeypatch.setattr(cli, "default_tools", _pub_tools)
    monkeypatch.setattr(cli, "tools_factory", lambda sandbox=None: _pub_tools)
    env.project()
    plan = {
        "goal": "test goal",
        "tasks": [
            {
                "id": "a",
                "title": "Publish",
                "role": "publisher",
                "acceptance_criteria": [
                    {
                        "description": "a.txt says ok",
                        "check": {"type": "file_contains", "path": "a.txt", "text": "ok"},
                    }
                ],
            }
        ],
    }
    env.store.save_plan(PID, Plan.model_validate(plan))
    return env


def _pending(env: Env) -> ApprovalRequest:
    (req,) = env.approvals.list(ApprovalStatus.PENDING)
    return req


def test_run_pauses_and_shows_how_to_decide(pub: Env) -> None:
    pub.provider.push(publish())
    r = pub.cli("run", PID)
    assert r.exit_code == cli.EXIT_PAUSED, r.output
    req = _pending(pub)
    assert f"approval needed: {req.id}" in r.output
    assert f"automa approve {PID} {req.id}" in r.output
    assert f"automa reject {PID} {req.id}" in r.output
    assert "not available yet" not in r.output
    assert _Publish.runs == []


def test_approve_then_run_executes_exactly_once(pub: Env) -> None:
    pub.provider.push(publish())
    assert pub.cli("run", PID).exit_code == cli.EXIT_PAUSED
    req = _pending(pub)
    assert pub.cli("approve", PID, req.id, input="y\n").exit_code == 0
    pub.provider.push(
        publish(),
        calls(("write_file", {"path": "a.txt", "content": "ok"})),
        "done",
    )
    r = pub.cli("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert _Publish.runs == ["hello world"]
    assert pub.approvals.is_used(req.id)
    assert pub.store.load(PID).status is ProjectStatus.DONE
    assert pub.store.task_states(PID)["a"].status is TaskStatus.PASSED
    assert "[approved, used]" in pub.cli("approvals", PID, "--all").output
    assert "pending approvals" not in pub.cli("status", PID).output


def test_reject_then_run_tells_the_model(pub: Env) -> None:
    pub.provider.push(publish())
    assert pub.cli("run", PID).exit_code == cli.EXIT_PAUSED
    req = _pending(pub)
    assert pub.cli("reject", PID, req.id, "--note", "not public yet").exit_code == 0
    pub.provider.push(
        publish(),
        calls(("write_file", {"path": "a.txt", "content": "ok"})),
        "done",
    )
    r = pub.cli("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert _Publish.runs == []
    seen = json.dumps([m.model_dump(mode="json") for m in pub.provider.requests[-2].messages])
    assert "not public yet" in seen


def test_declined_confirmation_keeps_the_project_paused(pub: Env) -> None:
    pub.provider.push(publish())
    assert pub.cli("run", PID).exit_code == cli.EXIT_PAUSED
    req = _pending(pub)
    assert pub.cli("approve", PID, req.id, input="n\n").exit_code == cli.EXIT_USER
    pub.provider.push(publish())
    r = pub.cli("run", PID)
    assert r.exit_code == cli.EXIT_PAUSED
    assert _pending(pub).id == req.id  # the same request is asked again, nothing ran
    assert _Publish.runs == []
