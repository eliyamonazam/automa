"""End-to-end tests: ``automa new`` -> ``automa status`` -> ``automa run`` -> ``automa status``.

Only the provider factory is replaced by a single scripted ``FakeProvider`` shared by
every command; everything else is the real stack: config loading, ``LLMGateway`` +
``BudgetGuard`` + ``CostLedger``, ``ExecutiveAgent``, ``Orchestrator``, ``WorkerAgent``,
``VerifierAgent``, role registry, ``ToolExecutor`` + ``PermissionPolicy``, ``run_command``
and ``ProjectStore``. No network, no real model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.budget import CostLedger
from automa.config import AutomaConfig
from automa.llm import (
    ChatRequest,
    ChatResponse,
    FakeProvider,
    FinishReason,
    Message,
    ProviderError,
    ToolCall,
    Usage,
)
from automa.llm import Role as MsgRole
from automa.plan import TaskStatus
from automa.project import ProjectStatus, ProjectStore

PID = "greet"
GOAL = "Build a tiny greeting CLI with a design note and a README"

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
input_per_mtok = 2.0
output_per_mtok = 4.0
[prices."fake-large"]
input_per_mtok = 4.0
output_per_mtok = 8.0
[limits]
project_usd = {project}
daily_usd = 5.0
per_call_usd = {per_call}
"""

QUESTION = "Which greeting should the CLI print?"
ANSWER = "Hello, Automa!"
GREET_PY = "print('Hello, Automa!')\n"
SUMMARY_MARK = "WORKER-SUMMARY-SHOULD-NOT-REACH-VERIFIER"

PLAN: dict[str, Any] = {
    "goal": GOAL,
    "tasks": [
        {
            "id": "design",
            "title": "Write the design note",
            "role": "architect",
            "acceptance_criteria": [
                {
                    "description": "design.md exists",
                    "check": {"type": "file_exists", "path": "design.md"},
                }
            ],
        },
        {
            "id": "build",
            "title": "Implement greet.py",
            "role": "developer",
            "depends_on": ["design"],
            "acceptance_criteria": [
                {
                    "description": "greet.py runs",
                    "check": {"type": "command", "command": "python3 greet.py"},
                },
                {
                    "description": "greet.py prints the greeting",
                    "check": {"type": "file_contains", "path": "greet.py", "text": "Hello"},
                },
                {"description": "the code is short and readable"},
            ],
        },
        {
            "id": "docs",
            "title": "Write the README",
            "role": "writer",
            "depends_on": ["build"],
            "acceptance_criteria": [
                {
                    "description": "README explains usage",
                    "check": {"type": "file_contains", "path": "README.md", "text": "greet.py"},
                }
            ],
        },
    ],
}


def calls(*items: tuple[str, dict[str, Any]]) -> ChatResponse:
    tcs = tuple(
        ToolCall(id=f"c{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(items)
    )
    return ChatResponse(
        model="fake",
        message=Message(role=MsgRole.ASSISTANT, content=None, tool_calls=tcs),
        usage=Usage(input_tokens=200, output_tokens=40),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def judgment(passed: bool, reason: str = "") -> str:
    """Verifier reply for the one model-evaluated criterion of ``build`` (index 2)."""
    return json.dumps({"criteria": [{"criterion": 2, "passed": passed, "reason": reason}]})


def design_attempt() -> list[Any]:
    return [calls(("write_file", {"path": "design.md", "content": "# Design\nOne file."})), "ok"]


def build_attempt(code: str = GREET_PY) -> list[Any]:
    return [
        calls(("write_file", {"path": "greet.py", "content": code})),
        calls(("run_command", {"argv": ["python3", "greet.py"]})),
        f"implemented greet.py {SUMMARY_MARK}",
    ]


def docs_attempt() -> list[Any]:
    readme = "# Greet\n\nRun `python3 greet.py`.\n"
    return [calls(("write_file", {"path": "README.md", "content": readme})), "wrote README"]


def text_of(request: ChatRequest) -> str:
    return "\n".join(m.content or "" for m in request.messages)


class E2E:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.ws = tmp_path / "ws"
        self.config = tmp_path / "automa.toml"
        self.write_config()
        self.provider = FakeProvider()
        monkeypatch.chdir(tmp_path)  # no stray .env / automa.toml from the repo
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)

        def factory(config: AutomaConfig) -> FakeProvider:
            return self.provider

        monkeypatch.setattr(cli, "make_provider", factory)

    def write_config(self, *, project: float = 5.0) -> None:
        self.config.write_text(
            TOML.format(ws=self.ws.as_posix(), project=project, per_call=min(0.25, project)),
            encoding="utf-8",
        )

    def cli(self, *args: str, input: str | None = None) -> Any:
        return CliRunner().invoke(cli.app, [*args, "--config", str(self.config)], input=input)

    def script(self, *items: Any) -> None:
        self.provider.push(*items)

    @property
    def store(self) -> ProjectStore:
        return ProjectStore.for_workspace(self.ws)

    def status_json(self) -> dict[str, Any]:
        r = self.cli("status", PID, "--json")
        assert r.exit_code == 0, r.output
        data: dict[str, Any] = json.loads(r.output)
        return data

    def ledger_spent_nusd(self) -> int:
        with CostLedger(self.ws / cli.LEDGER_FILENAME) as ledger:
            return ledger.spent_nusd(project_id=PID)

    def event_types(self) -> list[str]:
        return [e.type for e in self.store.events(PID).read()]

    def new_project(self) -> None:
        self.script(json.dumps({"questions": [QUESTION]}), json.dumps(PLAN))
        r = self.cli("new", GOAL, "--id", PID, input=ANSWER + "\n")
        assert r.exit_code == cli.EXIT_OK, r.output
        assert f"project: {PID}" in r.output


@pytest.fixture
def e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> E2E:
    return E2E(tmp_path, monkeypatch)


def test_full_lifecycle_new_status_run_status(e2e: E2E) -> None:
    # -- new: clarifying question answered on stdin, plan written ---------------------
    e2e.new_project()
    before = e2e.status_json()
    assert before["goal"] == GOAL
    assert before["clarifications"] == [{"question": QUESTION, "answer": ANSWER}]
    assert [t["id"] for t in before["plan"]["tasks"]] == ["design", "build", "docs"]
    assert before["status"] != ProjectStatus.DONE.value
    planning_requests = len(e2e.provider.requests)
    assert planning_requests == 2
    # the user's answer reached the planning call
    assert ANSWER in text_of(e2e.provider.requests[1])
    assert before["cost_usd"] > 0

    # -- run: dependency order, real tools, command check, model evaluation -----------
    e2e.script(*design_attempt(), *build_attempt(), judgment(True), *docs_attempt())
    r = e2e.cli("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "status: done (completed)" in r.output
    assert "attempts this run: 3" in r.output
    assert e2e.provider.remaining == 0

    files = e2e.store.files_dir(PID)
    assert (files / "design.md").is_file()
    assert (files / "greet.py").read_text(encoding="utf-8") == GREET_PY
    assert "greet.py" in (files / "README.md").read_text(encoding="utf-8")
    states = e2e.store.task_states(PID)
    assert {tid: s.status for tid, s in states.items()} == {
        "design": TaskStatus.PASSED,
        "build": TaskStatus.PASSED,
        "docs": TaskStatus.PASSED,
    }

    # roles map to tiers: architect/developer/writer use their configured models
    run_requests = e2e.provider.requests[planning_requests:]
    assert all(req.model.startswith("fake-") for req in run_requests)
    # exactly one model evaluation (only `build` has a criterion without a check),
    # and it never sees the worker's own summary
    evals = [req for req in run_requests if '"criteria"' in text_of(req)]
    assert len(evals) == 1
    assert SUMMARY_MARK not in text_of(evals[0])
    assert "Hello, Automa!" in text_of(evals[0])  # it does see the workspace files

    # -- status after run: consistent with the ledger and the event log ----------------
    after = e2e.status_json()
    assert after["status"] == ProjectStatus.DONE.value
    spent = e2e.ledger_spent_nusd()
    assert after["cost_usd"] == pytest.approx(spent / 1e9)
    assert after["cost_usd"] > before["cost_usd"]
    text = e2e.cli("status", PID)
    assert text.exit_code == 0 and "status: done" in text.output
    listing = e2e.cli("status")
    assert listing.exit_code == 0 and PID in listing.output and "done" in listing.output

    types = e2e.event_types()
    assert types, "events.jsonl must record the run"
    assert any(t.startswith("tool.") for t in types)

    # -- running a finished project again is a no-op ------------------------------------
    n = len(e2e.provider.requests)
    again = e2e.cli("run", PID)
    assert again.exit_code == cli.EXIT_OK, again.output
    assert len(e2e.provider.requests) == n
    assert e2e.ledger_spent_nusd() == spent


def test_lifecycle_with_rejection_provider_outage_and_resume(e2e: E2E) -> None:
    e2e.new_project()

    # first run: design passes, build is rejected by the model evaluation and retried
    # with feedback, then the provider goes down during the retry -> paused
    e2e.script(
        *design_attempt(),
        *build_attempt(),
        judgment(False, "add a docstring"),
        ProviderError("upstream down"),
    )
    r = e2e.cli("run", PID)
    assert r.exit_code == cli.EXIT_PROVIDER, r.output
    assert "paused at task build" in r.output and f"automa run {PID}" in r.output
    assert e2e.status_json()["status"] == ProjectStatus.PAUSED.value
    states = e2e.store.task_states(PID)
    assert states["design"].status is TaskStatus.PASSED
    assert states["docs"].status is TaskStatus.PENDING
    assert len(e2e.store.task_history(PID, "build")) == 1  # the rejected attempt is kept

    # resume: design is not redone; build retry carries the verifier's feedback
    mark = len(e2e.provider.requests)
    fixed = '"""Print the greeting."""\n' + GREET_PY
    e2e.script(*build_attempt(fixed), judgment(True), *docs_attempt())
    r = e2e.cli("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "status: done (completed)" in r.output
    assert len(e2e.store.task_history(PID, "design")) == 1
    assert len(e2e.store.task_history(PID, "build")) == 2
    assert "add a docstring" in text_of(e2e.provider.requests[mark])
    assert (e2e.store.files_dir(PID) / "greet.py").read_text(encoding="utf-8") == fixed
    assert e2e.status_json()["status"] == ProjectStatus.DONE.value
    assert e2e.provider.remaining == 0


def test_lifecycle_budget_stop_mid_task_and_resume_after_raising_cap(e2e: E2E) -> None:
    e2e.new_project()
    planned = e2e.ledger_spent_nusd()

    # A project cap that admits the first worker step but not the reservation for the
    # second: the guard stops the run in the middle of task ``design``.
    cap_usd = planned / 1e9 + 0.037
    e2e.write_config(project=cap_usd)
    e2e.script(*design_attempt(), *build_attempt(), judgment(True), *docs_attempt())
    r = e2e.cli("run", PID)
    assert r.exit_code == cli.EXIT_BUDGET, r.output
    assert "paused at task design" in r.output and "'project'" in r.output
    assert "raise the cap" in r.output
    assert len(e2e.provider.requests) == 3  # 2 planning calls + 1 worker step
    assert e2e.status_json()["status"] == ProjectStatus.PAUSED.value
    stopped = e2e.ledger_spent_nusd()
    assert planned < stopped <= round(cap_usd * 1e9)  # spend never passes the cap
    assert e2e.store.task_states(PID)["design"].status is not TaskStatus.PASSED

    # Same cap again: still refused before any provider call (the stop is not lifted
    # by simply re-running).
    mark = len(e2e.provider.requests)
    r = e2e.cli("run", PID)
    assert r.exit_code == cli.EXIT_BUDGET, r.output
    assert len(e2e.provider.requests) == mark
    assert e2e.ledger_spent_nusd() == stopped

    # Raise the cap and resume with a fresh script: the project completes.
    e2e.write_config(project=5.0)
    e2e.provider = FakeProvider()
    e2e.script(*design_attempt(), *build_attempt(), judgment(True), *docs_attempt())
    r = e2e.cli("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert e2e.status_json()["status"] == ProjectStatus.DONE.value
    assert e2e.provider.remaining == 0
    assert e2e.ledger_spent_nusd() > stopped


def test_new_and_run_never_reach_the_real_provider_without_key(
    e2e: E2E, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automa.llm.factory import provider_from_config

    monkeypatch.setattr(cli, "make_provider", provider_from_config)
    r = e2e.cli("new", GOAL, "--id", PID, "--no-ask")
    assert r.exit_code == cli.EXIT_CONFIG
    assert e2e.store.list_ids() == []
    assert not (e2e.ws / cli.LEDGER_FILENAME).exists()
