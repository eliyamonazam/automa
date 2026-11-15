"""``github_read`` connected (D76): in ``default_tools()``, listed only by the
``researcher`` role, every call still through the Approval Gate, and
``automa approvals``/``approve`` describe GitHub requests.

Local only: ``FakeProvider`` for the model, ``httpx.MockTransport`` and a fake resolver
for GitHub. No network, no real model calls.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from automa import cli
from automa.config import AutomaConfig
from automa.events import MemoryEventSink
from automa.llm import ChatResponse, FakeProvider, FinishReason, Message, ToolCall, Usage
from automa.llm import Role as MsgRole
from automa.plan import Plan, TaskStatus
from automa.project import ProjectStatus, ProjectStore
from automa.roles import RoleRegistry
from automa.tools import (
    ApprovalRequired,
    Decision,
    GitHubRead,
    GitHubReadArgs,
    RiskLevel,
    Tool,
    ToolExecutor,
    default_tools,
    describe_github_request,
    github_api_url,
)
from automa.tools.approvals import ApprovalRequest, ApprovalStatus, ApprovalStore
from automa.tools.workspace import Workspace

PID = "proj-gh"
PUBLIC = "140.82.112.6"

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

FILE_ARGS = {"kind": "file", "owner": "octo", "repo": "hello", "path": "docs/README.md"}
FILE_URL = "https://api.github.com/repos/octo/hello/contents/docs/README.md"


def recorded(arguments: dict[str, Any]) -> dict[str, Any]:
    """Arguments as the executor records them (validated and dumped)."""
    return GitHubReadArgs.model_validate(arguments).model_dump(mode="json")


# ------------------------------------------------------------------ tools and roles


def test_default_tools_have_one_github_read() -> None:
    names = [t.name for t in default_tools()]
    assert names.count("github_read") == 1
    # Fresh instances each time.
    a = next(t for t in default_tools() if t.name == "github_read")
    b = next(t for t in default_tools() if t.name == "github_read")
    assert isinstance(a, GitHubRead) and a is not b


def test_web_search_is_still_not_a_default_tool() -> None:
    assert "web_search" not in {t.name for t in default_tools()}


def test_only_the_researcher_lists_github_read() -> None:
    reg = RoleRegistry.builtin(tools=default_tools())
    assert {r.name for r in reg if "github_read" in r.tools} == {"researcher"}


def test_researcher_needs_approval_for_every_github_read() -> None:
    r = RoleRegistry.builtin(tools=default_tools()).get("researcher")
    assert r.max_risk is RiskLevel.WRITE
    d = r.policy().decide("github_read", RiskLevel.EXTERNAL)
    assert d.decision is Decision.NEEDS_APPROVAL
    assert r.policy().decide("fetch_url", RiskLevel.EXTERNAL).decision is Decision.NEEDS_APPROVAL
    assert "run_command" not in r.tools and "delete_file" not in r.tools


@pytest.mark.parametrize("name", ["architect", "developer", "writer"])
def test_other_roles_are_denied_github_read(name: str) -> None:
    r = RoleRegistry.builtin(tools=default_tools()).get(name)
    assert "github_read" not in r.tools
    assert r.policy().decide("github_read", RiskLevel.EXTERNAL).decision is Decision.DENY


def test_researcher_prompt_mentions_github_read_and_untrusted() -> None:
    text = RoleRegistry.builtin(tools=default_tools()).get("researcher").instructions
    assert "github_read" in text and "approval" in text and "untrusted" in text


def test_executor_without_store_hides_github_read(tmp_path: Path) -> None:
    r = RoleRegistry.builtin(tools=default_tools()).get("researcher")
    ex = ToolExecutor(default_tools(), r.policy(), Workspace(tmp_path), MemoryEventSink(), PID)
    shown = {s["function"]["name"] for s in ex.specs()}
    assert "github_read" not in shown and "fetch_url" not in shown
    # No Approval Gate store: the call never runs.
    with pytest.raises(ApprovalRequired):
        ex.execute("github_read", FILE_ARGS)


def test_other_role_executor_denies_github_read(tmp_path: Path) -> None:
    r = RoleRegistry.builtin(tools=default_tools()).get("developer")
    store = ApprovalStore(tmp_path / "approvals", PID)
    (tmp_path / "ws").mkdir()
    ex = ToolExecutor(
        default_tools(),
        r.policy(),
        Workspace(tmp_path / "ws"),
        MemoryEventSink(),
        PID,
        approvals=store,
        task_id="t",
    )
    assert "github_read" not in {s["function"]["name"] for s in ex.specs()}
    res = ex.execute("github_read", FILE_ARGS)
    assert not res.ok and store.list() == []


# ------------------------------------------------------------------ describe_github_request


def test_describe_file_request() -> None:
    v = describe_github_request(recorded({**FILE_ARGS, "ref": "v1.0"}))
    assert v.valid and v.kind == "file" and v.repo == "octo/hello"
    assert v.path == "docs/README.md" and v.ref == "v1.0" and v.number is None
    assert v.api_url == FILE_URL + "?ref=v1.0"
    assert v.warnings == ()


def test_describe_file_without_ref_warns_about_the_default_branch() -> None:
    v = describe_github_request(recorded(FILE_ARGS))
    assert v.api_url == FILE_URL
    assert any("default branch" in w for w in v.warnings)


def test_describe_root_listing_warns() -> None:
    v = describe_github_request(recorded({**FILE_ARGS, "path": "", "ref": "main"}))
    assert v.api_url == "https://api.github.com/repos/octo/hello/contents?ref=main"
    assert v.path == "" and any("repository root" in w for w in v.warnings)


@pytest.mark.parametrize(("kind", "segment"), [("issue", "issues"), ("pull", "pulls")])
def test_describe_issue_and_pull(kind: str, segment: str) -> None:
    v = describe_github_request(recorded({"kind": kind, "owner": "o", "repo": "r", "number": 7}))
    assert v.valid and v.kind == kind and v.number == 7 and v.path is None and v.ref is None
    assert v.api_url == f"https://api.github.com/repos/o/r/{segment}/7"
    assert v.warnings == ()


def test_describe_url_matches_the_tool() -> None:
    args = GitHubReadArgs.model_validate({**FILE_ARGS, "path": "a b/ü.md", "ref": "feat/x"})
    view = describe_github_request(args.model_dump(mode="json"))
    assert view.api_url == github_api_url(args)
    assert "%20" in (view.api_url or "") and "feat%2Fx" in (view.api_url or "")


def test_describe_accepts_raw_model_arguments() -> None:
    assert describe_github_request({"kind": "issue", "owner": "o", "repo": "r", "number": 1}).valid


@pytest.mark.parametrize(
    "arguments",
    [
        None,
        "octo/hello",
        ["x"],
        {},
        {"kind": "issue", "owner": "o", "repo": "r"},
        {"kind": "file", "owner": "o", "repo": "r", "path": "../etc"},
        {"kind": "file", "owner": "-bad", "repo": "r", "path": "x"},
        {"kind": "file", "owner": "o", "repo": "r", "path": "x", "ref": "a..b"},
        {**FILE_ARGS, "extra": 1},
        {"kind": "commit", "owner": "o", "repo": "r"},
    ],
)
def test_describe_invalid(arguments: Any) -> None:
    v = describe_github_request(arguments)
    assert not v.valid and v.error and v.api_url is None and v.repo is None
    assert v.warnings == ()


def test_describe_error_does_not_echo_the_input() -> None:
    v = describe_github_request({**FILE_ARGS, "path": "x/‮evil"})
    assert not v.valid and "‮" not in (v.error or "") and "evil" not in (v.error or "")


def test_describe_never_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("describe must not resolve")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    assert describe_github_request(recorded(FILE_ARGS)).valid


# ------------------------------------------------------------------ CLI


class GitHub:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        content = base64.b64encode(b"The answer is 42.\n").decode()
        return httpx.Response(
            200,
            json={"type": "file", "encoding": "base64", "content": content, "sha": "abc"},
        )

    @property
    def urls(self) -> list[str]:
        return [str(r.url) for r in self.requests]


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


class Env:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.ws = tmp_path / "ws"
        self.config = tmp_path / "automa.toml"
        self.config.write_text(TOML.format(ws=self.ws.as_posix()), encoding="utf-8")
        self.provider = FakeProvider()
        self.gh = GitHub()
        self.address = PUBLIC
        monkeypatch.chdir(tmp_path)
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_must_not_be_sent")
        monkeypatch.setattr(cli, "make_provider", lambda config: self._provider(config))

        def tools() -> list[Tool]:
            mocked = GitHubRead(
                transport=httpx.MockTransport(self.gh.handler),
                resolver=lambda h, p: [self.address],
            )
            return [t for t in default_tools() if t.name != "github_read"] + [mocked]

        monkeypatch.setattr(cli, "tools_factory", lambda sandbox=None: tools)

    def _provider(self, config: AutomaConfig) -> FakeProvider:
        return self.provider

    @property
    def store(self) -> ProjectStore:
        return ProjectStore.for_workspace(self.ws)

    @property
    def approvals(self) -> ApprovalStore:
        return self.store.approvals(PID)

    def cli(self, *args: str, input: str | None = None) -> Any:
        return CliRunner().invoke(cli.app, [*args, "--config", str(self.config)], input=input)

    def research_project(self) -> None:
        self.store.create("find the answer", project_id=PID)
        plan = {
            "goal": "find the answer",
            "tasks": [
                {
                    "id": "r",
                    "title": "Look it up",
                    "role": "researcher",
                    "acceptance_criteria": [
                        {
                            "description": "notes.md has the answer",
                            "check": {"type": "file_contains", "path": "notes.md", "text": "42"},
                        }
                    ],
                }
            ],
        }
        self.store.save_plan(PID, Plan.model_validate(plan))

    def request(self, arguments: Any, tool: str = "github_read") -> ApprovalRequest:
        self.store.create("goal", project_id=PID)
        return self.approvals.request(
            tool=tool,
            risk=RiskLevel.EXTERNAL,
            arguments=arguments,
            reason="external tool needs approval",
            task_id="r",
        )

    def pending(self) -> ApprovalRequest:
        (req,) = self.approvals.list(ApprovalStatus.PENDING)
        return req


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


def test_approvals_shows_the_github_request(env: Env) -> None:
    env.request(recorded(FILE_ARGS))
    r = env.cli("approvals", PID)
    assert r.exit_code == 0, r.output
    assert "GitHub request:" in r.output
    assert "file in octo/hello (anonymous, read-only)" in r.output
    assert f"GET {FILE_URL}" in r.output
    assert "path: docs/README.md" in r.output
    assert "! no ref: reads the default branch" in r.output
    assert "untrusted data" in r.output
    assert "web request:" not in r.output


def test_approvals_shows_number_and_ref(env: Env) -> None:
    env.request(recorded({"kind": "pull", "owner": "o", "repo": "r", "number": 12}))
    out = env.cli("approvals", PID).output
    assert "pull in o/r" in out and "number: 12" in out
    assert "GET https://api.github.com/repos/o/r/pulls/12" in out
    assert "path:" not in out.split("GitHub request:")[1].split("untrusted")[0]


def test_approvals_json_has_github_request(env: Env) -> None:
    env.request(recorded({**FILE_ARGS, "ref": "main"}))
    (row,) = json.loads(env.cli("approvals", PID, "--json").output)
    gh = row["github_request"]
    assert gh["valid"] and gh["repo"] == "octo/hello" and gh["kind"] == "file"
    assert gh["api_url"] == FILE_URL + "?ref=main" and gh["ref"] == "main"
    assert gh["warnings"] == []
    assert "web_request" not in row


def test_non_github_requests_have_no_github_section(env: Env) -> None:
    env.request({"url": "https://example.com/"}, tool="fetch_url")
    out = env.cli("approvals", PID).output
    assert "GitHub request:" not in out and "web request:" in out
    (row,) = json.loads(env.cli("approvals", PID, "--json").output)
    assert "github_request" not in row


def test_tampered_request_is_shown_as_invalid(env: Env) -> None:
    env.request({"kind": "file", "owner": "o", "repo": "r", "path": "../../etc/passwd"})
    out = env.cli("approvals", PID).output
    assert "invalid arguments (" in out and "the tool will refuse them" in out
    (row,) = json.loads(env.cli("approvals", PID, "--json").output)
    assert row["github_request"]["valid"] is False


def test_display_cannot_inject_terminal_controls(env: Env) -> None:
    env.request({**FILE_ARGS, "path": "a/‮b\x1b[31m"})
    out = env.cli("approvals", PID).output
    assert "‮" not in out and "\x1b" not in out


def test_approve_shows_the_github_request_before_confirming(env: Env) -> None:
    req = env.request(recorded(FILE_ARGS))
    r = env.cli("approve", PID, req.id, input="n\n")
    assert r.exit_code == cli.EXIT_USER
    before_prompt = r.output.split("Approve running")[0]
    assert "GitHub request:" in before_prompt and f"GET {FILE_URL}" in before_prompt
    assert env.approvals.get(req.id).status is ApprovalStatus.PENDING


def test_researcher_run_pauses_then_reads_once_after_approval(env: Env) -> None:
    env.research_project()
    env.provider.push(calls(("github_read", FILE_ARGS)))
    r = env.cli("run", PID)
    assert r.exit_code == cli.EXIT_PAUSED, r.output
    assert env.gh.requests == []
    req = env.pending()
    assert req.tool == "github_read" and req.task_id == "r"
    assert req.arguments["path"] == "docs/README.md"
    assert f"GET {FILE_URL}" in env.cli("approvals", PID).output

    assert env.cli("approve", PID, req.id, input="y\n").exit_code == 0
    env.provider.push(
        calls(("github_read", FILE_ARGS)),
        calls(("write_file", {"path": "notes.md", "content": "The answer is 42 (octo/hello)."})),
        "done",
    )
    r = env.cli("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert env.gh.urls == [FILE_URL]
    assert "authorization" not in {k.lower() for k in env.gh.requests[0].headers}
    assert env.approvals.is_used(req.id)
    assert env.store.load(PID).status is ProjectStatus.DONE
    assert env.store.task_states(PID)["r"].status is TaskStatus.PASSED
    seen = json.dumps([m.model_dump(mode="json") for m in env.provider.requests[-2].messages])
    assert "untrusted data" in seen and "The answer is 42." in seen


def test_approval_does_not_cover_another_path(env: Env) -> None:
    env.research_project()
    env.provider.push(calls(("github_read", FILE_ARGS)))
    assert env.cli("run", PID).exit_code == cli.EXIT_PAUSED
    assert env.cli("approve", PID, env.pending().id, "--yes").exit_code == 0
    other = {**FILE_ARGS, "path": "secrets.md"}
    env.provider.push(calls(("github_read", other)))
    assert env.cli("run", PID).exit_code == cli.EXIT_PAUSED
    assert env.gh.requests == []
    assert env.pending().arguments["path"] == "secrets.md"


def test_researcher_rejected_read_reaches_nothing(env: Env) -> None:
    env.research_project()
    env.provider.push(calls(("github_read", FILE_ARGS)))
    assert env.cli("run", PID).exit_code == cli.EXIT_PAUSED
    req = env.pending()
    assert env.cli("reject", PID, req.id, "--note", "use the files").exit_code == 0
    env.provider.push(
        calls(("github_read", FILE_ARGS)),
        calls(("write_file", {"path": "notes.md", "content": "42"})),
        "done",
    )
    assert env.cli("run", PID).exit_code == cli.EXIT_OK
    assert env.gh.requests == []


def test_private_api_address_is_refused_even_after_approval(env: Env) -> None:
    env.address = "10.0.0.5"
    env.research_project()
    env.provider.push(calls(("github_read", FILE_ARGS)))
    assert env.cli("run", PID).exit_code == cli.EXIT_PAUSED
    assert env.cli("approve", PID, env.pending().id, "--yes").exit_code == 0
    env.provider.push(
        calls(("github_read", FILE_ARGS)),
        calls(("write_file", {"path": "notes.md", "content": "42"})),
        "done",
    )
    assert env.cli("run", PID).exit_code == cli.EXIT_OK
    assert env.gh.requests == []
    seen = json.dumps([m.model_dump(mode="json") for m in env.provider.requests[-2].messages])
    assert "not a public address" in seen


def test_developer_task_cannot_use_github_read(env: Env) -> None:
    env.store.create("build", project_id=PID)
    plan = {
        "goal": "build",
        "tasks": [
            {
                "id": "d",
                "title": "Write it",
                "role": "developer",
                "acceptance_criteria": [
                    {
                        "description": "out.md exists",
                        "check": {"type": "file_contains", "path": "out.md", "text": "ok"},
                    }
                ],
            }
        ],
    }
    env.store.save_plan(PID, Plan.model_validate(plan))
    env.provider.push(
        calls(("github_read", FILE_ARGS)),
        calls(("write_file", {"path": "out.md", "content": "ok"})),
        "done",
    )
    r = env.cli("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert env.gh.requests == [] and env.approvals.list() == []
