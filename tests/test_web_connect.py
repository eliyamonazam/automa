"""``fetch_url`` connected (D73): the ``researcher`` role lists it, every call still goes
through the Approval Gate, and ``automa approvals``/``approve`` describe web requests.

Local only: ``FakeProvider`` for the model, ``httpx.MockTransport`` and a fake resolver
for the web. No network, no real model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from automa import cli
from automa.config import AutomaConfig
from automa.llm import ChatResponse, FakeProvider, FinishReason, Message, ToolCall, Usage
from automa.llm import Role as MsgRole
from automa.plan import Plan, TaskStatus
from automa.project import ProjectStatus, ProjectStore
from automa.roles import RoleRegistry
from automa.tools import (
    Decision,
    FetchUrl,
    RiskLevel,
    Tool,
    default_tools,
    describe_fetch_request,
)
from automa.tools.approvals import ApprovalRequest, ApprovalStatus, ApprovalStore

PID = "proj-web"

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


# ------------------------------------------------------------------ roles


def test_only_the_researcher_lists_fetch_url() -> None:
    reg = RoleRegistry.builtin(tools=default_tools())
    having = {r.name for r in reg if "fetch_url" in r.tools}
    assert having == {"researcher"}


def test_researcher_needs_approval_for_every_fetch() -> None:
    r = RoleRegistry.builtin(tools=default_tools()).get("researcher")
    assert r.max_risk is RiskLevel.WRITE
    d = r.policy().decide("fetch_url", RiskLevel.EXTERNAL)
    assert d.decision is Decision.NEEDS_APPROVAL
    # Still no command execution or deletion for the researcher.
    assert r.policy().decide("run_command", RiskLevel.EXECUTE).decision is Decision.DENY
    assert "run_command" not in r.tools and "delete_file" not in r.tools


@pytest.mark.parametrize("name", ["architect", "developer", "writer"])
def test_other_roles_are_denied_fetch_url(name: str) -> None:
    r = RoleRegistry.builtin(tools=default_tools()).get(name)
    assert r.policy().decide("fetch_url", RiskLevel.EXTERNAL).decision is Decision.DENY


def test_researcher_prompt_treats_pages_as_untrusted() -> None:
    text = RoleRegistry.builtin(tools=default_tools()).get("researcher").instructions.lower()
    assert "untrusted" in text and "approval" in text
    assert "no web access" not in text


# ------------------------------------------------------------------ describe_fetch_request


def test_describe_plain_https() -> None:
    v = describe_fetch_request("https://example.com/docs/page")
    assert v.valid and v.scheme == "https" and v.host == "example.com" and v.port == 443
    assert v.path == "/docs/page" and v.query_parameters == 0 and v.warnings == ()


def test_describe_empty_path_is_root() -> None:
    assert describe_fetch_request("https://example.com").path == "/"


def test_describe_query_string_warns_about_data_leaving() -> None:
    v = describe_fetch_request("https://example.com/a?token=SECRET&b=2")
    assert v.query_parameters == 2
    assert any("query string (2 parameter(s))" in w and "example.com" in w for w in v.warnings)


def test_describe_http_ip_and_port() -> None:
    v = describe_fetch_request("http://93.184.216.34:8080/x")
    joined = " | ".join(v.warnings)
    assert "not encrypted" in joined and "IP address" in joined and "port 8080" in joined
    assert v.port == 8080


def test_describe_ipv6_literal() -> None:
    v = describe_fetch_request("https://[2606:4700::1111]/")
    assert v.valid and any("IP address" in w for w in v.warnings)


def test_describe_default_port_written_out_is_not_a_warning() -> None:
    assert describe_fetch_request("https://example.com:443/").warnings == ()
    assert any("port 443" in w for w in describe_fetch_request("http://example.com:443/").warnings)


def test_describe_non_ascii_host_shows_the_wire_name() -> None:
    v = describe_fetch_request("https://пример.рф/")
    assert v.host == "xn--e1afmkfd.xn--p1ai"
    assert any("non-ASCII" in w and "xn--e1afmkfd.xn--p1ai" in w for w in v.warnings)


def test_describe_punycode_host_warns() -> None:
    v = describe_fetch_request("https://xn--pple-43d.com/")
    assert any("punycode" in w for w in v.warnings)


def test_describe_long_path_warns() -> None:
    v = describe_fetch_request("https://example.com/" + "a" * 300)
    assert any("long path" in w for w in v.warnings)


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "https://u:p@example.com/", "https://", "", None, 5, ["x"]]
)
def test_describe_invalid(url: Any) -> None:
    v = describe_fetch_request(url)
    assert not v.valid and v.error and v.host is None and v.warnings == ()


def test_describe_never_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("describe must not resolve")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    assert describe_fetch_request("https://example.com/").valid


# ------------------------------------------------------------------ CLI


class Web:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html><title>T</title><p>The answer is 42.</p></html>",
        )


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


URL = "https://example.com/facts?q=answer"


class Env:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.ws = tmp_path / "ws"
        self.config = tmp_path / "automa.toml"
        self.config.write_text(TOML.format(ws=self.ws.as_posix()), encoding="utf-8")
        self.provider = FakeProvider()
        self.web = Web()
        monkeypatch.chdir(tmp_path)
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(cli, "make_provider", lambda config: self._provider(config))

        def tools() -> list[Tool]:
            mocked = FetchUrl(
                transport=httpx.MockTransport(self.web.handler),
                resolver=lambda h, p: ["93.184.216.34"],
            )
            return [t for t in default_tools() if t.name != "fetch_url"] + [mocked]

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

    def request(self, url: object) -> ApprovalRequest:
        self.store.create("goal", project_id=PID)
        return self.approvals.request(
            tool="fetch_url",
            risk=RiskLevel.EXTERNAL,
            arguments={"url": url},
            reason="external tool needs approval",
            task_id="r",
        )

    def pending(self) -> ApprovalRequest:
        (req,) = self.approvals.list(ApprovalStatus.PENDING)
        return req


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


def test_approvals_shows_the_web_request(env: Env) -> None:
    env.request("http://example.com:8080/a?token=SECRET&x=1")
    r = env.cli("approvals", PID)
    assert r.exit_code == 0, r.output
    assert "web request:" in r.output
    assert "host: example.com (port 8080, http)" in r.output
    assert "path: /a" in r.output
    assert "! not encrypted" in r.output and "! non-default port 8080" in r.output
    assert "query string (2 parameter(s)) is sent to example.com" in r.output
    assert "untrusted data" in r.output
    # The exact arguments are still shown.
    assert "token=SECRET" in r.output


def test_approvals_json_has_web_request(env: Env) -> None:
    env.request("https://пример.рф/")
    (row,) = json.loads(env.cli("approvals", PID, "--json").output)
    web = row["web_request"]
    assert web["valid"] and web["host"] == "xn--e1afmkfd.xn--p1ai" and web["port"] == 443
    assert row["arguments"] == {"url": "https://пример.рф/"}


def test_non_web_requests_have_no_web_section(env: Env) -> None:
    env.store.create("goal", project_id=PID)
    env.approvals.request(
        tool="publish",
        risk=RiskLevel.EXTERNAL,
        arguments={"url": "https://example.com/"},
        reason="r",
        task_id="r",
    )
    assert "web request:" not in env.cli("approvals", PID).output
    (row,) = json.loads(env.cli("approvals", PID, "--json").output)
    assert "web_request" not in row


def test_tampered_request_is_shown_as_invalid(env: Env) -> None:
    env.request("file:///etc/passwd")
    out = env.cli("approvals", PID).output
    assert "invalid URL (only http and https URLs can be fetched)" in out
    env2 = env.cli("approvals", PID, "--json").output
    assert json.loads(env2)[0]["web_request"]["valid"] is False


def test_non_string_url_is_shown_as_invalid(env: Env) -> None:
    env.request(["https://example.com/"])
    assert "invalid URL (URL is not a string)" in env.cli("approvals", PID).output


def test_host_display_cannot_inject_terminal_controls(env: Env) -> None:
    # A path with a bidi override: escaped in the view (check_url allows it).
    env.request("https://example.com/‮abc")
    out = env.cli("approvals", PID).output
    assert "‮" not in out


def test_approve_shows_the_web_request_before_confirming(env: Env) -> None:
    req = env.request("https://example.com/a?q=1")
    r = env.cli("approve", PID, req.id, input="n\n")
    assert r.exit_code == cli.EXIT_USER
    before_prompt = r.output.split("Approve running")[0]
    assert "web request:" in before_prompt and "host: example.com" in before_prompt
    assert env.approvals.get(req.id).status is ApprovalStatus.PENDING


def test_researcher_run_pauses_then_fetches_once_after_approval(env: Env) -> None:
    env.research_project()
    env.provider.push(calls(("fetch_url", {"url": URL})))
    r = env.cli("run", PID)
    assert r.exit_code == cli.EXIT_PAUSED, r.output
    assert env.web.requests == []
    req = env.pending()
    assert req.tool == "fetch_url" and req.arguments["url"] == URL and req.task_id == "r"
    shown = env.cli("approvals", PID).output
    assert "host: example.com (port 443, https)" in shown

    assert env.cli("approve", PID, req.id, input="y\n").exit_code == 0
    env.provider.push(
        calls(("fetch_url", {"url": URL})),
        calls(("write_file", {"path": "notes.md", "content": "The answer is 42 (example.com)."})),
        "done",
    )
    r = env.cli("run", PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert env.web.requests == [URL]
    assert env.approvals.is_used(req.id)
    assert env.store.load(PID).status is ProjectStatus.DONE
    assert env.store.task_states(PID)["r"].status is TaskStatus.PASSED
    # The page reached the model as untrusted data.
    seen = json.dumps([m.model_dump(mode="json") for m in env.provider.requests[-2].messages])
    assert "untrusted data" in seen and "The answer is 42." in seen


def test_researcher_rejected_fetch_reaches_nothing(env: Env) -> None:
    env.research_project()
    env.provider.push(calls(("fetch_url", {"url": URL})))
    assert env.cli("run", PID).exit_code == cli.EXIT_PAUSED
    req = env.pending()
    assert env.cli("reject", PID, req.id, "--note", "use the files").exit_code == 0
    env.provider.push(
        calls(("fetch_url", {"url": URL})),
        calls(("write_file", {"path": "notes.md", "content": "42"})),
        "done",
    )
    assert env.cli("run", PID).exit_code == cli.EXIT_OK
    assert env.web.requests == []


def test_researcher_private_host_is_refused_even_after_approval(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def private_tools() -> list[Tool]:
        mocked = FetchUrl(
            transport=httpx.MockTransport(env.web.handler), resolver=lambda h, p: ["10.0.0.5"]
        )
        return [t for t in default_tools() if t.name != "fetch_url"] + [mocked]

    monkeypatch.setattr(cli, "tools_factory", lambda sandbox=None: private_tools)
    env.research_project()
    url = "https://intranet.example/"
    env.provider.push(calls(("fetch_url", {"url": url})))
    assert env.cli("run", PID).exit_code == cli.EXIT_PAUSED
    assert env.cli("approve", PID, env.pending().id, "--yes").exit_code == 0
    env.provider.push(
        calls(("fetch_url", {"url": url})),
        calls(("write_file", {"path": "notes.md", "content": "42"})),
        "done",
    )
    assert env.cli("run", PID).exit_code == cli.EXIT_OK
    assert env.web.requests == []
    seen = json.dumps([m.model_dump(mode="json") for m in env.provider.requests[-2].messages])
    assert "not a public address" in seen
