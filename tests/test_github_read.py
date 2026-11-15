"""``github_read`` core (D75): args, fixed API URLs, guards, rendering, Approval Gate.

No network: every test uses ``httpx.MockTransport`` and a fake resolver.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from automa.events import MemoryEventSink
from automa.roles import RoleRegistry
from automa.tools import (
    ApprovalRequired,
    GitHubRead,
    GitHubReadArgs,
    PermissionPolicy,
    RiskLevel,
    ToolError,
    ToolExecutor,
    Workspace,
    default_tools,
    github_api_url,
)
from automa.tools.approvals import ApprovalStore
from automa.tools.github import MAX_GITHUB_BYTES, UNTRUSTED_GITHUB_NOTE, clean_body

PID = "gh-proj"
PUBLIC = "140.82.112.6"


def public_resolver(host: str, port: int) -> list[str]:
    return [PUBLIC]


class Recorder:
    def __init__(self, handler):  # type: ignore[no-untyped-def]
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)


def tool(handler, resolver=public_resolver) -> tuple[GitHubRead, Recorder]:  # type: ignore[no-untyped-def]
    rec = Recorder(handler)
    return GitHubRead(transport=httpx.MockTransport(rec), resolver=resolver), rec


def ok(data: Any, **headers: str) -> httpx.Response:
    return httpx.Response(200, json=data, headers=headers)


ISSUE: dict[str, Any] = {
    "title": "Crash on start",
    "state": "open",
    "user": {"login": "alice"},
    "labels": [{"name": "bug"}, {"name": "p1"}],
    "comments": 3,
    "created_at": "2026-11-01T10:00:00Z",
    "updated_at": "2026-11-02T10:00:00Z",
    "body": "Steps:\n1. run\n2. crash",
}
PULL: dict[str, Any] = {
    "title": "Fix crash",
    "state": "closed",
    "merged": True,
    "draft": False,
    "user": {"login": "bob"},
    "base": {"ref": "main"},
    "head": {"label": "bob:fix"},
    "changed_files": 2,
    "additions": 10,
    "deletions": 4,
    "body": "Fixes #1",
}


def file_answer(text: bytes, **kw: Any) -> dict[str, Any]:
    data = {
        "type": "file",
        "encoding": "base64",
        "content": base64.encodebytes(text).decode(),
        "sha": "abc123",
    }
    data.update(kw)
    return data


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "files"
    root.mkdir()
    return Workspace(root)


def args(**kw: Any) -> GitHubReadArgs:
    base: dict[str, Any] = {"kind": "issue", "owner": "octo", "repo": "hello", "number": 1}
    base.update(kw)
    return GitHubReadArgs(**base)


def fargs(path: str = "README.md", **kw: Any) -> GitHubReadArgs:
    return GitHubReadArgs(kind="file", owner="octo", repo="hello", path=path, **kw)


# ------------------------------------------------------------------ contract


def test_tool_is_external_and_reversible() -> None:
    assert GitHubRead.risk is RiskLevel.EXTERNAL
    assert GitHubRead.irreversible is False
    assert GitHubRead.name == "github_read"
    assert "approval" in GitHubRead.description and "untrusted" in GitHubRead.description


def test_not_in_default_tools_and_no_role_lists_it() -> None:
    assert "github_read" not in {t.name for t in default_tools()}
    reg = RoleRegistry.builtin(tools=[*default_tools(), GitHubRead()])
    assert [r.name for r in reg if "github_read" in r.tools] == []


def test_spec_describes_the_arguments() -> None:
    props = GitHubRead().spec()["function"]["parameters"]["properties"]
    assert {"kind", "owner", "repo", "number", "path", "ref", "max_chars"} <= set(props)


def test_timeout_bounds() -> None:
    with pytest.raises(ValueError):
        GitHubRead(timeout_s=0)
    with pytest.raises(ValueError):
        GitHubRead(timeout_s=121)


# ------------------------------------------------------------------ arguments


@pytest.mark.parametrize("owner", ["", "-octo", "octo-", "oc to", "oc/to", "a" * 40, "o.c"])
def test_invalid_owner(owner: str) -> None:
    with pytest.raises(ValidationError):
        args(owner=owner)


@pytest.mark.parametrize("repo", ["", ".", "..", "a/b", "a b", "a?b", "a#b", "r" * 101, "a%2f"])
def test_invalid_repo(repo: str) -> None:
    with pytest.raises(ValidationError):
        args(repo=repo)


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "a/", "a//b", "../x", "a/../b", "./a", "a\\b", "a\nb", "a\x00b", "a‮b"],
)
def test_invalid_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        fargs(path)


@pytest.mark.parametrize("ref", ["", "-x", "a..b", "/main", "main/", "a b", "a?b", "a//b", ".x"])
def test_invalid_refs(ref: str) -> None:
    with pytest.raises(ValidationError):
        fargs(ref=ref)


def test_kind_and_fields_must_match() -> None:
    with pytest.raises(ValidationError):
        args(number=None)
    with pytest.raises(ValidationError):
        args(path="README.md")
    with pytest.raises(ValidationError):
        args(kind="pull", ref="main")
    with pytest.raises(ValidationError):
        GitHubReadArgs(kind="file", owner="octo", repo="hello")
    with pytest.raises(ValidationError):
        fargs(number=3)
    with pytest.raises(ValidationError):
        args(kind="commit")
    with pytest.raises(ValidationError):
        args(number=0)
    with pytest.raises(ValidationError):
        args(url="https://evil.example")


def test_api_urls_are_fixed_and_escaped() -> None:
    assert github_api_url(args()) == "https://api.github.com/repos/octo/hello/issues/1"
    assert github_api_url(args(kind="pull", number=7)).endswith("/repos/octo/hello/pulls/7")
    assert github_api_url(fargs("")) == "https://api.github.com/repos/octo/hello/contents"
    url = github_api_url(fargs("docs/a b?#.md", ref="release/1.0"))
    assert url == (
        "https://api.github.com/repos/octo/hello/contents/docs/a%20b%3F%23.md?ref=release%2F1.0"
    )
    assert github_api_url(fargs("دفتر.md")).startswith("https://api.github.com/repos/")


def test_audit_records_only_kind_and_repo() -> None:
    a = fargs("secret/plan.md", ref="feature-x")
    assert GitHubRead().audit(a) == {"kind": "file", "repo": "octo/hello"}


# ------------------------------------------------------------------ requests


def test_issue_is_rendered_with_untrusted_header(ws: Workspace) -> None:
    t, rec = tool(lambda r: ok(ISSUE))
    out = t.run(args(), ws)
    assert out.startswith(UNTRUSTED_GITHUB_NOTE)
    assert "Source: GitHub octo/hello issue #1" in out
    for part in ("Crash on start", "State: open", "Author: alice", "Labels: bug, p1", "2. crash"):
        assert part in out
    (req,) = rec.requests
    assert str(req.url) == "https://api.github.com/repos/octo/hello/issues/1"
    assert req.method == "GET"


def test_requests_are_anonymous(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    monkeypatch.setenv("GH_TOKEN", "ghp_secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    t, rec = tool(lambda r: ok(ISSUE))
    t.run(args(), ws)
    (req,) = rec.requests
    assert "authorization" not in req.headers and "cookie" not in req.headers
    assert req.headers["accept"] == "application/vnd.github+json"
    assert "ghp_secret" not in str(req.headers)


def test_issue_that_is_a_pull_request_is_marked(ws: Workspace) -> None:
    t, _ = tool(lambda r: ok({**ISSUE, "pull_request": {"url": "x"}}))
    assert "this issue is a pull request" in t.run(args(), ws)


def test_pull_request_is_rendered(ws: Workspace) -> None:
    t, rec = tool(lambda r: ok(PULL))
    out = t.run(args(kind="pull", number=2), ws)
    assert "pull request #2" in out and "State: closed (merged)" in out
    assert "Base: main  Head: bob:fix" in out and "2 file(s), +10 -4" in out
    assert str(rec.requests[0].url).endswith("/pulls/2")


def test_missing_body_and_odd_fields(ws: Workspace) -> None:
    t, _ = tool(lambda r: ok({"title": None, "user": "x", "labels": "y", "body": None}))
    out = t.run(args(), ws)
    assert "(no body)" in out and "Author: \n" in out


def test_text_file_is_decoded(ws: Workspace) -> None:
    t, rec = tool(lambda r: ok(file_answer("# سلام\nhello\n".encode())))
    out = t.run(fargs(ref="main"), ws)
    assert "file README.md @ main" in out and "# سلام\nhello" in out and "SHA: abc123" in out
    assert str(rec.requests[0].url).endswith("/contents/README.md?ref=main")


@pytest.mark.parametrize(
    "answer, message",
    [
        (file_answer(b"\x89PNG\x00\x01"), "binary"),
        (file_answer(b"\xff\xfe abc"), "not UTF-8"),
        (file_answer(b"", encoding="none", content=""), "too large"),
        ({"type": "symlink", "target": "/etc"}, "not a file"),
        ({"type": "submodule"}, "not a file"),
        ({"type": "file", "encoding": "base64", "content": "!!!"}, "invalid answer"),
        ("text", "invalid answer"),
    ],
)
def test_unshown_files(ws: Workspace, answer: Any, message: str) -> None:
    t, _ = tool(lambda r: ok(answer))
    with pytest.raises(ToolError, match=message):
        t.run(fargs(), ws)


def test_directory_listing(ws: Workspace) -> None:
    listing = [
        {"type": "dir", "name": "src"},
        {"type": "file", "name": "README.md", "size": 12},
        "junk",
    ]
    t, rec = tool(lambda r: ok(listing))
    out = t.run(fargs(""), ws)
    assert "directory /" in out and "Entries: 2" in out
    assert "dir       src" in out and "README.md (12 bytes)" in out
    assert str(rec.requests[0].url) == "https://api.github.com/repos/octo/hello/contents"


def test_issue_answer_must_be_an_object(ws: Workspace) -> None:
    t, _ = tool(lambda r: ok([1, 2]))
    with pytest.raises(ToolError, match="invalid answer"):
        t.run(args(), ws)
    t, _ = tool(lambda r: httpx.Response(200, content=b"<html>"))
    with pytest.raises(ToolError, match="invalid answer"):
        t.run(args(kind="pull"), ws)


@pytest.mark.parametrize(
    "status, headers, message",
    [
        (404, {}, "not found on GitHub"),
        (403, {"x-ratelimit-remaining": "0"}, "rate limit"),
        (429, {}, "rate limit"),
        (403, {}, "refused the request \\(HTTP 403\\)"),
        (401, {}, "refused"),
        (500, {}, "HTTP 500 from GitHub"),
    ],
)
def test_error_statuses_never_echo_the_body(
    ws: Workspace, status: int, headers: dict[str, str], message: str
) -> None:
    t, _ = tool(lambda r: httpx.Response(status, headers=headers, text="SECRET-BODY"))
    with pytest.raises(ToolError, match=message) as ei:
        t.run(args(), ws)
    assert "SECRET" not in str(ei.value)


def test_redirect_inside_the_api_is_followed(ws: Workspace) -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        if r.url.path == "/repos/octo/hello/issues/1":
            return httpx.Response(301, headers={"location": "/repositories/42/issues/1"})
        return ok(ISSUE)

    t, rec = tool(handler)
    assert "Crash on start" in t.run(args(), ws)
    assert [r.url.path for r in rec.requests] == [
        "/repos/octo/hello/issues/1",
        "/repositories/42/issues/1",
    ]


@pytest.mark.parametrize(
    "location",
    [
        "https://evil.example/x",
        "http://api.github.com/x",
        "https://api.github.com:8443/x",
        "https://user@api.github.com/x",
        "https://api.github.com.evil.example/x",
        "https://[::1]/x",
    ],
)
def test_redirect_outside_the_api_is_refused(ws: Workspace, location: str) -> None:
    t, rec = tool(lambda r: httpx.Response(302, headers={"location": location}))
    with pytest.raises(ToolError, match="outside https://api.github.com|invalid URL"):
        t.run(args(), ws)
    assert len(rec.requests) == 1


def test_redirect_loops_are_bounded(ws: Workspace) -> None:
    t, rec = tool(lambda r: httpx.Response(307, headers={"location": "/repos/o/r/issues/1"}))
    with pytest.raises(ToolError, match="too many redirects"):
        t.run(args(), ws)
    assert len(rec.requests) == 4


def test_redirect_without_location(ws: Workspace) -> None:
    t, _ = tool(lambda r: httpx.Response(301))
    with pytest.raises(ToolError, match="without a location"):
        t.run(args(), ws)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "169.254.169.254", "::1"])
def test_api_host_resolving_to_private_address_is_refused(ws: Workspace, address: str) -> None:
    t, rec = tool(lambda r: ok(ISSUE), resolver=lambda h, p: [address])
    with pytest.raises(ToolError, match="not a public address"):
        t.run(args(), ws)
    assert rec.requests == []


def test_resolver_is_asked_for_the_api_host(ws: Workspace) -> None:
    seen: list[tuple[str, int]] = []

    def resolver(host: str, port: int) -> list[str]:
        seen.append((host, port))
        return [PUBLIC]

    t, _ = tool(lambda r: ok(ISSUE), resolver=resolver)
    t.run(args(), ws)
    assert seen == [("api.github.com", 443)]


def test_too_large_answer_is_refused(ws: Workspace) -> None:
    t, _ = tool(lambda r: httpx.Response(200, content=b" " * (MAX_GITHUB_BYTES + 1)))
    with pytest.raises(ToolError, match="too large"):
        t.run(args(), ws)


def test_transport_errors_are_clean(ws: Workspace) -> None:
    def boom(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret detail", request=r)

    t, _ = tool(boom)
    with pytest.raises(ToolError) as ei:
        t.run(args(), ws)
    assert "ConnectError" in str(ei.value) and "secret" not in str(ei.value)

    def slow(r: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=r)

    t, _ = tool(slow)
    with pytest.raises(ToolError, match="timed out"):
        t.run(args(), ws)


def test_control_and_bidi_characters_are_removed(ws: Workspace) -> None:
    data = {**ISSUE, "title": "a\x1b[2Jb‮c", "body": "x\x1b]0;t\x07y​z\r\nn‌m\tq"}
    t, _ = tool(lambda r: ok(data))
    out = t.run(args(), ws)
    assert "\x1b" not in out and "‮" not in out and "​" not in out and "\r" not in out
    assert "x]0;tyz\nn‌m\tq" in out


def test_clean_body_truncates() -> None:
    assert clean_body("abcdef", 3) == "abc\n[... truncated]"


def test_output_is_bounded_by_max_chars(ws: Workspace) -> None:
    t, _ = tool(lambda r: ok({**ISSUE, "body": "y" * 50_000}))
    out = t.run(args(max_chars=500), ws)
    assert len(out) <= 500 + len("\n[... truncated]")


def test_run_requires_validated_args(ws: Workspace) -> None:
    with pytest.raises(TypeError):
        GitHubRead().run(object(), ws)  # type: ignore[arg-type]


# ------------------------------------------------------------------ Approval Gate


def clock() -> datetime:
    return datetime(2026, 11, 15, 12, 0, tzinfo=UTC)


def gated(
    tmp_path: Path,
    t: GitHubRead,
    store: ApprovalStore | None,
    allowed: frozenset[str] | None = None,
) -> tuple[ToolExecutor, MemoryEventSink]:
    root = tmp_path / "files"
    root.mkdir(exist_ok=True)
    sink = MemoryEventSink()
    ex = ToolExecutor(
        [t],
        PermissionPolicy(
            allowed_tools=frozenset({"github_read"}) if allowed is None else allowed,
            max_risk=RiskLevel.WRITE,
        ),
        Workspace(root),
        sink,
        PID,
        approvals=store,
        task_id="t1",
    )
    return ex, sink


CALL = {"kind": "file", "owner": "octo", "repo": "hello", "path": "secret/plan.md"}


def test_every_call_needs_approval_and_nothing_is_sent_before(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    t, rec = tool(lambda r: ok(file_answer(b"plan")))
    ex, sink = gated(tmp_path, t, store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("github_read", CALL)
    assert rec.requests == []
    req = store.get(ei.value.request_id or "")
    assert req.tool == "github_read" and req.arguments["path"] == "secret/plan.md"
    store.decide(req.id, approve=True)
    result = ex.execute("github_read", CALL)
    assert result.ok and "plan" in result.content and len(rec.requests) == 1
    with pytest.raises(ApprovalRequired):
        ex.execute("github_read", CALL)
    assert len(rec.requests) == 1
    calls = sink.of_type("tool.call")
    assert calls and all("secret" not in json.dumps(e.data) for e in calls)


def test_approval_covers_only_the_exact_arguments(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    t, rec = tool(lambda r: ok(ISSUE))
    ex, _ = gated(tmp_path, t, store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("github_read", CALL)
    store.decide(ei.value.request_id or "", approve=True)
    for other in (
        {**CALL, "path": "secret/other.md"},
        {**CALL, "ref": "dev"},
        {**CALL, "repo": "hello2"},
    ):
        with pytest.raises(ApprovalRequired):
            ex.execute("github_read", other)
    assert rec.requests == []


def test_rejected_call_sends_nothing(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    t, rec = tool(lambda r: ok(ISSUE))
    ex, _ = gated(tmp_path, t, store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("github_read", CALL)
    store.decide(ei.value.request_id or "", approve=False, note="no")
    result = ex.execute("github_read", CALL)
    assert not result.ok and rec.requests == []


def test_invalid_arguments_never_reach_the_gate(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    t, rec = tool(lambda r: ok(ISSUE))
    ex, _ = gated(tmp_path, t, store)
    result = ex.execute("github_read", {**CALL, "path": "../../etc/passwd"})
    assert not result.ok and store.list() == [] and rec.requests == []


def test_without_store_nothing_is_offered_or_sent(tmp_path: Path) -> None:
    t, rec = tool(lambda r: ok(ISSUE))
    ex, _ = gated(tmp_path, t, None)
    assert ex.specs() == []
    with pytest.raises(ApprovalRequired):
        ex.execute("github_read", CALL)
    assert rec.requests == []


def test_unlisted_github_read_is_denied(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    t, rec = tool(lambda r: ok(ISSUE))
    ex, _ = gated(tmp_path, t, store, allowed=frozenset())
    result = ex.execute("github_read", CALL)
    assert result.denied and store.list() == [] and rec.requests == []


def test_approved_private_address_is_still_refused(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    t, rec = tool(lambda r: ok(ISSUE), resolver=lambda h, p: ["192.168.1.1"])
    ex, _ = gated(tmp_path, t, store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("github_read", CALL)
    store.decide(ei.value.request_id or "", approve=True)
    result = ex.execute("github_read", CALL)
    assert not result.ok and "not a public address" in (result.error or "")
    assert rec.requests == []
