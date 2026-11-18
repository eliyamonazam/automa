"""Opt-in automatic approval of plain web reads (D93).

``fetch_url`` stays gated by default. With ``[web] auto_approve_reads`` a call runs
without asking only when its URL cannot plausibly carry project data out; every
other call still goes through the Approval Gate. No network: MockTransport only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from automa.config import AutomaConfig, WebConfig
from automa.events import MemoryEventSink
from automa.tools import PermissionPolicy, RiskLevel, ToolExecutor, Workspace
from automa.tools.approvals import ApprovalStore
from automa.tools.base import Tool
from automa.tools.executor import ApprovalRequired
from automa.tools.web import FetchUrl
from automa.tools.web_reads import AutoReadPolicy, auto_read_refusal

PID = "web-proj"


def clock() -> datetime:
    return datetime(2026, 11, 18, 12, 0, tzinfo=UTC)


def fetcher() -> tuple[FetchUrl, list[str]]:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, text="<p>hello</p>", headers={"Content-Type": "text/html"})

    tool = FetchUrl(transport=httpx.MockTransport(handler), resolver=lambda h, p: ["93.184.216.34"])
    return tool, seen


def executor(
    tmp_path: Path, tools: list[Tool], auto: AutoReadPolicy | None
) -> tuple[ToolExecutor, MemoryEventSink, ApprovalStore]:
    root = tmp_path / "files"
    root.mkdir(exist_ok=True)
    sink = MemoryEventSink()
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    names = frozenset(t.name for t in tools)
    ex = ToolExecutor(
        tools,
        PermissionPolicy(allowed_tools=names, max_risk=RiskLevel.READ),
        Workspace(root),
        sink,
        PID,
        approvals=store,
        task_id="t1",
        auto_reads=auto,
    )
    return ex, sink, store


def types(sink: MemoryEventSink) -> list[str]:
    return [e.type for e in sink.events]


# --- which URLs qualify -----------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://www.sqlite.org/wal.html",
        "https://www.postgresql.org/docs/current/mvcc.html",
        "https://en.wikipedia.org/wiki/SQLite",
        "https://example.com/",
    ],
)
def test_plain_https_pages_qualify(url: str) -> None:
    assert auto_read_refusal(url) is None


@pytest.mark.parametrize(
    ("url", "why"),
    [
        ("http://example.com/", "https"),
        ("https://example.com/search?q=project+secret", "query"),
        ("https://user:pw@example.com/", "credentials"),
        ("https://93.184.216.34/", "IP"),
        ("https://xn--exmple-cua.com/", "punycode"),
        ("https://example.com:8443/", "port"),
        ("https://example.com/" + "a/" * 100, "long"),
        ("https://evil.example/c2VjcmV0LXByb2plY3QtZGF0YS1lbmNvZGVkLWhlcmU", "encoded"),
        (
            "https://evil.example/%73%65%63%72%65%74%2d%64%61%74%61%2d%68%65%72%65%2d%6c%6f%6e%67",
            "encoded",
        ),
        ("ftp://example.com/", None),
        ("https://localhost/", "single-label"),
        (12, None),
    ],
)
def test_risky_urls_do_not_qualify(url: object, why: str | None) -> None:
    reason = auto_read_refusal(url)
    assert reason is not None
    if why:
        assert why.lower() in reason.lower()


# --- executor behaviour -------------------------------------------------------------


def test_off_by_default_every_fetch_needs_approval(tmp_path: Path) -> None:
    tool, seen = fetcher()
    ex, _, store = executor(tmp_path, [tool], None)
    with pytest.raises(ApprovalRequired):
        ex.execute("fetch_url", {"url": "https://www.sqlite.org/wal.html"})
    assert seen == [] and len(store.list()) == 1


def test_config_default_is_off() -> None:
    assert AutomaConfig().web.auto_approve_reads is False


def test_qualifying_read_runs_without_asking_and_is_logged(tmp_path: Path) -> None:
    tool, seen = fetcher()
    ex, sink, store = executor(tmp_path, [tool], AutoReadPolicy())
    result = ex.execute("fetch_url", {"url": "https://www.sqlite.org/wal.html"})
    assert result.ok and "hello" in result.content
    assert seen == ["https://www.sqlite.org/wal.html"]
    assert store.list() == []  # no approval request
    (auto,) = [e for e in sink.events if e.type == "tool.auto_approved"]
    assert auto.data == {"tool": "fetch_url", "host": "www.sqlite.org", "path": "/wal.html"}


def test_non_qualifying_read_still_needs_approval(tmp_path: Path) -> None:
    tool, seen = fetcher()
    ex, sink, store = executor(tmp_path, [tool], AutoReadPolicy())
    with pytest.raises(ApprovalRequired):
        ex.execute("fetch_url", {"url": "https://example.com/search?q=secret"})
    assert seen == [] and len(store.list()) == 1
    assert "tool.auto_read_refused" in types(sink)


def test_auto_reads_are_capped_per_task(tmp_path: Path) -> None:
    tool, seen = fetcher()
    ex, _, store = executor(tmp_path, [tool], AutoReadPolicy(max_per_task=2))
    assert ex.execute("fetch_url", {"url": "https://example.com/a"}).ok
    assert ex.execute("fetch_url", {"url": "https://example.com/b"}).ok
    with pytest.raises(ApprovalRequired):
        ex.execute("fetch_url", {"url": "https://example.com/c"})
    assert len(seen) == 2 and len(store.list()) == 1


def test_only_the_real_fetch_tool_is_auto_approved(tmp_path: Path) -> None:
    # A different external tool that happens to take a "url" argument is never
    # auto-approved, even when it is named like fetch_url in a subclass.
    class Lookalike(FetchUrl):
        pass

    tool = Lookalike(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="x")))
    ex, _, store = executor(tmp_path, [tool], AutoReadPolicy())
    with pytest.raises(ApprovalRequired):
        ex.execute("fetch_url", {"url": "https://example.com/"})
    assert len(store.list()) == 1


def test_auto_reads_never_widen_the_role(tmp_path: Path) -> None:
    tool, seen = fetcher()
    root = tmp_path / "files"
    root.mkdir()
    ex = ToolExecutor(
        [tool],
        PermissionPolicy(allowed_tools=frozenset(), max_risk=RiskLevel.READ),
        Workspace(root),
        MemoryEventSink(),
        PID,
        approvals=ApprovalStore(tmp_path / "approvals", PID, clock=clock),
        task_id="t1",
        auto_reads=AutoReadPolicy(),
    )
    result = ex.execute("fetch_url", {"url": "https://example.com/"})
    assert result.denied and seen == []


def test_executor_rejects_a_bad_auto_reads_value(tmp_path: Path) -> None:
    tool, _ = fetcher()
    with pytest.raises(TypeError):
        executor(tmp_path, [tool], {"max_per_task": 5})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad", [{"max_auto_reads_per_task": 0}, {"max_auto_reads_per_task": 101}, {"x": 1}]
)
def test_invalid_web_config_is_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        WebConfig.model_validate(bad)
