"""``web_search`` core (D74): args, provider contract, result checks, Approval Gate.

No network and no real search service: every test uses ``FakeSearchProvider``.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from automa.events import MemoryEventSink
from automa.roles import RoleRegistry
from automa.tools import (
    ApprovalRequired,
    FakeSearchProvider,
    PermissionPolicy,
    RiskLevel,
    SearchHit,
    SearchProviderError,
    ToolError,
    ToolExecutor,
    WebSearch,
    WebSearchArgs,
    Workspace,
    default_tools,
)
from automa.tools.approvals import ApprovalStore
from automa.tools.search import (
    MAX_QUERY_CHARS,
    MAX_RESULTS,
    MAX_SNIPPET_CHARS,
    MAX_TITLE_CHARS,
    UNTRUSTED_SEARCH_NOTE,
    clean_text,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "automa" / "tools" / "search.py"
PID = "search-proj"
Q = "python packaging guide"


def hit(n: int, **kw: str) -> SearchHit:
    data = {"title": f"Title {n}", "url": f"https://example.com/{n}", "snippet": f"About {n}"}
    data.update(kw)
    return SearchHit(**data)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "files"
    root.mkdir()
    return Workspace(root)


def run(t: WebSearch, ws: Workspace, query: str = Q, **kw: int) -> str:
    return t.run(WebSearchArgs(query=query, **kw), ws)


# ------------------------------------------------------------------ contract


def test_tool_is_external_and_reversible() -> None:
    assert WebSearch.risk is RiskLevel.EXTERNAL
    assert WebSearch.irreversible is False
    assert WebSearch.name == "web_search"
    assert "approval" in WebSearch.description and "untrusted" in WebSearch.description


def test_not_in_default_tools_and_no_role_lists_it() -> None:
    assert "web_search" not in {t.name for t in default_tools()}
    reg = RoleRegistry.builtin(tools=[*default_tools(), WebSearch()])
    assert [r.name for r in reg if "web_search" in r.tools] == []


def test_spec_describes_the_arguments() -> None:
    spec = WebSearch(FakeSearchProvider()).spec()
    props = spec["function"]["parameters"]["properties"]
    assert set(props) == {"query", "max_results"}


def test_provider_must_follow_the_protocol() -> None:
    with pytest.raises(TypeError):
        WebSearch(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["", "Fake", "1x", "a b", "x" * 40, "bad/name"])
def test_provider_name_is_validated(name: str) -> None:
    with pytest.raises(ValueError):
        WebSearch(FakeSearchProvider(name=name))


def test_without_provider_every_call_is_refused(ws: Workspace) -> None:
    t = WebSearch()
    assert t.provider_name is None
    with pytest.raises(ToolError, match="no web search service"):
        run(t, ws)


# ------------------------------------------------------------------ arguments


@pytest.mark.parametrize(
    "query",
    ["", "   ", "a\nb", "a\tb", "x\x00", "x\x1b[31m", "‮evil", "a​b", "q" * 401],
)
def test_invalid_queries_are_rejected(query: str) -> None:
    with pytest.raises(ValidationError):
        WebSearchArgs(query=query)


def test_valid_queries_are_accepted() -> None:
    assert WebSearchArgs(query="جست‌وجوی وب").query == "جست‌وجوی وب"  # ZWNJ allowed
    assert WebSearchArgs(query="q" * MAX_QUERY_CHARS).max_results == 5


@pytest.mark.parametrize("n", [0, -1, MAX_RESULTS + 1])
def test_max_results_bounds(n: int) -> None:
    with pytest.raises(ValidationError):
        WebSearchArgs(query=Q, max_results=n)


def test_unknown_arguments_are_rejected() -> None:
    with pytest.raises(ValidationError):
        WebSearchArgs(query=Q, provider="other")  # type: ignore[call-arg]


def test_audit_records_provider_and_size_not_the_query() -> None:
    t = WebSearch(FakeSearchProvider(name="fake"))
    audit = t.audit(WebSearchArgs(query="secret project codename"))
    assert audit == {"provider": "fake", "query_chars": "23"}
    assert WebSearch().audit(WebSearchArgs(query=Q))["provider"] == "none"


# ------------------------------------------------------------------ results


def test_results_are_rendered_with_untrusted_header(ws: Workspace) -> None:
    provider = FakeSearchProvider({Q: [hit(1), hit(2)]})
    out = run(WebSearch(provider), ws)
    assert out.startswith(UNTRUSTED_SEARCH_NOTE)
    assert f"Query: {Q}" in out and "Results: 2" in out
    assert "1. Title 1" in out and "URL: https://example.com/2" in out and "About 2" in out
    assert provider.queries == [(Q, 5)]


def test_max_results_is_passed_and_enforced(ws: Workspace) -> None:
    provider = FakeSearchProvider({Q: [hit(i) for i in range(8)]})
    out = run(WebSearch(provider), ws, max_results=3)
    assert provider.queries == [(Q, 3)]
    assert "Results: 3" in out and "example.com/3" not in out


def test_no_results(ws: Workspace) -> None:
    out = run(WebSearch(FakeSearchProvider()), ws)
    assert "Results: 0" in out and "No results." in out


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "ftp://example.com/x",
        "https://user:pw@example.com/",
        "https://exa mple.com/",
        "https://example.com/‮txt.exe",
        "https:///nohost",
        "https://example.com/" + "a" * 3000,
    ],
)
def test_unsafe_result_urls_are_dropped(ws: Workspace, url: str) -> None:
    provider = FakeSearchProvider({Q: [SearchHit(title="bad", url=url), hit(1)]})
    out = run(WebSearch(provider), ws)
    assert "Results: 1" in out and "Skipped results: 1" in out
    assert "bad" not in out.split("Skipped")[1]


def test_duplicate_urls_are_dropped(ws: Workspace) -> None:
    provider = FakeSearchProvider({Q: [hit(1), hit(1, title="again"), hit(2)]})
    out = run(WebSearch(provider), ws)
    assert "Results: 2" in out and "again" not in out and "Skipped results: 1" in out


def test_non_hit_items_are_dropped(ws: Workspace) -> None:
    class Loose(FakeSearchProvider):
        def search(self, query: str, *, max_results: int) -> Sequence[SearchHit]:
            return [{"url": "https://example.com/"}, hit(1)]  # type: ignore[list-item]

    out = run(WebSearch(Loose()), ws)
    assert "Results: 1" in out and "Skipped results: 1" in out


@pytest.mark.parametrize("answer", [None, "text", b"bytes", 42, {"a": 1}])
def test_invalid_provider_answer(ws: Workspace, answer: object) -> None:
    class Broken(FakeSearchProvider):
        def search(self, query: str, *, max_results: int) -> Sequence[SearchHit]:
            return answer  # type: ignore[return-value]

    with pytest.raises(ToolError, match="invalid answer"):
        run(WebSearch(Broken()), ws)


def test_titles_and_snippets_are_cleaned_and_shortened(ws: Workspace) -> None:
    long_hit = hit(
        1,
        title="Evil\x1b[2J\r\nTitle‮ " + "t" * 600,
        snippet="line1\nline2\x07 " + "s" * 3000,
    )
    out = run(WebSearch(FakeSearchProvider({Q: [long_hit]})), ws)
    assert "\x1b" not in out and "\r" not in out and "‮" not in out and "\x07" not in out
    title_line = next(line for line in out.splitlines() if line.startswith("1. "))
    assert len(title_line) <= MAX_TITLE_CHARS + 3 and title_line.endswith("…")
    snippet_line = out.splitlines()[-1]
    assert snippet_line.strip().startswith("line1 line2")
    assert len(snippet_line.strip()) <= MAX_SNIPPET_CHARS


def test_missing_title_is_shown(ws: Workspace) -> None:
    out = run(WebSearch(FakeSearchProvider({Q: [hit(1, title="  ")]})), ws)
    assert "1. (no title)" in out


def test_clean_text_keeps_zwnj_and_collapses_space() -> None:
    assert clean_text("می‌خواهم   \n  این", 100) == "می‌خواهم این"
    assert clean_text("abcdef", 4) == "abc…"


def test_provider_error_text_is_not_shown(ws: Workspace) -> None:
    t = WebSearch(FakeSearchProvider(error=SearchProviderError("401 key=sk-SECRET")))
    with pytest.raises(ToolError) as ei:
        run(t, ws)
    assert "SECRET" not in str(ei.value) and "failed" in str(ei.value)


def test_unexpected_provider_exception_is_contained(ws: Workspace) -> None:
    t = WebSearch(FakeSearchProvider(error=RuntimeError("token=SECRET")))
    with pytest.raises(ToolError) as ei:
        run(t, ws)
    assert "SECRET" not in str(ei.value) and "RuntimeError" in str(ei.value)


def test_output_never_exceeds_the_tool_limit(ws: Workspace) -> None:
    from automa.tools.base import MAX_TOOL_OUTPUT_CHARS

    hits = [hit(i, title="t" * 400, snippet="s" * 2000) for i in range(MAX_RESULTS)]
    out = run(WebSearch(FakeSearchProvider({Q: hits})), ws, max_results=MAX_RESULTS)
    assert len(out) <= MAX_TOOL_OUTPUT_CHARS + 20


def test_run_requires_validated_args(ws: Workspace) -> None:
    with pytest.raises(TypeError):
        WebSearch(FakeSearchProvider()).run(object(), ws)  # type: ignore[arg-type]


# ------------------------------------------------------------------ Approval Gate


def clock() -> datetime:
    return datetime(2026, 11, 15, 12, 0, tzinfo=UTC)


def gated(
    tmp_path: Path, t: WebSearch, store: ApprovalStore | None, allowed: frozenset[str] | None = None
) -> tuple[ToolExecutor, MemoryEventSink]:
    root = tmp_path / "files"
    root.mkdir(exist_ok=True)
    sink = MemoryEventSink()
    ex = ToolExecutor(
        [t],
        PermissionPolicy(
            allowed_tools=frozenset({"web_search"}) if allowed is None else allowed,
            max_risk=RiskLevel.READ,
        ),
        Workspace(root),
        sink,
        PID,
        approvals=store,
        task_id="t1",
    )
    return ex, sink


def test_every_call_needs_approval_and_nothing_is_sent_before(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    provider = FakeSearchProvider({"codename x": [hit(1)]})
    ex, sink = gated(tmp_path, WebSearch(provider), store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("web_search", {"query": "codename x"})
    assert provider.queries == []
    req = store.get(ei.value.request_id or "")
    assert req.tool == "web_search" and req.arguments["query"] == "codename x"
    store.decide(req.id, approve=True)
    result = ex.execute("web_search", {"query": "codename x"})
    assert result.ok and "Title 1" in result.content and len(provider.queries) == 1
    with pytest.raises(ApprovalRequired):
        ex.execute("web_search", {"query": "codename x"})
    assert len(provider.queries) == 1
    calls = sink.of_type("tool.call")
    assert calls and all("codename" not in str(e.data) for e in calls)


def test_approval_for_one_query_does_not_cover_another(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    provider = FakeSearchProvider()
    ex, _ = gated(tmp_path, WebSearch(provider), store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("web_search", {"query": "a"})
    store.decide(ei.value.request_id or "", approve=True)
    with pytest.raises(ApprovalRequired):
        ex.execute("web_search", {"query": "b"})
    with pytest.raises(ApprovalRequired):
        ex.execute("web_search", {"query": "a", "max_results": 10})
    assert provider.queries == []


def test_rejected_call_sends_nothing(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    provider = FakeSearchProvider()
    ex, _ = gated(tmp_path, WebSearch(provider), store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("web_search", {"query": "a"})
    store.decide(ei.value.request_id or "", approve=False, note="no")
    result = ex.execute("web_search", {"query": "a"})
    assert not result.ok and provider.queries == []


def test_invalid_query_never_reaches_the_gate(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    provider = FakeSearchProvider()
    ex, _ = gated(tmp_path, WebSearch(provider), store)
    result = ex.execute("web_search", {"query": "a\x1b[2Jb"})
    assert not result.ok and store.list() == [] and provider.queries == []


def test_without_store_nothing_is_offered_or_sent(tmp_path: Path) -> None:
    provider = FakeSearchProvider()
    ex, _ = gated(tmp_path, WebSearch(provider), None)
    assert ex.specs() == []
    with pytest.raises(ApprovalRequired):
        ex.execute("web_search", {"query": "a"})
    assert provider.queries == []


def test_unlisted_web_search_is_denied(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    provider = FakeSearchProvider()
    ex, _ = gated(tmp_path, WebSearch(provider), store, allowed=frozenset())
    result = ex.execute("web_search", {"query": "a"})
    assert result.denied and store.list() == [] and provider.queries == []


def test_approved_call_without_provider_fails_cleanly(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    ex, _ = gated(tmp_path, WebSearch(), store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("web_search", {"query": "a"})
    store.decide(ei.value.request_id or "", approve=True)
    result = ex.execute("web_search", {"query": "a"})
    assert not result.ok and "no web search service" in (result.error or "")


# ------------------------------------------------------------------ structure


def test_module_does_no_network_io() -> None:
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not imported & {"httpx", "socket", "urllib3", "requests", "subprocess", "os"}
