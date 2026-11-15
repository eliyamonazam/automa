"""``fetch_url`` (D72): URL/host checks, redirects, content handling, Approval Gate.

No network: every test uses ``httpx.MockTransport`` and a fake resolver.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from automa.events import MemoryEventSink
from automa.roles import RoleRegistry
from automa.tools import (
    ApprovalRequired,
    FetchUrl,
    FetchUrlArgs,
    PermissionPolicy,
    RiskLevel,
    ToolError,
    ToolExecutor,
    Workspace,
    default_tools,
)
from automa.tools.approvals import ApprovalStore
from automa.tools.web import (
    MAX_FETCH_BYTES,
    MAX_REDIRECTS,
    UNTRUSTED_NOTE,
    check_host,
    check_url,
    html_to_text,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "automa" / "tools" / "web.py"
PID = "web-proj"
PUBLIC = "93.184.216.34"


def public_resolver(host: str, port: int) -> list[str]:
    return [PUBLIC]


class Recorder:
    def __init__(self, handler):  # type: ignore[no-untyped-def]
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)


def tool(handler, resolver=public_resolver) -> tuple[FetchUrl, Recorder]:  # type: ignore[no-untyped-def]
    rec = Recorder(handler)
    return FetchUrl(transport=httpx.MockTransport(rec), resolver=resolver), rec


def html(body: str, status: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(
        status,
        content=body.encode(),
        headers={"content-type": "text/html; charset=utf-8", **headers},
    )


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "files"
    root.mkdir()
    return Workspace(root)


def run(t: FetchUrl, ws: Workspace, url: str, **kw: int) -> str:
    return t.run(FetchUrlArgs(url=url, **kw), ws)


# ------------------------------------------------------------------ tool contract


def test_tool_is_external_and_reversible() -> None:
    t = FetchUrl()
    assert t.name == "fetch_url" and t.risk is RiskLevel.EXTERNAL and not t.irreversible
    spec = t.spec()["function"]
    assert spec["name"] == "fetch_url" and spec["parameters"]["additionalProperties"] is False


def test_no_builtin_role_has_fetch_url() -> None:
    reg = RoleRegistry.builtin(tools=default_tools())
    assert all("fetch_url" not in r.tools for r in reg) and len(reg) > 0


def test_timeout_is_bounded() -> None:
    for bad in (0, -1, 121):
        with pytest.raises(ValueError):
            FetchUrl(timeout_s=bad)


def test_audit_records_only_the_host() -> None:
    args = FetchUrlArgs(url="https://example.com/a?token=SECRET")
    assert FetchUrl().audit(args) == {"host": "example.com"}


# ------------------------------------------------------------------ URL validation


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/x",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "https://user:pw@example.com/",
        "https://user@example.com/",
        "https:///nohost",
        "https://exa mple.com/",
        "https://example.com/\nx",
        "https://example.com:99999/",
        "example.com",
        "https://" + "a" * 2050 + ".com",
    ],
)
def test_invalid_urls_are_rejected_by_args(url: str) -> None:
    with pytest.raises(ValidationError):
        FetchUrlArgs(url=url)


def test_max_chars_bounds() -> None:
    with pytest.raises(ValidationError):
        FetchUrlArgs(url="https://example.com", max_chars=10)
    with pytest.raises(ValidationError):
        FetchUrlArgs(url="https://example.com", max_chars=10**6)
    with pytest.raises(ValidationError):
        FetchUrlArgs(url="https://example.com", extra=1)  # type: ignore[call-arg]


def test_check_url_ports() -> None:
    assert check_url("https://example.com/x") == ("example.com", 443)
    assert check_url("http://example.com/x") == ("example.com", 80)
    assert check_url("HTTP://Example.com:8080/") == ("example.com", 8080)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.3.4",
        "192.168.1.1",
        "169.254.169.254",
        "0.0.0.0",  # noqa: S104
        "100.64.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",
        "224.0.0.1",
        "not-an-ip",
    ],
)
def test_non_public_resolution_is_refused(address: str) -> None:
    with pytest.raises(ToolError, match="not a public address"):
        check_host("evil.example", 443, lambda h, p: [address])


def test_any_private_address_among_many_is_refused() -> None:
    with pytest.raises(ToolError):
        check_host("mixed.example", 443, lambda h, p: [PUBLIC, "10.0.0.1"])


def test_literal_ip_hosts_are_checked_without_resolver() -> None:
    def boom(h: str, p: int) -> list[str]:
        raise AssertionError("resolver must not be called for a literal IP")

    with pytest.raises(ToolError):
        check_host("127.0.0.1", 80, boom)
    check_host(PUBLIC, 80, boom)


def test_unresolvable_host_is_refused() -> None:
    with pytest.raises(ToolError, match="cannot resolve"):
        check_host("nothing.example", 443, lambda h, p: [])


def test_private_host_makes_no_request(ws: Workspace) -> None:
    t, rec = tool(lambda r: html("x"), resolver=lambda h, p: ["127.0.0.1"])
    with pytest.raises(ToolError):
        run(t, ws, "http://localhost/admin")
    assert rec.requests == []


# ------------------------------------------------------------------ fetching


def test_html_page_is_reduced_to_text(ws: Workspace) -> None:
    page = (
        "<html><head><title> Hello  Page </title><style>p{color:red}</style>"
        "<script>steal()</script></head><body><h1>Head</h1><p>First &amp; para</p>"
        "<noscript>no</noscript><p>Second</p></body></html>"
    )
    t, rec = tool(lambda r: html(page))
    out = run(t, ws, "https://example.com/page")
    assert out.startswith(UNTRUSTED_NOTE)
    assert "URL: https://example.com/page" in out and "Status: 200" in out
    assert "Title: Hello Page" in out
    assert "Head\n" in out and "First & para" in out and "Second" in out
    assert "steal" not in out and "color" not in out and "<p>" not in out and "no\n" not in out
    req = rec.requests[0]
    assert req.method == "GET" and req.headers["user-agent"].startswith("automa/")
    assert "cookie" not in req.headers and "authorization" not in req.headers


def test_plain_text_and_json_are_returned(ws: Workspace) -> None:
    t, _ = tool(
        lambda r: httpx.Response(200, text="a <b> c", headers={"content-type": "text/plain"})
    )
    assert run(t, ws, "https://example.com/t").endswith("a <b> c")
    t, _ = tool(lambda r: httpx.Response(200, json={"k": 1}))
    assert '"k"' in run(t, ws, "https://example.com/j")


def test_charset_is_respected(ws: Workspace) -> None:
    body = "héllo".encode("latin-1")
    t, _ = tool(
        lambda r: httpx.Response(
            200, content=body, headers={"content-type": "text/plain; charset=latin-1"}
        )
    )
    assert run(t, ws, "https://example.com/").endswith("héllo")
    t, _ = tool(
        lambda r: httpx.Response(
            200, content=b"ok", headers={"content-type": "text/plain; charset=nope"}
        )
    )
    assert run(t, ws, "https://example.com/").endswith("ok")


def test_binary_content_is_refused(ws: Workspace) -> None:
    t, _ = tool(
        lambda r: httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})
    )
    with pytest.raises(ToolError, match="unsupported content type"):
        run(t, ws, "https://example.com/x.png")


def test_http_error_status(ws: Workspace) -> None:
    t, _ = tool(lambda r: html("gone", status=404))
    with pytest.raises(ToolError, match="HTTP 404"):
        run(t, ws, "https://example.com/missing")


def test_text_is_truncated_to_max_chars(ws: Workspace) -> None:
    t, _ = tool(
        lambda r: httpx.Response(200, text="x" * 5000, headers={"content-type": "text/plain"})
    )
    out = run(t, ws, "https://example.com/", max_chars=100)
    assert out.endswith("x" * 100 + "\n[... truncated]")


def test_body_size_is_capped(ws: Workspace) -> None:
    big = b"y" * (MAX_FETCH_BYTES * 2)
    t, _ = tool(lambda r: httpx.Response(200, content=big, headers={"content-type": "text/plain"}))
    out = run(t, ws, "https://example.com/")
    assert "[... truncated]" in out


def test_timeout_and_transport_errors(ws: Workspace) -> None:
    def timeout(r: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=r)

    def broken(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=r)

    with pytest.raises(ToolError, match="timed out"):
        run(tool(timeout)[0], ws, "https://example.com/")
    with pytest.raises(ToolError, match="request failed: ConnectError"):
        run(tool(broken)[0], ws, "https://example.com/")


# ------------------------------------------------------------------ redirects


def test_redirect_is_followed_and_rechecked(ws: Workspace) -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        if r.url.path == "/old":
            return httpx.Response(302, headers={"location": "/new"})
        return html("<p>moved here</p>")

    t, rec = tool(handler)
    out = run(t, ws, "https://example.com/old")
    assert "URL: https://example.com/new" in out and "moved here" in out
    assert [str(r.url) for r in rec.requests] == [
        "https://example.com/old",
        "https://example.com/new",
    ]


def test_redirect_to_private_host_is_refused(ws: Workspace) -> None:
    def resolver(host: str, port: int) -> list[str]:
        return ["10.0.0.1"] if host == "internal.example" else [PUBLIC]

    t, rec = tool(
        lambda r: httpx.Response(302, headers={"location": "http://internal.example/secret"}),
        resolver=resolver,
    )
    with pytest.raises(ToolError, match="not a public address"):
        run(t, ws, "https://example.com/")
    assert len(rec.requests) == 1


def test_redirect_to_metadata_ip_is_refused(ws: Workspace) -> None:
    t, rec = tool(
        lambda r: httpx.Response(301, headers={"location": "http://169.254.169.254/latest"})
    )
    with pytest.raises(ToolError):
        run(t, ws, "https://example.com/")
    assert len(rec.requests) == 1


def test_redirect_to_other_scheme_is_refused(ws: Workspace) -> None:
    t, _ = tool(lambda r: httpx.Response(302, headers={"location": "file:///etc/passwd"}))
    with pytest.raises(ToolError, match="only http and https"):
        run(t, ws, "https://example.com/")


def test_redirect_without_location(ws: Workspace) -> None:
    t, _ = tool(lambda r: httpx.Response(302))
    with pytest.raises(ToolError, match="without a location"):
        run(t, ws, "https://example.com/")


def test_too_many_redirects(ws: Workspace) -> None:
    t, rec = tool(lambda r: httpx.Response(302, headers={"location": "/again"}))
    with pytest.raises(ToolError, match="too many redirects"):
        run(t, ws, "https://example.com/")
    assert len(rec.requests) == MAX_REDIRECTS + 1


# ------------------------------------------------------------------ html_to_text


def test_html_to_text_collapses_whitespace() -> None:
    title, text = html_to_text("<div>  a\n\n   b </div><br><ul><li>x</li><li>y</li></ul>")
    assert title == "" and text == "a\n\nb\n\nx\n\ny"


def test_html_to_text_handles_broken_markup() -> None:
    _, text = html_to_text("<p>open <b>bold <script>x()")
    assert "open" in text and "x()" not in text


# ------------------------------------------------------------------ Approval Gate


def clock() -> datetime:
    return datetime(2026, 11, 15, 12, 0, tzinfo=UTC)


def gated(
    tmp_path: Path, t: FetchUrl, store: ApprovalStore | None
) -> tuple[ToolExecutor, MemoryEventSink]:
    root = tmp_path / "files"
    root.mkdir(exist_ok=True)
    sink = MemoryEventSink()
    ex = ToolExecutor(
        [t],
        PermissionPolicy(allowed_tools=frozenset({"fetch_url"}), max_risk=RiskLevel.READ),
        Workspace(root),
        sink,
        PID,
        approvals=store,
        task_id="t1",
    )
    return ex, sink


def test_every_call_needs_approval_and_nothing_is_fetched_before(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    t, rec = tool(lambda r: html("<p>page</p>"))
    ex, sink = gated(tmp_path, t, store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("fetch_url", {"url": "https://example.com/a?token=SECRET"})
    assert rec.requests == []
    req = store.get(ei.value.request_id or "")
    assert req.tool == "fetch_url" and req.arguments["url"].endswith("token=SECRET")
    store.decide(req.id, approve=True)
    result = ex.execute("fetch_url", {"url": "https://example.com/a?token=SECRET"})
    assert result.ok and "page" in result.content and len(rec.requests) == 1
    # The approval is used up: the same call needs a new one.
    with pytest.raises(ApprovalRequired):
        ex.execute("fetch_url", {"url": "https://example.com/a?token=SECRET"})
    assert len(rec.requests) == 1
    calls = sink.of_type("tool.call")
    assert calls and all("SECRET" not in str(e.data) for e in calls)


def test_approval_for_one_url_does_not_cover_another(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    t, rec = tool(lambda r: html("<p>page</p>"))
    ex, _ = gated(tmp_path, t, store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("fetch_url", {"url": "https://example.com/a"})
    store.decide(ei.value.request_id or "", approve=True)
    with pytest.raises(ApprovalRequired):
        ex.execute("fetch_url", {"url": "https://example.com/b"})
    assert rec.requests == []


def test_rejected_call_fetches_nothing(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    t, rec = tool(lambda r: html("<p>page</p>"))
    ex, _ = gated(tmp_path, t, store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("fetch_url", {"url": "https://example.com/a"})
    store.decide(ei.value.request_id or "", approve=False, note="no")
    result = ex.execute("fetch_url", {"url": "https://example.com/a"})
    assert not result.ok and rec.requests == []


def test_invalid_url_never_reaches_the_gate(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    t, rec = tool(lambda r: html("x"))
    ex, _ = gated(tmp_path, t, store)
    result = ex.execute("fetch_url", {"url": "file:///etc/passwd"})
    assert not result.ok and store.list() == [] and rec.requests == []


def test_without_store_nothing_is_offered_or_fetched(tmp_path: Path) -> None:
    t, rec = tool(lambda r: html("x"))
    ex, _ = gated(tmp_path, t, None)
    assert ex.specs() == []
    with pytest.raises(ApprovalRequired):
        ex.execute("fetch_url", {"url": "https://example.com/"})
    assert rec.requests == []


def test_approved_but_private_is_still_refused(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    t, rec = tool(lambda r: html("x"), resolver=lambda h, p: ["192.168.0.10"])
    ex, _ = gated(tmp_path, t, store)
    with pytest.raises(ApprovalRequired) as ei:
        ex.execute("fetch_url", {"url": "http://router.example/"})
    store.decide(ei.value.request_id or "", approve=True)
    result = ex.execute("fetch_url", {"url": "http://router.example/"})
    assert not result.ok and "not a public address" in (result.error or "")
    assert rec.requests == []


def test_unlisted_fetch_url_is_denied(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals", PID, clock=clock)
    t, rec = tool(lambda r: html("x"))
    root = tmp_path / "files"
    root.mkdir()
    ex = ToolExecutor(
        [t],
        PermissionPolicy(allowed_tools=frozenset(), max_risk=RiskLevel.EXECUTE),
        Workspace(root),
        MemoryEventSink(),
        PID,
        approvals=store,
        task_id="t1",
    )
    result = ex.execute("fetch_url", {"url": "https://example.com/"})
    assert result.denied and store.list() == [] and rec.requests == []


# ------------------------------------------------------------------ structure


def test_client_never_trusts_environment_or_follows_redirects() -> None:
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    clients = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Client"
    ]
    assert len(clients) == 1
    kw = {k.arg: k.value for k in clients[0].keywords}
    assert isinstance(kw["trust_env"], ast.Constant) and kw["trust_env"].value is False
    assert (
        isinstance(kw["follow_redirects"], ast.Constant) and kw["follow_redirects"].value is False
    )
