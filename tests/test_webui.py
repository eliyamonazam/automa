"""Tests for the read-only web UI core (``automa.webui``, D77): views and HTML rendering."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from automa.config import Tier
from automa.plan import (
    AcceptanceCriterion,
    Issue,
    Plan,
    ResultStatus,
    Task,
    TaskResult,
    Verdict,
    VerdictOutcome,
)
from automa.project import ProjectStore
from automa.tools.base import RiskLevel
from automa.webui import (
    CONTENT_SECURITY_POLICY,
    approval_view,
    build_overview,
    build_project_view,
    render_overview,
    render_project,
    text,
)

T0 = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)


def _task(tid: str, deps: list[str] | None = None, title: str | None = None) -> Task:
    return Task(
        id=tid,
        title=title or f"task {tid}",
        role="writer",
        depends_on=deps or [],
        acceptance_criteria=[AcceptanceCriterion(description="it is good")],
    )


def _result(tid: str, attempt: int, status: ResultStatus) -> TaskResult:
    return TaskResult(
        task_id=tid,
        attempt=attempt,
        status=status,
        tier=Tier.MEDIUM,
        model="fake-medium",
        error="gave up" if status is ResultStatus.FAILED else None,
        started_at=T0,
        finished_at=T0,
    )


@pytest.fixture
def store(tmp_path: Path) -> ProjectStore:
    return ProjectStore.for_workspace(tmp_path)


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else b"<dir>"
        for p in sorted(root.rglob("*"))
    }


def _request(store: ProjectStore, pid: str, tool: str = "publish", **args: object) -> None:
    store.approvals(pid).request(
        tool=tool,
        risk=RiskLevel.EXTERNAL,
        arguments=args or {"text": "hello"},
        reason="external tool needs approval",
        task_id="a",
    )


# -- views ------------------------------------------------------------------------


def test_overview_lists_projects_with_cost_and_pending(store: ProjectStore) -> None:
    store.create("first goal", project_id="p1")
    store.create("second   goal\nwith lines", project_id="p2")
    _request(store, "p2")
    view = build_overview(store, {"p1": 2_500_000_000, "other": 7})
    assert [r.id for r in view.projects] == ["p2", "p1"]  # newest first (D138)
    p2, p1 = view.projects
    assert p1.cost_usd == 2.5 and p2.cost_usd == 0
    assert p1.pending_approvals == 0 and p2.pending_approvals == 1
    assert p2.goal_preview == "second goal with lines"
    assert p2.title == "second goal"
    assert view.errors == ()


def test_overview_of_empty_workspace_creates_nothing(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path / "ws")
    view = build_overview(store, {})
    assert view.projects == ()
    assert not (tmp_path / "ws").exists()
    assert "No projects yet" in render_overview(view)


def test_overview_reports_corrupt_project_without_hiding_others(store: ProjectStore) -> None:
    store.create("good", project_id="good")
    store.create("bad", project_id="bad")
    store.project_file("bad").write_text("{not json", encoding="utf-8")
    view = build_overview(store, {})
    assert [r.id for r in view.projects] == ["good"]
    assert len(view.errors) == 1 and view.errors[0].startswith("bad:")
    assert "Unreadable data" in render_overview(view)


def test_project_view_with_plan_attempts_and_approvals(store: ProjectStore) -> None:
    store.create("build it", project_id="p1")
    store.save_plan("p1", Plan(goal="build it", tasks=[_task("b", ["a"]), _task("a")]))
    store.save_task_result("p1", _result("a", 1, ResultStatus.FAILED))
    store.save_verdict(
        "p1",
        Verdict(
            task_id="a",
            attempt=1,
            outcome=VerdictOutcome.FAIL,
            issues=[Issue(description="not yet")],
            created_at=T0,
        ),
    )
    store.save_task_result("p1", _result("a", 2, ResultStatus.COMPLETED))
    _request(store, "p1")
    view = build_project_view(store, "p1", 1_000_000_000)
    assert view.cost_usd == 1.0
    assert view.plan_version == 1
    assert [(t.order, t.id, t.attempts, t.last_result) for t in view.tasks] == [
        (1, "a", 2, "completed"),
        (2, "b", 0, None),
    ]
    assert view.tasks[1].depends_on == ("a",)
    assert len(view.pending) == 1 and view.pending[0].tool == "publish"
    html = render_project(view)
    assert "Plan v1 (2 tasks)" in html and "completed" in html
    assert "Pending approvals (1)" in html


def test_project_view_shows_only_pending_approvals(store: ProjectStore) -> None:
    store.create("g", project_id="p1")
    _request(store, "p1", text="one")
    approvals = store.approvals("p1")
    first = approvals.list()[0]
    approvals.decide(first.id, approve=True)
    _request(store, "p1", text="two")
    view = build_project_view(store, "p1", 0)
    assert [a.arguments for a in view.pending] == [{"text": "two"}]


def test_project_view_without_plan(store: ProjectStore) -> None:
    store.create("g", project_id="p1")
    view = build_project_view(store, "p1", 0)
    assert view.plan_version is None and view.tasks == ()
    assert "No plan." in render_project(view)


def test_project_view_corrupt_plan_is_an_error_line(store: ProjectStore) -> None:
    store.create("g", project_id="p1")
    store.plan_file("p1").write_text("[]", encoding="utf-8")
    view = build_project_view(store, "p1", 0)
    assert view.plan_version is None
    assert view.errors and view.errors[0].startswith("plan:")


def test_project_view_corrupt_approval_is_an_error_line(store: ProjectStore) -> None:
    store.create("g", project_id="p1")
    _request(store, "p1")
    path = next(store.approvals("p1").directory.glob("*.json"))
    path.write_text("{", encoding="utf-8")
    view = build_project_view(store, "p1", 0)
    assert view.pending == ()
    assert view.errors and view.errors[0].startswith("approvals:")


def test_views_and_rendering_are_read_only(store: ProjectStore, tmp_path: Path) -> None:
    store.create("g", project_id="p1")
    store.save_plan("p1", Plan(goal="g", tasks=[_task("a")]))
    _request(store, "p1")
    before = _tree(tmp_path)
    render_overview(build_overview(store, {}))
    render_project(build_project_view(store, "p1", 0))
    assert _tree(tmp_path) == before


def test_fetch_url_request_target_is_shown(store: ProjectStore) -> None:
    store.create("g", project_id="p1")
    _request(store, "p1", tool="fetch_url", url="https://example.com:8443/a/b?x=1&y=2")
    view = build_project_view(store, "p1", 0).pending[0]
    assert "host: example.com" in view.request_lines
    assert "port: 8443" in view.request_lines
    assert "query parameters: 2" in view.request_lines


def test_invalid_fetch_url_says_the_tool_refuses(store: ProjectStore) -> None:
    store.create("g", project_id="p1")
    _request(store, "p1", tool="fetch_url", url="ftp://example.com/")
    view = build_project_view(store, "p1", 0).pending[0]
    assert "the tool will refuse it" in view.request_lines[0]


def test_github_request_target_is_shown(store: ProjectStore) -> None:
    store.create("g", project_id="p1")
    _request(store, "p1", tool="github_read", kind="issue", owner="octo", repo="hello", number=3)
    view = build_project_view(store, "p1", 0).pending[0]
    assert "api url: https://api.github.com/repos/octo/hello/issues/3" in view.request_lines


def test_other_tools_have_no_target_lines(store: ProjectStore) -> None:
    store.create("g", project_id="p1")
    _request(store, "p1")
    req = store.approvals("p1").list()[0]
    assert approval_view(req).request_lines == ()


# -- rendering safety ---------------------------------------------------------------


@pytest.mark.parametrize(
    "evil",
    [
        "<script>alert(1)</script>",
        '"><img src=x onerror=alert(1)>',
        "a‮evil‬",  # bidi override
        "bell\x07 and esc\x1b[31m",
        "</pre><a href='javascript:x'>",
    ],
)
def test_recorded_text_is_escaped_everywhere(store: ProjectStore, evil: str) -> None:
    store.create(f"goal {evil}", project_id="p1")
    store.save_plan("p1", Plan(goal="g", tasks=[_task("a", title=f"title {evil}")]))
    store.approvals("p1").request(
        tool="publish",
        risk=RiskLevel.EXTERNAL,
        arguments={"text": evil, evil: 1},
        reason=f"reason {evil}",
        task_id="a",
    )
    pages = [
        render_overview(build_overview(store, {})),
        render_project(build_project_view(store, "p1", 0)),
    ]
    for html in pages:
        assert "<script" not in html
        assert "<img" not in html
        assert "<a href='javascript" not in html
        assert "‮" not in html and "\x07" not in html and "\x1b" not in html
    if "\u202e" in evil:
        assert "\\u202e" in pages[1]


def test_text_keeps_persian_and_joiners() -> None:
    value = "می‌خواهم «سلام»"
    assert text(value) == value


def test_text_escapes_quotes_and_controls() -> None:
    assert text("a\"b'c<d>&") == "a&quot;b&#x27;c&lt;d&gt;&amp;"
    assert text("x⁦y\x00") == "x\\u2066y\\x00"


def test_pages_have_no_scripts_forms_or_external_resources(store: ProjectStore) -> None:
    store.create("g", project_id="p1")
    _request(store, "p1", tool="fetch_url", url="https://example.com/")
    pages = [
        render_overview(build_overview(store, {})),
        render_project(build_project_view(store, "p1", 0)),
    ]
    for html in pages:
        assert html.startswith("<!doctype html>")
        assert CONTENT_SECURITY_POLICY in html
        assert "default-src 'none'" in CONTENT_SECURITY_POLICY
        assert "form-action 'none'" in CONTENT_SECURITY_POLICY
        assert "<script" not in html and "<form" not in html
        assert "<link" not in html and "<iframe" not in html
        hrefs = re.findall(r'href="([^"]*)"', html)
        assert all(h.startswith("/") and not h.startswith("//") for h in hrefs)
        assert 'src="' not in html


def test_project_page_points_to_cli_to_decide(store: ProjectStore) -> None:
    store.create("g", project_id="p1")
    _request(store, "p1")
    html = render_project(build_project_view(store, "p1", 0))
    assert "decide in the terminal" in html and "automa approvals" not in html
    assert "approve" not in html.lower().replace("approvals", "")


# -- cleaner home page (D138) --------------------------------------------------------


def test_project_title_from_goal() -> None:
    from automa.webui.views import project_title

    assert project_title("\n  Build a site  \nmore details") == "Build a site"
    long = "یک فایل index.html تک‌صفحه‌ای بساز: ابزار تبدیل اعداد فارسی و انگلیسی با دکمه کپی"
    t = project_title(long)
    assert t.endswith("…") and len(t) <= 61 and t.startswith("یک فایل")
    assert project_title("a\r\nb") == "a"
    assert project_title("") == ""


def test_overview_hides_benchmarks_unless_asked(tmp_path: Path) -> None:
    from automa.webui.server import WebApp, allowed_hosts

    store = ProjectStore.for_workspace(tmp_path)
    store.create("Real idea\nwith details", project_id="p-20261119-100000-aaaaaa")
    store.create("bench goal", project_id="bench-20261119-090000-cli-word-count")
    app = WebApp(tmp_path, tmp_path / "ledger.sqlite3", allowed_hosts("127.0.0.1", 8765))
    home = app.handle("GET", "/", "127.0.0.1:8765").body.decode()
    assert "Real idea" in home and "bench-20261119" not in home
    assert "1 test runs hidden" in home and 'href="/?tests=1"' in home
    shown = app.handle("GET", "/?tests=1", "127.0.0.1:8765").body.decode()
    assert "bench-20261119" in shown and "Hide test runs" in shown
    odd = app.handle("GET", "/?a=1&b=2&c=3&d=4&e=5&tests=1", "127.0.0.1:8765")
    assert odd.status == 200 and "bench-20261119" not in odd.body.decode()


def test_welcome_when_empty_in_action_mode() -> None:
    from automa.webui.render import render_overview
    from automa.webui.views import Overview

    doc = render_overview(Overview(), actions=True, ideas=True)
    assert "Welcome" in doc and 'href="/new"' in doc and "For example" in doc
