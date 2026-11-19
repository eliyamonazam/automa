"""Tests for readable money on the web UI (M13, D133).

Every amount reads first in rounded toman ("about 145 thousand toman") when a rate is
set, with dollars to two decimals in parentheses; planning and doing the tasks are
shown apart. Display only: nothing here touches BudgetGuard or the Ledger's exact
figures. Model calls only through ``FakeProvider``; no network.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from automa import cli
from automa.budget import NANO_PER_USD
from automa.webui import money as mo
from automa.webui import progress as pg
from automa.webui import render as rd
from automa.webui import server as ws
from automa.webui.views import Overview, ProjectRow, ProjectView, build_project_view
from test_webui_ideas import HOST, LEDGER, PLAN, PORT, TOKEN, RealEnv, _settings

RATE = 323_000.0
T0 = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)


# --- toman_parts / toman_text --------------------------------------------------------


@pytest.mark.parametrize(
    ("usd", "number", "unit"),
    [
        (0, "0", "toman"),
        (0.0000001, "under 100", "toman"),  # 0.03 toman
        (0.0001, "under 100", "toman"),  # 32.3 toman
        (0.0002, "100", "toman"),  # 64.6 toman
        (0.0012, "400", "toman"),  # 387.6 toman
        (0.003, "1", "thousand toman"),  # 969 toman -> rounds to 1,000
        (0.0031, "1", "thousand toman"),  # 1,001.3 toman
        (0.005, "1.6", "thousand toman"),  # 1,615 toman
        (0.0304, "9.8", "thousand toman"),  # 9,819.2 toman
        (0.0309, "10", "thousand toman"),  # 9,980.7 toman -> 10.0 -> whole number
        (0.1, "32", "thousand toman"),  # 32,300 toman
        (0.45, "145", "thousand toman"),  # 145,350 toman
        (3.0959, "1", "million toman"),  # 999,975.7 toman -> 1,000 thousand
        (3.0, "969", "thousand toman"),
        (5, "1.6", "million toman"),  # 1,615,000
        (40, "13", "million toman"),  # 12,920,000
        (10_000, "3,230", "million toman"),
    ],
)
def test_toman_parts(usd: float, number: str, unit: str) -> None:
    assert mo.toman_parts(usd, RATE) == (number, unit)


def test_toman_parts_never_shows_a_trailing_zero_decimal() -> None:
    assert mo.toman_parts(Decimal(2) / Decimal(323), RATE) == ("2", "thousand toman")


def test_toman_text_wording() -> None:
    assert mo.toman_text(0.45, RATE) == "about 145 thousand toman"
    assert mo.toman_text(0, RATE) == "0 toman"
    assert mo.toman_text(0.0001, RATE) == "under 100 toman"


def test_toman_half_up_rounding() -> None:
    # 1,500 toman exactly -> 1.5 thousand; 150 toman -> 200 (half up, not to even)
    assert mo.toman_parts(Decimal(1500), 1) == ("1.5", "thousand toman")
    assert mo.toman_parts(Decimal(150), 1) == ("200", "toman")
    assert mo.toman_parts(Decimal(250), 1) == ("300", "toman")
    assert mo.toman_parts(Decimal(12_500), 1) == ("13", "thousand toman")


# --- dollars ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("usd", "shown"),
    [
        (0, "$0.00"),
        (0.0001, "under $0.01"),
        (0.0049, "under $0.01"),
        (0.005, "$0.01"),
        (0.125, "$0.13"),
        (0.45, "$0.45"),
        (3, "$3.00"),
        (1234.5, "$1,234.50"),
    ],
)
def test_dollars_two_decimals(usd: float, shown: str) -> None:
    assert mo.dollars(usd) == shown


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_amounts_are_refused(bad: float) -> None:
    with pytest.raises(ValueError):
        mo.dollars(bad)
    with pytest.raises(ValueError):
        mo.money(0.1, bad)


def test_negative_amounts_read_as_zero() -> None:
    assert mo.money(-1, RATE) == "0 toman ($0.00)"


# --- money / money_range -----------------------------------------------------------------


def test_money_toman_first_then_dollars() -> None:
    assert mo.money(0.45, RATE) == "about 145 thousand toman ($0.45)"
    assert mo.money(5, RATE) == "about 1.6 million toman ($5.00)"


def test_money_without_rate_is_dollars_only() -> None:
    assert mo.money(0.45, None) == "$0.45"
    assert "toman" not in mo.money(0.45, None)


def test_money_range_shares_the_unit() -> None:
    assert mo.money_range(0.1, 0.19, RATE) == "about 32 to 61 thousand toman ($0.10 to $0.19)"


def test_money_range_different_units() -> None:
    assert (
        mo.money_range(0.0025, 0.004, RATE)
        == "about 800 toman to 1.3 thousand toman (under $0.01 to under $0.01)"
    )
    assert (
        mo.money_range(2, 4, RATE)
        == "about 646 thousand toman to 1.3 million toman ($2.00 to $4.00)"
    )


def test_money_range_equal_rounded_ends() -> None:
    assert mo.money_range(0.1, 0.1001, RATE) == "about 32 thousand toman ($0.10 to $0.10)"


def test_money_range_tiny_low_end() -> None:
    assert mo.money_range(0, 0.1, RATE) == "0 toman to 32 thousand toman ($0.00 to $0.10)"
    assert (
        mo.money_range(0.0001, 0.0012, RATE)
        == "under 100 toman to 400 toman (under $0.01 to under $0.01)"
    )


def test_money_range_without_rate() -> None:
    assert mo.money_range(0.1, 0.19, None) == "about $0.10 to $0.19"


def test_nusd_helpers_use_exact_amounts() -> None:
    assert mo.nusd_money(450_000_000, RATE) == "about 145 thousand toman ($0.45)"
    assert mo.nusd_money(3 * NANO_PER_USD, None) == "$3.00"
    assert mo.nusd_money_range(100_000_000, 190_000_000, RATE).startswith("about 32 to 61")


# --- rendering ---------------------------------------------------------------------------


def _view(cost: float, planning: float = 0.0) -> ProjectView:
    return ProjectView(
        id="p1",
        goal="g",
        status="planned",
        created_at=T0,
        updated_at=T0,
        cost_usd=cost,
        planning_usd=planning,
    )


def test_overview_card_shows_toman_first() -> None:
    row = ProjectRow(
        id="p1", status="done", goal_preview="g", created_at=T0, cost_usd=0.45, pending_approvals=0
    )
    page = rd.render_overview(Overview(projects=(row,)), rate=RATE)
    assert "cost: about 145 thousand toman ($0.45)" in page
    assert "$0.4500" not in page


def test_overview_without_rate_two_decimals() -> None:
    row = ProjectRow(
        id="p1", status="done", goal_preview="g", created_at=T0, cost_usd=0.45, pending_approvals=0
    )
    page = rd.render_overview(Overview(projects=(row,)))
    assert "cost: $0.45" in page and "toman" not in page


def test_project_page_splits_planning_and_tasks() -> None:
    page = rd.render_project(_view(0.45, planning=0.05), rate=RATE)
    assert '<p class="big"><strong>about 145 thousand toman ($0.45)</strong>' in page
    assert "Planning: about 16 thousand toman ($0.05)" in page
    assert "Tasks: about 129 thousand toman ($0.40)" in page


def test_project_page_with_nothing_spent_has_no_split() -> None:
    page = rd.render_project(_view(0.0), rate=RATE)
    assert "<strong>0 toman ($0.00)</strong>" in page and "Nothing spent yet." in page
    assert "Planning:" not in page


def test_project_page_split_never_exceeds_the_total() -> None:
    page = rd.render_project(_view(0.1, planning=5.0), rate=None)
    assert "Planning: $0.10" in page
    assert "Tasks: $0.00" in page


def test_build_project_view_clamps_planning(tmp_path: Path) -> None:
    from automa.project import ProjectStore

    store = ProjectStore.for_workspace(tmp_path / "ws")
    store.create("g", project_id="p1")
    view = build_project_view(store, "p1", 100, planning_nusd=500)
    assert view.planning_usd == view.cost_usd
    assert build_project_view(store, "p1", 100, planning_nusd=-5).planning_usd == 0.0


def _progress(spent: float, other: float = 0.0) -> pg.ProgressView:
    cost = pg.CostProgress(
        estimated_at=T0,
        spent_usd=spent,
        other_usd=other,
        low_usd=0.1,
        high_usd=0.19,
        worst_usd=9.0,
        verdict="within",
    )
    return pg.ProgressView(passed=1, total=1, live=False, cost=cost)


def test_progress_amounts_in_toman() -> None:
    page = rd.render_project(_view(0.15), progress=_progress(0.15, other=0.02), rate=RATE)
    assert "<strong>about 48 thousand toman ($0.15)</strong>" in page
    assert (
        "This run within the estimated range: about 32 to 61 thousand toman ($0.10 to $0.19)"
        in page
    )
    # the meter keeps exact values (it is not text)
    assert 'value="0.150000"' in page
    assert not re.search(r"\$\d+\.\d{4}", page)


def test_unusual_line_in_toman() -> None:
    progress = _progress(0.15).model_copy(
        update={"unusual": pg.UnusualSpending(count=1, largest_usd=0.12, ratio=6.0)}
    )
    page = rd.render_project(_view(0.15), progress=progress, rate=RATE)
    assert "largest about 39 thousand toman ($0.12), 6.0× the usual cost" in page


# --- server and the real cost-range line ------------------------------------------------


def _planned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[RealEnv, str]:
    e = RealEnv(tmp_path, monkeypatch, project=3.0)
    e.provider.push(PLAN)
    r = e.post("/new", {"csrf": TOKEN, "goal": "Build a site"})
    assert r.status == 303, r.body
    (pid,) = e.store.list_ids()
    return e, pid


def _with_rate(e: RealEnv) -> cli.AutomaConfig:
    return e.config.model_copy(
        update={"display": e.config.display.model_copy(update={"toman_per_usd": RATE})}
    )


def test_web_cost_range_reads_in_toman(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    e, pid = _planned(tmp_path, monkeypatch)
    line = cli._web_cost_range(_with_rate(e), e.store, pid)
    assert "project safety cap about 969 thousand toman ($3.00)" in line
    assert re.search(r"toman \(\$\d+\.\d\d to \$\d+\.\d\d\)|under \$0\.01", line), line
    assert "≈" not in line and not re.search(r"\$\d+\.\d{4}", line)


def test_web_cost_range_without_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    e, pid = _planned(tmp_path, monkeypatch)
    line = cli._web_cost_range(e.config, e.store, pid)
    assert "project safety cap $3.00: spending never goes above it" in line
    assert "toman" not in line


def test_cli_new_line_keeps_four_decimals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The terminal is unchanged: exact dollars."""
    e, pid = _planned(tmp_path, monkeypatch)
    line = cli._new_cost_range(_with_rate(e), e.store, pid)
    assert re.search(r"about \$\d+\.\d{4} to \$\d+\.\d{4}", line), line


def test_served_project_page_shows_planning_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    e, pid = _planned(tmp_path, monkeypatch)
    app = ws.WebApp(
        e.ws,
        e.ws / LEDGER,
        ws.allowed_hosts("127.0.0.1", PORT),
        actions=_settings(),
        planner=cli._WebIdeaPlanner(_with_rate(e)),
        toman_per_usd=RATE,
    )
    body = app.handle("GET", f"/projects/{pid}", HOST).body.decode()
    assert '<div class="costcard">' in body
    # the plan call has no task: everything so far is planning
    assert "Tasks: 0 toman ($0.00)" in body
    assert "Planning: " in body
    overview = app.handle("GET", "/", HOST).body.decode()
    assert "cost: " in overview and "toman" in overview
    # the plan call was just recorded: it counts for today and this month
    assert (
        'Spent today:</span> <span class="value"><strong>' in overview
        and 'Spent today:</span> <span class="value"><strong>0 toman' not in overview
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_web_app_refuses_a_bad_rate(tmp_path: Path, bad: float) -> None:
    with pytest.raises(ValueError):
        ws.WebApp(
            tmp_path / "ws",
            tmp_path / "ws" / LEDGER,
            ws.allowed_hosts("127.0.0.1", PORT),
            toman_per_usd=bad,
        )


def test_create_server_passes_the_rate(tmp_path: Path) -> None:
    server = ws.create_server(tmp_path / "ws", tmp_path / "ws" / LEDGER, port=0, toman_per_usd=RATE)
    try:
        handler = server.RequestHandlerClass
        assert handler is not None
    finally:
        server.server_close()


# --- cost card (D139) ------------------------------------------------------------------


def test_cost_card_lists_each_task_that_cost_something() -> None:
    from automa.webui.views import TaskRow

    tasks = (
        TaskRow(order=1, id="a", title="Write <page>", role="coder"),
        TaskRow(order=2, id="b", title="Check it", role="coder"),
    )
    view = _view(0.45, planning=0.05).model_copy(update={"tasks": tasks})
    page = rd.render_project(view, rate=RATE, task_costs={"a": 0.4, "b": 0.0, "zz": 9.0})
    assert "<summary>Cost of each task</summary>" in page
    assert "Write &lt;page&gt;</td><td>about 129 thousand toman ($0.40)" in page
    assert "Check it</td>" not in page  # nothing spent on it
    assert "$9.00" not in page  # not a task of this plan


def test_cost_card_without_task_costs_has_no_table() -> None:
    page = rd.render_project(_view(0.45), rate=RATE)
    assert "Cost of each task" not in page


def test_task_costs_sums_by_task() -> None:
    from automa.budget.ledger import LedgerEntry

    def entry(task: str | None, nusd: int) -> LedgerEntry:
        return LedgerEntry.model_construct(task_id=task, cost_nusd=nusd)

    got = ws._task_costs([entry("a", 100_000_000), entry("a", 50_000_000), entry(None, 7)])
    assert got == {"a": pytest.approx(0.15)}


def test_overview_shows_today_and_month() -> None:
    page = rd.render_overview(Overview(projects=()), rate=RATE, totals=(0.1, 0.45))
    assert (
        'Spent today:</span> <span class="value"><strong>about 32 thousand toman ($0.10)</strong>'
        in page
    )
    assert (
        'This month:</span> <span class="value"><strong>about 145 thousand toman ($0.45)</strong>'
        in page
    )
    assert "Spent today" not in rd.render_overview(Overview(projects=()))
