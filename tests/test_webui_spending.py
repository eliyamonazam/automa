# ruff: noqa: S106 - test passwords
"""Tests for the "My spending this month" page (M15, D184): each account sees its own
month, projects and monthly cap (exactly the guard's figure, D183); only an owner
sees everyone. Display only; the Ledger is read, never written.

Only in-process calls; no network, no model calls.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from automa.budget import CostLedger
from automa.budget.ledger import LedgerEntry
from automa.config import AutomaConfig
from automa.users import UserRole
from automa.webui import server as ws
from automa.webui import spending as sp
from test_ledger import FixedClock, make_pair
from test_webui_isolation import IsoEnv
from test_webui_login import HOST, LEDGER, PW, PW_KID

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)
TEHRAN = "Asia/Tehran"


def _spend(root: Path, pid: str | None, usd: float, when: datetime, user: str | None) -> None:
    with CostLedger(root / LEDGER, clock=FixedClock(when)) as led:
        led.record(*make_pair(usd, project=pid), user=user)


def _config(root: Path, **users: float) -> AutomaConfig:
    return AutomaConfig.model_validate(
        {
            "workspace": str(root),
            "users": {name: {"monthly_usd": cap} for name, cap in users.items()},
        }
    )


class SpendEnv(IsoEnv):
    def __init__(self, root: Path, *, lang: str = "en", config: bool = True) -> None:
        super().__init__(root)
        self.app = ws.WebApp(
            root,
            root / LEDGER,
            ws.allowed_hosts("127.0.0.1", 8765),
            actions=self.app._base_actions,
            auth=self.auth,
            language=lang,  # type: ignore[arg-type]
            timezone=TEHRAN,
            clock=lambda: NOW,
            budget_config=_config(root, kid=2.0) if config else None,
        )


@pytest.fixture
def env(tmp_path: Path) -> SpendEnv:
    return SpendEnv(tmp_path)


def _body(r: ws.Response) -> str:
    return r.body.decode("utf-8", "replace")


def _main(r: ws.Response) -> str:
    m = re.search(r'<main class="wrap">(.*)</main>', _body(r), re.S)
    assert m is not None
    return m.group(1)


def _people(r: ws.Response) -> str:
    m = re.search(r'<div class="cards people">(.*?)</div>\s*</main>', _body(r), re.S)
    return "" if m is None else m.group(1)


def _seed(root: Path) -> None:
    _spend(root, "pdad", 0.25, NOW, "dad")
    _spend(root, "pdad", 0.50, NOW - timedelta(days=60), "dad")  # an older month
    _spend(root, "pkid", 1.50, NOW, "kid")
    _spend(root, "p1", 7.00, NOW, None)  # a CLI project: nobody's


# -- who sees what ------------------------------------------------------------------------


def test_member_sees_only_their_month_and_projects(env: SpendEnv) -> None:
    _seed(env.root)
    dad = env.signed_in("dad", PW_KID)
    r = env.get("/spending", dad)
    assert r.status == 200
    main = _main(r)
    assert "My spending this month" in main
    assert "$0.25" in main and "all time $0.75" in main
    assert "bourse screen" in main and 'href="/projects/pdad"' in main
    # Nothing of anyone else: not their projects, totals or the "everyone" part.
    for other in ("$1.50", "$7.00", "$8.75", "my web page", "secret family", "Everyone"):
        assert other not in main, other
    assert 'class="cards people"' not in main


def test_member_cap_is_the_default_member_cap(env: SpendEnv) -> None:
    _seed(env.root)
    main = _main(env.get("/spending", env.signed_in("dad", PW_KID)))
    # member_monthly_usd default 3.0; member project cap min(project_usd, 1.0).
    assert "<li>Limit: <strong>$3.00</strong></li>" in main
    assert "<li>Counted so far: <strong>$0.25</strong></li>" in main
    assert "<li>Left: <strong>$2.75</strong></li>" in main
    assert "Limit of each project: $1.00" in main
    assert 'aria-valuenow="8"' in main  # 0.25 / 3.00


def test_configured_cap_and_full_cap_message(env: SpendEnv) -> None:
    _spend(env.root, "pkid", 2.10, NOW, "kid")
    main = _main(env.get("/spending", env.signed_in("kid", "kid-pass-123")))
    assert "Limit: <strong>$2.00</strong>" in main and "Left: <strong>$0.00</strong>" in main
    assert "limit is used up" in main and 'class="bar full"' in main
    assert 'aria-valuenow="100"' in main


def test_near_cap_warning(env: SpendEnv) -> None:
    _spend(env.root, "pkid", 1.70, NOW, "kid")
    main = _main(env.get("/spending", env.signed_in("kid", "kid-pass-123")))
    assert 'class="bar near"' in main and "Most of this month" in main
    assert "used up" not in main


def test_owner_sees_everyone(env: SpendEnv) -> None:
    _seed(env.root)
    r = env.get("/spending", env.signed_in("eli", PW))
    main = _main(r)
    assert "Everyone" in main and "Total this month: $8.75" in main
    people = _people(r)
    # owner first, then members by name, then the projects of nobody.
    order = [people.index(x) for x in (">eli<", ">dad<", ">kid<", "No account")]
    assert order == sorted(order)
    assert "$7.00" in people and "$1.50 of $2.00 limit" in people
    assert "$0.25 of $3.00 limit" in people
    assert "no monthly limit" in people  # the owner has none unless configured
    # The owner's own card: no projects of their own this month.
    assert "You have no monthly limit" in main and "Nothing spent this month yet" in main


def test_owner_without_spending_and_no_unowned_row(env: SpendEnv) -> None:
    people = _people(env.get("/spending", env.signed_in("eli", PW)))
    assert "No account" not in people and people.count('class="card person"') == 3


def test_disabled_and_removed_accounts_are_marked(env: SpendEnv) -> None:
    env.users.disable("kid")
    env.store.create("old goal", project_id="pold", owner="gone")
    _spend(env.root, "pold", 0.40, NOW, "gone")
    people = _people(env.get("/spending", env.signed_in("eli", PW)))
    assert "disabled" in people and "account removed" in people and "$0.40" in people


def test_page_needs_login_when_accounts_exist(env: SpendEnv) -> None:
    r = env.get("/spending")
    assert r.status == 303 and dict(r.headers)["Location"] == "/login"


def test_no_accounts_shows_the_whole_workspace(tmp_path: Path) -> None:
    from automa.project import ProjectStore

    ProjectStore.for_workspace(tmp_path).create("cli goal", project_id="pc")
    _spend(tmp_path, "pc", 0.30, NOW, None)
    app = ws.WebApp(
        tmp_path,
        tmp_path / LEDGER,
        ws.allowed_hosts("127.0.0.1", 8765),
        clock=lambda: NOW,
        budget_config=_config(tmp_path),
    )
    main = _main(app.handle("GET", "/spending", HOST))
    assert "Spending this month" in main and "$0.30" in main
    assert "Monthly limit" not in main and "Limit of each project:" in main
    assert "Everyone" not in main


def test_without_config_no_caps_are_claimed(tmp_path: Path) -> None:
    env = SpendEnv(tmp_path, config=False)
    _seed(tmp_path)
    main = _main(env.get("/spending", env.signed_in("dad", PW_KID)))
    # Caps unknown here: neither a limit nor "no limit" is claimed.
    assert "monthly limit" not in main.lower() and "Limit" not in main
    assert "$0.25" in main
    assert "Limit of each project" not in main


def test_unreadable_ledger_shows_an_error_not_zero(env: SpendEnv) -> None:
    (env.root / LEDGER).write_bytes(b"not a database" * 100)
    main = _main(env.get("/spending", env.signed_in("eli", PW)))
    assert "could not be read" in main and "$0.00" not in main


def test_ledger_is_not_created_by_the_page(env: SpendEnv) -> None:
    assert env.get("/spending", env.signed_in("eli", PW)).status == 200
    assert not (env.root / LEDGER).exists()


def test_head_and_post(env: SpendEnv) -> None:
    eli = env.signed_in("eli", PW)
    head = env.app.handle("HEAD", "/spending", HOST, [("Cookie", f"automa_session={eli}")])
    assert head.status == 200
    post = env.post("/spending", {"csrf": env.csrf_of(eli)}, cookie=eli)
    assert post.status in (404, 405)


def test_top_bar_month_links_to_the_page(env: SpendEnv) -> None:
    _seed(env.root)
    body = _body(env.get("/", env.signed_in("dad", PW_KID)))
    assert '<a class="month" href="/spending"' in body


def test_project_titles_are_escaped(env: SpendEnv) -> None:
    env.store.create("<script>x</script> dad 2", project_id="pdad2", owner="dad")
    _spend(env.root, "pdad2", 0.10, NOW, "dad")
    main = _main(env.get("/spending", env.signed_in("dad", PW_KID)))
    assert "<script>" not in main and "&lt;script&gt;" in main


# -- the two month figures ------------------------------------------------------------------


def test_cap_counts_only_rows_recorded_for_the_account(env: SpendEnv) -> None:
    # A row of dad's project recorded before D183 (no user): his month, not his cap.
    _spend(env.root, "pdad", 0.40, NOW, None)
    _spend(env.root, "pdad", 0.10, NOW, "dad")
    main = _main(env.get("/spending", env.signed_in("dad", PW_KID)))
    assert "<strong>$0.50</strong>" in main  # this month
    assert "Counted so far: <strong>$0.10</strong>" in main


def test_cap_uses_the_utc_month_like_the_guard(tmp_path: Path) -> None:
    # 2026-11-30 22:00 UTC is already December 1st in Tehran (+03:30).
    late = datetime(2026, 11, 30, 22, 0, tzinfo=UTC)
    entries = [
        LedgerEntry(
            id=1,
            ts=late,
            project_id="pa",
            task_id=None,
            tier="small",
            model="m",
            response_model="m",
            input_tokens=1,
            output_tokens=1,
            cost_nusd=500_000_000,
            estimate_nusd=600_000_000,
            reported_cost_nusd=None,
            user="dad",
        )
    ]
    projects = {"pa": sp.ProjectInfo(owner="dad", title="A")}
    viewer = sp.AccountInfo(name="dad", role="member", cap_usd=3.0)
    from automa.l10n import resolve_timezone

    view = sp.build_spending(
        entries,
        projects,
        viewer=viewer,
        now=datetime(2026, 11, 30, 23, 0, tzinfo=UTC),
        tz=resolve_timezone(TEHRAN),
    )
    assert view.me.month_nusd == 500_000_000  # local December
    assert view.me.cap_used_nusd == 500_000_000  # UTC November, the guard's month


def test_deleted_project_rows_go_to_the_recorded_account(env: SpendEnv) -> None:
    _spend(env.root, "pgone", 0.20, NOW, "dad")
    main = _main(env.get("/spending", env.signed_in("dad", PW_KID)))
    assert "(deleted project)" in main and "$0.20" in main
    home = _body(env.get("/", env.signed_in("dad", PW_KID)))
    bar = re.search(r'<header class="topbar">.*?</header>', home, re.S)
    assert bar is not None and "$0.20" in bar.group(0)


def test_many_projects_are_cut(env: SpendEnv) -> None:
    for i in range(sp.MAX_PROJECTS_SHOWN + 3):
        pid = f"pm{i:02d}"
        env.store.create(f"goal {i}", project_id=pid, owner="dad")
        _spend(env.root, pid, 0.01 * (i + 1), NOW, "dad")
    main = _main(env.get("/spending", env.signed_in("dad", PW_KID)))
    assert "and 3 more projects" in main
    assert main.count('class="file-row spend-row"') == sp.MAX_PROJECTS_SHOWN


# -- Persian ----------------------------------------------------------------------------


def test_persian_page(tmp_path: Path) -> None:
    env = SpendEnv(tmp_path, lang="fa")
    _seed(tmp_path)
    main = _main(env.get("/spending", env.signed_in("eli", PW)))
    assert "خرج من این ماه" in main and "همه" in main and "بدون حساب" in main
    assert "۷٫۰۰ دلار" in main and "۱٫۵۰ دلار از سقف ۲٫۰۰ دلار" in main
    for english in ("Everyone", "Monthly limit", "this month", "limit"):
        assert english not in main, english


def test_messages_have_the_same_fields() -> None:
    with pytest.raises(ValueError):
        sp.st("today", "en")
    with pytest.raises(ValueError):
        sp.st("title", "de")  # type: ignore[arg-type]
    assert sp.st("projects.more", "fa", count=3) == "و ۳ پروژهٔ دیگه"


def test_roles_from_users_store_drive_caps(env: SpendEnv) -> None:
    env.users.add("mom", "mom-pass-1234", UserRole.OWNER)
    people = _people(env.get("/spending", env.signed_in("eli", PW)))
    assert ">mom<" in people and people.count("no monthly limit") == 2
