"""Per-user spending (M15 part 2, D183): ``user`` in the Ledger, the monthly user cap
in the BudgetGuard, the stricter member caps and their wiring into ``automa run``.

No network: FakeProvider only, SQLite in memory or under ``tmp_path``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from automa import cli
from automa.budget import (
    BudgetError,
    BudgetExceeded,
    BudgetGuard,
    BudgetScope,
    CostLedger,
    cap_to_nusd,
    usd_to_nusd,
)
from automa.budget.accounts import config_for_project, owner_role, project_budget
from automa.budget.ledger import LedgerError, month_of
from automa.config import AutomaConfig, BudgetLimits, ConfigError, Tier, load_config
from automa.events import Event
from automa.llm import CallContext, CallEstimate, ChatResponse, LLMResult, Message, Usage
from automa.project import ProjectStatus
from automa.users import UserRole, UserStore
from automa.webui.stops import _budget_stop, describe_stop
from test_cli_run import PID, Env, task_dict, write

LIMITS = BudgetLimits(project_usd=1.0, daily_usd=2.0, per_call_usd=0.5)


class FixedClock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(datetime(2026, 11, 9, 12, 0, tzinfo=UTC))


@pytest.fixture
def ledger(clock: FixedClock) -> Iterator[CostLedger]:
    with CostLedger(":memory:", clock=clock) as led:
        yield led


def est(max_cost: float, project: str = "p1") -> CallEstimate:
    return CallEstimate(
        tier=Tier.SMALL,
        model="m-small",
        max_input_tokens=100,
        max_output_tokens=50,
        max_cost_usd=max_cost,
        context=CallContext(project_id=project, task_id="t1"),
    )


def res(cost: float) -> LLMResult:
    return LLMResult(
        tier=Tier.SMALL,
        response=ChatResponse(
            model="m-small", message=Message.assistant("ok"), usage=Usage(input_tokens=1)
        ),
        cost_usd=cost,
    )


def spend(guard: BudgetGuard, cost: float, project: str = "p1") -> None:
    e = est(cost, project)
    guard.authorize(e)
    guard.record(e, res(cost))


# -- ledger ----------------------------------------------------------------------------


def test_ledger_records_user_and_filters_by_user_and_month(
    ledger: CostLedger, clock: FixedClock
) -> None:
    ledger.record(est(0.1), res(0.1), user="sara")
    ledger.record(est(0.2, "p2"), res(0.2), user="sara")
    ledger.record(est(0.4), res(0.4), user="ali")
    ledger.record(est(0.8), res(0.8))
    clock.now = datetime(2026, 12, 1, 0, 0, tzinfo=UTC)
    ledger.record(est(1.6), res(1.6), user="sara")
    assert ledger.spent_nusd(user="sara") == usd_to_nusd(0.1) + usd_to_nusd(0.2) + usd_to_nusd(1.6)
    assert ledger.spent_nusd(user="sara", month="2026-11") == usd_to_nusd(0.1) + usd_to_nusd(0.2)
    assert ledger.spent_nusd(user="sara", month="2026-12") == usd_to_nusd(1.6)
    assert ledger.spent_nusd(month="2026-11") == sum(usd_to_nusd(c) for c in (0.1, 0.2, 0.4, 0.8))
    assert ledger.spent_nusd(user="nobody") == 0
    users = [e.user for e in ledger.entries()]
    assert users == ["sara", "sara", "ali", None, "sara"]
    assert [e.cost_nusd for e in ledger.entries(user="sara", month="2026-11")] == [
        usd_to_nusd(0.1),
        usd_to_nusd(0.2),
    ]


@pytest.mark.parametrize("bad", ["", "Sara", "a", "x y", "sara'--", "1abc"])
def test_ledger_refuses_invalid_user_names(ledger: CostLedger, bad: str) -> None:
    with pytest.raises(LedgerError):
        ledger.record(est(0.1), res(0.1), user=bad)
    with pytest.raises(LedgerError):
        ledger.spent_nusd(user=bad)
    assert ledger.entries() == []


@pytest.mark.parametrize("bad", ["2026-13", "2026-1", "26-11", "2026-11-01", "%", ""])
def test_ledger_refuses_invalid_months(ledger: CostLedger, bad: str) -> None:
    with pytest.raises(LedgerError):
        ledger.spent_nusd(month=bad)


def test_month_of() -> None:
    assert month_of(date(2026, 1, 31)) == "2026-01"
    assert month_of(date(2026, 12, 1)) == "2026-12"


_V1_SCHEMA = """
CREATE TABLE calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, day TEXT NOT NULL,
    project_id TEXT, task_id TEXT, tier TEXT NOT NULL, model TEXT NOT NULL,
    response_model TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
    cost_nusd INTEGER NOT NULL, estimate_nusd INTEGER NOT NULL, reported_cost_nusd INTEGER
);
CREATE INDEX calls_project ON calls (project_id);
CREATE INDEX calls_day ON calls (day);
INSERT INTO calls VALUES (1, '2026-11-01T10:00:00+00:00', '2026-11-01', 'old', NULL, 'small',
    'm', 'm', 10, 5, 7000, 9000, NULL);
PRAGMA user_version = 1;
"""


def _v1_file(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(_V1_SCHEMA)
    conn.close()
    return path


def test_version_1_ledger_is_migrated_without_losing_rows(
    tmp_path: Path, clock: FixedClock
) -> None:
    path = _v1_file(tmp_path / "ledger.sqlite3")
    with CostLedger(path, clock=clock) as led:
        (old,) = led.entries()
        assert old.project_id == "old" and old.user is None and old.cost_nusd == 7000
        led.record(est(0.1), res(0.1), user="sara")
        assert led.spent_nusd(user="sara") == usd_to_nusd(0.1)
        assert led.spent_nusd() == 7000 + usd_to_nusd(0.1)
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone() == (2,)
    conn.close()
    with CostLedger(path, clock=clock) as again:  # reopening a v2 file is a no-op
        assert len(again.entries()) == 2


def test_version_1_ledger_can_be_read_only_without_migrating(tmp_path: Path) -> None:
    path = _v1_file(tmp_path / "ledger.sqlite3")
    with CostLedger(path, read_only=True) as led:
        (old,) = led.entries()
        assert old.user is None
        assert led.spent_nusd() == 7000
        assert led.spent_nusd(user="sara") == 0
        assert led.entries(user="sara") == []
        assert led.get(1).user is None
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone() == (1,)
    conn.close()


def test_unknown_future_ledger_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 3")
    conn.close()
    with pytest.raises(LedgerError, match="version 3"):
        CostLedger(path)


# -- guard -----------------------------------------------------------------------------


def test_guard_records_user_on_every_row(ledger: CostLedger) -> None:
    guard = BudgetGuard(LIMITS, ledger, user="sara")
    spend(guard, 0.1)
    assert [e.user for e in ledger.entries()] == ["sara"]
    assert guard.user == "sara" and guard.user_monthly_nusd is None
    assert BudgetGuard(LIMITS, ledger).user is None


def test_guard_without_monthly_cap_keeps_every_other_cap(ledger: CostLedger) -> None:
    guard = BudgetGuard(LIMITS, ledger, user="sara")
    spend(guard, 0.5)
    spend(guard, 0.5)
    with pytest.raises(BudgetExceeded) as exc:
        guard.authorize(est(0.01))
    assert exc.value.scope is BudgetScope.PROJECT


def test_monthly_cap_refuses_a_call_that_would_exceed_it(ledger: CostLedger) -> None:
    guard = BudgetGuard(LIMITS, ledger, user="sara", user_monthly_usd=0.6)
    spend(guard, 0.4, "p1")
    with pytest.raises(BudgetExceeded) as exc:
        guard.authorize(est(0.3, "p2"))  # another project of the same account
    err = exc.value
    assert err.scope is BudgetScope.USER_MONTHLY
    assert err.key == "sara:2026-11"
    assert err.limit_nusd == cap_to_nusd(0.6)
    assert err.spent_nusd == usd_to_nusd(0.4)
    assert guard.is_stopped(BudgetScope.USER_MONTHLY, "sara:2026-11")
    (stop,) = [e for e in guard.events() if e.type == "budget.stop"]
    assert stop.data["scope"] == "user_monthly" and stop.data["key"] == "sara:2026-11"
    # latched: even a tiny call is refused now
    with pytest.raises(BudgetExceeded) as again:
        guard.authorize(est(0.000001, "p3"))
    assert again.value.latched and again.value.scope is BudgetScope.USER_MONTHLY
    assert guard.pending_nusd() == 0


def test_monthly_cap_counts_spending_from_other_processes(ledger: CostLedger) -> None:
    # Another run of the same account (another guard, same ledger file).
    ledger.record(est(0.5, "elsewhere"), res(0.5), user="sara")
    ledger.record(est(0.5, "elsewhere"), res(0.5), user="ali")
    guard = BudgetGuard(LIMITS, ledger, user="sara", user_monthly_usd=0.7)
    with pytest.raises(BudgetExceeded) as exc:
        guard.authorize(est(0.3))
    assert exc.value.scope is BudgetScope.USER_MONTHLY
    other = BudgetGuard(LIMITS, ledger, user="reza", user_monthly_usd=0.7)
    other.authorize(est(0.3))  # nothing spent by reza yet


def test_monthly_cap_counts_in_flight_reservations(ledger: CostLedger) -> None:
    guard = BudgetGuard(LIMITS, ledger, user="sara", user_monthly_usd=0.5)
    guard.authorize(est(0.3, "p1"))
    with pytest.raises(BudgetExceeded) as exc:
        guard.authorize(est(0.3, "p2"))
    assert exc.value.scope is BudgetScope.USER_MONTHLY


def test_monthly_cap_starts_fresh_next_month(ledger: CostLedger, clock: FixedClock) -> None:
    guard = BudgetGuard(LIMITS, ledger, user="sara", user_monthly_usd=0.5)
    spend(guard, 0.5, "p1")
    assert guard.is_stopped(BudgetScope.USER_MONTHLY, "sara:2026-11")
    with pytest.raises(BudgetExceeded):
        guard.authorize(est(0.01, "p2"))
    clock.now = datetime(2026, 12, 1, 0, 0, 1, tzinfo=UTC)
    spend(guard, 0.1, "p2")  # December: a new key, nothing spent yet


def test_monthly_cap_reached_by_actual_cost_latches_on_record(ledger: CostLedger) -> None:
    guard = BudgetGuard(LIMITS, ledger, user="sara", user_monthly_usd=0.5)
    e = est(0.2)
    guard.authorize(e)
    guard.record(e, res(0.5))  # cost more than estimated: recorded anyway, then latched
    assert ledger.spent_nusd(user="sara") == usd_to_nusd(0.5)
    assert guard.is_stopped(BudgetScope.USER_MONTHLY, "sara:2026-11")
    warned = {
        e.data["threshold_pct"]
        for e in guard.events()
        if e.type == "budget.warning" and e.data["scope"] == "user_monthly"
    }
    assert warned == {50, 80}


def test_monthly_cap_never_loosens_project_or_daily_caps(ledger: CostLedger) -> None:
    guard = BudgetGuard(LIMITS, ledger, user="sara", user_monthly_usd=1000.0)
    spend(guard, 0.5)
    spend(guard, 0.5)
    with pytest.raises(BudgetExceeded) as exc:
        guard.authorize(est(0.1))
    assert exc.value.scope is BudgetScope.PROJECT
    with pytest.raises(BudgetExceeded) as per_call:
        guard.authorize(est(0.6, "p9"))
    assert per_call.value.scope is BudgetScope.PER_CALL


def test_guard_refuses_bad_user_settings(ledger: CostLedger) -> None:
    with pytest.raises(BudgetError, match="needs a user"):
        BudgetGuard(LIMITS, ledger, user_monthly_usd=1.0)
    with pytest.raises(BudgetError):
        BudgetGuard(LIMITS, ledger, user="Bad Name")
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(BudgetError):
            BudgetGuard(LIMITS, ledger, user="sara", user_monthly_usd=bad)


# -- configuration and accounts ----------------------------------------------------------


def _config(**extra: Any) -> AutomaConfig:
    data: dict[str, Any] = {
        "limits": {"project_usd": 5.0, "daily_usd": 2.0, "per_call_usd": 0.25},
        "workspace": "ws",
    }
    data.update(extra)
    return AutomaConfig.model_validate(data)


def test_config_users_section(tmp_path: Path) -> None:
    toml = tmp_path / "automa.toml"
    toml.write_text(
        "[limits]\nmember_project_usd = 0.8\nmember_monthly_usd = 2.5\n"
        "[users.sara]\nmonthly_usd = 4.0\n",
        encoding="utf-8",
    )
    config = load_config(toml, env_file=None, environ={})
    assert config.users["sara"].monthly_usd == 4.0
    assert config.limits.member_project_usd == 0.8
    assert config.limits.member_monthly_usd == 2.5
    assert AutomaConfig().users == {}
    assert AutomaConfig().limits.member_project_usd == 1.0
    assert AutomaConfig().limits.member_monthly_usd == 3.0


@pytest.mark.parametrize(
    "body",
    [
        "[users.Sara]\nmonthly_usd = 1.0\n",
        '[users."a b"]\nmonthly_usd = 1.0\n',
        "[users.sara]\nmonthly_usd = 0\n",
        "[users.sara]\nmonthly_usd = -1\n",
        "[users.sara]\nmonthly_usd = inf\n",
        "[users.sara]\n",
        "[users.sara]\nmonthly_usd = 1.0\nproject_usd = 9.0\n",
        "[limits]\nmember_project_usd = 0\n",
        "[limits]\nmember_monthly_usd = -2\n",
        "[limits]\nper_call_usd = 0.5\nmember_project_usd = 0.4\n",
    ],
)
def test_config_rejects_bad_user_caps(tmp_path: Path, body: str) -> None:
    toml = tmp_path / "automa.toml"
    toml.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(toml, env_file=None, environ={})


def test_project_without_owner_keeps_the_configuration() -> None:
    config = _config()
    budget = project_budget(config, None, None)
    assert budget.limits is config.limits
    assert budget.user is None and budget.user_monthly_usd is None and not budget.member


def test_owner_account_is_recorded_and_capped_only_when_configured() -> None:
    plain = project_budget(_config(), "ilia", UserRole.OWNER)
    assert plain.user == "ilia" and plain.user_monthly_usd is None and not plain.member
    capped = project_budget(_config(users={"ilia": {"monthly_usd": 20}}), "ilia", UserRole.OWNER)
    assert capped.user_monthly_usd == 20 and capped.limits.project_usd == 5.0


def test_member_gets_the_stricter_project_cap_and_a_monthly_cap() -> None:
    config = _config()
    budget = project_budget(config, "sara", UserRole.MEMBER)
    assert budget.member and budget.user == "sara"
    assert budget.limits.project_usd == 1.0  # min(5.0, member_project_usd 1.0)
    assert budget.limits.daily_usd == 2.0 and budget.limits.per_call_usd == 0.25
    assert budget.user_monthly_usd == 3.0
    assert config.limits.project_usd == 5.0  # the configuration itself is untouched
    own = project_budget(_config(users={"sara": {"monthly_usd": 1.5}}), "sara", UserRole.MEMBER)
    assert own.user_monthly_usd == 1.5


def test_member_project_cap_is_never_looser_than_project_usd() -> None:
    config = _config(limits={"project_usd": 0.5, "member_project_usd": 4.0})
    assert project_budget(config, "sara", UserRole.MEMBER).limits.project_usd == 0.5


def test_unknown_role_counts_as_member() -> None:
    budget = project_budget(_config(), "sara", None)
    assert budget.member and budget.user_monthly_usd == 3.0


def test_owner_role_fails_closed(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    assert owner_role(ws, None) is None
    assert owner_role(ws, "ilia") is UserRole.MEMBER  # no users file
    store = UserStore(ws, scrypt_n=2**14)
    store.add("ilia", "a-long-password", UserRole.OWNER)
    store.add("sara", "a-long-password", UserRole.MEMBER)
    assert owner_role(ws, "ilia") is UserRole.OWNER
    assert owner_role(ws, "sara") is UserRole.MEMBER
    assert owner_role(ws, "gone") is UserRole.MEMBER
    (ws / "users.json").chmod(0o644)  # unreadable for the store: refused
    assert owner_role(ws, "ilia") is UserRole.MEMBER


def test_config_for_project(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    store = UserStore(ws, scrypt_n=2**14)
    store.add("ilia", "a-long-password", UserRole.OWNER)
    store.add("sara", "a-long-password", UserRole.MEMBER)
    config = _config(workspace=str(ws))
    same, none = config_for_project(config, None)
    assert same is config and none.user is None
    member_config, budget = config_for_project(config, "sara")
    assert member_config.limits.project_usd == 1.0 and budget.member
    assert member_config.tiers == config.tiers and config.limits.project_usd == 5.0


# -- web stop panel ------------------------------------------------------------------------


def test_stop_panel_explains_the_monthly_user_cap() -> None:
    now = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
    event = Event(
        ts=now,
        type="budget.stop",
        project_id="p1",
        data={"scope": "user_monthly", "key": "sara:2026-11"},
    )
    stop = _budget_stop(event)
    assert stop.kind == "user_monthly_cap"
    line = describe_stop(stop, lang="fa", tz=UTC, now=now)
    assert "سقف خرج این ماه" in line.text and line.tone == "bad" and not line.resume


# -- automa run ------------------------------------------------------------------------------


MEMBER_TOML = "\n[users.sara]\nmonthly_usd = 0.5\n"


def _member_project(env: Env, owner: str, role: UserRole) -> None:
    store = UserStore(env.ws, scrypt_n=2**14)
    store.add("ilia", "a-long-password", UserRole.OWNER)
    store.add(owner, "a-long-password", role)
    env.store.create("test goal", project_id=PID, owner=owner)
    from automa.plan import Plan

    env.store.save_plan(PID, Plan.model_validate({"goal": "test goal", "tasks": [task_dict("a")]}))


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


def test_run_of_member_project_is_stopped_by_the_monthly_cap(env: Env) -> None:
    env.config.write_text(env.config.read_text(encoding="utf-8") + MEMBER_TOML, encoding="utf-8")
    _member_project(env, "sara", UserRole.MEMBER)
    with CostLedger.for_workspace(env.ws) as led:  # spent earlier this month, elsewhere
        led.record(est(0.5, "other"), res(0.5), user="sara")
    env.script(*write("a"))
    r = env.run(PID, "--yes")
    assert r.exit_code == cli.EXIT_BUDGET, r.output
    assert env.provider.requests == []  # refused before the provider
    assert env.status() is ProjectStatus.PAUSED
    stops = [e for e in env.store.events(PID).read() if e.type == "budget.stop"]
    assert stops and stops[-1].data["scope"] == "user_monthly"


def test_run_of_member_project_records_the_account(env: Env) -> None:
    _member_project(env, "sara", UserRole.MEMBER)
    env.script(*write("a"))
    r = env.run(PID, "--yes")
    assert r.exit_code == cli.EXIT_OK, r.output
    with CostLedger.for_workspace(env.ws) as led:
        rows = led.entries(project_id=PID)
    assert rows and {e.user for e in rows} == {"sara"}


def test_run_of_project_without_owner_is_unchanged(env: Env) -> None:
    env.project(task_dict("a"))
    env.script(*write("a"))
    r = env.run(PID, "--yes")
    assert r.exit_code == cli.EXIT_OK, r.output
    with CostLedger.for_workspace(env.ws) as led:
        assert {e.user for e in led.entries(project_id=PID)} == {None}


def test_run_of_member_project_uses_the_member_project_cap(env: Env) -> None:
    # project_usd = 5.0 in the test config; the member cap is far smaller.
    env.config.write_text(
        env.config.read_text(encoding="utf-8").replace(
            "per_call_usd = 0.25", "per_call_usd = 0.25\nmember_project_usd = 0.3"
        ),
        encoding="utf-8",
    )
    _member_project(env, "sara", UserRole.MEMBER)
    with CostLedger.for_workspace(env.ws) as led:
        led.record(est(0.3, PID), res(0.3), user="sara")  # this project already at 0.3
    env.script(*write("a"))
    r = env.run(PID, "--yes")
    assert r.exit_code == cli.EXIT_BUDGET, r.output
    assert env.provider.requests == []
    stops = [e for e in env.store.events(PID).read() if e.type == "budget.stop"]
    assert stops[-1].data["scope"] == "project"
    assert stops[-1].data["limit_usd"] == pytest.approx(0.3)


def test_cli_run_cli_has_no_unguarded_budget_guard() -> None:
    """Every project guard in the CLI is built for the project's owner (D183)."""
    import ast
    import inspect
    import textwrap

    for fn in (cli._plan_with_executive, cli.review_run, cli._WebIdeaPlanner._call):
        names = {
            n.func.id
            for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn))))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "_project_guard" in names and "BudgetGuard" not in names


def test_example_config_documents_the_account_caps(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "automa.toml.example"
    text = example.read_text(encoding="utf-8")
    assert "member_project_usd" in text and "member_monthly_usd" in text
    text = text.replace("# [users.sara]\n# monthly_usd = 5.0", "[users.sara]\nmonthly_usd = 5.0")
    toml = tmp_path / "automa.toml"
    toml.write_text(text, encoding="utf-8")
    config = load_config(toml, env_file=None, environ={})
    assert config.users["sara"].monthly_usd == 5.0
