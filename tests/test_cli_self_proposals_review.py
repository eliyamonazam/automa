"""``automa self proposals`` / ``accept`` / ``reject``: review stored proposal sets (D87).

No model, no network and no repository: sets are stored directly with
``ProposalStore.save``. The provider factory is a ``FakeProvider`` that must never be
built by these commands.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from automa import cli
from automa.selfdev import parse_roadmap
from automa.selfdev_proposals import (
    DecisionStatus,
    ItemProposal,
    ProposalSet,
    ProposalStore,
    Screened,
)
from test_cli import Env

ROADMAP = """# Roadmap

## M0 — base
- [x] skeleton
- [ ] first open item

## M1 — next
- [ ] later item
"""

COMMIT = "a" * 40


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


def store(env: Env, clock: datetime | None = None) -> ProposalStore:
    if clock is None:
        return ProposalStore.for_workspace(env.ws)
    return ProposalStore.for_workspace(env.ws, clock=lambda: clock)


def save(env: Env, *texts: tuple[str, str], clock: datetime | None = None) -> ProposalSet:
    screened = Screened(
        proposals=[ItemProposal(milestone=m, text=t, reason="because") for m, t in texts],
        dropped_duplicates=1,
    )
    return store(env, clock).save(screened, roadmap_text=ROADMAP, commit=COMMIT)


TWO = (("M0 — base", "add a cost summary to status"), ("M1 — next", "document the exit codes"))


def decision_files(env: Env) -> list[str]:
    root = env.ws / "selfdev" / "proposals"
    return sorted(p.name for p in root.iterdir() if ".d" in p.name) if root.exists() else []


def all_files(env: Env) -> list[str]:
    return sorted(str(p.relative_to(env.ws)) for p in env.ws.rglob("*")) if env.ws.exists() else []


# -- listing and showing ---------------------------------------------------------------


def test_list_empty(env: Env) -> None:
    r = env.run("self", "proposals")
    assert r.exit_code == 0, r.output
    assert "no proposal sets yet" in r.output
    assert not env.ws.exists() or all_files(env) == []


def test_list_empty_json(env: Env) -> None:
    r = env.run("self", "proposals", "--json")
    assert r.exit_code == 0
    assert json.loads(r.output) == []


def test_list_sets_with_counts(env: Env) -> None:
    t0 = datetime(2026, 11, 16, 10, 0, tzinfo=UTC)
    a = save(env, *TWO, clock=t0)
    b = save(env, ("M0 — base", "a third idea for the base"), clock=t0 + timedelta(hours=1))
    store(env).reject(a.id, 2)
    r = env.run("self", "proposals")
    assert r.exit_code == 0, r.output
    lines = r.output.splitlines()
    assert lines[0].startswith(a.id) and "1 pending, 0 accepted, 1 rejected" in lines[0]
    assert lines[1].startswith(b.id) and "1 pending, 0 accepted, 0 rejected" in lines[1]
    assert "commit aaaaaaaaaaaa" in lines[0]


def test_list_json(env: Env) -> None:
    a = save(env, *TWO)
    store(env).accept(a.id, 1)
    data = json.loads(env.run("self", "proposals", "--json").output)
    assert [s["id"] for s in data] == [a.id]
    assert data[0]["decisions"]["1"]["status"] == "accepted"
    assert "2" not in data[0]["decisions"]
    assert data[0]["proposals"][1]["line"] == "- [ ] document the exit codes"


def test_show_one_set(env: Env) -> None:
    a = save(env, *TWO)
    store(env).accept(a.id, 2)
    r = env.run("self", "proposals", a.id)
    assert r.exit_code == 0, r.output
    assert f"proposal set: {a.id}" in r.output
    assert "1. [M0 — base] pending" in r.output
    assert "2. [M1 — next] accepted" in r.output
    assert "   - [ ] add a cost summary to status" in r.output
    assert "why: because" in r.output
    assert "1 pending, 1 accepted, 0 rejected" in r.output
    assert f"automa self accept {a.id}" in r.output


def test_show_fully_decided_has_no_hint(env: Env) -> None:
    a = save(env, *TWO)
    store(env).accept(a.id, 1)
    store(env).reject(a.id, 2)
    r = env.run("self", "proposals", a.id)
    assert r.exit_code == 0
    assert "decide with" not in r.output


def test_show_empty_set(env: Env) -> None:
    a = save(env)
    r = env.run("self", "proposals", a.id)
    assert r.exit_code == 0
    assert "no new item proposed" in r.output
    assert "decide with" not in r.output


def test_show_json(env: Env) -> None:
    a = save(env, *TWO)
    data = json.loads(env.run("self", "proposals", a.id, "--json").output)
    assert data["id"] == a.id and data["commit"] == COMMIT
    assert data["decisions"] == {}
    assert data["dropped_duplicates"] == 1


@pytest.mark.parametrize("bad", ["nope", "sdp-2026", "../x", "sdp-20261116-100000-zzzzzz"])
def test_show_invalid_id(env: Env, bad: str) -> None:
    save(env, *TWO)
    r = env.run("self", "proposals", bad)
    assert r.exit_code == cli.EXIT_USER
    assert "invalid proposal set id" in r.output


def test_show_unknown_id(env: Env) -> None:
    save(env, *TWO)
    r = env.run("self", "proposals", "sdp-20000101-000000-abcdef")
    assert r.exit_code == cli.EXIT_USER
    assert "no proposal set" in r.output


def test_broken_store_fails_closed(env: Env) -> None:
    save(env, *TWO)
    (env.ws / "selfdev" / "proposals" / "stray.txt").write_text("x", encoding="utf-8")
    for args in (("self", "proposals"), ("self", "proposals", "--json")):
        r = env.run(*args)
        assert r.exit_code == cli.EXIT_USER
        assert "unexpected entry" in r.output


def test_tampered_set_fails_closed(env: Env) -> None:
    a = save(env, *TWO)
    path = env.ws / "selfdev" / "proposals" / f"{a.id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["proposals"][0]["text"] = "- [x] sneaky"
    path.write_text(json.dumps(raw), encoding="utf-8")
    r = env.run("self", "proposals", a.id)
    assert r.exit_code == cli.EXIT_USER
    assert "invalid proposal set file" in r.output
    r = env.run("self", "accept", a.id, "1", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert decision_files(env) == []


def test_reading_writes_nothing_and_needs_no_provider(env: Env) -> None:
    a = save(env, *TWO)
    before = all_files(env)
    for args in (
        ("self", "proposals"),
        ("self", "proposals", "--json"),
        ("self", "proposals", a.id),
        ("self", "proposals", a.id, "--json"),
    ):
        assert env.run(*args).exit_code == 0
    assert all_files(env) == before
    assert env.factory_calls == []


# -- accept / reject -----------------------------------------------------------------------


def test_accept_with_yes_records_and_prints_lines(env: Env) -> None:
    a = save(env, *TWO)
    r = env.run("self", "accept", a.id, "2", "1", "--yes")
    assert r.exit_code == 0, r.output
    assert "2: accepted" in r.output and "1: accepted" in r.output
    tail = r.output.split("by hand, under its milestone:\n", 1)[1].splitlines()
    assert tail == [
        "## M1 — next",
        "- [ ] document the exit codes",
        "## M0 — base",
        "- [ ] add a cost summary to status",
    ]
    d = store(env).decisions(a.id)
    assert {n: x.status for n, x in d.items()} == {
        1: DecisionStatus.ACCEPTED,
        2: DecisionStatus.ACCEPTED,
    }
    assert env.factory_calls == []


def test_accepted_line_round_trips_into_roadmap(env: Env) -> None:
    a = save(env, *TWO)
    r = env.run("self", "accept", a.id, "1", "--yes")
    line = r.output.strip().splitlines()[-1]
    updated = ROADMAP.replace("- [ ] first open item\n", f"- [ ] first open item\n{line}\n")
    roadmap = parse_roadmap(updated)
    assert any(i.text == "add a cost summary to status" for i in roadmap.items)


def test_accept_writes_only_decision_files(env: Env, tmp_path: Path) -> None:
    a = save(env, *TWO)
    (tmp_path / "ROADMAP.md").write_text(ROADMAP, encoding="utf-8")
    before = all_files(env)
    r = env.run("self", "accept", a.id, "1", "--yes")
    assert r.exit_code == 0
    after = all_files(env)
    assert sorted(set(after) - set(before)) == [f"selfdev/proposals/{a.id}.d1.json"]
    assert (tmp_path / "ROADMAP.md").read_text(encoding="utf-8") == ROADMAP


def test_reject_with_yes(env: Env) -> None:
    a = save(env, *TWO)
    r = env.run("self", "reject", a.id, "1", "--yes")
    assert r.exit_code == 0, r.output
    assert "1: rejected" in r.output
    assert "by hand" not in r.output
    assert store(env).decisions(a.id)[1].status is DecisionStatus.REJECTED
    assert store(env).pending(a.id) == [2]


def test_confirm_yes(env: Env) -> None:
    a = save(env, *TWO)
    r = env.run("self", "accept", a.id, "1", input="y\n")
    assert r.exit_code == 0, r.output
    assert "Accept this proposal?" in r.output
    assert "- [ ] add a cost summary to status" in r.output.split("Accept")[0]
    assert decision_files(env) == [f"{a.id}.d1.json"]


@pytest.mark.parametrize("command", ["accept", "reject"])
@pytest.mark.parametrize("answer", ["n\n", "\n", ""])
def test_confirm_no_changes_nothing(env: Env, command: str, answer: str) -> None:
    a = save(env, *TWO)
    r = env.run("self", command, a.id, "1", "2", input=answer)
    assert r.exit_code == cli.EXIT_USER
    if answer:  # end of input aborts (exit 1) before the message
        assert "nothing was changed" in r.output
    assert "these 2 proposals" in r.output
    assert decision_files(env) == []


@pytest.mark.parametrize("command", ["accept", "reject"])
def test_decision_is_final(env: Env, command: str) -> None:
    a = save(env, *TWO)
    store(env).reject(a.id, 1)
    r = env.run("self", command, a.id, "2", "1", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "already decided" in r.output and "1 is rejected" in r.output
    # all-or-nothing: proposal 2 was not decided either
    assert decision_files(env) == [f"{a.id}.d1.json"]


@pytest.mark.parametrize("numbers", [["0"], ["3"], ["1", "3"], ["-1"]])
def test_out_of_range_changes_nothing(env: Env, numbers: list[str]) -> None:
    a = save(env, *TWO)
    r = env.run("self", "accept", a.id, *numbers, "--yes")
    assert r.exit_code != 0
    assert decision_files(env) == []


def test_out_of_range_message(env: Env) -> None:
    a = save(env, *TWO)
    r = env.run("self", "accept", a.id, "1", "3", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "has proposals 1..2; no proposal 3" in r.output


def test_empty_set_has_nothing_to_decide(env: Env) -> None:
    a = save(env)
    r = env.run("self", "reject", a.id, "1", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "has no proposals" in r.output


def test_repeated_number_changes_nothing(env: Env) -> None:
    a = save(env, *TWO)
    r = env.run("self", "accept", a.id, "1", "1", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "only once" in r.output
    assert decision_files(env) == []


def test_numbers_required(env: Env) -> None:
    a = save(env, *TWO)
    r = env.run("self", "accept", a.id, "--yes")
    assert r.exit_code != 0
    assert decision_files(env) == []


def test_non_integer_number_rejected(env: Env) -> None:
    a = save(env, *TWO)
    r = env.run("self", "accept", a.id, "one", "--yes")
    assert r.exit_code != 0
    assert decision_files(env) == []


@pytest.mark.parametrize("bad", ["nope", "../../etc", "sdp-20000101-000000-abcdef"])
def test_decide_bad_or_unknown_set(env: Env, bad: str) -> None:
    save(env, *TWO)
    r = env.run("self", "accept", bad, "1", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "nothing was changed" in r.output
    assert decision_files(env) == []


def test_decide_without_store(env: Env) -> None:
    r = env.run("self", "reject", "sdp-20000101-000000-abcdef", "1", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert not (env.ws / "selfdev" / "proposals").exists()


def test_concurrent_decision_reports_rest(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    a = save(env, *TWO)
    real = ProposalStore.accept
    calls: list[int] = []

    def flaky(self: ProposalStore, set_id: str, number: int) -> object:
        calls.append(number)
        if number == 2:
            ProposalStore.reject(self, set_id, number)  # someone else decided first
        return real(self, set_id, number)

    monkeypatch.setattr(ProposalStore, "accept", flaky)
    r = env.run("self", "accept", a.id, "1", "2", "--yes")
    assert r.exit_code == cli.EXIT_USER
    assert "1: accepted" in r.output
    assert "not decided: 2" in r.output
    assert "by hand" not in r.output
    assert store(env).decisions(a.id)[2].status is DecisionStatus.REJECTED


def test_control_characters_in_tampered_reason_are_refused(env: Env) -> None:
    # The store validates every field on load, so bidi/control text never reaches output.
    a = save(env, *TWO)
    path = env.ws / "selfdev" / "proposals" / f"{a.id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["proposals"][0]["reason"] = "ok‮evil"
    path.write_text(json.dumps(raw), encoding="utf-8")
    r = env.run("self", "proposals", a.id)
    assert "‮" not in r.output


def test_commands_have_no_model_path() -> None:
    import ast
    import inspect

    for fn in (cli.self_proposals, cli._self_decide, cli.self_accept, cli.self_reject):
        tree = ast.parse(inspect.getsource(fn).lstrip())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert not names & {"make_provider", "LLMGateway", "BudgetGuard", "CostLedger"}
