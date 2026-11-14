"""``automa review accept|reject`` and the ``review:`` line in ``automa status`` (D55).

Local only: no provider is ever made (the fake factory counts calls); no network, no
model calls. Memory is written only by ``accept``, only after the full text was shown
and confirmed (or ``--yes``), always with ``source = review``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.memory import MemoryScope, MemorySource
from automa.memory.store import MAX_ENTRIES_PER_LAYER
from automa.project import ProjectStatus
from automa.reviews import DecisionStatus
from test_cli_review import PID, Env, save_review, tree

PROPOSALS = (
    ("project", "Uses click for the CLI."),
    ("lesson", "Install pytest first."),
    ("project", "Tests live in tests/."),
)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


def run(env: Env, *args: str, input: str | None = None) -> Any:
    return CliRunner().invoke(cli.app, [*args, "--config", str(env.config)], input=input)


def setup(env: Env, *proposals: tuple[str, str]) -> str:
    env.project()
    return save_review(env, *(proposals or PROPOSALS))


def snapshot(env: Env) -> set[str]:
    return tree(env.ws)


# --- help ----------------------------------------------------------------------


def test_review_help_lists_accept_and_reject(env: Env) -> None:
    out = CliRunner().invoke(cli.app, ["review", "--help"]).output
    assert "accept" in out and "reject" in out


# --- accept --------------------------------------------------------------------


def test_accept_shows_full_text_asks_and_writes_memory(env: Env) -> None:
    rid = setup(env)
    result = run(env, "review", "accept", PID, "1", "2", input="y\n")
    assert result.exit_code == 0, result.output
    out = result.output
    assert f"review: {rid} (project {PID})" in out
    assert f"1. [project] -> project memory of {PID}" in out
    assert "2. [lesson] -> lessons (used by all future projects)" in out
    assert "Uses click for the CLI." in out and "Install pytest first." in out
    assert "Add these 2 proposals to memory?" in out
    project = env.memory.entries(MemoryScope.PROJECT, PID)
    lessons = env.memory.entries(MemoryScope.LESSON)
    assert [e.text for e in project] == ["Uses click for the CLI."]
    assert [e.text for e in lessons] == ["Install pytest first."]
    assert all(e.source is MemorySource.REVIEW for e in [*project, *lessons])
    assert f"1: accepted ({project[0].id})" in out
    assert f"2: accepted ({lessons[0].id})" in out
    assert env.reviews.pending(PID, rid) == [3]
    assert env.made_providers == 0


def test_accept_records_events(env: Env) -> None:
    setup(env)
    assert run(env, "review", "accept", PID, "1", "--yes").exit_code == 0
    assert env.events().count("review.proposal_accepted") == 1


def test_accept_text_appears_before_the_question(env: Env) -> None:
    setup(env)
    out = run(env, "review", "accept", PID, "3", input="y\n").output
    assert out.index("Tests live in tests/.") < out.index("Add this proposal to memory?")


def test_accept_declined_writes_nothing(env: Env) -> None:
    rid = setup(env)
    before = snapshot(env)
    result = run(env, "review", "accept", PID, "1", "2", input="n\n")
    assert result.exit_code == 1
    assert "not accepted; nothing was changed" in result.output
    assert snapshot(env) == before
    assert env.all_memory() == 0
    assert env.reviews.pending(PID, rid) == [1, 2, 3]


def test_accept_default_answer_is_no(env: Env) -> None:
    setup(env)
    result = run(env, "review", "accept", PID, "1", input="\n")
    assert result.exit_code == 1
    assert env.all_memory() == 0


def test_accept_without_input_writes_nothing(env: Env) -> None:
    setup(env)
    result = run(env, "review", "accept", PID, "1")
    assert result.exit_code != 0
    assert env.all_memory() == 0


def test_accept_yes_skips_the_question(env: Env) -> None:
    setup(env)
    result = run(env, "review", "accept", PID, "2", "-y")
    assert result.exit_code == 0, result.output
    assert "to memory?" not in result.output
    assert len(env.memory.entries(MemoryScope.LESSON)) == 1


def test_accept_identical_existing_memory_is_not_duplicated(env: Env) -> None:
    rid = setup(env)
    existing = env.memory.add(MemoryScope.LESSON, "install pytest first.", source=MemorySource.USER)
    result = run(env, "review", "accept", PID, "2", "--yes")
    assert result.exit_code == 0, result.output
    assert f"2: accepted ({existing.id}, already in memory)" in result.output
    assert len(env.memory.entries(MemoryScope.LESSON)) == 1
    decision = env.reviews.decisions(PID, rid)[2]
    assert decision.memory_existed is True


def test_accept_order_follows_the_arguments(env: Env) -> None:
    setup(env)
    out = run(env, "review", "accept", PID, "3", "1", "--yes").output
    assert out.index("3: accepted") < out.index("1: accepted")


def test_accept_on_a_selected_review(env: Env) -> None:
    first = setup(env)
    second = save_review(env, ("lesson", "Something else."))
    result = run(env, "review", "accept", PID, "1", "--review", first, "--yes")
    assert result.exit_code == 0, result.output
    assert env.reviews.pending(PID, first) == [2, 3]
    assert env.reviews.pending(PID, second) == [1]


def test_accept_defaults_to_the_latest_review(env: Env) -> None:
    first = setup(env)
    second = save_review(env, ("lesson", "Something else."))
    assert run(env, "review", "accept", PID, "1", "--yes").exit_code == 0
    assert env.reviews.pending(PID, first) == [1, 2, 3]
    assert env.reviews.pending(PID, second) == []


def test_accept_after_the_project_ran_again(env: Env) -> None:
    rid = setup(env)
    env.store.set_status(PID, ProjectStatus.RUNNING)
    assert run(env, "review", "accept", PID, "1", "--yes").exit_code == 0
    assert env.reviews.pending(PID, rid) == [2, 3]


def test_accept_needs_no_api_key_or_provider(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    setup(env)

    def boom(config: Any) -> Any:
        raise AssertionError("no provider may be made")

    monkeypatch.setattr(cli, "make_provider", boom)
    assert run(env, "review", "accept", PID, "1", "--yes").exit_code == 0


def test_accept_escapes_control_characters(env: Env) -> None:
    setup(env, ("lesson", "Safe ‮evil‬ text"))
    result = run(env, "review", "accept", PID, "1", input="n\n")
    assert "‮" not in result.output
    assert "Safe" in result.output


def test_accept_multiline_text_is_indented(env: Env) -> None:
    setup(env, ("lesson", "line one\nline two"))
    out = run(env, "review", "accept", PID, "1", input="n\n").output
    assert "     line one\n     line two" in out


def test_accept_persian_text(env: Env) -> None:
    setup(env, ("lesson", "اول pytest رو نصب کن."))
    result = run(env, "review", "accept", PID, "1", "--yes")
    assert result.exit_code == 0
    assert "اول pytest رو نصب کن." in result.output
    assert env.memory.entries(MemoryScope.LESSON)[0].text == "اول pytest رو نصب کن."


# --- reject --------------------------------------------------------------------


def test_reject_records_decision_and_writes_no_memory(env: Env) -> None:
    rid = setup(env)
    result = run(env, "review", "reject", PID, "1", "3", input="y\n")
    assert result.exit_code == 0, result.output
    assert "Reject these 2 proposals?" in result.output
    assert "Uses click for the CLI." in result.output
    assert "1: rejected" in result.output and "3: rejected" in result.output
    assert env.all_memory() == 0
    decisions = env.reviews.decisions(PID, rid)
    assert {n: d.status for n, d in decisions.items()} == {
        1: DecisionStatus.REJECTED,
        3: DecisionStatus.REJECTED,
    }
    assert env.events().count("review.proposal_rejected") == 2


def test_reject_declined_records_nothing(env: Env) -> None:
    rid = setup(env)
    before = snapshot(env)
    result = run(env, "review", "reject", PID, "1", input="n\n")
    assert result.exit_code == 1
    assert "not rejected; nothing was changed" in result.output
    assert snapshot(env) == before
    assert env.reviews.pending(PID, rid) == [1, 2, 3]


def test_reject_yes(env: Env) -> None:
    rid = setup(env)
    assert run(env, "review", "reject", PID, "2", "--yes").exit_code == 0
    assert env.reviews.pending(PID, rid) == [1, 3]


def test_rejected_proposal_cannot_be_accepted(env: Env) -> None:
    setup(env)
    assert run(env, "review", "reject", PID, "1", "--yes").exit_code == 0
    result = run(env, "review", "accept", PID, "1", "--yes")
    assert result.exit_code == 1
    assert "proposal 1 is rejected" in result.output
    assert env.all_memory() == 0


def test_accepted_proposal_cannot_be_decided_again(env: Env) -> None:
    setup(env)
    assert run(env, "review", "accept", PID, "1", "--yes").exit_code == 0
    for command in ("accept", "reject"):
        result = run(env, "review", command, PID, "1", "--yes")
        assert result.exit_code == 1
        assert "proposal 1 is accepted" in result.output
    assert len(env.memory.entries(MemoryScope.PROJECT, PID)) == 1


# --- free checks before anything is written ------------------------------------


@pytest.mark.parametrize("command", ["accept", "reject"])
def test_one_decided_number_blocks_all(env: Env, command: str) -> None:
    rid = setup(env)
    assert run(env, "review", "reject", PID, "2", "--yes").exit_code == 0
    before = snapshot(env)
    result = run(env, "review", command, PID, "1", "2", "--yes")
    assert result.exit_code == 1
    assert "nothing was changed" in result.output
    assert snapshot(env) == before
    assert env.reviews.pending(PID, rid) == [1, 3]


@pytest.mark.parametrize("command", ["accept", "reject"])
@pytest.mark.parametrize("number", ["0", "4", "99"])
def test_out_of_range_number(env: Env, command: str, number: str) -> None:
    setup(env)
    before = snapshot(env)
    result = run(env, "review", command, PID, "1", number, "--yes")
    assert result.exit_code == 1
    assert "has proposals 1..3" in result.output
    assert f"no proposal {number}" in result.output
    assert snapshot(env) == before


@pytest.mark.parametrize("command", ["accept", "reject"])
def test_duplicate_numbers(env: Env, command: str) -> None:
    setup(env)
    before = snapshot(env)
    result = run(env, "review", command, PID, "1", "1", "--yes")
    assert result.exit_code == 1
    assert "only once" in result.output
    assert snapshot(env) == before


@pytest.mark.parametrize("command", ["accept", "reject"])
def test_non_integer_number(env: Env, command: str) -> None:
    setup(env)
    before = snapshot(env)
    result = run(env, "review", command, PID, "one", "--yes")
    assert result.exit_code == 2  # usage error from the parser
    assert snapshot(env) == before


@pytest.mark.parametrize("command", ["accept", "reject"])
def test_no_numbers_is_a_usage_error(env: Env, command: str) -> None:
    setup(env)
    result = run(env, "review", command, PID, "--yes")
    assert result.exit_code == 2
    assert env.all_memory() == 0


def test_empty_review_has_no_proposals(env: Env) -> None:
    env.project()
    save_review(env)  # no proposals: setup() would use the defaults
    result = run(env, "review", "accept", PID, "1", "--yes")
    assert result.exit_code == 1
    assert "has no proposals" in result.output


@pytest.mark.parametrize("command", ["accept", "reject"])
def test_no_review_yet(env: Env, command: str) -> None:
    env.project()
    result = run(env, "review", command, PID, "1", "--yes")
    assert result.exit_code == 1
    assert f"automa review run {PID}" in result.output
    assert not env.review_dir().exists()


@pytest.mark.parametrize("command", ["accept", "reject"])
def test_unknown_project(env: Env, command: str) -> None:
    result = run(env, "review", command, "nope", "1", "--yes")
    assert result.exit_code == 1
    assert not env.ws.exists() or env.all_memory() == 0


@pytest.mark.parametrize("command", ["accept", "reject"])
@pytest.mark.parametrize("rid", ["rev-20260101-000000-abcdef", "../x", "bad"])
def test_unknown_or_invalid_review_id(env: Env, command: str, rid: str) -> None:
    setup(env)
    before = snapshot(env)
    result = run(env, "review", command, PID, "1", "--review", rid, "--yes")
    assert result.exit_code == 1
    assert snapshot(env) == before


@pytest.mark.parametrize("command", ["accept", "reject"])
def test_corrupt_review_file_fails_closed(env: Env, command: str) -> None:
    rid = setup(env)
    (env.review_dir() / f"{rid}.json").write_text("{not json", encoding="utf-8")
    result = run(env, "review", command, PID, "1", "--yes")
    assert result.exit_code == 1
    assert env.all_memory() == 0


@pytest.mark.parametrize("command", ["accept", "reject"])
def test_unexpected_file_in_reviews_dir_fails_closed(env: Env, command: str) -> None:
    setup(env)
    (env.review_dir() / "stray.txt").write_text("x", encoding="utf-8")
    result = run(env, "review", command, PID, "1", "--yes")
    assert result.exit_code == 1
    assert "unexpected entry" in result.output
    assert env.all_memory() == 0


def test_accept_with_broken_memory_is_a_config_error(env: Env) -> None:
    rid = setup(env)
    lessons = env.ws / "memory" / "lessons"
    lessons.mkdir(parents=True, exist_ok=True)
    (lessons / "garbage.json").write_text("{", encoding="utf-8")
    result = run(env, "review", "accept", PID, "1", "--yes")
    assert result.exit_code == 2
    assert env.reviews.pending(PID, rid) == [1, 2, 3]


def test_reject_does_not_need_memory(env: Env) -> None:
    rid = setup(env)
    lessons = env.ws / "memory" / "lessons"
    lessons.mkdir(parents=True, exist_ok=True)
    (lessons / "garbage.json").write_text("{", encoding="utf-8")
    assert run(env, "review", "reject", PID, "1", "--yes").exit_code == 0
    assert env.reviews.pending(PID, rid) == [2, 3]


def test_accept_into_a_full_layer_fails_before_writing(env: Env) -> None:
    rid = setup(env)
    for i in range(MAX_ENTRIES_PER_LAYER):
        env.memory.add(MemoryScope.LESSON, f"lesson number {i}", source=MemorySource.USER)
    result = run(env, "review", "accept", PID, "1", "2", "--yes")
    assert result.exit_code == 1
    assert "lesson memory has room for 0 more" in result.output
    assert "nothing was changed" in result.output
    assert env.memory.entries(MemoryScope.PROJECT, PID) == []
    assert env.reviews.pending(PID, rid) == [1, 2, 3]


def test_full_layer_still_accepts_an_already_remembered_text(env: Env) -> None:
    setup(env)
    for i in range(MAX_ENTRIES_PER_LAYER - 1):
        env.memory.add(MemoryScope.LESSON, f"lesson number {i}", source=MemorySource.USER)
    env.memory.add(MemoryScope.LESSON, "Install pytest first.", source=MemorySource.USER)
    result = run(env, "review", "accept", PID, "2", "--yes")
    assert result.exit_code == 0, result.output
    assert "already in memory" in result.output


def test_failure_midway_reports_what_was_not_decided(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automa.reviews import ReviewStore, ReviewStoreError

    rid = setup(env)
    original = ReviewStore.accept

    def flaky(self: ReviewStore, pid: str, review_id: str, number: int) -> Any:
        if number == 2:
            raise ReviewStoreError("disk full")
        return original(self, pid, review_id, number)

    monkeypatch.setattr(ReviewStore, "accept", flaky)
    result = run(env, "review", "accept", PID, "1", "2", "3", "--yes")
    assert result.exit_code == 1
    assert "1: accepted" in result.output
    assert "proposal 2: disk full" in result.output
    assert "not decided: 2, 3" in result.output
    assert env.reviews.pending(PID, rid) == [2, 3]


# --- automa status: review line -------------------------------------------------


def test_status_without_review_on_done_project_suggests_one(env: Env) -> None:
    env.project()
    before = snapshot(env)
    out = run(env, "status", PID).output
    assert f"review: none (run one with: automa review run {PID})" in out
    assert snapshot(env) == before


def test_status_without_review_on_running_project(env: Env) -> None:
    env.project(ProjectStatus.RUNNING)
    out = run(env, "status", PID).output
    assert "review: none\n" in out


def test_status_shows_latest_review_and_pending(env: Env) -> None:
    rid = setup(env)
    assert run(env, "review", "reject", PID, "1", "--yes").exit_code == 0
    out = run(env, "status", PID).output
    assert f"review: {rid}: 3 proposal(s), 2 pending" in out
    assert f"decide with: automa review accept|reject {PID} <n>" in out


def test_status_all_decided_has_no_hint(env: Env) -> None:
    rid = setup(env, ("lesson", "Only one."))
    assert run(env, "review", "accept", PID, "1", "--yes").exit_code == 0
    out = run(env, "status", PID).output
    assert f"review: {rid}: 1 proposal(s), 0 pending\n" in out


def test_status_counts_reviews(env: Env) -> None:
    setup(env)
    rid = save_review(env, ("lesson", "Second."))
    out = run(env, "status", PID).output
    assert f"review: {rid}: 1 proposal(s), 1 pending (2 reviews)" in out


def test_status_json_review_key(env: Env) -> None:
    rid = setup(env)
    assert run(env, "review", "accept", PID, "2", "--yes").exit_code == 0
    data = json.loads(run(env, "status", PID, "--json").output)
    assert data["review"] == {
        "count": 1,
        "latest": rid,
        "proposals": 3,
        "pending": [1, 3],
        "error": None,
    }


def test_status_json_without_review(env: Env) -> None:
    env.project()
    data = json.loads(run(env, "status", PID, "--json").output)
    assert data["review"]["count"] == 0
    assert data["review"]["latest"] is None


def test_status_reports_broken_reviews_without_failing(env: Env) -> None:
    setup(env)
    (env.review_dir() / "stray.txt").write_text("x", encoding="utf-8")
    result = run(env, "status", PID)
    assert result.exit_code == 0
    assert "review: error:" in result.output
    data = json.loads(run(env, "status", PID, "--json").output)
    assert data["review"]["error"]


def test_status_needs_no_provider(env: Env) -> None:
    setup(env)
    assert run(env, "status", PID).exit_code == 0
    assert env.made_providers == 0
