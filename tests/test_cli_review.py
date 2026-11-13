"""``automa review run|show`` (D54): the post-project review from the CLI.

Local only: ``FakeProvider`` replaces the provider; no network, no real model calls.
``review run`` stores proposals only (never memory); ``review show`` is read-only.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.agents import Review
from automa.agents.reviewer import REVIEW_SYSTEM_PROMPT
from automa.budget.ledger import CostLedger
from automa.config import AutomaConfig, ConfigError, Tier
from automa.llm import FakeProvider, ProviderError
from automa.memory import MemoryScope, MemorySource, MemoryStore
from automa.plan import (
    AcceptanceCriterion,
    Plan,
    ResultStatus,
    Task,
    TaskResult,
    TaskState,
    TaskStatus,
    Verdict,
    VerdictOutcome,
)
from automa.project import ProjectStatus, ProjectStore
from automa.reviews import MAX_REVIEWS_PER_PROJECT, ReviewStore

PID = "proj-1"
T0 = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)

TOML = """\
workspace = "{ws}"
[tiers]
small = "fake-small"
medium = "fake-medium"
large = "fake-large"
[prices."fake-small"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[prices."fake-medium"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[prices."fake-large"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[limits]
project_usd = 5.0
daily_usd = 2.0
per_call_usd = {per_call}
"""

NO_MEDIUM_TOML = """\
workspace = "{ws}"
[tiers]
small = "fake-small"
[prices."fake-small"]
input_per_mtok = 1.0
output_per_mtok = 2.0
"""


def reply(*proposals: tuple[str, str]) -> str:
    return json.dumps({"proposals": [{"scope": s, "text": t} for s, t in proposals]})


TWO = reply(("project", "Uses click for the CLI."), ("lesson", "Install pytest first."))


def tree(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        os.path.relpath(os.path.join(d, f), root)
        for d, dirs, files in os.walk(root)
        for f in [*dirs, *files]
    }


class Env:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.ws = tmp_path / "ws"
        self.config = tmp_path / "automa.toml"
        self.write_config()
        self.provider = FakeProvider()
        self.made_providers = 0
        monkeypatch.chdir(tmp_path)
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)

        def factory(config: AutomaConfig) -> FakeProvider:
            self.made_providers += 1
            return self.provider

        monkeypatch.setattr(cli, "make_provider", factory)

    def write_config(self, per_call: float = 0.25, text: str | None = None) -> None:
        body = text if text is not None else TOML
        self.config.write_text(
            body.format(ws=self.ws.as_posix(), per_call=per_call), encoding="utf-8"
        )

    @property
    def store(self) -> ProjectStore:
        return ProjectStore.for_workspace(self.ws)

    @property
    def memory(self) -> MemoryStore:
        return MemoryStore.for_workspace(self.ws)

    @property
    def reviews(self) -> ReviewStore:
        return ReviewStore.for_workspace(self.ws)

    def project(self, status: ProjectStatus = ProjectStatus.DONE, pid: str = PID) -> str:
        store = self.store
        store.create("build a CLI", project_id=pid)
        task = Task(
            id="a",
            title="Task a",
            role="developer",
            acceptance_criteria=[AcceptanceCriterion(description="works")],
        )
        store.save_plan(pid, Plan(goal="build a CLI", tasks=[task]))
        store.save_task_result(
            pid,
            TaskResult(
                task_id="a",
                attempt=1,
                status=ResultStatus.COMPLETED,
                summary="wrote it",
                tier=Tier.MEDIUM,
                model="fake-medium",
                steps=2,
                cost_nusd=1_000,
                started_at=T0,
                finished_at=T0 + timedelta(seconds=3),
            ),
        )
        store.save_verdict(
            pid,
            Verdict(task_id="a", attempt=1, outcome=VerdictOutcome.PASS, created_at=T0),
        )
        store.save_task_state(
            pid, TaskState(task_id="a", status=TaskStatus.PASSED, attempts=1, updated_at=T0)
        )
        store.set_status(pid, status)
        return pid

    def invoke(self, *args: str) -> Any:
        return CliRunner().invoke(cli.app, [*args, "--config", str(self.config)])

    def review_dir(self, pid: str = PID) -> Path:
        return self.store.project_dir(pid) / "reviews"

    def events(self, pid: str = PID) -> list[str]:
        path = self.store.project_dir(pid) / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line)["type"] for line in path.read_text().splitlines()]

    def all_memory(self) -> int:
        m = self.memory
        return (
            len(m.entries(MemoryScope.USER))
            + len(m.entries(MemoryScope.LESSON))
            + sum(len(m.entries(MemoryScope.PROJECT, p)) for p in m.project_ids())
        )

    def spent(self, pid: str = PID) -> int:
        with CostLedger.for_workspace(self.ws) as ledger:
            return ledger.spent_nusd(project_id=pid)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


def save_review(env: Env, *proposals: tuple[str, str], pid: str = PID) -> str:
    review = Review(
        project_id=pid,
        project_status=env.store.load(pid).status,
        proposals=[{"scope": s, "text": t} for s, t in proposals],  # type: ignore[misc]
        created_at=T0,
    )
    return env.reviews.save(review).id


# --- help ----------------------------------------------------------------------


def test_review_help_lists_run_and_show(env: Env) -> None:
    result = CliRunner().invoke(cli.app, ["review", "--help"])
    assert result.exit_code == 0
    assert "run" in result.output and "show" in result.output
    top = CliRunner().invoke(cli.app, ["--help"])
    assert "review" in top.output


# --- review run: happy path ----------------------------------------------------


def test_run_stores_proposals_and_writes_no_memory(env: Env) -> None:
    env.project()
    env.provider.push(TWO)
    result = env.invoke("review", "run", PID)
    assert result.exit_code == 0, result.output
    out = result.output
    assert f"project: {PID} (done)" in out
    assert "cost: $" in out
    assert "proposals: 2 (1 project, 1 lesson)" in out
    assert "1. [project] pending" in out and "Uses click for the CLI." in out
    assert "2. [lesson] pending" in out and "Install pytest first." in out
    assert "nothing has been added to memory" in out
    stored = env.reviews.all(PID)
    assert len(stored) == 1
    assert [p.text for p in stored[0].proposals] == [
        "Uses click for the CLI.",
        "Install pytest first.",
    ]
    assert f"review: {stored[0].id}" in out
    assert env.all_memory() == 0
    assert env.reviews.pending(PID, stored[0].id) == [1, 2]
    assert env.store.load(PID).status is ProjectStatus.DONE


def test_run_goes_through_the_gateway_and_counts_toward_the_project(env: Env) -> None:
    env.project()
    env.provider.push(TWO)
    assert env.spent() == 0
    assert env.invoke("review", "run", PID).exit_code == 0
    assert len(env.provider.requests) == 1
    request = env.provider.requests[0]
    assert request.model == "fake-medium"
    assert request.messages[0].content == REVIEW_SYSTEM_PROMPT
    assert request.tools in (None, [], ())
    assert env.spent() > 0
    types = env.events()
    assert "review.ready" in types and "review.saved" in types
    assert types.index("review.ready") < types.index("review.saved")


def test_run_failed_project(env: Env) -> None:
    env.project(ProjectStatus.FAILED)
    env.provider.push(reply(("lesson", "Check the toolchain before starting.")))
    result = env.invoke("review", "run", PID)
    assert result.exit_code == 0, result.output
    assert f"project: {PID} (failed)" in result.output
    assert env.reviews.latest(PID).project_status is ProjectStatus.FAILED  # type: ignore[union-attr]


def test_run_json(env: Env) -> None:
    env.project()
    env.provider.push(TWO)
    result = env.invoke("review", "run", PID, "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert set(data) == {"review", "decisions", "pending", "cost_nusd"}
    assert data["review"]["project_id"] == PID
    assert [p["scope"] for p in data["review"]["proposals"]] == ["project", "lesson"]
    assert data["decisions"] == {}
    assert data["pending"] == [1, 2]
    assert data["cost_nusd"] == env.spent() > 0


def test_run_empty_reply(env: Env) -> None:
    env.project()
    env.provider.push(reply())
    result = env.invoke("review", "run", PID)
    assert result.exit_code == 0, result.output
    assert "proposals: 0" in result.output
    assert "nothing worth remembering" in result.output
    assert len(env.reviews.all(PID)) == 1


def test_run_drops_duplicates_of_existing_memory(env: Env) -> None:
    env.project()
    env.memory.add(MemoryScope.LESSON, "Install pytest first.", source=MemorySource.USER)
    env.provider.push(TWO)
    result = env.invoke("review", "run", PID)
    assert result.exit_code == 0, result.output
    assert "proposals: 1 (1 project, 0 lesson); 1 duplicate(s)" in result.output
    # Existing memory is never sent to the model.
    sent = "\n".join(str(m.content) for m in env.provider.requests[0].messages)
    assert "Install pytest first." not in sent
    assert env.all_memory() == 1


def test_run_escapes_model_text(env: Env) -> None:
    env.project()
    env.provider.push(reply(("lesson", "safe ‮evil‬ end")))
    result = env.invoke("review", "run", PID)
    assert result.exit_code == 0, result.output
    assert "‮" not in result.output
    assert "\\u202e" in result.output


def test_run_keeps_persian_text(env: Env) -> None:
    env.project()
    text = "قبل از شروع، می‌خواستیم تست‌ها را اجرا کنیم."
    env.provider.push(reply(("lesson", text)))
    result = env.invoke("review", "run", PID)
    assert result.exit_code == 0, result.output
    assert text in result.output


def test_run_multiline_proposal_is_indented(env: Env) -> None:
    env.project()
    env.provider.push(reply(("project", "line one\nline two")))
    result = env.invoke("review", "run", PID)
    assert "     line one\n     line two" in result.output


# --- review run: refused before any model call -------------------------------


@pytest.mark.parametrize(
    "status",
    [ProjectStatus.NEW, ProjectStatus.PLANNING, ProjectStatus.RUNNING, ProjectStatus.PAUSED],
)
def test_run_refuses_unfinished_project(env: Env, status: ProjectStatus) -> None:
    env.project(status)
    result = env.invoke("review", "run", PID)
    assert result.exit_code == cli.EXIT_USER
    assert "only done or failed" in result.output
    assert env.made_providers == 0 and env.provider.requests == []
    assert not env.review_dir().exists()
    assert env.store.load(PID).status is status


def test_run_unknown_project(env: Env) -> None:
    before = tree(env.ws)
    result = env.invoke("review", "run", "nope")
    assert result.exit_code == cli.EXIT_USER
    assert env.made_providers == 0
    assert tree(env.ws) == before


@pytest.mark.parametrize("pid", ["../x", "A B", ""])
def test_run_invalid_project_id(env: Env, pid: str) -> None:
    result = env.invoke("review", "run", pid)
    assert result.exit_code == cli.EXIT_USER
    assert env.made_providers == 0


def test_run_refuses_second_review_without_again(env: Env) -> None:
    env.project()
    first = save_review(env, ("lesson", "x"))
    result = env.invoke("review", "run", PID)
    assert result.exit_code == cli.EXIT_USER
    assert first in result.output and "--again" in result.output
    assert env.made_providers == 0 and env.provider.requests == []
    assert len(env.reviews.all(PID)) == 1


def test_run_again_stores_another_review(env: Env) -> None:
    env.project()
    save_review(env, ("lesson", "x"))
    env.provider.push(TWO)
    result = env.invoke("review", "run", PID, "--again")
    assert result.exit_code == 0, result.output
    assert len(env.reviews.all(PID)) == 2
    assert env.all_memory() == 0


def test_run_review_limit(env: Env) -> None:
    env.project()
    for _ in range(MAX_REVIEWS_PER_PROJECT):
        save_review(env)
    result = env.invoke("review", "run", PID, "--again")
    assert result.exit_code == cli.EXIT_USER
    assert "no more can be stored" in result.output
    assert env.made_providers == 0


def test_run_missing_tier_is_config_error(env: Env) -> None:
    env.project()
    env.write_config(text=NO_MEDIUM_TOML)
    result = env.invoke("review", "run", PID)
    assert result.exit_code == cli.EXIT_CONFIG
    assert "medium" in result.output
    assert env.made_providers == 0
    assert not env.review_dir().exists()


def test_run_provider_config_error(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    env.project()

    def no_key(config: AutomaConfig) -> FakeProvider:
        raise ConfigError("AUTOMA_API_KEY is not set")

    monkeypatch.setattr(cli, "make_provider", no_key)
    result = env.invoke("review", "run", PID)
    assert result.exit_code == cli.EXIT_CONFIG
    assert "AUTOMA_API_KEY" in result.output
    assert not env.review_dir().exists()


def test_run_broken_memory_is_config_error(env: Env) -> None:
    env.project()
    lessons = env.ws / "memory" / "lessons"
    lessons.mkdir(parents=True)
    (lessons / "junk.txt").write_text("x")
    result = env.invoke("review", "run", PID)
    assert result.exit_code == cli.EXIT_CONFIG
    assert env.provider.requests == []
    assert not env.review_dir().exists()


def test_run_broken_reviews_dir_fails_closed(env: Env) -> None:
    env.project()
    env.review_dir().mkdir()
    (env.review_dir() / "junk.json").write_text("{}")
    result = env.invoke("review", "run", PID)
    assert result.exit_code == cli.EXIT_USER
    assert "unexpected entry" in result.output
    assert env.made_providers == 0


def test_run_symlinked_reviews_dir_fails_closed(env: Env, tmp_path: Path) -> None:
    env.project()
    target = tmp_path / "elsewhere"
    target.mkdir()
    env.review_dir().symlink_to(target)
    result = env.invoke("review", "run", PID)
    assert result.exit_code == cli.EXIT_USER
    assert env.made_providers == 0
    assert list(target.iterdir()) == []


# --- review run: stops during the model call ----------------------------------


def test_run_budget_stop_stores_nothing(env: Env) -> None:
    env.project()
    env.write_config(per_call=0.000001)
    env.provider.push(TWO)
    result = env.invoke("review", "run", PID)
    assert result.exit_code == cli.EXIT_BUDGET
    assert "budget stop" in result.output and "Nothing was stored" in result.output
    assert env.provider.requests == []  # the guard stops before the provider
    assert not env.review_dir().exists()
    assert env.store.load(PID).status is ProjectStatus.DONE
    assert env.all_memory() == 0


def test_run_invalid_replies_exit_planning(env: Env) -> None:
    env.project()
    env.provider.push("not json", reply(("user", "prefers tabs")), "{}x")
    result = env.invoke("review", "run", PID)
    assert result.exit_code == cli.EXIT_PLANNING
    assert "nothing was stored" in result.output
    assert len(env.provider.requests) == 3
    assert not env.review_dir().exists()
    assert env.spent() > 0  # the failed attempts were still paid and recorded


def test_run_provider_error(env: Env) -> None:
    env.project()
    env.provider.push(ProviderError("boom"))
    result = env.invoke("review", "run", PID)
    assert result.exit_code == cli.EXIT_PROVIDER
    assert "provider error" in result.output
    assert not env.review_dir().exists()


def test_run_interrupted(env: Env) -> None:
    env.project()
    env.provider.push(KeyboardInterrupt())
    result = env.invoke("review", "run", PID)
    assert result.exit_code == cli.EXIT_INTERRUPTED
    assert not env.review_dir().exists()
    assert env.store.load(PID).status is ProjectStatus.DONE


def test_run_status_changed_during_review_is_not_stored(env: Env) -> None:
    env.project()

    def rerun_meanwhile(request: Any) -> str:
        env.store.set_status(PID, ProjectStatus.RUNNING)
        return TWO

    env.provider.push(rerun_meanwhile)
    result = env.invoke("review", "run", PID)
    assert result.exit_code == cli.EXIT_USER
    assert "review not stored" in result.output
    assert env.reviews.all(PID) == []


# --- review show -----------------------------------------------------------------


def test_show_without_reviews_creates_nothing(env: Env) -> None:
    env.project()
    before = tree(env.ws)
    result = env.invoke("review", "show", PID)
    assert result.exit_code == 0
    assert f"automa review run {PID}" in result.output
    assert tree(env.ws) == before
    as_json = env.invoke("review", "show", PID, "--json")
    assert as_json.exit_code == 0 and json.loads(as_json.output) is None
    assert tree(env.ws) == before


def test_show_needs_no_provider(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    env.project()
    save_review(env, ("lesson", "x"))

    def forbidden(config: AutomaConfig) -> FakeProvider:
        raise AssertionError("review show must not create a provider")

    monkeypatch.setattr(cli, "make_provider", forbidden)
    assert env.invoke("review", "show", PID).exit_code == 0


def test_show_latest_with_decisions(env: Env) -> None:
    env.project()
    rid = save_review(env, ("project", "A fact."), ("lesson", "A lesson."), ("lesson", "Other."))
    accepted = env.reviews.accept(PID, rid, 2)
    env.reviews.reject(PID, rid, 3)
    before = tree(env.ws)
    result = env.invoke("review", "show", PID)
    assert result.exit_code == 0, result.output
    out = result.output
    assert f"review: {rid}" in out
    assert "1. [project] pending" in out
    assert f"2. [lesson] accepted ({accepted.memory_id})" in out
    assert "3. [lesson] rejected" in out
    assert "A fact." in out and "Other." in out
    assert "reviews:" not in out  # only one review
    assert tree(env.ws) == before


def test_show_marks_memory_that_already_existed(env: Env) -> None:
    env.project()
    env.memory.add(MemoryScope.LESSON, "A lesson.", source=MemorySource.USER)
    rid = save_review(env, ("lesson", "A lesson."))
    env.reviews.accept(PID, rid, 1)
    assert "already in memory" in env.invoke("review", "show", PID).output


def test_show_json(env: Env) -> None:
    env.project()
    rid = save_review(env, ("project", "A fact."), ("lesson", "A lesson."))
    env.reviews.accept(PID, rid, 1)
    result = env.invoke("review", "show", PID, "--json")
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["review"]["id"] == rid
    assert set(data["decisions"]) == {"1"}
    assert data["decisions"]["1"]["status"] == "accepted"
    assert data["pending"] == [2]


def test_show_specific_and_multiple_reviews(env: Env) -> None:
    env.project()
    first = save_review(env, ("lesson", "first"))
    second = save_review(env, ("lesson", "second"))
    latest = env.invoke("review", "show", PID)
    assert f"review: {second}" in latest.output
    assert f"reviews: 2 (others: {first})" in latest.output
    chosen = env.invoke("review", "show", PID, "--review", first)
    assert chosen.exit_code == 0
    assert f"review: {first}" in chosen.output and "first" in chosen.output


def test_show_unknown_review(env: Env) -> None:
    env.project()
    save_review(env, ("lesson", "x"))
    result = env.invoke("review", "show", PID, "--review", "rev-20260101-000000-abcdef")
    assert result.exit_code == cli.EXIT_USER
    bad = env.invoke("review", "show", PID, "--review", "../../etc")
    assert bad.exit_code == cli.EXIT_USER
    assert "invalid review id" in bad.output


def test_show_unknown_project(env: Env) -> None:
    before = tree(env.ws)
    assert env.invoke("review", "show", "nope").exit_code == cli.EXIT_USER
    assert tree(env.ws) == before


def test_show_broken_review_fails_closed(env: Env) -> None:
    env.project()
    rid = save_review(env, ("lesson", "x"))
    (env.review_dir() / f"{rid}.json").write_text("{broken")
    result = env.invoke("review", "show", PID)
    assert result.exit_code == cli.EXIT_USER


def test_show_escapes_text(env: Env) -> None:
    env.project()
    save_review(env, ("lesson", "a ‮b‬ c"))
    result = env.invoke("review", "show", PID)
    assert "‮" not in result.output and "\\u202e" in result.output
