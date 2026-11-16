"""``automa self propose``: new roadmap items proposed and stored, never applied (D86).

Local only: a temporary git repository, ``FakeProvider`` behind the real guarded
``LLMGateway`` + ``BudgetGuard`` + ``CostLedger``. No network, no real model call.
"""

from __future__ import annotations

import ast
import inspect
import json
import shutil
from pathlib import Path

import pytest

from automa import cli
from automa.agents.proposer import DEFAULT_BUDGET_ID
from automa.budget import CostLedger
from automa.events import JsonlEventLog
from automa.llm import ProviderError
from automa.llm.factory import provider_from_config
from automa.selfdev_project import SnapshotError, read_commit_texts
from automa.selfdev_proposals import MAX_SETS, ProposalStore, Screened, roadmap_digest
from test_cli import Env
from test_selfdev_project import ROADMAP, commit_all, git, tree

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

VISION = "Automa builds software by itself and never overspends.\n"


def reply(*items: tuple[str, str, str]) -> str:
    return json.dumps(
        {"proposals": [{"milestone": m, "text": t, "reason": r} for m, t, r in items]}
    )


GOOD = reply(
    ("M0 — base", "add a cost summary to the status command", "owners ask for it"),
    ("M0 — base", "document the exit codes in the README", ""),
)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "ROADMAP.md").write_text(ROADMAP, encoding="utf-8")
    (r / "docs").mkdir()
    (r / "docs" / "VISION.md").write_text(VISION, encoding="utf-8")
    (r / ".gitignore").write_text(".env\n", encoding="utf-8")
    commit_all(r, "first")
    return r


def propose(env: Env, repo: Path, *args: str) -> object:
    return env.run("self", "propose", "--repo", str(repo), *args)


def store(env: Env) -> ProposalStore:
    return ProposalStore.for_workspace(env.ws)


def nothing_stored(env: Env) -> bool:
    return not (env.ws / "selfdev" / "proposals").exists() or store(env).all() == []


# -- happy path ----------------------------------------------------------------------


def test_proposes_and_stores_a_set(env: Env, repo: Path) -> None:
    env.provider.push(GOOD)
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_OK, r.output
    sets = store(env).all()
    assert len(sets) == 1
    pset = sets[0]
    head = git(repo, "rev-parse", "HEAD")
    assert pset.commit == head
    assert pset.roadmap_sha256 == roadmap_digest(ROADMAP)
    assert [p.text for p in pset.proposals] == [
        "add a cost summary to the status command",
        "document the exit codes in the README",
    ]
    assert f"proposal set: {pset.id} (commit {head[:12]})" in r.output
    assert "1. [M0 — base]" in r.output
    assert "   - [ ] add a cost summary to the status command" in r.output
    assert "   why: owners ask for it" in r.output
    assert "2. [M0 — base]" in r.output
    assert "cost: $" in r.output
    assert "nothing was written to the repository" in r.output
    assert store(env).pending(pset.id) == [1, 2]
    assert len(env.provider.requests) == 1


def test_prompt_contains_vision_and_roadmap(env: Env, repo: Path) -> None:
    env.provider.push(GOOD)
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_OK, r.output
    prompt = env.provider.requests[0].messages[-1].content or ""
    assert "never overspends" in prompt
    assert "first open item" in prompt and "needs the owner" in prompt


def test_json_output(env: Env, repo: Path) -> None:
    env.provider.push(GOOD)
    r = propose(env, repo, "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    out = json.loads(r.output)
    pset = store(env).all()[0]
    assert out["id"] == pset.id
    assert out["commit"] == git(repo, "rev-parse", "HEAD")
    assert out["attempts"] == 1
    assert out["cost_nusd"] > 0
    assert [p["number"] for p in out["proposals"]] == [1, 2]
    assert out["proposals"][0]["line"] == "- [ ] add a cost summary to the status command"
    assert out["dropped_duplicates"] == 0


def test_duplicates_and_unknown_milestones_are_dropped(env: Env, repo: Path) -> None:
    env.provider.push(
        reply(
            ("M0 — base", "first open item", "already there"),
            ("M0 — base", "second open item", "already there too"),
            ("M9 — nowhere", "a new idea for nowhere", ""),
            ("m0 — base", "a fresh idea worth doing", ""),
        )
    )
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_OK, r.output
    pset = store(env).all()[0]
    assert [(p.milestone, p.text) for p in pset.proposals] == [
        ("M0 — base", "a fresh idea worth doing")
    ]
    assert (pset.dropped_duplicates, pset.dropped_unknown_milestone) == (2, 1)
    assert "dropped: 2 duplicate(s), 1 with an unknown milestone" in r.output


def test_empty_reply_is_stored_as_an_empty_set(env: Env, repo: Path) -> None:
    env.provider.push(reply())
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "no new item proposed" in r.output
    assert store(env).all()[0].proposals == []


def test_invalid_reply_is_retried(env: Env, repo: Path) -> None:
    env.provider.push("not json", GOOD)
    r = propose(env, repo, "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert json.loads(r.output)["attempts"] == 2
    assert len(env.provider.requests) == 2


def test_ref_reads_an_earlier_commit(env: Env, repo: Path) -> None:
    first = git(repo, "rev-parse", "HEAD")
    (repo / "ROADMAP.md").write_text(ROADMAP + "- [ ] added later\n", encoding="utf-8")
    commit_all(repo, "second")
    env.provider.push(GOOD)
    r = propose(env, repo, "--ref", first[:10])
    assert r.exit_code == cli.EXIT_OK, r.output
    pset = store(env).all()[0]
    assert pset.commit == first
    assert pset.roadmap_sha256 == roadmap_digest(ROADMAP)
    assert "added later" not in (env.provider.requests[0].messages[-1].content or "")


def test_uncommitted_changes_are_not_used(env: Env, repo: Path) -> None:
    (repo / "ROADMAP.md").write_text("## M\n- [?] broken but not committed\n", encoding="utf-8")
    (repo / "docs" / "VISION.md").write_text("uncommitted vision\n", encoding="utf-8")
    env.provider.push(GOOD)
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_OK, r.output
    prompt = env.provider.requests[0].messages[-1].content or ""
    assert "uncommitted vision" not in prompt and "never overspends" in prompt


def test_custom_paths(env: Env, repo: Path) -> None:
    (repo / "PLAN.md").write_text("## Only\n- [ ] one item here\n", encoding="utf-8")
    (repo / "IDEA.md").write_text("an idea\n", encoding="utf-8")
    commit_all(repo, "custom")
    env.provider.push(reply(("Only", "another item for the only milestone", "")))
    r = propose(env, repo, "--roadmap", "PLAN.md", "--vision", "IDEA.md")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "add its line to PLAN.md by hand" in r.output
    assert store(env).all()[0].proposals[0].milestone == "Only"


def test_default_repo_is_the_current_directory(
    env: Env, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(repo)
    env.provider.push(GOOD)
    r = env.run("self", "propose")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert len(store(env).all()) == 1


def test_two_runs_make_two_sets(env: Env, repo: Path) -> None:
    env.provider.push(GOOD, GOOD)
    assert propose(env, repo).exit_code == cli.EXIT_OK
    assert propose(env, repo).exit_code == cli.EXIT_OK
    assert len(store(env).all()) == 2


# -- the repository and ROADMAP are never written ------------------------------------


def test_repository_is_never_changed(env: Env, repo: Path) -> None:
    (repo / ".env").write_text("AUTOMA_API_KEY=sk-not-committed\n", encoding="utf-8")
    before = tree(repo)
    head = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain")
    env.provider.push(GOOD)
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert tree(repo) == before
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "status", "--porcelain") == status
    assert "sk-not-committed" not in r.output
    assert "sk-not-committed" not in (env.provider.requests[0].messages[-1].content or "")


def test_only_the_proposal_store_and_logs_are_written(env: Env, repo: Path) -> None:
    env.provider.push(GOOD)
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_OK, r.output
    selfdev = env.ws / "selfdev"
    names = sorted(p.relative_to(selfdev).as_posix() for p in selfdev.rglob("*"))
    pset = store(env).all()[0]
    assert names == ["events.jsonl", "proposals", f"proposals/{pset.id}.json"]
    assert env.store.list_ids() == []  # no project is created


def test_events_are_logged(env: Env, repo: Path) -> None:
    env.provider.push("not json", GOOD)
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_OK, r.output
    events = list(JsonlEventLog(env.ws / "selfdev" / "events.jsonl").read())
    types = [e.type for e in events]
    assert types == ["selfdev.proposals_rejected", "selfdev.proposals_ready"]
    assert all(e.project_id == DEFAULT_BUDGET_ID for e in events)


def test_cost_is_recorded_under_the_proposal_budget_id(env: Env, repo: Path) -> None:
    env.provider.push(GOOD)
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_OK, r.output
    with CostLedger.for_workspace(env.ws) as ledger:
        assert ledger.spent_nusd(project_id=DEFAULT_BUDGET_ID) > 0


def test_cost_line_shows_this_run_only(env: Env, repo: Path) -> None:
    env.provider.push(GOOD, GOOD)
    first = json.loads(propose(env, repo, "--json").output)["cost_nusd"]  # type: ignore[attr-defined]
    second = json.loads(propose(env, repo, "--json").output)["cost_nusd"]  # type: ignore[attr-defined]
    assert first > 0 and second == first


# -- errors: nothing stored ----------------------------------------------------------


def test_missing_key_fails_before_reading(
    env: Env, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "make_provider", provider_from_config)
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_CONFIG
    assert "AUTOMA_API_KEY" in r.output
    assert nothing_stored(env)


def test_missing_tier_fails_before_anything(env: Env, repo: Path) -> None:
    env.config.write_text(f'workspace = "{env.ws.as_posix()}"\n', encoding="utf-8")
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_CONFIG
    assert env.factory_calls == []
    assert nothing_stored(env)


def test_missing_roadmap_file(env: Env, repo: Path) -> None:
    r = propose(env, repo, "--roadmap", "NOPE.md")
    assert r.exit_code == cli.EXIT_USER
    assert "repository:" in r.output and "not a tracked file" in r.output
    assert "nothing was stored" in r.output
    assert env.provider.requests == [] and nothing_stored(env)


def test_missing_vision_file(env: Env, repo: Path) -> None:
    r = propose(env, repo, "--vision", "docs/NOPE.md")
    assert r.exit_code == cli.EXIT_USER
    assert env.provider.requests == [] and nothing_stored(env)


@pytest.mark.parametrize("bad", ["../ROADMAP.md", "/etc/passwd", ".git/HEAD", ""])
def test_unsafe_paths_are_refused(env: Env, repo: Path, bad: str) -> None:
    for option in ("--roadmap", "--vision"):
        r = propose(env, repo, option, bad)
        assert r.exit_code == cli.EXIT_USER
    assert env.provider.requests == [] and nothing_stored(env)


def test_ambiguous_roadmap_exits_10(env: Env, repo: Path) -> None:
    (repo / "ROADMAP.md").write_text("## M\n- [?] what is this\n", encoding="utf-8")
    commit_all(repo, "bad")
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_ROADMAP_INVALID
    assert "roadmap:" in r.output
    assert env.provider.requests == [] and nothing_stored(env)


def test_roadmap_without_milestones_is_refused_before_the_model(env: Env, repo: Path) -> None:
    (repo / "ROADMAP.md").write_text("# Roadmap\n\nnothing here\n", encoding="utf-8")
    commit_all(repo, "empty")
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_USER
    assert "no milestones" in r.output
    assert env.provider.requests == [] and nothing_stored(env)


def test_empty_vision_is_refused_before_the_model(env: Env, repo: Path) -> None:
    (repo / "docs" / "VISION.md").write_text("  \n", encoding="utf-8")
    commit_all(repo, "empty vision")
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_USER
    assert "VISION is empty" in r.output
    assert env.provider.requests == [] and nothing_stored(env)


def test_binary_vision_is_refused(env: Env, repo: Path) -> None:
    (repo / "docs" / "VISION.md").write_bytes(b"\xff\xfe\x00bad")
    commit_all(repo, "binary")
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_USER
    assert "not valid UTF-8" in r.output
    assert env.provider.requests == []


def test_not_a_repository(env: Env, tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    r = propose(env, plain)
    assert r.exit_code == cli.EXIT_USER
    assert env.provider.requests == [] and nothing_stored(env)


@pytest.mark.parametrize("ref", ["--upload-pack=x", "no-such-branch", "a b", "-x"])
def test_bad_ref(env: Env, repo: Path, ref: str) -> None:
    r = propose(env, repo, f"--ref={ref}")
    assert r.exit_code == cli.EXIT_USER
    assert env.provider.requests == [] and nothing_stored(env)


def test_symlink_in_repository_is_refused(env: Env, repo: Path) -> None:
    (repo / "link").symlink_to("ROADMAP.md")
    commit_all(repo, "link")
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_USER
    assert "symlinks are not supported" in r.output
    assert env.provider.requests == []


def test_full_store_is_refused_before_the_model(
    env: Env, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import automa.selfdev_proposals as sp

    monkeypatch.setattr(sp, "MAX_SETS", 1)
    env.provider.push(GOOD, GOOD)
    assert propose(env, repo).exit_code == cli.EXIT_OK
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_USER
    assert "already 1 proposal sets" in r.output
    assert len(env.provider.requests) == 1
    assert MAX_SETS == 200


def test_broken_store_is_refused_before_the_model(env: Env, repo: Path) -> None:
    root = env.ws / "selfdev" / "proposals"
    root.mkdir(parents=True)
    (root / "junk.txt").write_text("x", encoding="utf-8")
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_USER
    assert "unexpected entry" in r.output
    assert env.provider.requests == []


def test_per_call_cap_blocks_before_the_provider(env: Env, repo: Path) -> None:
    env.write_config(project=1.0, daily=1.0, per_call=0.000001)
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_BUDGET
    assert "per_call cap" in r.output and "nothing was stored" in r.output
    assert env.provider.requests == [] and nothing_stored(env)


def test_project_cap_bounds_proposal_spending(env: Env, repo: Path) -> None:
    # Every run is attributed to the same budget id, so repeated runs hit its cap.
    env.write_config(project=0.01, daily=1.0, per_call=0.01)
    codes = []
    for _ in range(30):
        env.provider.push(GOOD)
        r = propose(env, repo)
        codes.append(r.exit_code)
        if r.exit_code != cli.EXIT_OK:
            break
    assert codes[-1] == cli.EXIT_BUDGET and "project cap" in r.output  # type: ignore[attr-defined]
    assert codes.count(cli.EXIT_OK) >= 1
    assert len(store(env).all()) == codes.count(cli.EXIT_OK)
    with CostLedger.for_workspace(env.ws) as ledger:
        assert ledger.spent_nusd(project_id=DEFAULT_BUDGET_ID) <= 10_000_000


def test_no_valid_reply_exits_4(env: Env, repo: Path) -> None:
    env.provider.push("nope", "still nope", "{}x")
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_PLANNING
    assert "no valid proposals" in r.output and "nothing was stored" in r.output
    assert len(env.provider.requests) == 3
    assert nothing_stored(env)


def test_provider_error_exits_5(env: Env, repo: Path) -> None:
    env.provider.push(ProviderError("HTTP 502", status=502, retryable=False))
    r = propose(env, repo)
    assert r.exit_code == cli.EXIT_PROVIDER
    assert nothing_stored(env)


# -- read_commit_texts ---------------------------------------------------------------


def test_read_commit_texts_reads_the_commit(repo: Path) -> None:
    read = read_commit_texts(repo, ["ROADMAP.md", "docs/VISION.md"])
    assert read.commit == git(repo, "rev-parse", "HEAD")
    assert read.texts == {"ROADMAP.md": ROADMAP, "docs/VISION.md": VISION}


def test_read_commit_texts_same_path_twice(repo: Path) -> None:
    read = read_commit_texts(repo, ["ROADMAP.md", "ROADMAP.md"])
    assert read.texts == {"ROADMAP.md": ROADMAP}


def test_read_commit_texts_size_limit(repo: Path) -> None:
    with pytest.raises(SnapshotError, match="larger than 10 bytes"):
        read_commit_texts(repo, ["ROADMAP.md"], max_bytes=10)


def test_read_commit_texts_needs_paths(repo: Path) -> None:
    with pytest.raises(SnapshotError, match="no files"):
        read_commit_texts(repo, [])


def test_read_commit_texts_does_not_read_a_directory(repo: Path) -> None:
    with pytest.raises(SnapshotError, match="not a tracked file"):
        read_commit_texts(repo, ["docs"])


def test_check_room_on_a_missing_store(tmp_path: Path) -> None:
    ProposalStore.for_workspace(tmp_path).check_room()
    assert not (tmp_path / "selfdev").exists()


def test_check_room_does_not_write(tmp_path: Path) -> None:
    s = ProposalStore.for_workspace(tmp_path)
    s.save(Screened(), roadmap_text=ROADMAP)
    before = sorted(p.name for p in s.root.iterdir())
    s.check_room()
    assert sorted(p.name for p in s.root.iterdir()) == before


# -- structure -----------------------------------------------------------------------


def _calls(fn: object) -> set[str]:
    tree_ = ast.parse(inspect.getsource(fn))  # type: ignore[arg-type]
    return {
        n.func.id
        for n in ast.walk(tree_)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }


def test_propose_uses_the_guarded_gateway() -> None:
    calls = _calls(cli.self_propose)
    assert {"BudgetGuard", "LLMGateway", "ProposalAgent", "make_provider"} <= calls
    src = inspect.getsource(cli.self_propose)
    assert ".complete(" not in src
    assert "write_text" not in src and "open(" not in src


def test_help_lists_self_propose() -> None:
    from typer.testing import CliRunner

    r = CliRunner().invoke(cli.app, ["self", "--help"])
    assert r.exit_code == 0
    assert "propose" in r.output
