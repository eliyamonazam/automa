"""``automa self new``: next roadmap item on a repository snapshot, planned (D83).

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
from automa.budget import CostLedger
from automa.llm import ProviderError
from automa.llm.factory import provider_from_config
from automa.project import ProjectStatus
from automa.selfdev import build_goal, next_item, read_roadmap
from automa.selfdev_project import STAGING_PREFIX
from test_cli import GOOD_PLAN, NO_QUESTIONS, Env
from test_selfdev_project import ROADMAP, commit_all, git, tree

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "ROADMAP.md").write_text(ROADMAP, encoding="utf-8")
    (r / "CLAUDE.md").write_text("rules\n", encoding="utf-8")
    (r / "src").mkdir()
    (r / "src" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (r / ".gitignore").write_text(".env\n", encoding="utf-8")
    commit_all(r, "first")
    return r


def self_new(env: Env, repo: Path, *args: str, input: str | None = None) -> object:
    return env.run("self", "new", "--repo", str(repo), *args, input=input)


def nothing_created(env: Env) -> bool:
    if not env.ws.exists():
        return True
    projects = env.ws / "projects"
    staging = [p for p in projects.iterdir()] if projects.exists() else []
    return env.store.list_ids() == [] and not any(
        p.name.startswith(STAGING_PREFIX) for p in staging
    )


# -- happy path ---------------------------------------------------------------------


def test_creates_snapshot_project_and_plans_it(env: Env, repo: Path) -> None:
    env.provider.push(GOOD_PLAN)
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_OK, r.output
    head = git(repo, "rev-parse", "HEAD")
    assert "project: self-1" in r.output
    assert "item: line 6 [ ] M0 — base" in r.output
    assert "  first open item" in r.output
    assert f"snapshot: commit {head[:12]}, 4 file(s)" in r.output
    assert "the repository itself is not changed" in r.output
    assert "plan v1: 2 task(s)" in r.output
    assert "cost: $" in r.output
    assert "status: planning" in r.output
    item = next_item(read_roadmap(repo / "ROADMAP.md"))
    assert item is not None
    assert env.store.load("self-1").goal == build_goal(item)
    assert env.store.has_plan("self-1")
    files = tree(env.store.files_dir("self-1"))
    assert files["ROADMAP.md"] == ROADMAP.encode()
    assert set(files) == {"ROADMAP.md", "CLAUDE.md", "src/mod.py", ".gitignore"}
    assert len(env.provider.requests) == 1
    prompt = env.provider.requests[0].messages[-1].content or ""
    assert "first open item" in prompt
    assert "second open item" not in prompt


def test_records_selfdev_item_event(env: Env, repo: Path) -> None:
    env.provider.push(GOOD_PLAN)
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_OK, r.output
    events = list(env.store.events("self-1").read())
    types = [e.type for e in events]
    assert types.index("selfdev.item") < types.index("plan.saved")
    data = next(e.data for e in events if e.type == "selfdev.item")
    assert data["line"] == 6 and data["commit"] == git(repo, "rev-parse", "HEAD")


def test_ask_saves_answers(env: Env, repo: Path) -> None:
    env.provider.push(json.dumps({"questions": ["Which module?"]}), GOOD_PLAN)
    r = self_new(env, repo, "--id", "self-1", input="the cli\n")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "Which module?" in r.output
    items = env.store.load_clarifications("self-1")
    assert [(c.question, c.answer) for c in items] == [("Which module?", "the cli")]
    assert len(env.provider.requests) == 2


def test_ask_is_the_default(env: Env, repo: Path) -> None:
    env.provider.push(NO_QUESTIONS, GOOD_PLAN)
    r = self_new(env, repo, "--id", "self-1")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert len(env.provider.requests) == 2


def test_generated_id(env: Env, repo: Path) -> None:
    env.provider.push(GOOD_PLAN)
    r = self_new(env, repo, "--no-ask")
    assert r.exit_code == cli.EXIT_OK, r.output
    ids = env.store.list_ids()
    assert len(ids) == 1 and f"project: {ids[0]}" in r.output


def test_default_repo_is_the_current_directory(
    env: Env, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(repo)
    env.provider.push(GOOD_PLAN)
    r = env.run("self", "new", "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "first open item" in r.output


def test_ref_selects_an_earlier_commit(env: Env, repo: Path) -> None:
    first = git(repo, "rev-parse", "HEAD")
    (repo / "ROADMAP.md").write_text(
        ROADMAP.replace("- [ ] first open item", "- [x] first open item"), encoding="utf-8"
    )
    commit_all(repo, "second")
    env.provider.push(GOOD_PLAN, GOOD_PLAN)
    r_head = self_new(env, repo, "--id", "now", "--no-ask")
    assert r_head.exit_code == cli.EXIT_OK, r_head.output
    assert "  second open item" in r_head.output
    r_old = self_new(env, repo, "--id", "old", "--no-ask", "--ref", first[:10])
    assert r_old.exit_code == cli.EXIT_OK, r_old.output
    assert "  first open item" in r_old.output
    assert f"commit {first[:12]}" in r_old.output


def test_roadmap_path_inside_the_snapshot(env: Env, repo: Path) -> None:
    (repo / "docs").mkdir()
    (repo / "docs" / "PLAN.md").write_text("## M\n- [~] from the docs plan\n", encoding="utf-8")
    commit_all(repo, "docs")
    env.provider.push(GOOD_PLAN)
    r = self_new(env, repo, "--id", "self-1", "--no-ask", "--roadmap", "docs/PLAN.md")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "item: line 2 [~] M" in r.output
    assert "from the docs plan" in env.store.load("self-1").goal


def test_repository_is_never_changed(env: Env, repo: Path) -> None:
    (repo / ".env").write_text("AUTOMA_API_KEY=sk-not-committed\n", encoding="utf-8")
    (repo / "notes.txt").write_text("untracked\n", encoding="utf-8")
    before = tree(repo)
    head = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain")
    env.provider.push(GOOD_PLAN)
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert tree(repo) == before
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "status", "--porcelain") == status
    files = tree(env.store.files_dir("self-1"))
    assert ".env" not in files and "notes.txt" not in files
    assert not any(name.startswith(".git/") for name in files)
    assert "sk-not-committed" not in r.output
    prompt = env.provider.requests[0].messages[-1].content or ""
    assert "sk-not-committed" not in prompt


def test_control_characters_in_the_item_are_escaped(env: Env, repo: Path) -> None:
    (repo / "ROADMAP.md").write_text("## M\n- [ ] evil ‮ text\n", encoding="utf-8")
    commit_all(repo, "bidi")
    env.provider.push(GOOD_PLAN)
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "‮" not in r.output
    assert "evil \\u202e text" in r.output


# -- nothing open / broken roadmap: nothing is created -------------------------------


def test_nothing_open_exits_9(env: Env, repo: Path) -> None:
    (repo / "ROADMAP.md").write_text("## M\n- [x] done\n- [!] owner\n", encoding="utf-8")
    commit_all(repo, "done")
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_SELF_NOTHING_OPEN
    assert "no open item left" in r.output
    assert nothing_created(env)
    assert env.provider.requests == []


def test_missing_roadmap_exits_10(env: Env, repo: Path) -> None:
    r = self_new(env, repo, "--id", "self-1", "--no-ask", "--roadmap", "NOPE.md")
    assert r.exit_code == cli.EXIT_ROADMAP_INVALID
    assert "roadmap:" in r.output and "nothing was created" in r.output
    assert nothing_created(env)
    assert env.provider.requests == []


def test_ambiguous_roadmap_exits_10(env: Env, repo: Path) -> None:
    (repo / "ROADMAP.md").write_text("## M\n- [?] what is this\n", encoding="utf-8")
    commit_all(repo, "bad")
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_ROADMAP_INVALID
    assert nothing_created(env)


def test_uncommitted_roadmap_change_is_not_used(env: Env, repo: Path) -> None:
    (repo / "ROADMAP.md").write_text("## M\n- [?] broken but not committed\n", encoding="utf-8")
    env.provider.push(GOOD_PLAN)
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "  first open item" in r.output


@pytest.mark.parametrize("bad", ["../ROADMAP.md", "/etc/passwd", ".git/HEAD", ""])
def test_unsafe_roadmap_path_exits_10(env: Env, repo: Path, bad: str) -> None:
    r = self_new(env, repo, "--id", "self-1", "--no-ask", "--roadmap", bad)
    assert r.exit_code == cli.EXIT_ROADMAP_INVALID
    assert nothing_created(env)


# -- snapshot / id errors: user errors, nothing is created ---------------------------


def test_not_a_repository(env: Env, tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    r = self_new(env, plain, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_USER
    assert "snapshot:" in r.output and "nothing was created" in r.output
    assert nothing_created(env)
    assert env.provider.requests == []


def test_missing_repository(env: Env, tmp_path: Path) -> None:
    r = self_new(env, tmp_path / "nowhere", "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_USER
    assert nothing_created(env)


@pytest.mark.parametrize("ref", ["--upload-pack=x", "no-such-branch", "a b", "-x"])
def test_bad_ref(env: Env, repo: Path, ref: str) -> None:
    r = self_new(env, repo, "--id", "self-1", "--no-ask", f"--ref={ref}")
    assert r.exit_code == cli.EXIT_USER
    assert nothing_created(env)


def test_symlink_in_repository_is_refused(env: Env, repo: Path) -> None:
    (repo / "link").symlink_to("CLAUDE.md")
    commit_all(repo, "link")
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_USER
    assert "symlinks are not supported" in r.output
    assert nothing_created(env)


def test_existing_id_is_refused(env: Env, repo: Path) -> None:
    env.store.create("old goal", project_id="self-1")
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_USER
    assert "already exists" in r.output
    assert env.store.load("self-1").goal == "old goal"
    assert env.provider.requests == []


@pytest.mark.parametrize("bad", ["../x", "Upper", "a/b"])
def test_unsafe_id_is_refused(env: Env, repo: Path, bad: str) -> None:
    r = self_new(env, repo, "--id", bad, "--no-ask")
    assert r.exit_code == cli.EXIT_USER
    assert "invalid project id" in r.output
    assert nothing_created(env)


# -- config / provider / budget ------------------------------------------------------


def test_missing_key_fails_before_the_snapshot(
    env: Env, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "make_provider", provider_from_config)
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_CONFIG
    assert "AUTOMA_API_KEY" in r.output
    assert nothing_created(env)


def test_missing_tier_fails_before_the_snapshot(env: Env, repo: Path) -> None:
    env.config.write_text(f'workspace = "{env.ws.as_posix()}"\n', encoding="utf-8")
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_CONFIG
    assert env.factory_calls == []
    assert nothing_created(env)


def test_per_call_cap_blocks_before_the_provider(env: Env, repo: Path) -> None:
    env.write_config(project=1.0, daily=1.0, per_call=0.001)
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_BUDGET
    assert "per_call cap" in r.output
    assert env.provider.requests == []
    assert env.store.load("self-1").status is ProjectStatus.PAUSED


def test_planning_failed(env: Env, repo: Path) -> None:
    env.provider.push("not json", "still not", "{}")
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_PLANNING
    assert env.store.load("self-1").status is ProjectStatus.FAILED
    assert not env.store.has_plan("self-1")


def test_provider_error(env: Env, repo: Path) -> None:
    env.provider.push(ProviderError("HTTP 502", status=502, retryable=True))
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_PROVIDER
    assert len(env.provider.requests) == 1


def test_cost_is_recorded_for_the_project(env: Env, repo: Path) -> None:
    env.provider.push(GOOD_PLAN)
    r = self_new(env, repo, "--id", "self-1", "--no-ask")
    assert r.exit_code == cli.EXIT_OK, r.output
    with CostLedger.for_workspace(env.ws) as ledger:
        assert ledger.spent_nusd(project_id="self-1") > 0


# -- structure -----------------------------------------------------------------------


def _calls(fn: object) -> set[str]:
    tree_ = ast.parse(inspect.getsource(fn))  # type: ignore[arg-type]
    return {
        n.func.id
        for n in ast.walk(tree_)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }


def test_new_and_self_new_share_the_guarded_planning_path() -> None:
    for fn in (cli.new, cli.self_new):
        calls = _calls(fn)
        assert {"_prepare_executive", "_plan_with_executive"} <= calls
        assert "LLMGateway" not in calls and "make_provider" not in calls
    assert "create_selfdev_project" in _calls(cli.self_new)
    planning = _calls(cli._plan_with_executive)
    assert {"LLMGateway", "BudgetGuard"} <= planning


def test_help_lists_self_new() -> None:
    from typer.testing import CliRunner

    r = CliRunner().invoke(cli.app, ["self", "--help"])
    assert r.exit_code == 0
    assert "new" in r.output and "next" in r.output
