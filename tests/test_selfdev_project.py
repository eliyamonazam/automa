"""Tests for self-development project creation on a repository snapshot (D82)."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest

from automa import selfdev_project
from automa.project.store import ProjectExists, ProjectStore
from automa.selfdev import RoadmapError, build_goal, next_item, read_roadmap
from automa.selfdev_project import (
    STAGING_PREFIX,
    SnapshotError,
    create_selfdev_project,
    snapshot_repo,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 16, 7, 0, tzinfo=UTC)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

ROADMAP = """# Roadmap

## M0 — base
- [x] skeleton
- [!] needs the owner's key
- [ ] first open item
- [ ] second open item
"""


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.com",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.com",
    )
    return env


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(  # noqa: S603 - fixed argv in tests
        ["git", "-C", str(repo), *args],  # noqa: S607
        capture_output=True,
        env=_env(),
        check=True,
        text=True,
    )
    return done.stdout.strip()


def commit_all(repo: Path, message: str = "c") -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "ROADMAP.md").write_text(ROADMAP, encoding="utf-8")
    (r / "CLAUDE.md").write_text("rules\n", encoding="utf-8")
    (r / "src" / "pkg").mkdir(parents=True)
    (r / "src" / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (r / "run.sh").write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    os.chmod(r / "run.sh", 0o755)  # noqa: S103 - executable bit under test
    (r / ".gitignore").write_text(".env\nworkspace/\n", encoding="utf-8")
    commit_all(r, "first")
    return r


@pytest.fixture
def store(tmp_path: Path) -> ProjectStore:
    return ProjectStore.for_workspace(tmp_path / "ws")


def tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() or p.is_symlink()
    }


# -- snapshot_repo -----------------------------------------------------------------


def test_snapshot_copies_tracked_files_of_the_commit(repo: Path, tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    snap = snapshot_repo(repo, dest)
    assert snap.commit == git(repo, "rev-parse", "HEAD")
    files = tree(dest)
    assert set(files) == {".gitignore", "CLAUDE.md", "ROADMAP.md", "run.sh", "src/pkg/mod.py"}
    assert files["src/pkg/mod.py"] == b"x = 1\n"
    assert snap.files == 5
    assert snap.bytes == sum(len(v) for v in files.values())
    assert os.stat(dest / "run.sh").st_mode & 0o777 == 0o755
    assert os.stat(dest / "CLAUDE.md").st_mode & 0o777 == 0o644


def test_snapshot_never_copies_git_untracked_ignored_or_uncommitted(
    repo: Path, tmp_path: Path
) -> None:
    (repo / ".env").write_text("ANTHROPIC_API_KEY=sk-secret\n", encoding="utf-8")
    (repo / "workspace").mkdir()
    (repo / "workspace" / "ledger.sqlite3").write_bytes(b"data")
    (repo / "notes.txt").write_text("untracked\n", encoding="utf-8")
    (repo / "CLAUDE.md").write_text("changed but not committed\n", encoding="utf-8")
    dest = tmp_path / "dest"
    dest.mkdir()
    snapshot_repo(repo, dest)
    files = tree(dest)
    assert ".env" not in files and "notes.txt" not in files
    assert not any(name.startswith(("workspace", ".git/")) for name in files)
    assert not (dest / ".git").exists()
    assert files["CLAUDE.md"] == b"rules\n"
    assert all(b"sk-secret" not in v for v in files.values())


def test_snapshot_of_an_earlier_ref(repo: Path, tmp_path: Path) -> None:
    first = git(repo, "rev-parse", "HEAD")
    (repo / "later.txt").write_text("later\n", encoding="utf-8")
    commit_all(repo, "second")
    dest = tmp_path / "dest"
    dest.mkdir()
    snap = snapshot_repo(repo, dest, ref="HEAD~1")
    assert snap.commit == first
    assert "later.txt" not in tree(dest)


@pytest.mark.parametrize("ref", ["", "-x", "--output=/tmp/x", "a..b", "a b", "x" * 300, "a;b"])
def test_snapshot_refuses_invalid_ref(repo: Path, tmp_path: Path, ref: str) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(SnapshotError, match="invalid git ref"):
        snapshot_repo(repo, dest, ref=ref)


def test_snapshot_unknown_ref_and_not_a_repo(repo: Path, tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(SnapshotError, match="rev-parse failed"):
        snapshot_repo(repo, dest, ref="no-such-branch")
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(SnapshotError, match="rev-parse failed"):
        snapshot_repo(plain, dest)
    with pytest.raises(SnapshotError, match="does not exist"):
        snapshot_repo(tmp_path / "missing", dest)
    assert list(dest.iterdir()) == []


def test_snapshot_empty_repository_has_no_commit(tmp_path: Path) -> None:
    r = tmp_path / "empty"
    r.mkdir()
    git(r, "init", "-q")
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(SnapshotError):
        snapshot_repo(r, dest)


def test_snapshot_destination_must_be_an_empty_real_directory(repo: Path, tmp_path: Path) -> None:
    with pytest.raises(SnapshotError, match="does not exist"):
        snapshot_repo(repo, tmp_path / "nope")
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("x", encoding="utf-8")
    with pytest.raises(SnapshotError, match="not empty"):
        snapshot_repo(repo, full)
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(SnapshotError, match="real directory"):
        snapshot_repo(repo, link)
    afile = tmp_path / "afile"
    afile.write_text("x", encoding="utf-8")
    with pytest.raises(SnapshotError, match="real directory"):
        snapshot_repo(repo, afile)
    assert list(real.iterdir()) == []


def test_snapshot_refuses_symlinks(repo: Path, tmp_path: Path) -> None:
    (repo / "link").symlink_to("/etc/passwd")
    commit_all(repo)
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(SnapshotError, match="symlinks"):
        snapshot_repo(repo, dest)


def test_snapshot_refuses_submodules(repo: Path, tmp_path: Path) -> None:
    sha = git(repo, "rev-parse", "HEAD")
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{sha},vendor/sub")
    git(repo, "commit", "-q", "-m", "sub")
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(SnapshotError, match="submodules"):
        snapshot_repo(repo, dest)


def test_snapshot_does_not_run_repository_filters(repo: Path, tmp_path: Path) -> None:
    marker = tmp_path / "filter-ran"
    (repo / ".gitattributes").write_text("*.txt filter=evil\n", encoding="utf-8")
    (repo / "a.txt").write_text("raw\n", encoding="utf-8")
    commit_all(repo)
    git(repo, "config", "filter.evil.smudge", f"touch {marker}; cat")
    git(repo, "config", "filter.evil.clean", "cat")
    git(repo, "config", "core.fsmonitor", f"touch {marker}; false")
    dest = tmp_path / "dest"
    dest.mkdir()
    snapshot_repo(repo, dest)
    assert (dest / "a.txt").read_bytes() == b"raw\n"
    assert not marker.exists()


def test_snapshot_ignores_outer_git_environment(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q")
    (other / "evil.txt").write_text("other\n", encoding="utf-8")
    commit_all(other)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'core.fsmonitor=touch /tmp/x'")
    dest = tmp_path / "dest"
    dest.mkdir()
    snap = snapshot_repo(repo, dest)
    assert "evil.txt" not in tree(dest)
    assert snap.commit == git(repo, "rev-parse", "HEAD")


def test_snapshot_limits(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    monkeypatch.setattr(selfdev_project, "MAX_SNAPSHOT_FILES", 2)
    with pytest.raises(SnapshotError, match="more than 2 tracked files"):
        snapshot_repo(repo, dest)
    monkeypatch.setattr(selfdev_project, "MAX_SNAPSHOT_FILES", 100)
    monkeypatch.setattr(selfdev_project, "MAX_SNAPSHOT_BYTES", 10)
    with pytest.raises(SnapshotError, match="larger than 10 bytes"):
        snapshot_repo(repo, dest)
    assert list(dest.iterdir()) == []


def test_snapshot_without_git(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(SnapshotError, match="git is not installed"):
        snapshot_repo(repo, dest)


def test_snapshot_leaves_the_repository_unchanged(repo: Path, tmp_path: Path) -> None:
    before = git(repo, "status", "--porcelain", "--ignored")
    head = git(repo, "rev-parse", "HEAD")
    dest = tmp_path / "dest"
    dest.mkdir()
    snapshot_repo(repo, dest)
    assert git(repo, "status", "--porcelain", "--ignored") == before
    assert git(repo, "rev-parse", "HEAD") == head


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"/abs",
        b"a/../b",
        b"./a",
        b".git/config",
        b"x/.GIT/hooks",
        b"a\\b",
        b"\xff",
        b"a\x1bb",
        "a‮b".encode(),
        b"a//b",
        b"x" * 401,
    ],
)
def test_safe_path_refuses(raw: bytes) -> None:
    with pytest.raises(SnapshotError):
        selfdev_project._safe_path(raw)


def test_safe_path_accepts_normal_and_persian_names() -> None:
    assert selfdev_project._safe_path(b"docs/ARCHITECTURE.md") == PurePosixPath(
        "docs/ARCHITECTURE.md"
    )
    name = "اسناد/نیم‌فاصله.md"
    assert str(selfdev_project._safe_path(name.encode())) == name
    assert (
        str(selfdev_project._safe_path(b".github/workflows/ci.yml")) == ".github/workflows/ci.yml"
    )


# -- create_selfdev_project ---------------------------------------------------------


def _staging_left(store: ProjectStore) -> list[Path]:
    if not store.root.exists():
        return []
    return [p for p in store.root.iterdir() if p.name.startswith(STAGING_PREFIX)]


def test_create_project_for_the_next_item(repo: Path, store: ProjectStore) -> None:
    created = create_selfdev_project(store, repo, project_id="self-1", now=NOW)
    assert created is not None
    assert created.project.id == "self-1"
    assert created.item.line == 6 and created.item.text == "first open item"
    roadmap = read_roadmap(repo / "ROADMAP.md")
    item = next_item(roadmap)
    assert item is not None
    assert created.goal == build_goal(item) == store.load("self-1").goal
    files = tree(store.files_dir("self-1"))
    assert files["ROADMAP.md"] == ROADMAP.encode()
    assert "src/pkg/mod.py" in files
    assert _staging_left(store) == []
    assert store.list_ids() == ["self-1"]


def test_create_project_records_selfdev_item_event(repo: Path, store: ProjectStore) -> None:
    created = create_selfdev_project(store, repo, project_id="self-1", now=NOW)
    assert created is not None
    lines = (store.project_dir("self-1") / "events.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in lines.splitlines()]
    assert [e["type"] for e in events] == ["project.created", "selfdev.item"]
    data = events[1]["data"]
    assert data == {
        "roadmap": "ROADMAP.md",
        "line": 6,
        "milestone": "M0 — base",
        "status": "todo",
        "text": "first open item",
        "commit": created.snapshot.commit,
        "files": created.snapshot.files,
        "bytes": created.snapshot.bytes,
    }
    assert events[1]["ts"].startswith("2026-11-16T07:00:00")


def test_item_comes_from_the_committed_roadmap(repo: Path, store: ProjectStore) -> None:
    (repo / "ROADMAP.md").write_text(
        ROADMAP.replace("- [ ] first open item", "- [x] first open item"), encoding="utf-8"
    )
    created = create_selfdev_project(store, repo, project_id="self-1")
    assert created is not None
    assert created.item.text == "first open item"
    commit_all(repo)
    again = create_selfdev_project(store, repo, project_id="self-2")
    assert again is not None and again.item.text == "second open item"


def test_partial_item_goal(repo: Path, store: ProjectStore) -> None:
    (repo / "ROADMAP.md").write_text(
        ROADMAP.replace("- [ ] first open item", "- [~] first open item"), encoding="utf-8"
    )
    commit_all(repo)
    created = create_selfdev_project(store, repo)
    assert created is not None
    assert "Status: partly done; continue it" in created.goal
    assert created.project.id.startswith("p-")


def test_no_open_item_creates_nothing(repo: Path, store: ProjectStore) -> None:
    (repo / "ROADMAP.md").write_text("## M0\n- [x] done\n- [!] waiting\n", encoding="utf-8")
    commit_all(repo)
    assert create_selfdev_project(store, repo) is None
    assert store.list_ids() == []
    assert _staging_left(store) == []


@pytest.mark.parametrize(
    "content",
    ["## M0\n- [?] unknown\n", "## M0\n  - [ ] nested\n", "```\n- [ ] open fence\n"],
)
def test_broken_roadmap_creates_nothing(repo: Path, store: ProjectStore, content: str) -> None:
    (repo / "ROADMAP.md").write_text(content, encoding="utf-8")
    commit_all(repo)
    with pytest.raises(RoadmapError):
        create_selfdev_project(store, repo)
    assert store.list_ids() == []
    assert _staging_left(store) == []


def test_missing_or_unsafe_roadmap_path(repo: Path, store: ProjectStore) -> None:
    with pytest.raises(RoadmapError, match="not found"):
        create_selfdev_project(store, repo, roadmap_path="docs/ROADMAP.md")
    for bad in ("../ROADMAP.md", "/etc/passwd", ".git/HEAD", ""):
        with pytest.raises(RoadmapError, match="invalid roadmap path"):
            create_selfdev_project(store, repo, roadmap_path=bad)
    with pytest.raises(RoadmapError, match="regular file"):
        create_selfdev_project(store, repo, roadmap_path="src")
    assert store.list_ids() == []
    assert _staging_left(store) == []


def test_roadmap_in_subdirectory(repo: Path, store: ProjectStore) -> None:
    (repo / "docs").mkdir()
    (repo / "docs" / "PLAN.md").write_text("## A\n- [ ] sub item\n", encoding="utf-8")
    commit_all(repo)
    created = create_selfdev_project(store, repo, roadmap_path="docs/PLAN.md")
    assert created is not None and created.item.text == "sub item"


def test_snapshot_failure_creates_nothing(repo: Path, store: ProjectStore) -> None:
    (repo / "link").symlink_to("CLAUDE.md")
    commit_all(repo)
    with pytest.raises(SnapshotError):
        create_selfdev_project(store, repo)
    assert store.list_ids() == []
    assert _staging_left(store) == []


def test_existing_project_id_is_never_overwritten(repo: Path, store: ProjectStore) -> None:
    store.create("other goal", project_id="taken")
    (store.files_dir("taken") / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ProjectExists):
        create_selfdev_project(store, repo, project_id="taken")
    assert store.load("taken").goal == "other goal"
    assert tree(store.files_dir("taken")) == {"keep.txt": b"keep"}
    assert _staging_left(store) == []


def test_invalid_repo_name(repo: Path, store: ProjectStore) -> None:
    with pytest.raises(RoadmapError, match="repository name"):
        create_selfdev_project(store, repo, repo_name="bad name")
    assert store.list_ids() == []


def test_move_failure_keeps_project_layout(
    repo: Path, store: ProjectStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = os.replace

    def boom(src: str | Path, dst: str | Path) -> None:
        if Path(src).name.startswith(STAGING_PREFIX):
            raise OSError("cross-device")
        real_replace(src, dst)

    monkeypatch.setattr(selfdev_project.os, "replace", boom)
    with pytest.raises(SnapshotError, match="cannot move the snapshot into project p1"):
        create_selfdev_project(store, repo, project_id="p1")
    assert store.files_dir("p1").is_dir()
    assert list(store.files_dir("p1").iterdir()) == []
    assert _staging_left(store) == []


def test_repository_is_unchanged(repo: Path, store: ProjectStore) -> None:
    before = git(repo, "status", "--porcelain", "--ignored")
    head = git(repo, "rev-parse", "HEAD")
    roadmap = (repo / "ROADMAP.md").read_bytes()
    create_selfdev_project(store, repo)
    assert git(repo, "status", "--porcelain", "--ignored") == before
    assert git(repo, "rev-parse", "HEAD") == head
    assert (repo / "ROADMAP.md").read_bytes() == roadmap


def test_real_repository_snapshot(tmp_path: Path) -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    store = ProjectStore.for_workspace(tmp_path / "ws")
    try:
        created = create_selfdev_project(store, REPO_ROOT, project_id="self-real")
    except SnapshotError as e:  # e.g. a shallow/partial checkout without blobs
        pytest.skip(str(e))
    files = store.files_dir("self-real")
    assert (files / "CLAUDE.md").is_file()
    assert (files / "src" / "automa" / "selfdev.py").is_file()
    head = git(REPO_ROOT, "rev-parse", "HEAD")
    if created is not None:
        assert created.snapshot.commit == head
    assert not (files / ".git").exists()
    assert not (files / ".env").exists()


def test_module_has_no_model_budget_network_or_tool_dependency() -> None:
    tree_ = ast.parse(Path(selfdev_project.__file__).read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree_):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    forbidden = ("httpx", "socket", "subprocess", "automa.llm", "automa.budget", "automa.agents")
    for name in names:
        assert not name.startswith(forbidden), name
        # git only through the read-only runner of the single process-spawning module.
        assert not name.startswith("automa.tools") or name == "automa.tools.command", name
