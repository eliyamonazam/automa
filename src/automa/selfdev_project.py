"""Self-development project creation: one roadmap item on a snapshot of the repo (D82).

``create_selfdev_project`` makes an Automa project whose goal is ``build_goal`` for the
next roadmap item and whose ``files/`` workspace is a *snapshot* of one commit of the
repository — never the repository's own working tree:

* ``snapshot_repo`` copies the tracked files of one commit (``git ls-tree`` +
  ``git cat-file --batch``: raw blobs, so no smudge/textconv filter or hook of the
  repository runs). git runs through ``automa.tools.command.run_git_read`` (read-only
  subcommands, minimal environment), the only place that starts processes.
  Untracked and ignored files (``.env``, workspaces) and ``.git`` itself are never
  copied, so no secret outside the commit and no git remote or
  credential reaches the project. Symlinks, submodules, unsafe paths and oversized
  trees are refused instead of guessed.
* The item is read from the ROADMAP of that same snapshot, so goal and files always
  describe the same commit, and the project is created only after the snapshot and
  the goal are valid (no orphan project on failure).
* A ``selfdev.item`` event records the roadmap line, item text (escaped), commit and
  snapshot size.

No model call, no network, and nothing is written to the repository. Tools of the
project stay under Policy, Approval Gate and BudgetGuard exactly as for any project.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from automa.events import Event
from automa.project.store import Project, ProjectStore
from automa.selfdev import RoadmapError, RoadmapItem, build_goal, next_item, read_roadmap, visible
from automa.tools.command import GitReadError, run_git_read

MAX_SNAPSHOT_FILES = 5_000
MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
MAX_PATH_CHARS = 400
GIT_TIMEOUT_S = 60.0
DEFAULT_ROADMAP_PATH = "ROADMAP.md"
STAGING_PREFIX = ".selfdev-"

_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@{}^~-]{0,199}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
_REGULAR_MODES = {"100644": 0o644, "100755": 0o755}


class SnapshotError(ValueError):
    """The repository snapshot could not be made safely."""


class RepoSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    commit: str = Field(pattern=_SHA_RE.pattern)
    files: int = Field(ge=0)
    bytes: int = Field(ge=0)


class SelfdevProject(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    project: Project
    item: RoadmapItem
    snapshot: RepoSnapshot
    goal: str


def _git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    try:
        return run_git_read(repo, args, stdin=stdin, timeout_s=GIT_TIMEOUT_S)
    except GitReadError as e:
        raise SnapshotError(str(e)) from None


def _safe_path(raw: bytes) -> PurePosixPath:
    try:
        name = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise SnapshotError("tracked path is not valid UTF-8") from None
    if not name or len(name) > MAX_PATH_CHARS or visible(name) != name or "\\" in name:
        raise SnapshotError(f"unsafe tracked path: {visible(name[:100])}")
    path = PurePosixPath(name)
    parts = path.parts
    if str(path) != name or path.is_absolute() or any(p in ("", ".", "..") for p in parts):
        raise SnapshotError(f"unsafe tracked path: {visible(name[:100])}")
    if any(p.lower() == ".git" for p in parts):
        raise SnapshotError(f"tracked path inside .git: {visible(name[:100])}")
    return path


def _resolve_commit(repo: Path, ref: str) -> str:
    if not _REF_RE.fullmatch(ref) or ".." in ref:
        raise SnapshotError("invalid git ref")
    out = _git(repo, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
    commit = out.decode("ascii", "replace").strip()
    if not _SHA_RE.fullmatch(commit):
        raise SnapshotError("git rev-parse returned an unexpected value")
    return commit


def _list_tree(repo: Path, commit: str) -> list[tuple[PurePosixPath, int, str, int]]:
    """(path, file mode, blob sha, size) of every tracked file of ``commit``."""
    out = _git(repo, "ls-tree", "-r", "-z", "--long", "--full-tree", commit)
    entries: list[tuple[PurePosixPath, int, str, int]] = []
    seen: set[str] = set()
    total = 0
    for record in out.split(b"\0"):
        if not record:
            continue
        meta, sep, raw_path = record.partition(b"\t")
        fields = meta.decode("ascii", "replace").split()
        if not sep or len(fields) != 4:
            raise SnapshotError("unexpected git ls-tree output")
        mode, kind, sha, size_text = fields
        path = _safe_path(raw_path)
        if mode == "120000":
            raise SnapshotError(f"symlinks are not supported: {visible(str(path))}")
        if kind == "commit" or mode == "160000":
            raise SnapshotError(f"submodules are not supported: {visible(str(path))}")
        if kind != "blob" or mode not in _REGULAR_MODES or not _SHA_RE.fullmatch(sha):
            raise SnapshotError(f"unsupported tree entry: {visible(str(path))}")
        key = str(path).casefold()
        if key in seen:
            raise SnapshotError(f"paths differ only in case: {visible(str(path))}")
        seen.add(key)
        if not size_text.isdigit():
            raise SnapshotError("unexpected git ls-tree size")
        size = int(size_text)
        total += size
        entries.append((path, _REGULAR_MODES[mode], sha, size))
        if len(entries) > MAX_SNAPSHOT_FILES:
            raise SnapshotError(f"more than {MAX_SNAPSHOT_FILES} tracked files")
        if total > MAX_SNAPSHOT_BYTES:
            raise SnapshotError(f"tracked files are larger than {MAX_SNAPSHOT_BYTES} bytes")
    dirs = {str(parent).casefold() for path, *_ in entries for parent in path.parents}
    if dirs & seen:
        raise SnapshotError("a tracked file has the same path as a directory")
    return entries


def _read_blobs(repo: Path, entries: list[tuple[PurePosixPath, int, str, int]]) -> list[bytes]:
    """Raw blob contents (no filters), in the order of ``entries``."""
    if not entries:
        return []
    out = _git(repo, "cat-file", "--batch", stdin=b"".join(f"{e[2]}\n".encode() for e in entries))
    blobs: list[bytes] = []
    pos = 0
    for _, _, sha, size in entries:
        end = out.find(b"\n", pos)
        if end < 0:
            raise SnapshotError("truncated git cat-file output")
        header = out[pos:end].decode("ascii", "replace").split()
        if header != [sha, "blob", str(size)]:
            raise SnapshotError("unexpected git cat-file output")
        start = end + 1
        data = out[start : start + size]
        if len(data) != size or out[start + size : start + size + 1] != b"\n":
            raise SnapshotError("truncated git cat-file output")
        blobs.append(data)
        pos = start + size + 1
    if pos != len(out):
        raise SnapshotError("unexpected git cat-file output")
    return blobs


def _require_empty_dir(dest: Path) -> None:
    try:
        info = os.lstat(dest)
    except FileNotFoundError:
        raise SnapshotError("snapshot destination does not exist") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SnapshotError("snapshot destination must be a real directory")
    if any(dest.iterdir()):
        raise SnapshotError("snapshot destination is not empty")


def _write_file(root: Path, path: PurePosixPath, data: bytes, mode: int) -> None:
    target = root.joinpath(*path.parts)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Every directory was created here; a symlink can only appear by a race.
    if parent.is_symlink() or not parent.resolve().is_relative_to(root.resolve()):
        raise SnapshotError("snapshot destination changed while writing")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(target, flags, mode)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.chmod(target, mode)


def snapshot_repo(repo: Path, dest: Path, *, ref: str = "HEAD") -> RepoSnapshot:
    """Copy the tracked files of commit ``ref`` of ``repo`` into the empty dir ``dest``.

    Only blobs of the commit are written (raw, unfiltered); ``.git``, untracked and
    ignored files never are. Raises ``SnapshotError`` (and leaves ``dest`` partly
    filled; callers use a staging directory) on anything unsafe or ambiguous.
    """
    _require_empty_dir(dest)
    repo_info = repo.resolve()
    if not repo_info.is_dir():
        raise SnapshotError("repository directory does not exist")
    commit = _resolve_commit(repo_info, ref)
    entries = _list_tree(repo_info, commit)
    blobs = _read_blobs(repo_info, entries)
    for (path, mode, _, _), data in zip(entries, blobs, strict=True):
        try:
            _write_file(dest, path, data, mode)
        except OSError as exc:
            raise SnapshotError(f"cannot write snapshot: {type(exc).__name__}") from None
    return RepoSnapshot(commit=commit, files=len(entries), bytes=sum(e[3] for e in entries))


def _roadmap_in(snapshot_dir: Path, roadmap_path: str) -> Path:
    path = _safe_path(roadmap_path.encode("utf-8"))
    return snapshot_dir.joinpath(*path.parts)


def create_selfdev_project(
    store: ProjectStore,
    repo: Path,
    *,
    roadmap_path: str = DEFAULT_ROADMAP_PATH,
    ref: str = "HEAD",
    project_id: str | None = None,
    repo_name: str = "automa",
    now: datetime | None = None,
) -> SelfdevProject | None:
    """Create a project for the next roadmap item on a snapshot of ``repo`` at ``ref``.

    Returns ``None`` (and creates nothing) when the snapshot's roadmap has no open
    item. Raises ``SnapshotError``, ``RoadmapError`` or ``ProjectStoreError`` without
    leaving a project or staging directory behind.
    """
    store.root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=store.root))
    try:
        snapshot = snapshot_repo(repo, staging, ref=ref)
        try:
            roadmap = read_roadmap(_roadmap_in(staging, roadmap_path))
        except SnapshotError as e:
            raise RoadmapError(f"invalid roadmap path: {e}") from None
        item = next_item(roadmap)
        if item is None:
            return None
        goal = build_goal(item, repo_name=repo_name)
        project = store.create(goal, project_id=project_id)
        files = store.files_dir(project.id)
        try:
            files.rmdir()  # empty, made by create(); same filesystem as staging
            os.replace(staging, files)
        except OSError as exc:
            files.mkdir(exist_ok=True)  # keep the project's layout valid
            raise SnapshotError(
                f"cannot move the snapshot into project {project.id}: {type(exc).__name__}"
            ) from None
        store.events(project.id).emit(
            Event(
                ts=now or datetime.now(UTC),
                type="selfdev.item",
                project_id=project.id,
                data={
                    "roadmap": visible(roadmap_path),
                    "line": item.line,
                    "milestone": visible(item.milestone),
                    "status": item.status.name.lower(),
                    "text": visible(item.text),
                    "commit": snapshot.commit,
                    "files": snapshot.files,
                    "bytes": snapshot.bytes,
                },
            )
        )
        return SelfdevProject(project=project, item=item, snapshot=snapshot, goal=goal)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
