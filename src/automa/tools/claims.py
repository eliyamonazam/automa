"""``PathClaims``: which running task may write which path in ``files/`` (D37).

In parallel mode (``max_parallel`` > 1) tasks share one ``files/`` directory. The
first write of a path by a task *claims* it for that task until the task stops
being driven in the current run; while claimed, no other task may write, delete
or create anything that overlaps it (the same path, a parent directory or a path
inside it). So two parallel tasks can never write the same file at the same time
— the losing write fails with a clear, model-readable error instead of silently
replacing the other task's output.

Claims live in memory for one run only and are released when the owning task
finishes, stops or fails, which keeps parallel runs equivalent to some sequential
order of the same writes. They are thread-safe; a claim of several paths is
all-or-nothing.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

from automa.tools.workspace import normalize


def paths_overlap(a: str, b: str) -> bool:
    """Whether two normalized relative paths are equal or one lies inside the other."""
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


class PathClaims:
    """Thread-safe map of claimed path → owner (a task id)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owners: dict[str, str] = {}

    def claim(self, paths: Iterable[str], owner: str) -> tuple[str, str] | None:
        """Claim every path for ``owner``, or none of them.

        Returns ``None`` on success (paths ``owner`` already holds are fine), else
        ``(path, other_owner)`` for the first conflict. Paths are normalized; an
        invalid path raises ``WorkspaceError``.
        """
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner is required")
        wanted = sorted({normalize(p) for p in paths})
        with self._lock:
            for path in wanted:
                for held, other in self._owners.items():
                    if other != owner and paths_overlap(path, held):
                        return path, other
            for path in wanted:
                self._owners.setdefault(path, owner)
        return None

    def release(self, owner: str) -> int:
        """Release everything ``owner`` holds. Returns how many paths were released."""
        with self._lock:
            mine = [p for p, o in self._owners.items() if o == owner]
            for p in mine:
                del self._owners[p]
        return len(mine)

    def owned_by(self, owner: str) -> list[str]:
        with self._lock:
            return sorted(p for p, o in self._owners.items() if o == owner)

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._owners)
