"""Real task costs from the ledger, for the cost range (D175).

``passed_task_roles`` reads which tasks of the workspace passed and with which role
(from each project's current plan and task states); ``task_totals`` joins that with
the ledger: the real total cost of each passed task, grouped by role. Read-only and
advisory: a project that cannot be read is skipped, never an error.
"""

from __future__ import annotations

from pathlib import Path

from automa.budget.ledger import CostLedger, LedgerError
from automa.finance.estimate import task_totals_by_role
from automa.plan import TaskStatus
from automa.project.store import ProjectStore, ProjectStoreError


def passed_task_roles(store: ProjectStore) -> dict[tuple[str, str], str]:
    """(project, task) -> role of every passed task of the current plans."""
    out: dict[tuple[str, str], str] = {}
    try:
        ids = store.list_ids()
    except (ProjectStoreError, OSError):
        return out
    for pid in ids:
        try:
            plan = store.load_plan(pid)
            states = store.task_states(pid)
        except (ProjectStoreError, OSError, ValueError):
            continue
        for task in plan.tasks:
            state = states.get(task.id)
            if state is not None and state.status is TaskStatus.PASSED:
                out[(pid, task.id)] = task.role
    return out


def task_totals(store: ProjectStore, ledger: CostLedger) -> dict[str, list[int]]:
    """Real total cost of every passed task, by role; ``{}`` if anything is unreadable."""
    roles_of = passed_task_roles(store)
    if not roles_of:
        return {}
    try:
        entries = ledger.entries()
    except (LedgerError, OSError, ValueError):
        return {}
    return task_totals_by_role(entries, roles_of)


def workspace_task_totals(store: ProjectStore, path: Path) -> dict[str, list[int]]:
    """:func:`task_totals` with the ledger at ``path`` opened read-only; a missing or
    unreadable ledger gives ``{}``."""
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        with CostLedger(path, read_only=True) as ledger:
            return task_totals(store, ledger)
    except (LedgerError, OSError, ValueError):
        return {}
