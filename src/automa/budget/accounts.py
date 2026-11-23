"""Spending caps of a project that belongs to a web UI account (M15, D183).

``project_budget`` is pure: from the configuration, the project's ``owner`` (D182)
and that account's role it returns the limits and the per-user monthly cap the
``BudgetGuard`` of that project must enforce. Every rule only ever *adds* a stop:

* no owner (a CLI project, or one made before accounts): the configuration as is;
* an ``owner`` account: its name is recorded on every ledger row, and the monthly
  cap applies only when ``[users.<name>] monthly_usd`` is set;
* a ``member`` account: the project cap is ``min(project_usd, member_project_usd)``
  and the monthly cap is ``[users.<name>] monthly_usd`` or ``member_monthly_usd``.

``owner_role`` reads the account from ``users.json`` and fails closed: an account
that is missing, or a users file that cannot be read, counts as a ``member``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from automa.config import AutomaConfig, BudgetLimits
from automa.users import UserRole, UsersError, UserStore


@dataclass(frozen=True)
class ProjectBudget:
    """What the guard of one project enforces on top of the plain configuration."""

    limits: BudgetLimits
    user: str | None
    user_monthly_usd: float | None
    member: bool


def owner_role(workspace: Path, owner: str | None) -> UserRole | None:
    """The role of the account ``owner`` (``None`` when the project has no owner)."""
    if owner is None:
        return None
    try:
        account = UserStore(workspace).get(owner)
    except UsersError:
        return UserRole.MEMBER
    if account is None or account.role is not UserRole.OWNER:
        return UserRole.MEMBER
    return UserRole.OWNER


def project_budget(config: AutomaConfig, owner: str | None, role: UserRole | None) -> ProjectBudget:
    """The caps for a project of ``owner`` with ``role`` (see the module docstring)."""
    limits = config.limits
    if owner is None:
        return ProjectBudget(limits=limits, user=None, user_monthly_usd=None, member=False)
    configured = config.users.get(owner)
    if role is UserRole.OWNER:
        monthly = None if configured is None else configured.monthly_usd
        return ProjectBudget(limits=limits, user=owner, user_monthly_usd=monthly, member=False)
    project_usd = min(limits.project_usd, limits.member_project_usd)
    stricter = BudgetLimits.model_validate({**limits.model_dump(), "project_usd": project_usd})
    monthly = limits.member_monthly_usd if configured is None else configured.monthly_usd
    return ProjectBudget(limits=stricter, user=owner, user_monthly_usd=monthly, member=True)


def config_for_project(
    config: AutomaConfig, owner: str | None
) -> tuple[AutomaConfig, ProjectBudget]:
    """``config`` with the project's limits, and the caps for its ``BudgetGuard``.

    Everything that shows or compares against the project cap (estimates, the
    "near the cap" question, the brake) then sees the same, stricter numbers.
    """
    budget = project_budget(config, owner, owner_role(config.workspace, owner))
    if budget.limits is config.limits:
        return config, budget
    return config.model_copy(update={"limits": budget.limits}), budget
