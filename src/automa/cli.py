"""Command-line interface: ``automa new "<goal>"``, ``automa status [project-id]``.

Every model call goes through ``LLMGateway`` + ``BudgetGuard`` (never a bare provider).
``status`` is read-only: it needs no API key, makes no model call and creates no files.

Exit codes (stable, for scripts):

* 0 — success
* 1 — usage/user error (unknown project, aborted prompt, invalid id, …)
* 2 — configuration error (missing key, tier without a model, invalid ``automa.toml``)
* 3 — budget stop: the project is ``paused`` until the user decides
* 4 — planning failed: the model gave no valid plan within the retry limit
* 5 — provider error (network, HTTP status from the gateway)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from automa import __version__
from automa.agents import (
    Clarification,
    ExecutiveAgent,
    ExecutiveSettings,
    PlanningFailed,
    plan_project,
)
from automa.budget.guard import BudgetError, BudgetExceeded, BudgetGuard
from automa.budget.ledger import CostLedger, nusd_to_usd
from automa.config import AutomaConfig, ConfigError, load_config
from automa.llm.factory import provider_from_config
from automa.llm.gateway import LLMGateway
from automa.llm.provider import Provider, ProviderError
from automa.plan import Plan
from automa.project import Project, ProjectStatus, ProjectStore, ProjectStoreError

EXIT_OK = 0
EXIT_USER = 1
EXIT_CONFIG = 2
EXIT_BUDGET = 3
EXIT_PLANNING = 4
EXIT_PROVIDER = 5

LEDGER_FILENAME = "ledger.sqlite3"
GOAL_PREVIEW_CHARS = 60

app = typer.Typer(help="Automa command-line interface.", no_args_is_help=True)


# Indirection point: tests replace this with a FakeProvider factory. The provider is
# always wrapped in a guarded LLMGateway below; nothing here calls it directly.
make_provider: Callable[[AutomaConfig], Provider] = provider_from_config


def _fail(message: str, code: int) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


def _load(config_path: Path | None) -> AutomaConfig:
    try:
        return load_config(config_path)
    except ConfigError as e:
        raise _fail(str(e), EXIT_CONFIG) from None


def _usd(nusd: int) -> str:
    return f"${nusd_to_usd(nusd):.4f}"


def _project_spent_nusd(workspace: Path, project_id: str) -> int:
    """Spending recorded for ``project_id``; 0 if no ledger exists yet (never creates one)."""
    path = workspace / LEDGER_FILENAME
    if not path.is_file():
        return 0
    with CostLedger(path) as ledger:
        return ledger.spent_nusd(project_id=project_id)


def _all_spent_nusd(workspace: Path) -> dict[str, int]:
    path = workspace / LEDGER_FILENAME
    if not path.is_file():
        return {}
    with CostLedger(path) as ledger:
        return {
            pid: ledger.spent_nusd(project_id=pid)
            for pid in {e.project_id for e in ledger.entries() if e.project_id}
        }


def _preview(text: str, limit: int = GOAL_PREVIEW_CHARS) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def _plan_lines(plan: Plan) -> list[str]:
    lines = [f"plan v{plan.version}: {len(plan.tasks)} task(s)"]
    for i, task in enumerate(plan.topological_order(), start=1):
        deps = f" (after: {', '.join(task.depends_on)})" if task.depends_on else ""
        lines.append(f"  {i}. [{task.role}] {task.id}: {task.title}{deps}")
    return lines


@app.callback()
def _root() -> None:
    """Automa command-line interface."""


@app.command()
def version() -> None:
    """Print the installed Automa version."""
    typer.echo(__version__)


@app.command()
def new(
    goal: Annotated[str, typer.Argument(help="What the project should achieve.")],
    project_id: Annotated[
        str | None, typer.Option("--id", help="Project id (a-z, 0-9, '-'); default: generated.")
    ] = None,
    ask: Annotated[
        bool,
        typer.Option("--ask/--no-ask", help="Ask the Executive's clarifying questions first."),
    ] = True,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """Create a project for GOAL, answer clarifying questions and write its plan."""
    goal = goal.strip()
    if not goal:
        raise _fail("goal must not be empty", EXIT_USER)
    config = _load(config_path)
    settings = ExecutiveSettings()
    try:
        # Validate everything a model call needs *before* creating the project, so a
        # missing key or tier never leaves an orphan project behind.
        config.model_for(settings.tier)
        provider = make_provider(config)
    except ConfigError as e:
        raise _fail(str(e), EXIT_CONFIG) from None

    store = ProjectStore.for_workspace(config.workspace)
    try:
        project = store.create(goal, project_id=project_id)
    except ProjectStoreError as e:
        raise _fail(str(e), EXIT_USER) from None
    pid = project.id
    typer.echo(f"project: {pid}")

    events = store.events(pid)
    with CostLedger.for_workspace(config.workspace) as ledger:
        guard = BudgetGuard(config.limits, ledger, events=events)
        gateway = LLMGateway(config, provider, guard=guard)
        agent = ExecutiveAgent(gateway, settings=settings, events=events)
        try:
            clarifications = _clarify(agent, store, project) if ask else []
            plan = plan_project(agent, store, pid, clarifications=clarifications)
        except BudgetError as e:
            detail = f" ({e.scope.value} cap)" if isinstance(e, BudgetExceeded) else ""
            raise _fail(
                f"budget stop{detail}: {e}. Project {pid} is paused; "
                "raise the cap or wait, then resume.",
                EXIT_BUDGET,
            ) from None
        except PlanningFailed as e:
            raise _fail(f"{e}\n{e.details}", EXIT_PLANNING) from None
        except ProviderError as e:
            raise _fail(f"provider error: {e}", EXIT_PROVIDER) from None
        finally:
            spent = ledger.spent_nusd(project_id=pid)
            typer.echo(f"cost: {_usd(spent)}")
    for line in _plan_lines(plan):
        typer.echo(line)
    typer.echo(f"status: {store.load(pid).status.value}")


def _clarify(agent: ExecutiveAgent, store: ProjectStore, project: Project) -> list[Clarification]:
    """Ask the Executive's questions, read answers from the user, persist them."""
    try:
        questions = agent.clarify(project.goal, project_id=project.id)
    except BudgetError:
        store.set_status(project.id, ProjectStatus.PAUSED)
        raise
    except PlanningFailed:
        store.set_status(project.id, ProjectStatus.FAILED)
        raise
    items: list[Clarification] = []
    if questions:
        typer.echo("The Executive has some questions (press Enter to skip one):")
    for i, question in enumerate(questions, start=1):
        answer = typer.prompt(f"{i}. {question}", default="", show_default=False)
        items.append(Clarification(question=question, answer=answer))
    if items:
        store.save_clarifications(project.id, items)
    return items


@app.command()
def status(
    project_id: Annotated[
        str | None, typer.Argument(help="Project id; omit to list all projects.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """Show one project (goal, status, plan, cost) or list all projects. Read-only."""
    config = _load(config_path)
    store = ProjectStore.for_workspace(config.workspace)
    try:
        if project_id is None:
            _status_all(store, config.workspace, as_json)
        else:
            _status_one(store, config.workspace, project_id, as_json)
    except ProjectStoreError as e:
        raise _fail(str(e), EXIT_USER) from None


def _status_all(store: ProjectStore, workspace: Path, as_json: bool) -> None:
    projects = store.list_projects() if store.root.is_dir() else []
    spent = _all_spent_nusd(workspace)
    if as_json:
        rows = [
            {
                "id": p.id,
                "status": p.status.value,
                "goal": p.goal,
                "created_at": p.created_at.isoformat(),
                "cost_usd": nusd_to_usd(spent.get(p.id, 0)),
            }
            for p in projects
        ]
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not projects:
        typer.echo('no projects yet; create one with: automa new "<goal>"')
        return
    for p in projects:
        typer.echo(f"{p.id}  {p.status.value:<8}  {_usd(spent.get(p.id, 0))}  {_preview(p.goal)}")


def _status_one(store: ProjectStore, workspace: Path, project_id: str, as_json: bool) -> None:
    project = store.load(project_id)
    plan = store.load_plan(project_id) if store.has_plan(project_id) else None
    clarifications = store.load_clarifications(project_id)
    spent = _project_spent_nusd(workspace, project_id)
    if as_json:
        data: dict[str, Any] = {
            "id": project.id,
            "goal": project.goal,
            "status": project.status.value,
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat(),
            "cost_usd": nusd_to_usd(spent),
            "clarifications": [c.model_dump() for c in clarifications],
            "plan": plan.model_dump(mode="json") if plan is not None else None,
        }
        typer.echo(json.dumps(data, ensure_ascii=False, indent=2))
        return
    typer.echo(f"project: {project.id}")
    typer.echo(f"goal: {project.goal}")
    typer.echo(f"status: {project.status.value}")
    typer.echo(f"created: {project.created_at.isoformat()}")
    typer.echo(f"updated: {project.updated_at.isoformat()}")
    typer.echo(f"cost: {_usd(spent)}")
    if clarifications:
        typer.echo(f"clarifications: {len(clarifications)}")
        for c in clarifications:
            typer.echo(f"  Q: {c.question}")
            typer.echo(f"  A: {c.answer or '(no answer)'}")
    if plan is None:
        typer.echo("plan: none")
    else:
        for line in _plan_lines(plan):
            typer.echo(line)


def main() -> None:
    """Console-script entry point."""
    app()
