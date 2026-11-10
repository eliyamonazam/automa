"""Command-line interface: ``automa new``, ``run``, ``status``, ``bench list|run``.

Every model call goes through ``LLMGateway`` + ``BudgetGuard`` (never a bare provider).
``status`` is read-only: it needs no API key, makes no model call and creates no files.

Exit codes (stable, for scripts):

* 0 — success
* 1 — usage/user error (unknown project, aborted prompt, invalid id, …)
* 2 — configuration error (missing key, tier without a model, invalid ``automa.toml``)
* 3 — budget stop: the project is ``paused`` until the user decides
* 4 — planning failed: the model gave no valid plan within the retry limit
* 5 — provider error (network, HTTP status from the gateway)
* 6 — run finished but some task failed for good (dependents blocked)
* 7 — run paused for another reason (approval needed, no valid verification,
  tool layer failure); fix the cause, then ``automa run`` again
* 8 — ``automa bench run``: the benchmark ran and was recorded, but its success
  criteria were not all met
* 130 — interrupted (Ctrl+C); the project is ``paused`` and can be resumed
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from automa import __version__
from automa.agents import (
    Clarification,
    ExecutiveAgent,
    ExecutiveSettings,
    PlanningFailed,
    VerifierAgent,
    WorkerAgent,
    plan_project,
)
from automa.bench import run_benchmark
from automa.benchmarks import (
    BenchmarkError,
    BenchmarkResultError,
    BenchmarkSuite,
    prepare_results_dir,
    write_result,
)
from automa.budget.guard import BudgetError, BudgetExceeded, BudgetGuard
from automa.budget.ledger import CostLedger, nusd_to_usd
from automa.config import AutomaConfig, ConfigError, load_config
from automa.llm.factory import provider_from_config
from automa.llm.gateway import LLMGateway
from automa.llm.provider import Provider, ProviderError
from automa.orchestrator import (
    Orchestrator,
    OrchestratorError,
    OrchestratorSettings,
    RunReport,
    StopReason,
)
from automa.plan import Plan
from automa.project import Project, ProjectStatus, ProjectStore, ProjectStoreError
from automa.roles import RoleError, RoleRegistry
from automa.tools import default_tools

EXIT_OK = 0
EXIT_USER = 1
EXIT_CONFIG = 2
EXIT_BUDGET = 3
EXIT_PLANNING = 4
EXIT_PROVIDER = 5
EXIT_TASKS_FAILED = 6
EXIT_PAUSED = 7
EXIT_BENCH_FAILED = 8
EXIT_INTERRUPTED = 130

# Exit code of each way a run can end (see the module docstring).
RUN_EXIT_CODES: dict[StopReason, int] = {
    StopReason.COMPLETED: EXIT_OK,
    StopReason.TASKS_FAILED: EXIT_TASKS_FAILED,
    StopReason.BUDGET: EXIT_BUDGET,
    StopReason.PROVIDER: EXIT_PROVIDER,
    StopReason.CONFIG: EXIT_CONFIG,
    StopReason.APPROVAL: EXIT_PAUSED,
    StopReason.VERIFICATION: EXIT_PAUSED,
    StopReason.TOOL_LAYER: EXIT_PAUSED,
}

LEDGER_FILENAME = "ledger.sqlite3"
GOAL_PREVIEW_CHARS = 60

app = typer.Typer(help="Automa command-line interface.", no_args_is_help=True)
bench_app = typer.Typer(help="Benchmark suite: list and run sample goals.", no_args_is_help=True)
app.add_typer(bench_app, name="bench")

DEFAULT_SUITE_DIR = Path("benchmarks")
RESULTS_DIRNAME = "results"


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
        roles = RoleRegistry.builtin(tools=default_tools())
    except (ConfigError, RoleError) as e:
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
        agent = ExecutiveAgent(gateway, settings=settings, known_roles=roles.names(), events=events)
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
def run(
    project_id: Annotated[str, typer.Argument(help="Project to run (or resume).")],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Run even if the project is marked running (only after a crash/kill; "
            "never while another run is active).",
        ),
    ] = False,
    max_attempts: Annotated[
        int | None,
        typer.Option("--max-attempts", help="Attempts per task before it fails (1-50)."),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """Run a planned project (worker → verifier → retry) or resume it after a stop.

    All progress is saved after every step, so running the same command again after
    any stop (budget, provider error, Ctrl+C, crash) continues where it left off.
    """
    config = _load(config_path)
    store = ProjectStore.for_workspace(config.workspace)
    try:
        project = store.load(project_id)
    except ProjectStoreError as e:
        raise _fail(str(e), EXIT_USER) from None
    pid = project.id
    if not store.has_plan(pid):
        raise _fail(
            f'project {pid} has no plan yet; create one with: automa new "<goal>"', EXIT_USER
        )
    if project.status is ProjectStatus.RUNNING and not force:
        raise _fail(
            f"project {pid} is marked running: another run may be active. If it crashed "
            "or was killed, run again with --force.",
            EXIT_USER,
        )
    try:
        settings = (
            OrchestratorSettings()
            if max_attempts is None
            else OrchestratorSettings(max_attempts=max_attempts)
        )
    except ValidationError as e:
        raise _fail(f"invalid --max-attempts: {e.errors()[0]['msg']}", EXIT_USER) from None
    try:
        # Validate what every model call needs before touching the project.
        provider = make_provider(config)
        roles = RoleRegistry.builtin(tools=default_tools())
    except (ConfigError, RoleError) as e:
        raise _fail(str(e), EXIT_CONFIG) from None

    typer.echo(f"project: {pid} ({project.status.value})")
    events = store.events(pid)
    with CostLedger.for_workspace(config.workspace) as ledger:
        spent_before = ledger.spent_nusd(project_id=pid)
        guard = BudgetGuard(config.limits, ledger, events=events)
        gateway = LLMGateway(config, provider, guard=guard)
        orchestrator = Orchestrator(
            store,
            WorkerAgent(gateway, events=events),
            VerifierAgent(gateway, events=events),
            roles,
            config,
            settings=settings,
        )
        try:
            report = orchestrator.run(pid)
        except KeyboardInterrupt:
            # The orchestrator has already marked the project paused.
            typer.echo(f"interrupted: project {pid} is paused; resume with: automa run {pid}")
            raise typer.Exit(EXIT_INTERRUPTED) from None
        except OrchestratorError as e:
            raise _fail(str(e), EXIT_USER) from None
        finally:
            spent = ledger.spent_nusd(project_id=pid)
            typer.echo(f"cost: {_usd(spent - spent_before)} this run, {_usd(spent)} total")
    for line in _report_lines(report):
        typer.echo(line)
    raise typer.Exit(RUN_EXIT_CODES[report.reason])


_REASON_HINTS: dict[StopReason, str] = {
    StopReason.BUDGET: "raise the cap in automa.toml or wait for the daily reset",
    StopReason.PROVIDER: "check the provider/network",
    StopReason.CONFIG: "fix automa.toml (e.g. map the task's tier to a model)",
    StopReason.APPROVAL: "this action needs approval (approval gate: not available yet)",
    StopReason.VERIFICATION: "the verifier gave no valid evaluation; the result is kept",
    StopReason.TOOL_LAYER: "the tool layer failed; see the events log",
}


def _report_lines(report: RunReport) -> list[str]:
    lines = [
        f"status: {report.status.value} ({report.reason.value})",
        f"attempts this run: {report.attempts_run}",
        f"passed: {', '.join(report.passed) or '-'}",
    ]
    if report.failed:
        lines.append(f"failed: {', '.join(report.failed)}")
    if report.blocked:
        lines.append(f"blocked: {', '.join(report.blocked)}")
    if report.status is ProjectStatus.PAUSED:
        where = f" at task {report.task_id}" if report.task_id else ""
        lines.append(f"paused{where}: {_preview(report.detail, 200) or report.reason.value}")
        hint = _REASON_HINTS.get(report.reason)
        if hint:
            lines.append(f"next: {hint}, then: automa run {report.project_id}")
    return lines


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


def _suite(suite_dir: Path) -> BenchmarkSuite:
    try:
        return BenchmarkSuite.from_dir(suite_dir)
    except BenchmarkError as e:
        raise _fail(f"benchmark suite: {e}", EXIT_USER) from None


@bench_app.command("list")
def bench_list(
    suite_dir: Annotated[
        Path, typer.Option("--suite", help="Directory of <id>.toml benchmark files.")
    ] = DEFAULT_SUITE_DIR,
) -> None:
    """List the benchmarks in the suite. Read-only; no config or API key needed."""
    suite = _suite(suite_dir)
    for b in suite:
        typer.echo(
            f"{b.id}  {b.kind.value:<8}  ${b.budget_usd:.2f}  "
            f"{len(b.success_criteria)} criteria  {b.title}"
        )


@bench_app.command("run")
def bench_run(
    benchmark_id: Annotated[str, typer.Argument(help="Benchmark id (see: automa bench list).")],
    suite_dir: Annotated[
        Path, typer.Option("--suite", help="Directory of <id>.toml benchmark files.")
    ] = DEFAULT_SUITE_DIR,
    results_dir: Annotated[
        Path | None,
        typer.Option("--results", help="Where to record the result (default: <suite>/results)."),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """Run one benchmark as a new project (plan, run, judge) and record the result.

    The project's budget is the smaller of the benchmark's budget_usd and the
    configured project cap. Success is decided only by the benchmark's
    deterministic checks.
    """
    suite = _suite(suite_dir)
    try:
        benchmark = suite.get(benchmark_id)
    except BenchmarkError as e:
        raise _fail(f"{e} (known: {', '.join(suite.ids())})", EXIT_USER) from None
    results = results_dir if results_dir is not None else suite_dir / RESULTS_DIRNAME
    config = _load(config_path)
    try:
        # Everything a run needs is checked before a project is created or a cent spent.
        config.model_for(ExecutiveSettings().tier)
        provider = make_provider(config)
        roles = RoleRegistry.builtin(tools=default_tools())
    except (ConfigError, RoleError) as e:
        raise _fail(str(e), EXIT_CONFIG) from None
    try:
        prepare_results_dir(results, benchmark.id)
    except BenchmarkResultError as e:
        raise _fail(str(e), EXIT_USER) from None

    store = ProjectStore.for_workspace(config.workspace)
    typer.echo(f"benchmark: {benchmark.id} ({benchmark.kind.value}) — {benchmark.title}")
    with CostLedger.for_workspace(config.workspace) as ledger:
        try:
            record = run_benchmark(
                benchmark,
                config=config,
                provider=provider,
                store=store,
                ledger=ledger,
                roles=roles,
            )
        except KeyboardInterrupt:
            typer.echo("interrupted: no result recorded (the project is kept and paused)")
            raise typer.Exit(EXIT_INTERRUPTED) from None
        except ProjectStoreError as e:
            raise _fail(str(e), EXIT_USER) from None
        except OrchestratorError as e:
            raise _fail(str(e), EXIT_USER) from None
    try:
        path = write_result(results, record)
    except BenchmarkResultError as e:
        raise _fail(f"could not record the result: {e}", EXIT_USER) from None
    typer.echo(f"project: {record.project_id} ({record.project_status.value})")
    typer.echo(f"stop: {record.stop.value}")
    for c in record.checks:
        mark = "PASS" if c.passed else "FAIL"
        desc = benchmark.success_criteria[c.criterion].description
        typer.echo(f"  [{mark}] {c.criterion}. {_preview(desc, 80)}")
    passed = sum(1 for c in record.checks if c.passed)
    typer.echo(
        f"result: {'PASSED' if record.passed else 'FAILED'} "
        f"(score {record.score:.2f}, {passed}/{len(record.checks)} criteria)"
    )
    typer.echo(
        f"cost: {_usd(record.cost_nusd)} of ${record.budget_usd:.2f}, "
        f"{record.model_calls} model call(s), {record.attempts} attempt(s)"
    )
    typer.echo(f"recorded: {path}")
    raise typer.Exit(EXIT_OK if record.passed else EXIT_BENCH_FAILED)


def main() -> None:
    """Console-script entry point."""
    app()
