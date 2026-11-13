"""Command-line interface: ``automa new``, ``run``, ``status``, ``approvals``,
``approve``/``reject``, ``workflow save|create|edit|show``/``workflows`` and
``skills``, ``memory list|add|forget`` and ``bench list|run|report``.

Every model call goes through ``LLMGateway`` + ``BudgetGuard`` (never a bare provider).
``status`` and ``approvals`` are read-only: they need no API key, make no model call
and create no files. ``approve``/``reject`` only record a decision on one pending
Approval Gate request; the action itself runs (at most once) on the next ``automa run``.
``workflow save`` copies the plan of a finished project into a workflow template and
``workflows`` lists them; neither needs an API key or makes a model call. ``automa new``
prints similar templates as a suggestion only (D39); ``automa new --workflow <id>``
uses a template's plan instead of the Executive (no API key, no model call, D40).
``workflow create``/``edit`` save a user-written or edited plan as a new template and
``workflow show`` prints one (``--plan-json`` gives an editable plan file); none of
them needs an API key or makes a model call (D41, D42). ``skills`` lists the available
skills and the roles that use them, read-only (D46). ``memory list`` shows the three
memory layers (read-only); ``memory add`` writes a user-written entry and ``memory forget``
deletes one after confirmation; none of them needs an API key or makes a model call (D49).
``new`` and ``run`` load the relevant memory before any model call and add it to the
Executive's/Worker's system prompts as data (``--no-memory`` or ``[memory] in_prompts =
false`` turns it off; a broken memory is a configuration error); ``bench run`` never
reads memory (D51).

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
import unicodedata
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
    BenchmarkReport,
    BenchmarkResultError,
    BenchmarkSuite,
    build_report,
    prepare_results_dir,
    scan_results,
    write_result,
)
from automa.budget.guard import BudgetError, BudgetExceeded, BudgetGuard
from automa.budget.ledger import CostLedger, nusd_to_usd
from automa.config import (
    MAX_PARALLEL,
    AutomaConfig,
    ConfigError,
    Tier,
    describe_parallel,
    load_config,
)
from automa.events import Event
from automa.llm.factory import provider_from_config
from automa.llm.gateway import LLMGateway
from automa.llm.provider import Provider, ProviderError
from automa.memory import (
    MemoryContext,
    MemoryEntry,
    MemoryScope,
    MemorySource,
    MemoryStore,
    MemoryStoreError,
)
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
from automa.skills import (
    Skill,
    SkillError,
    SkillRegistry,
    builtin_skills,
    load_skills,
    merge_skills,
    workspace_skills,
    workspace_skills_dir,
)
from automa.tools import DockerSandbox, SandboxError, check_docker, default_tools, tools_factory
from automa.tools.approvals import (
    ApprovalError,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalStore,
)
from automa.workflows import (
    WorkflowError,
    WorkflowIncompatible,
    WorkflowMatch,
    WorkflowStore,
    WorkflowTemplate,
    plan_from_template,
    similarity,
)

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
# ``automa workflow create/edit --plan``: largest plan file read (plans are small).
MAX_PLAN_FILE_BYTES = 1_000_000
# ``automa approvals`` shows at most this much of each request's arguments (the full
# arguments are always shown by ``automa approve`` before asking for confirmation).
ARGUMENTS_PREVIEW_CHARS = 2_000
# Zero-width (non-)joiners are format characters needed by Persian/Arabic text.
_ALLOWED_FORMAT_CHARS = frozenset({"\u200c", "\u200d"})

app = typer.Typer(help="Automa command-line interface.", no_args_is_help=True)
bench_app = typer.Typer(help="Benchmark suite: list and run sample goals.", no_args_is_help=True)
app.add_typer(bench_app, name="bench")
workflow_app = typer.Typer(
    help="Workflow templates: plans of finished projects, saved for reuse.",
    no_args_is_help=True,
)
app.add_typer(workflow_app, name="workflow")
memory_app = typer.Typer(
    help="Memory: user preferences, per-project knowledge and general lessons.",
    no_args_is_help=True,
)
app.add_typer(memory_app, name="memory")

DEFAULT_SUITE_DIR = Path("benchmarks")
RESULTS_DIRNAME = "results"


# Indirection point: tests replace this with a FakeProvider factory. The provider is
# always wrapped in a guarded LLMGateway below; nothing here calls it directly.
make_provider: Callable[[AutomaConfig], Provider] = provider_from_config
# Indirection point for the Docker preflight (tests use a fake docker client).
docker_check: Callable[[DockerSandbox], str] = check_docker


def _fail(message: str, code: int) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


def _require_sandbox(config: AutomaConfig) -> None:
    """With ``mode = "docker"``, stop before any work unless Docker answers (fail closed)."""
    sandbox = config.sandbox.docker_sandbox()
    if sandbox is None:
        return
    try:
        docker_check(sandbox)
    except SandboxError as e:
        raise _fail(
            f"sandbox mode is docker but Docker is not usable: {e}; nothing was run "
            '(start Docker, or set [sandbox] mode = "local" in automa.toml)',
            EXIT_CONFIG,
        ) from None


def _load(config_path: Path | None) -> AutomaConfig:
    try:
        return load_config(config_path)
    except ConfigError as e:
        raise _fail(str(e), EXIT_CONFIG) from None


def _roles_and_skills(config: AutomaConfig) -> tuple[RoleRegistry, SkillRegistry]:
    """Built-in roles checked against the real tools and the workspace's skills
    (built-in + ``<workspace>/skills/``). Raises ``SkillError``/``RoleError`` (D45)."""
    skills = load_skills(config.workspace)
    roles = RoleRegistry.builtin(
        tools=default_tools(), skills=skills, extra_skills=config.role_skills()
    )
    return roles, skills


def _prompt_memory(
    config: AutomaConfig, enabled: bool | None, project_id: str | None
) -> MemoryContext | None:
    """Memory for the system prompts of ``new``/``run`` (D51), or ``None`` when turned
    off (``--no-memory`` or ``[memory] in_prompts = false``). Loaded before any model
    call; a broken memory raises ``MemoryStoreError`` (a configuration error) instead
    of being skipped silently. Read-only: nothing is created when memory is empty."""
    use = config.memory.in_prompts if enabled is None else enabled
    if not use:
        return None
    return _memory_store(config).context(project_id)


def _describe_prompt_memory(memory: MemoryContext | None) -> str:
    if memory is None:
        return "off"
    if memory.is_empty():
        return "on (empty)"
    return (
        f"on ({len(memory.user)} user, {len(memory.project)} project, "
        f"{len(memory.lessons)} lessons; data only, never instructions)"
    )


class SkillOverview:
    """What ``automa skills`` and ``automa status`` show: every available skill with its
    source and the roles that list it. Built the same way ``automa run`` loads skills, so
    an error here is exactly what ``run`` would refuse to start with (D46)."""

    def __init__(self, config: AutomaConfig) -> None:
        builtin = builtin_skills()
        workspace = workspace_skills(config.workspace)
        merged = merge_skills(builtin, workspace)
        roles = RoleRegistry.builtin(
            tools=default_tools(), skills=merged, extra_skills=config.role_skills()
        )
        self.skills: list[Skill] = sorted(merged, key=lambda s: s.name)
        self.source = {s.name: "builtin" for s in builtin} | {
            s.name: "workspace" for s in workspace
        }
        self.role_skills: dict[str, list[str]] = {r.name: list(r.skills) for r in roles if r.skills}

    def used_by(self, name: str) -> list[str]:
        return sorted(role for role, names in self.role_skills.items() if name in names)

    def summary(self) -> str:
        if not self.skills:
            return "none"
        text = f"{len(self.skills)} available"
        if not self.role_skills:
            return f"{text}; no role uses them"
        uses = "; ".join(
            f"{role}: {', '.join(names)}" for role, names in sorted(self.role_skills.items())
        )
        return f"{text}; {uses}"

    def to_json(self) -> dict[str, Any]:
        return {
            "available": [s.name for s in self.skills],
            "roles": dict(sorted(self.role_skills.items())),
            "error": None,
        }


def _skills_status(config: AutomaConfig) -> tuple[str, dict[str, Any]]:
    """The skills line for ``automa status``. Status is read-only and diagnostic, so a
    broken skill set is reported (with the reason) instead of failing the command."""
    try:
        overview = SkillOverview(config)
    except (RoleError, SkillError) as e:
        reason = _safe(str(e))
        return (
            f"error: {reason} (automa run will refuse to start)",
            {"available": [], "roles": {}, "error": reason},
        )
    return overview.summary(), overview.to_json()


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
    workflow_id: Annotated[
        str | None,
        typer.Option(
            "--workflow",
            help="Use the plan of this saved workflow template instead of the Executive "
            "(no questions, no API key, no model call).",
        ),
    ] = None,
    memory: Annotated[
        bool | None,
        typer.Option(
            "--memory/--no-memory",
            help="Add memory (user preferences, this project's memory, lessons) to the "
            "system prompts as data; default: [memory] in_prompts in automa.toml (on).",
        ),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """Create a project for GOAL, answer clarifying questions and write its plan."""
    goal = goal.strip()
    if not goal:
        raise _fail("goal must not be empty", EXIT_USER)
    config = _load(config_path)
    if workflow_id is not None:
        _new_from_workflow(config, goal, project_id, workflow_id)
        return
    settings = ExecutiveSettings()
    try:
        # Validate everything a model call needs *before* creating the project, so a
        # missing key or tier never leaves an orphan project behind.
        config.model_for(settings.tier)
        provider = make_provider(config)
        roles, _ = _roles_and_skills(config)
        # A new project has no project memory yet: user preferences and lessons only.
        prompt_memory = _prompt_memory(config, memory, None)
    except (ConfigError, RoleError, SkillError, MemoryStoreError) as e:
        raise _fail(str(e), EXIT_CONFIG) from None

    store = ProjectStore.for_workspace(config.workspace)
    try:
        project = store.create(goal, project_id=project_id)
    except ProjectStoreError as e:
        raise _fail(str(e), EXIT_USER) from None
    pid = project.id
    typer.echo(f"project: {pid}")
    typer.echo(f"memory: {_describe_prompt_memory(prompt_memory)}")
    _print_suggestions(config, goal)

    events = store.events(pid)
    with CostLedger.for_workspace(config.workspace) as ledger:
        guard = BudgetGuard(config.limits, ledger, events=events)
        gateway = LLMGateway(config, provider, guard=guard)
        agent = ExecutiveAgent(
            gateway,
            settings=settings,
            known_roles=roles.names(),
            events=events,
            memory=prompt_memory,
        )
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


def _compat_context(config: AutomaConfig) -> tuple[dict[str, Tier], frozenset[Tier]]:
    """Current roles (with their default tiers) and configured tiers: what a workflow
    plan is checked against before it is saved or used (D40, D41)."""
    try:
        roles = RoleRegistry.builtin(tools=default_tools())
    except RoleError as e:
        raise _fail(str(e), EXIT_CONFIG) from None
    return {role.name: role.tier for role in roles}, frozenset(config.tiers)


def _new_from_workflow(
    config: AutomaConfig, goal: str, project_id: str | None, workflow_id: str
) -> None:
    """``automa new --workflow``: the plan comes from a stored template (D40).

    Everything is checked before the project is created (template readable, roles
    known, starting tiers configured), so a bad template never leaves an orphan
    project. No provider is built and no model is called.
    """
    role_tiers, configured = _compat_context(config)
    flows = WorkflowStore.for_workspace(config.workspace)
    try:
        template = flows.load(workflow_id)
        plan_from_template(template, goal, role_tiers=role_tiers, configured_tiers=configured)
    except WorkflowIncompatible as e:
        raise _fail(f"{_safe(str(e))}; nothing was created", EXIT_CONFIG) from None
    except WorkflowError as e:
        raise _fail(f"{_safe(str(e))}; nothing was created", EXIT_USER) from None

    store = ProjectStore.for_workspace(config.workspace)
    try:
        project = store.create(goal, project_id=project_id)
    except ProjectStoreError as e:
        raise _fail(str(e), EXIT_USER) from None
    pid = project.id
    typer.echo(f"project: {pid}")
    try:
        plan = flows.use(
            store, pid, template.id, role_tiers=role_tiers, configured_tiers=configured
        )
    except WorkflowIncompatible as e:
        raise _fail(f"{_safe(str(e))}; project {pid} has no plan", EXIT_CONFIG) from None
    except (WorkflowError, ProjectStoreError) as e:
        raise _fail(f"{_safe(str(e))}; project {pid} has no plan", EXIT_USER) from None
    score, _ = similarity(goal, template.goal)
    typer.echo(
        f"workflow: {template.id} ({_workflow_source(template)}, "
        f"{round(score * 100)}% similar goal: {_safe(_preview(template.goal))})"
    )
    typer.echo(
        "note: task titles, descriptions, inputs and checks are copied from the template "
        "unchanged; the Executive was not asked"
    )
    typer.echo(f"cost: {_usd(0)} (no model call)")
    for line in _plan_lines(plan):
        typer.echo(_safe(line))
    typer.echo(f"status: {store.load(pid).status.value}")


def _print_suggestions(config: AutomaConfig, goal: str) -> None:
    """Show stored templates similar to ``goal`` (suggestion only, D39).

    Deterministic and free: no model call, nothing is reused or written. A broken
    workflows directory never blocks a new project; it only produces a warning.
    """
    try:
        matches = WorkflowStore.for_workspace(config.workspace).suggest(goal)
    except WorkflowError as e:
        typer.echo(f"warning: workflow suggestions unavailable: {_safe(str(e))}", err=True)
        return
    if not matches:
        return
    typer.echo("similar workflows (suggestion only; the plan below is made from scratch):")
    for m in matches:
        typer.echo(f"  {_match_line(m)}")
    typer.echo("  see them with: automa workflows")


def _match_line(m: WorkflowMatch) -> str:
    t = m.template
    shared = _safe(", ".join(m.shared))
    return (
        f"{t.id}  {round(m.score * 100)}%  {t.task_count} task(s)  "
        f"{_safe(_preview(t.goal))}  [shared: {shared}]"
    )


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


def _run_settings(
    config: AutomaConfig, *, max_attempts: int | None, parallel: int | None
) -> OrchestratorSettings:
    """Orchestrator settings for ``automa run``: CLI options override automa.toml (D36)."""
    values: dict[str, int] = {"max_parallel": config.orchestrator.max_parallel}
    for option, field, value in (
        ("--max-attempts", "max_attempts", max_attempts),
        ("--parallel", "max_parallel", parallel),
    ):
        if value is None:
            continue
        try:
            OrchestratorSettings.model_validate({field: value})
        except ValidationError as e:
            raise _fail(f"invalid {option}: {e.errors()[0]['msg']}", EXIT_USER) from None
        values[field] = value
    return OrchestratorSettings.model_validate(values)


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
    parallel: Annotated[
        int | None,
        typer.Option(
            "--parallel",
            help=f"Independent tasks run at the same time (1-{MAX_PARALLEL}); "
            "overrides [orchestrator] max_parallel in automa.toml (default 1).",
        ),
    ] = None,
    memory: Annotated[
        bool | None,
        typer.Option(
            "--memory/--no-memory",
            help="Add memory (user preferences, this project's memory, lessons) to the "
            "system prompts as data; default: [memory] in_prompts in automa.toml (on).",
        ),
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
    settings = _run_settings(config, max_attempts=max_attempts, parallel=parallel)
    try:
        # Validate what every model call needs before touching the project.
        provider = make_provider(config)
        roles, skills = _roles_and_skills(config)
        prompt_memory = _prompt_memory(config, memory, pid)
    except (ConfigError, RoleError, SkillError, MemoryStoreError) as e:
        raise _fail(str(e), EXIT_CONFIG) from None
    _require_sandbox(config)

    typer.echo(f"project: {pid} ({project.status.value})")
    typer.echo(f"sandbox: {config.sandbox.describe()}")
    typer.echo(f"parallel: {describe_parallel(settings.max_parallel)}")
    typer.echo(f"memory: {_describe_prompt_memory(prompt_memory)}")
    events = store.events(pid)
    with CostLedger.for_workspace(config.workspace) as ledger:
        spent_before = ledger.spent_nusd(project_id=pid)
        guard = BudgetGuard(config.limits, ledger, events=events)
        gateway = LLMGateway(config, provider, guard=guard)
        orchestrator = Orchestrator(
            store,
            WorkerAgent(gateway, events=events, skills=skills, memory=prompt_memory),
            VerifierAgent(gateway, events=events),
            roles,
            config,
            settings=settings,
            tools=tools_factory(config.sandbox.docker_sandbox()),
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
    StopReason.APPROVAL: "review the request with: automa approvals {pid}",
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
        if report.reason is StopReason.APPROVAL and report.approval_id:
            lines.append(f"approval needed: {report.approval_id}")
            lines.append(
                f"next: automa approvals {report.project_id}, then "
                f"automa approve {report.project_id} {report.approval_id} (or: automa reject "
                f"{report.project_id} {report.approval_id}), then: automa run {report.project_id}"
            )
            return lines
        hint = _REASON_HINTS.get(report.reason)
        if hint:
            hint = hint.format(pid=report.project_id)
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
            _status_all(store, config, as_json)
        else:
            _status_one(store, config, project_id, as_json)
    except (ProjectStoreError, ApprovalError) as e:
        raise _fail(str(e), EXIT_USER) from None


def _sandbox_json(config: AutomaConfig) -> dict[str, Any]:
    sb = config.sandbox.docker_sandbox()
    if sb is None:
        return {"mode": config.sandbox.mode.value}
    return {
        "mode": config.sandbox.mode.value,
        "image": sb.image,
        "memory_mb": sb.memory_mb,
        "cpus": sb.cpus,
        "pids_limit": sb.pids_limit,
    }


def _status_all(store: ProjectStore, config: AutomaConfig, as_json: bool) -> None:
    workspace = config.workspace
    projects = store.list_projects() if store.root.is_dir() else []
    spent = _all_spent_nusd(workspace)
    pending = {p.id: _pending_approvals(store, p.id) for p in projects}
    if as_json:
        rows = [
            {
                "id": p.id,
                "status": p.status.value,
                "goal": p.goal,
                "created_at": p.created_at.isoformat(),
                "cost_usd": nusd_to_usd(spent.get(p.id, 0)),
                "pending_approvals": len(pending[p.id]),
            }
            for p in projects
        ]
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not projects:
        typer.echo('no projects yet; create one with: automa new "<goal>"')
        return
    for p in projects:
        waiting = f"  [{len(pending[p.id])} approval(s) pending]" if pending[p.id] else ""
        typer.echo(
            f"{p.id}  {p.status.value:<8}  {_usd(spent.get(p.id, 0))}  {_preview(p.goal)}{waiting}"
        )
    typer.echo(f"sandbox: {config.sandbox.describe()}")
    typer.echo(f"parallel: {config.orchestrator.describe()}")
    typer.echo(f"skills: {_skills_status(config)[0]}")
    typer.echo(f"memory: {_memory_status(config)[0]}")


def _status_one(store: ProjectStore, config: AutomaConfig, project_id: str, as_json: bool) -> None:
    workspace = config.workspace
    project = store.load(project_id)
    plan = store.load_plan(project_id) if store.has_plan(project_id) else None
    clarifications = store.load_clarifications(project_id)
    spent = _project_spent_nusd(workspace, project_id)
    pending = _pending_approvals(store, project_id)
    skills_line, skills_json = _skills_status(config)
    memory_line, memory_json = _memory_status(config, project.id)
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
            "pending_approvals": [_approval_json(store, r) for r in pending],
            "sandbox": _sandbox_json(config),
            "max_parallel": config.orchestrator.max_parallel,
            "skills": skills_json,
            "memory": memory_json,
        }
        typer.echo(json.dumps(data, ensure_ascii=False, indent=2))
        return
    typer.echo(f"project: {project.id}")
    typer.echo(f"goal: {project.goal}")
    typer.echo(f"status: {project.status.value}")
    typer.echo(f"created: {project.created_at.isoformat()}")
    typer.echo(f"updated: {project.updated_at.isoformat()}")
    typer.echo(f"cost: {_usd(spent)}")
    typer.echo(f"sandbox: {config.sandbox.describe()}")
    typer.echo(f"parallel: {config.orchestrator.describe()}")
    typer.echo(f"skills: {skills_line}")
    typer.echo(f"memory: {memory_line}")
    if pending:
        typer.echo(f"pending approvals: {len(pending)}")
        for r in pending:
            typer.echo(f"  {_approval_summary(r)}")
        typer.echo(f"  review with: automa approvals {project.id}")
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


# -- Workflow templates ----------------------------------------------------------


def _workflow_json(t: WorkflowTemplate) -> dict[str, Any]:
    data: dict[str, Any] = t.model_dump(mode="json")
    if t.stats is not None:
        data["stats"]["cost_usd"] = nusd_to_usd(t.stats.cost_nusd)
    return data


def _workflow_source(t: WorkflowTemplate) -> str:
    if t.source_project is not None:
        return f"from {t.source_project}"
    return f"user, edited from {t.parent}" if t.parent else "user-defined"


def _workflow_line(t: WorkflowTemplate) -> str:
    s = t.stats
    proven = f"{s.attempts} attempt(s)  {_usd(s.cost_nusd)}" if s is not None else "not run yet"
    return (
        f"{t.id}  {t.task_count} task(s)  {proven}  "
        f"{_workflow_source(t)}  {_safe(_preview(t.goal))}"
    )


@workflow_app.command("save")
def workflow_save(
    project_id: Annotated[str, typer.Argument(help="A finished (done) project to save.")],
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """Save the plan of a finished project (every task passed) as a workflow template.

    No API key or model call. A project is saved at most once; templates are never
    overwritten.
    """
    config = _load(config_path)
    projects = ProjectStore.for_workspace(config.workspace)
    flows = WorkflowStore.for_workspace(config.workspace)
    try:
        template = flows.save_from_project(projects, project_id)
    except (WorkflowError, ProjectStoreError) as e:
        raise _fail(str(e), EXIT_USER) from None
    if as_json:
        typer.echo(json.dumps(_workflow_json(template), ensure_ascii=False, indent=2))
        return
    typer.echo(f"workflow: {template.id}")
    typer.echo(_workflow_line(template))
    for line in _plan_lines(template.plan):
        typer.echo(line)
    typer.echo(
        "note: the template is a copy of this project's plan, including its task texts "
        "and inputs; it stays in your workspace."
    )


def _read_plan_file(path: Path) -> Any:
    """The JSON value in a user-written plan file (validated later by ``user_plan``).

    Read-only and bounded: a missing file, a non-regular file, a file over
    ``MAX_PLAN_FILE_BYTES``, invalid UTF-8 or invalid JSON is a user error.
    """
    try:
        if not path.is_file():
            raise _fail(f"plan file not found or not a regular file: {path}", EXIT_USER)
        with path.open("rb") as fh:
            raw = fh.read(MAX_PLAN_FILE_BYTES + 1)
    except OSError as e:
        raise _fail(f"cannot read plan file {path}: {e.strerror or e}", EXIT_USER) from None
    if len(raw) > MAX_PLAN_FILE_BYTES:
        raise _fail(f"plan file {path} is larger than {MAX_PLAN_FILE_BYTES} bytes", EXIT_USER)
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        raise _fail(f"plan file {path} is not valid UTF-8", EXIT_USER) from None
    except json.JSONDecodeError as e:
        raise _fail(
            f"plan file {path} is not valid JSON (line {e.lineno}, column {e.colno}: {e.msg})",
            EXIT_USER,
        ) from None


def _print_saved_user_template(template: WorkflowTemplate, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(_workflow_json(template), ensure_ascii=False, indent=2))
        return
    typer.echo(f"workflow: {template.id}")
    typer.echo(_workflow_line(template))
    for line in _plan_lines(template.plan):
        typer.echo(_safe(line))
    typer.echo(
        "note: this plan has not been run yet; its checks still pass the Permission "
        "Policy and its tasks the budget and Approval Gate when used"
    )
    typer.echo(f"use it with: automa new GOAL --workflow {template.id}")


def _user_template_errors(action: Callable[[], WorkflowTemplate]) -> WorkflowTemplate:
    try:
        return action()
    except WorkflowIncompatible as e:
        raise _fail(f"{_safe(str(e))}; nothing was saved", EXIT_CONFIG) from None
    except WorkflowError as e:
        raise _fail(f"{_safe(str(e))}; nothing was saved", EXIT_USER) from None


@workflow_app.command("create")
def workflow_create(
    goal: Annotated[str, typer.Option("--goal", help="What the workflow is for.")],
    plan_file: Annotated[
        Path,
        typer.Option(
            "--plan",
            help="JSON file with the plan (the shape of plan.json; its goal and version "
            "are ignored).",
        ),
    ],
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """Save a plan you wrote as a new workflow template (D41).

    The plan is validated like any plan and checked against the current roles and
    configured tiers; nothing is saved on any error. No API key or model call.
    """
    config = _load(config_path)
    data = _read_plan_file(plan_file)
    role_tiers, configured = _compat_context(config)
    flows = WorkflowStore.for_workspace(config.workspace)
    template = _user_template_errors(
        lambda: flows.create(goal, data, role_tiers=role_tiers, configured_tiers=configured)
    )
    _print_saved_user_template(template, as_json)


@workflow_app.command("edit")
def workflow_edit(
    workflow_id: Annotated[str, typer.Argument(help="The template to edit.")],
    goal: Annotated[str | None, typer.Option("--goal", help="A new goal.")] = None,
    plan_file: Annotated[
        Path | None,
        typer.Option("--plan", help="JSON file with a new plan (see: workflow show --plan-json)."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """Save an edited copy of a template as a new template (the original is kept).

    The new template records the original as its parent and has no stats: it has
    not been run as is (D41). No API key or model call.
    """
    config = _load(config_path)
    if goal is None and plan_file is None:
        raise _fail("nothing to change: give --goal and/or --plan", EXIT_USER)
    data = _read_plan_file(plan_file) if plan_file is not None else None
    role_tiers, configured = _compat_context(config)
    flows = WorkflowStore.for_workspace(config.workspace)
    template = _user_template_errors(
        lambda: flows.revise(
            workflow_id,
            goal=goal,
            plan=data,
            role_tiers=role_tiers,
            configured_tiers=configured,
        )
    )
    _print_saved_user_template(template, as_json)


def _editable_plan(t: WorkflowTemplate) -> dict[str, Any]:
    """The template's plan as accepted by ``workflow edit --plan`` (no goal/version:
    those come from the template)."""
    data: dict[str, Any] = t.plan.model_dump(mode="json")
    data.pop("goal", None)
    data.pop("version", None)
    return data


@workflow_app.command("show")
def workflow_show(
    workflow_id: Annotated[str, typer.Argument(help="The template to show.")],
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    plan_json: Annotated[
        bool,
        typer.Option(
            "--plan-json",
            help="Print only the plan, as a file for: automa workflow edit ID --plan FILE.",
        ),
    ] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """Show one workflow template: goal, origin, stats, edit history and plan.

    Read-only: no API key, no model call, nothing is written.
    """
    if as_json and plan_json:
        raise _fail("use either --json or --plan-json, not both", EXIT_USER)
    config = _load(config_path)
    flows = WorkflowStore.for_workspace(config.workspace)
    try:
        template = flows.load(workflow_id)
    except WorkflowError as e:
        raise _fail(_safe(str(e)), EXIT_USER) from None
    if plan_json:
        typer.echo(json.dumps(_editable_plan(template), ensure_ascii=False, indent=2))
        return
    history_ids: list[str] | None
    history_error: str | None = None
    try:
        history_ids = [t.id for t in flows.history(template.id)]
    except WorkflowError as e:
        history_ids, history_error = None, str(e)
    if as_json:
        data = _workflow_json(template)
        data["history"] = history_ids
        typer.echo(json.dumps(data, ensure_ascii=False, indent=2))
        return
    typer.echo(f"workflow: {template.id}")
    typer.echo(f"source: {_workflow_source(template)}")
    typer.echo(f"created: {template.created_at.isoformat()}")
    s = template.stats
    if s is None:
        typer.echo("stats: not run yet")
    else:
        typer.echo(
            f"stats: {s.attempts} attempt(s) for {template.task_count} task(s), {_usd(s.cost_nusd)}"
        )
    if history_ids is not None:
        if len(history_ids) > 1:
            typer.echo(f"history: {' <- '.join(history_ids)}")
    else:
        typer.echo(f"warning: history unavailable: {_safe(history_error or '')}", err=True)
    typer.echo(f"goal: {_safe(template.goal)}")
    for line in _plan_lines(template.plan):
        typer.echo(_safe(line))
    typer.echo(
        f"edit with: automa workflow show {template.id} --plan-json > plan.json, then "
        f"automa workflow edit {template.id} --plan plan.json"
    )


@app.command()
def workflows(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """List saved workflow templates, newest first. Read-only."""
    config = _load(config_path)
    try:
        templates = WorkflowStore.for_workspace(config.workspace).templates()
    except WorkflowError as e:
        raise _fail(_safe(str(e)), EXIT_USER) from None
    if as_json:
        rows = [_workflow_json(t) for t in templates]
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not templates:
        typer.echo("no workflows yet; save a finished project with: automa workflow save <id>")
        return
    for t in templates:
        typer.echo(_workflow_line(t))


# -- Skills ----------------------------------------------------------------------


@app.command()
def skills(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """List available skills (built-in and <workspace>/skills/) and the roles that use
    them. Read-only: no API key, no model call; a broken skill set is exit code 2."""
    config = _load(config_path)
    try:
        overview = SkillOverview(config)
    except (RoleError, SkillError) as e:
        raise _fail(_safe(str(e)), EXIT_CONFIG) from None
    if as_json:
        rows = [
            {
                "name": s.name,
                "source": overview.source[s.name],
                "description": s.description,
                "license": s.license,
                "used_by": overview.used_by(s.name),
                "body_chars": len(s.body),
            }
            for s in overview.skills
        ]
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not overview.skills:
        where = workspace_skills_dir(config.workspace) / "<name>" / "SKILL.md"
        typer.echo(f"no skills yet; add one as: {where}")
        return
    for s in overview.skills:
        used = ", ".join(overview.used_by(s.name)) or "-"
        typer.echo(f"{s.name}  {overview.source[s.name]:<9}  roles: {used}  {_safe(s.description)}")


# -- Memory ----------------------------------------------------------------------


def _memory_store(config: AutomaConfig) -> MemoryStore:
    return MemoryStore.for_workspace(config.workspace)


def _memory_status(
    config: AutomaConfig, project_id: str | None = None
) -> tuple[str, dict[str, Any]]:
    """The memory line for ``automa status``: entry counts per layer. Read-only and
    diagnostic, so a broken memory is reported (with the reason) instead of failing."""
    store = _memory_store(config)
    try:
        user = len(store.entries(MemoryScope.USER))
        lessons = len(store.entries(MemoryScope.LESSON))
        if project_id is None:
            project = sum(len(store.entries(MemoryScope.PROJECT, p)) for p in store.project_ids())
        else:
            project = len(store.entries(MemoryScope.PROJECT, project_id))
    except MemoryStoreError as e:
        reason = _safe(str(e))
        return f"error: {reason}", {"user": 0, "project": 0, "lessons": 0, "error": reason}
    where = "all projects" if project_id is None else "this project"
    line = f"{user} user, {project} project ({where}), {lessons} lessons"
    return line, {"user": user, "project": project, "lessons": lessons, "error": None}


def _memory_json(entry: MemoryEntry) -> dict[str, Any]:
    return entry.model_dump(mode="json")


def _memory_lines(entry: MemoryEntry) -> list[str]:
    """Header line, then the full text indented (escaped: review entries may come
    from a model)."""
    header = (
        f"{entry.id}  {entry.scope.value:<7}  {entry.project_id or '-'}  "
        f"{entry.source.value:<6}  {entry.created_at.isoformat()}"
    )
    return [header, *(f"  {_safe(line)}" for line in entry.text.split("\n"))]


def _memory_project(config: AutomaConfig, project_id: str) -> str:
    """``project_id`` of an existing project (memory is only added to real projects)."""
    try:
        return ProjectStore.for_workspace(config.workspace).load(project_id).id
    except ProjectStoreError as e:
        raise _fail(str(e), EXIT_USER) from None


def _selected_memory(
    store: MemoryStore, scope: MemoryScope | None, project_id: str | None
) -> list[MemoryEntry]:
    want = {scope} if scope is not None else set(MemoryScope)
    if project_id is not None:
        if scope in (MemoryScope.USER, MemoryScope.LESSON):
            raise MemoryStoreError(f"{scope.value} memory is not tied to a project; drop --project")
        context = store.context(project_id)
    else:
        projects = [
            e
            for pid in (store.project_ids() if MemoryScope.PROJECT in want else [])
            for e in store.entries(MemoryScope.PROJECT, pid)
        ]
        context = MemoryContext(
            user=tuple(store.entries(MemoryScope.USER)) if MemoryScope.USER in want else (),
            project=tuple(projects),
            lessons=tuple(store.entries(MemoryScope.LESSON)) if MemoryScope.LESSON in want else (),
        )
    result: list[MemoryEntry] = []
    if MemoryScope.USER in want:
        result += context.user
    if MemoryScope.PROJECT in want:
        result += context.project
    if MemoryScope.LESSON in want:
        result += context.lessons
    return result


_SCOPE_OPTION = typer.Option("--scope", help="Memory layer: user, project or lesson.")


@memory_app.command("list")
def memory_list(
    scope: Annotated[MemoryScope | None, _SCOPE_OPTION] = None,
    project_id: Annotated[
        str | None,
        typer.Option(
            "--project",
            help="Only the memory that applies to this project (user, its own, lessons).",
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """List memory entries (user, project, lessons) with their full text. Read-only:
    no API key, no model call, no files written; a broken memory is exit code 1."""
    config = _load(config_path)
    store = _memory_store(config)
    try:
        selected = _selected_memory(store, scope, project_id)
    except MemoryStoreError as e:
        raise _fail(_safe(str(e)), EXIT_USER) from None
    if as_json:
        typer.echo(json.dumps([_memory_json(e) for e in selected], ensure_ascii=False, indent=2))
        return
    if not selected:
        typer.echo('no memory yet; add some with: automa memory add --scope user "<text>"')
        return
    for entry in selected:
        for line in _memory_lines(entry):
            typer.echo(line)


@memory_app.command("add")
def memory_add(
    text: Annotated[str, typer.Argument(help="What to remember (at most 2000 characters).")],
    scope: Annotated[MemoryScope, _SCOPE_OPTION] = MemoryScope.USER,
    project_id: Annotated[
        str | None,
        typer.Option(
            "--project",
            help="Project of the entry: required for --scope project, optional for lesson.",
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """Remember one thing written by you (source: user). No API key, no model call."""
    config = _load(config_path)
    if scope is MemoryScope.PROJECT and project_id is None:
        raise _fail("project memory needs --project <id>", EXIT_USER)
    if scope is MemoryScope.USER and project_id is not None:
        raise _fail("user memory is not tied to a project; drop --project", EXIT_USER)
    if project_id is not None:
        project_id = _memory_project(config, project_id)
    try:
        entry = _memory_store(config).add(
            scope, text, project_id=project_id, source=MemorySource.USER
        )
    except MemoryStoreError as e:
        raise _fail(_safe(str(e)), EXIT_USER) from None
    if as_json:
        typer.echo(json.dumps(_memory_json(entry), ensure_ascii=False, indent=2))
        return
    typer.echo(f"remembered: {entry.id}")


@memory_app.command("forget")
def memory_forget(
    memory_id: Annotated[str, typer.Argument(help="Entry id (see: automa memory list).")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """Delete one memory entry after showing it. No API key, no model call."""
    config = _load(config_path)
    store = _memory_store(config)
    try:
        entry = store.find(memory_id)
    except MemoryStoreError as e:
        raise _fail(_safe(str(e)), EXIT_USER) from None
    for line in _memory_lines(entry):
        typer.echo(line)
    if not yes and not typer.confirm("Forget this memory for good?", default=False):
        raise _fail("not forgotten; nothing was changed", EXIT_USER)
    try:
        store.forget(entry.id)
    except MemoryStoreError as e:
        raise _fail(_safe(str(e)), EXIT_USER) from None
    typer.echo(f"forgotten: {entry.id}")


# -- Approval Gate -------------------------------------------------------------


def _safe(text: str) -> str:
    """``text`` with control/format characters escaped, so recorded content (which may
    come from the model) can't move the cursor, recolour or reorder the terminal and
    hide what is being approved. Newlines and tabs are kept."""
    out = []
    for ch in text:
        if ch in "\n\t" or ch in _ALLOWED_FORMAT_CHARS:
            out.append(ch)
        elif ch.isprintable() and unicodedata.category(ch) != "Cf":
            out.append(ch)
        else:
            code = ord(ch)
            out.append(f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}")
    return "".join(out)


def _arguments_text(req: ApprovalRequest) -> str:
    return _safe(json.dumps(req.arguments, ensure_ascii=False, indent=2, sort_keys=True))


def _pending_approvals(store: ProjectStore, project_id: str) -> list[ApprovalRequest]:
    return store.approvals(project_id).list(ApprovalStatus.PENDING)


def _approval_state(approvals: ApprovalStore, req: ApprovalRequest) -> str:
    if req.decided and approvals.is_used(req.id):
        return f"{req.status.value}, used"
    return req.status.value


def _approval_json(store: ProjectStore, req: ApprovalRequest) -> dict[str, Any]:
    approvals = store.approvals(req.project_id)
    data = req.model_dump(mode="json")
    data["used"] = req.decided and approvals.is_used(req.id)
    data["path"] = str(approvals.directory / f"{req.id}.json")
    return data


def _approval_summary(req: ApprovalRequest) -> str:
    task = _safe(req.task_id or "-")
    return f"{req.id}  {req.status.value:<8}  task {task}  {req.tool} ({req.risk.value})"


def _approval_lines(
    approvals: ApprovalStore, req: ApprovalRequest, *, full: bool = False
) -> list[str]:
    lines = [
        f"{req.id}  [{_approval_state(approvals, req)}]",
        f"  project: {req.project_id}",
        f"  task: {_safe(req.task_id or '-')}",
        f"  tool: {req.tool} (risk: {req.risk.value})",
        f"  requested: {req.created_at.isoformat(timespec='seconds')}",
        f"  reason: {_safe(_preview(req.reason, 200))}",
    ]
    if req.decided_at is not None:
        lines.append(f"  decided: {req.decided_at.isoformat(timespec='seconds')}")
    if req.note:
        lines.append(f"  note: {_safe(req.note)}")
    args = _arguments_text(req)
    truncated = not full and len(args) > ARGUMENTS_PREVIEW_CHARS
    if truncated:
        args = args[:ARGUMENTS_PREVIEW_CHARS]
    lines.append("  arguments:")
    lines.extend(f"    {line}" for line in args.splitlines())
    if truncated:
        lines.append(
            f"    … (truncated; full arguments: {approvals.directory / (req.id + '.json')})"
        )
    return lines


@app.command()
def approvals(
    project_id: Annotated[
        str | None, typer.Argument(help="Project id; omit for all projects.")
    ] = None,
    show_all: Annotated[
        bool, typer.Option("--all", help="Also show decided requests (default: pending only).")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to automa.toml.")
    ] = None,
) -> None:
    """List Approval Gate requests (with their exact arguments). Read-only."""
    config = _load(config_path)
    store = ProjectStore.for_workspace(config.workspace)
    status = None if show_all else ApprovalStatus.PENDING
    try:
        if project_id is not None:
            pids = [store.load(project_id).id]
        else:
            pids = [p.id for p in store.list_projects()] if store.root.is_dir() else []
        found = [(pid, store.approvals(pid).list(status)) for pid in pids]
        if as_json:
            rows = [_approval_json(store, r) for _, reqs in found for r in reqs]
            typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
            return
        total = 0
        for pid, reqs in found:
            approvals_store = store.approvals(pid)
            for req in reqs:
                if total:
                    typer.echo("")
                for line in _approval_lines(approvals_store, req):
                    typer.echo(line)
                if not req.decided:
                    typer.echo(
                        f"  decide: automa approve {pid} {req.id} | automa reject {pid} {req.id}"
                    )
                total += 1
    except (ProjectStoreError, ApprovalError) as e:
        raise _fail(str(e), EXIT_USER) from None
    if total == 0:
        typer.echo("no approval requests" if show_all else "no pending approval requests")


def _decide(
    project_id: str,
    approval_id: str,
    *,
    approve: bool,
    note: str | None,
    yes: bool,
    config_path: Path | None,
) -> None:
    config = _load(config_path)
    store = ProjectStore.for_workspace(config.workspace)
    try:
        pid = store.load(project_id).id
        approvals_store = store.approvals(pid)
        req = approvals_store.get(approval_id)
    except (ProjectStoreError, ApprovalError) as e:
        raise _fail(str(e), EXIT_USER) from None
    if req.decided:
        raise _fail(f"approval request {req.id} is already {req.status.value}", EXIT_USER)
    if note is not None:
        note = note.strip() or None
    if approve:
        # The user approves exactly what they see: show the full request first.
        for line in _approval_lines(approvals_store, req, full=True):
            typer.echo(line)
        if not yes and not typer.confirm(
            f"Approve running {req.tool} once with exactly these arguments?", default=False
        ):
            raise _fail("not approved; nothing was recorded", EXIT_USER)
    try:
        decided = approvals_store.decide(req.id, approve=approve, note=note)
        store.events(pid).emit(
            Event(
                ts=decided.decided_at or decided.created_at,
                type="approval.decided",
                data={
                    "approval": decided.id,
                    "task": decided.task_id,
                    "tool": decided.tool,
                    "status": decided.status.value,
                    "note": decided.note is not None,
                },
            )
        )
    except (ProjectStoreError, ApprovalError) as e:
        raise _fail(str(e), EXIT_USER) from None
    typer.echo(f"{decided.id}: {decided.status.value}")
    if approve:
        typer.echo(f"the call runs at most once, on the next: automa run {pid}")
    else:
        typer.echo(f"the model will be told it was rejected on the next: automa run {pid}")


_NOTE_OPTION = typer.Option("--note", help="Note recorded with the decision (shown to the model).")
_CONFIG_OPTION = typer.Option("--config", help="Path to automa.toml.")


@app.command()
def approve(
    project_id: Annotated[str, typer.Argument(help="Project of the request.")],
    approval_id: Annotated[str, typer.Argument(help="Request id (see: automa approvals).")],
    note: Annotated[str | None, _NOTE_OPTION] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
    config_path: Annotated[Path | None, _CONFIG_OPTION] = None,
) -> None:
    """Approve one pending request: that exact call may run once on the next run."""
    _decide(project_id, approval_id, approve=True, note=note, yes=yes, config_path=config_path)


@app.command()
def reject(
    project_id: Annotated[str, typer.Argument(help="Project of the request.")],
    approval_id: Annotated[str, typer.Argument(help="Request id (see: automa approvals).")],
    note: Annotated[str | None, _NOTE_OPTION] = None,
    config_path: Annotated[Path | None, _CONFIG_OPTION] = None,
) -> None:
    """Reject one pending request: the call never runs; the model is told why."""
    _decide(project_id, approval_id, approve=False, note=note, yes=True, config_path=config_path)


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
        roles, skills = _roles_and_skills(config)
    except (ConfigError, RoleError, SkillError) as e:
        raise _fail(str(e), EXIT_CONFIG) from None
    _require_sandbox(config)
    try:
        prepare_results_dir(results, benchmark.id)
    except BenchmarkResultError as e:
        raise _fail(str(e), EXIT_USER) from None

    store = ProjectStore.for_workspace(config.workspace)
    typer.echo(f"benchmark: {benchmark.id} ({benchmark.kind.value}) — {benchmark.title}")
    typer.echo(f"sandbox: {config.sandbox.describe()}")
    with CostLedger.for_workspace(config.workspace) as ledger:
        try:
            record = run_benchmark(
                benchmark,
                config=config,
                provider=provider,
                store=store,
                ledger=ledger,
                roles=roles,
                skills=skills,
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


def _bench_report_lines(report: BenchmarkReport) -> list[str]:
    lines: list[str] = []
    for s in report.benchmarks:
        note = "" if s.in_suite else "  [not in suite]"
        lines.append(f"{s.benchmark_id} — {s.title}{note}")
        lines.append(
            f"  runs: {s.runs}, passed: {s.passed_runs} ({s.pass_rate:.0%}), "
            f"mean score {s.mean_score:.2f}, mean attempts {s.mean_attempts:.1f}"
        )
        lines.append(
            f"  cost: total {_usd(s.total_cost_nusd)}, mean {_usd(s.mean_cost_nusd)} per run"
        )
        lines.append("  stops: " + ", ".join(f"{k} {v}" for k, v in s.stops.items()))
        if s.failure_causes:
            lines.append(
                "  failure causes: " + ", ".join(f"{k} {v}" for k, v in s.failure_causes.items())
            )
        for c in s.criteria_failures:
            lines.append(
                f"  criterion {c.criterion} failed in {c.failed_runs}/{s.runs} run(s) "
                f"({c.failed_in_completed_runs} completed): {_preview(c.description, 70)}"
            )
        last = "PASSED" if s.last_passed else "FAILED"
        lines.append(
            f"  last: {s.last_project_id} {last} at {s.last_run_at.isoformat(timespec='seconds')}"
        )
        lines.append(f"  versions: {', '.join(s.versions)}")
    for p in report.problems:
        lines.append(f"problem: {p.path}: {p.reason}")
    lines.append(
        f"total: {report.total_runs} run(s), {report.total_passed} passed, "
        f"cost {_usd(report.total_cost_nusd)}"
    )
    return lines


@bench_app.command("report")
def bench_report(
    benchmark_id: Annotated[
        str | None, typer.Option("--benchmark", help="Only this benchmark id.")
    ] = None,
    suite_dir: Annotated[
        Path, typer.Option("--suite", help="Directory of <id>.toml benchmark files.")
    ] = DEFAULT_SUITE_DIR,
    results_dir: Annotated[
        Path | None,
        typer.Option("--results", help="Recorded results (default: <suite>/results)."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Analyse recorded benchmark runs: pass rate, cost, stops, most-failed criteria.

    Read-only: needs no config or API key, makes no model call, writes nothing.
    Exits 1 if any result file could not be used (listed as a problem).
    """
    suite: BenchmarkSuite | None
    try:
        suite = BenchmarkSuite.from_dir(suite_dir)
    except BenchmarkError as e:
        if results_dir is None:
            raise _fail(f"benchmark suite: {e}", EXIT_USER) from None
        suite = None  # descriptions are optional when the results are given explicitly
    results = results_dir if results_dir is not None else suite_dir / RESULTS_DIRNAME
    try:
        scan = scan_results(results, benchmark_id=benchmark_id)
    except BenchmarkResultError as e:
        raise _fail(str(e), EXIT_USER) from None
    report = build_report(scan.records, suite=suite, problems=scan.problems)
    if as_json:
        typer.echo(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    elif report.total_runs == 0 and not report.problems:
        typer.echo("no benchmark results recorded yet (run: automa bench run <id>)")
    else:
        for line in _bench_report_lines(report):
            typer.echo(line)
    raise typer.Exit(EXIT_USER if report.problems else EXIT_OK)


def main() -> None:
    """Console-script entry point."""
    app()
