"""Benchmark runner: plan, run and judge one benchmark as a fresh project.

Flow of ``run_benchmark`` (see D27):

1. create a new project for the benchmark's goal (never reuses one);
2. the Executive asks its clarifying questions; each one is answered with the
   benchmark's fixed ``answer`` (runs are unattended) and the plan is written;
3. the ``Orchestrator`` runs the plan with ``max_attempts`` from the benchmark;
4. the benchmark's success criteria are evaluated on the project's ``files/``
   with ``evaluate_benchmark`` (deterministic checks only, whatever the stop);
5. a ``BenchmarkRunRecord`` is returned (the caller writes it).

Every model call goes through one ``LLMGateway`` + ``BudgetGuard`` whose limits
are ``benchmark_limits``: the project cap is the *smaller* of the user's cap and
the benchmark's ``budget_usd`` and the per-call cap is lowered to match. The
benchmark can only tighten the user's limits, never loosen them.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

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
from automa.benchmarks import Benchmark, evaluate_benchmark
from automa.benchmarks.results import BenchmarkRunRecord, BenchmarkStop
from automa.budget.guard import BudgetError, BudgetGuard
from automa.budget.ledger import CostLedger
from automa.config import AutomaConfig, BudgetLimits, ConfigError
from automa.events import Event
from automa.llm.gateway import LLMGateway
from automa.llm.provider import Provider, ProviderError
from automa.orchestrator import Orchestrator, OrchestratorSettings, StopReason
from automa.project import ProjectStatus, ProjectStore
from automa.quality import QualityLog
from automa.roles import RoleRegistry
from automa.skills import SkillRegistry

Clock = Callable[[], datetime]
PROJECT_PREFIX = "bench"
_MAX_ID = 64

STOP_FOR_REASON: dict[StopReason, BenchmarkStop] = {
    StopReason.COMPLETED: BenchmarkStop.COMPLETED,
    StopReason.TASKS_FAILED: BenchmarkStop.TASKS_FAILED,
    StopReason.BUDGET: BenchmarkStop.BUDGET,
    StopReason.APPROVAL: BenchmarkStop.APPROVAL,
    StopReason.PROVIDER: BenchmarkStop.PROVIDER,
    StopReason.VERIFICATION: BenchmarkStop.VERIFICATION,
    StopReason.TOOL_LAYER: BenchmarkStop.TOOL_LAYER,
    StopReason.CONFIG: BenchmarkStop.CONFIG,
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def benchmark_limits(limits: BudgetLimits, benchmark: Benchmark) -> BudgetLimits:
    """The user's limits, tightened by the benchmark's budget (never loosened)."""
    project = min(limits.project_usd, benchmark.budget_usd)
    return BudgetLimits(
        project_usd=project,
        daily_usd=limits.daily_usd,
        per_call_usd=min(limits.per_call_usd, project),
    )


def benchmark_project_id(benchmark_id: str, now: datetime) -> str:
    """``bench-<UTC timestamp>-<benchmark id>``, cut to the project id length limit."""
    stamp = now.astimezone(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{PROJECT_PREFIX}-{stamp}-{benchmark_id}"[:_MAX_ID].rstrip("-")


def _plan(
    agent: ExecutiveAgent, store: ProjectStore, pid: str, benchmark: Benchmark
) -> tuple[BenchmarkStop | None, str]:
    """Clarify (answered with ``benchmark.answer``) and plan. Returns (stop, detail)."""
    goal = store.load(pid).goal
    try:
        try:
            questions = agent.clarify(goal, project_id=pid)
        except BudgetError:
            store.set_status(pid, ProjectStatus.PAUSED)
            raise
        except PlanningFailed:
            store.set_status(pid, ProjectStatus.FAILED)
            raise
        items = [Clarification(question=q, answer=benchmark.answer) for q in questions]
        if items:
            store.save_clarifications(pid, items)
        plan_project(agent, store, pid, clarifications=items)
    except BudgetError as e:
        return BenchmarkStop.BUDGET, str(e)
    except PlanningFailed as e:
        return BenchmarkStop.PLANNING, str(e)
    except ProviderError as e:
        return BenchmarkStop.PROVIDER, str(e)
    except ConfigError as e:
        return BenchmarkStop.CONFIG, str(e)
    return None, ""


def run_benchmark(
    benchmark: Benchmark,
    *,
    config: AutomaConfig,
    provider: Provider,
    store: ProjectStore,
    ledger: CostLedger,
    roles: RoleRegistry,
    skills: SkillRegistry | None = None,
    quality: QualityLog | None = None,
    project_id: str | None = None,
    clock: Clock | None = None,
) -> BenchmarkRunRecord:
    """Run ``benchmark`` end to end as a new project and return its record.

    Raises ``ProjectStoreError`` if the project cannot be created (nothing has been
    spent then). ``KeyboardInterrupt`` propagates (the project stays resumable but
    no record is produced: an interrupted run is not a measurement).
    """
    clock = clock or _utc_now
    started = clock()
    pid = project_id or benchmark_project_id(benchmark.id, started)
    store.create(benchmark.goal, project_id=pid)
    events = store.events(pid)
    limits = benchmark_limits(config.limits, benchmark)
    guard = BudgetGuard(limits, ledger, events=events)
    gateway = LLMGateway(config, provider, guard=guard)
    events.emit(
        Event(
            ts=clock(),
            type="benchmark.started",
            project_id=pid,
            data={"benchmark": benchmark.id, "budget_usd": limits.project_usd},
        )
    )

    executive = ExecutiveAgent(
        gateway, settings=ExecutiveSettings(), known_roles=roles.names(), events=events
    )
    stop, detail = _plan(executive, store, pid, benchmark)
    passed: list[str] = []
    failed: list[str] = []
    blocked: list[str] = []
    if stop is None:
        orchestrator = Orchestrator(
            store,
            WorkerAgent(gateway, events=events, skills=skills),
            VerifierAgent(gateway, events=events),
            roles,
            config,
            settings=OrchestratorSettings(max_attempts=benchmark.max_attempts),
            quality=quality,
        )
        report = orchestrator.run(pid)
        stop, detail = STOP_FOR_REASON[report.reason], report.detail
        passed, failed, blocked = list(report.passed), list(report.failed), list(report.blocked)

    planned = store.has_plan(pid)
    attempts = 0
    if planned:
        plan = store.load_plan(pid)
        attempts = sum(len(store.task_history(pid, t.id)) for t in plan.tasks)

    outcome = evaluate_benchmark(
        benchmark,
        store.files_dir(pid),
        events,
        project_id=pid,
        sandbox=config.sandbox.docker_sandbox(),
    )
    record = BenchmarkRunRecord.build(
        outcome=outcome,
        title=benchmark.title,
        kind=benchmark.kind,
        project_id=pid,
        automa_version=__version__,
        started_at=started,
        finished_at=max(clock(), started),
        stop=stop,
        detail=detail,
        project_status=store.load(pid).status,
        planned=planned,
        budget_usd=limits.project_usd,
        cost_nusd=ledger.spent_nusd(project_id=pid),
        model_calls=len(ledger.entries(project_id=pid)),
        attempts=attempts,
        tasks_passed=passed,
        tasks_failed=failed,
        tasks_blocked=blocked,
    )
    events.emit(
        Event(
            ts=clock(),
            type="benchmark.finished",
            project_id=pid,
            data={
                "benchmark": benchmark.id,
                "passed": record.passed,
                "score": record.score,
                "stop": record.stop.value,
            },
        )
    )
    return record
