"""The Executive is told to size the plan to the work (M13, D134).

The owner's first real project (one small HTML file) was split into four tasks,
including a separate ``final-review`` task, and spent $0.35 on execution. Each
task re-reads the files and gets its own evaluation, so the plan prompt now asks
for the fewest tasks, one task for a small goal and no separate review task.
Local only: FakeProvider behind a real LLMGateway + BudgetGuard + CostLedger.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from automa.agents import ExecutiveAgent
from automa.agents.executive import PLAN_SYSTEM_PROMPT
from automa.budget import BudgetGuard, CostLedger
from automa.config import AutomaConfig
from automa.llm import FakeProvider, LLMGateway, Role


def _config() -> AutomaConfig:
    return AutomaConfig.model_validate(
        {
            "tiers": {"small": "fake-small", "medium": "fake-medium", "large": "fake-large"},
            "prices": {
                m: {"input_per_mtok": 1.0, "output_per_mtok": 2.0}
                for m in ("fake-small", "fake-medium", "fake-large")
            },
        }
    )


ONE_TASK_PLAN = json.dumps(
    {
        "goal": "g",
        "tasks": [
            {
                "id": "build-page",
                "title": "Build the page",
                "description": "Write index.html with a pomodoro timer.",
                "role": "developer",
                "acceptance_criteria": [
                    {
                        "description": "index.html exists",
                        "check": {"type": "file_exists", "path": "index.html"},
                    },
                    {
                        "description": "has a start button",
                        "check": {"type": "file_contains", "path": "index.html", "text": "Start"},
                    },
                    {"description": "the timer counts down from 25:00"},
                ],
            }
        ],
    }
)


def _rig(tmp_path: Path, script: list[Any]) -> tuple[FakeProvider, LLMGateway, CostLedger]:
    cfg = _config()
    provider = FakeProvider(script)
    ledger = CostLedger(tmp_path / "ledger.sqlite3")
    gateway = LLMGateway(cfg, provider, guard=BudgetGuard(cfg.limits, ledger))
    return provider, gateway, ledger


def _plan_section() -> str:
    return PLAN_SYSTEM_PROMPT.split("Plan size", 1)[1].lower()


def test_prompt_asks_for_the_fewest_tasks() -> None:
    assert "as FEW tasks as will do the work correctly" in PLAN_SYSTEM_PROMPT
    assert "smallest number of tasks" in PLAN_SYSTEM_PROMPT


def test_prompt_no_longer_asks_for_many_small_tasks() -> None:
    assert "small, independently verifiable tasks" not in PLAN_SYSTEM_PROMPT


def test_small_goal_is_one_task() -> None:
    assert "one or two files" in _plan_section()
    assert "is one task" in _plan_section()


@pytest.mark.parametrize("word", ["final review", "final test", "qa", "verification"])
def test_prompt_forbids_separate_review_tasks(word: str) -> None:
    section = _plan_section()
    assert "never add a separate" in section
    assert f'"{word}"' in section


def test_prompt_says_the_verifier_checks_every_task() -> None:
    assert "Verifier already checks every task" in PLAN_SYSTEM_PROMPT


def test_prompt_keeps_criteria_with_the_task_that_makes_the_files() -> None:
    assert "into the task that makes" in PLAN_SYSTEM_PROMPT


def test_prompt_explains_why_tasks_cost_money() -> None:
    assert "every task costs money" in PLAN_SYSTEM_PROMPT


def test_earlier_rules_are_kept() -> None:
    # D92 and the Verifier rules must survive the rewrite.
    assert "WITHOUT a shell" in PLAN_SYSTEM_PROMPT
    assert "python3 -c" in PLAN_SYSTEM_PROMPT
    assert "The Verifier reads" in PLAN_SYSTEM_PROMPT
    assert "JSON must not contain comments" in PLAN_SYSTEM_PROMPT


def test_size_rules_reach_the_model_through_the_gateway(tmp_path: Path) -> None:
    provider, gateway, ledger = _rig(tmp_path, [ONE_TASK_PLAN])
    agent = ExecutiveAgent(gateway, known_roles={"developer", "writer"})
    plan = agent.plan("one HTML page with a pomodoro timer", project_id="p1")
    assert plan.task_ids == ["build-page"]
    assert len(plan.tasks[0].acceptance_criteria) == 3
    system = provider.requests[0].messages[0]
    assert system.role is Role.SYSTEM
    text = system.content or ""
    assert "Never add a separate" in text
    assert "is ONE task" in text
    # Allowed roles still follow the prompt.
    assert text.rstrip().endswith("Allowed roles: developer, writer.")
    # The planning call is still metered by the Guard's ledger.
    assert len(ledger.entries()) == 1


def test_a_one_task_plan_is_valid(tmp_path: Path) -> None:
    provider, gateway, _ = _rig(tmp_path, [ONE_TASK_PLAN])
    plan = ExecutiveAgent(gateway).plan("g", project_id="p1")
    assert len(plan.tasks) == 1
    assert len(provider.requests) == 1  # accepted on the first try, no retry
