"""Command checks that need a shell are rejected at planning time (D92).

Found by the first real benchmark run: the Executive wrote checks such as
``! grep -qi '@import' style.css`` and ``test $(grep -c ...) -ge 3``. Checks run
without a shell, so they could never pass and every worker attempt was wasted.
"""

from __future__ import annotations

import json

import pytest

from automa.plan.models import PlanError, parse_plan, shell_construct


@pytest.mark.parametrize(
    ("command", "found"),
    [
        ("! grep -qi '@import' style.css", "!"),
        ("test $(grep -c x index.html) -ge 3", "$(...)"),
        ("test `wc -l a`", "backticks"),
        ("grep -o x index.html | wc -l", "|"),
        ("pytest -q && ruff check .", "&&"),
        ("pytest -q; echo done", ";"),
        ("python3 app.py > out.txt", ">"),
        ("python3 app.py < in.txt", "<"),
        ("test -n $HOME", "$HOME"),
        ("pytest -q || true", "||"),
        ("sleep 1 &", "&"),
    ],
)
def test_shell_constructs_are_found(command: str, found: str) -> None:
    assert shell_construct(command) == found


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "grep -q '@import' style.css",
        "grep -qE 'https?://' style.css",
        "grep -cE '\\$[0-9]+\\.[0-9]{2}' index.html",
        "python3 -c 'import sys; sys.exit(0 if 3 <= 5 else 1)'",
        "python3 wc.py README.md",
        "grep -q 'a|b' file.txt",
        "grep -q 'price: $5' menu.html",
        "node --version",
    ],
)
def test_plain_commands_are_accepted(command: str) -> None:
    assert shell_construct(command) is None


def _plan(command: str) -> str:
    return json.dumps(
        {
            "goal": "g",
            "tasks": [
                {
                    "id": "css",
                    "title": "stylesheet",
                    "description": "write style.css",
                    "role": "developer",
                    "acceptance_criteria": [
                        {
                            "description": "no imports",
                            "check": {"type": "command", "command": command, "exit_code": 0},
                        }
                    ],
                }
            ],
        }
    )


def test_parse_plan_rejects_shell_checks_with_a_fixable_message() -> None:
    with pytest.raises(PlanError) as exc:
        parse_plan(_plan("! grep -qi '@import' style.css"))
    details = exc.value.details
    assert "tasks[css].acceptance_criteria.0.check.command" in details
    assert "'!'" in details
    assert "WITHOUT a shell" in details and "exit_code 1" in details


def test_parse_plan_accepts_plain_command_checks() -> None:
    plan = parse_plan(_plan("grep -q '@import' style.css"))
    assert plan.tasks[0].id == "css"


def test_stored_plans_with_shell_checks_still_load() -> None:
    # Only model replies are rejected; plan.json files written before D92 must
    # still load so old projects stay readable.
    from automa.plan.models import Plan

    data = json.loads(_plan("! grep -q x y"))
    assert Plan.model_validate(data).tasks[0].acceptance_criteria[0].check is not None


def test_executive_prompt_explains_commands_run_without_a_shell() -> None:
    from automa.agents.executive import PLAN_SYSTEM_PROMPT

    assert "WITHOUT a shell" in PLAN_SYSTEM_PROMPT
    assert "python3 -c" in PLAN_SYSTEM_PROMPT
