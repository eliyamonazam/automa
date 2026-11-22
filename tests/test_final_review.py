"""Final review of the whole product before ``done`` (D168).

Real orchestrator, worker and verifier with a scripted FakeProvider (Rig); no
network, no real model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from automa.delivery import DeliveryState, build_delivery, render_markdown
from automa.final_review import (
    FIX_TASK_PREFIX,
    load_review,
    parse_reply,
    review_message,
)
from automa.orchestrator import StopReason
from automa.plan import Plan, TaskStatus
from automa.project import ProjectStatus
from test_orchestrator import PID, Rig, task_dict, write

OK = json.dumps({"passed": True, "issues": []})


def bad(*issues: str) -> str:
    return json.dumps({"passed": False, "issues": list(issues)})


def plan(**extra: Any) -> Plan:
    return Plan.model_validate(
        {
            "goal": "یک فایل b بساز",
            "tasks": [task_dict("a"), task_dict("b", deps=["a"])],
            "deliverable": {
                "summary": "فایل b",
                "main_files": ["b.txt"],
                "how_to_use": "بازش کن",
            },
        }
        | extra
    )


def review_requests(rig: Rig) -> list[Any]:
    return [r for r in rig.provider.requests if "finished product" in (r.messages[0].content or "")]


def test_passing_review_marks_done_once(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [*write("a"), *write("b"), OK], plan())
    report = rig.run()
    assert report.status is ProjectStatus.DONE and report.reason is StopReason.COMPLETED
    reqs = review_requests(rig)
    assert len(reqs) == 1
    user = reqs[0].messages[1].content or ""
    assert "یک فایل b بساز" in user and "فایل b" in user and "`b.txt`" in user and "ok" in user
    rev = load_review(rig.store.project_dir(PID))
    assert rev is not None and rev.passed and rev.plan_version == 1
    events = [e.type for e in rig.store.events(PID).read()]
    assert "final_review.started" in events and "final_review.finished" in events
    # running a done project again does not review again
    n = len(rig.provider.requests)
    rig.run()
    assert len(rig.provider.requests) == n
    assert build_delivery(rig.store, PID).state is DeliveryState.READY  # type: ignore[union-attr]


def test_failed_review_adds_one_fix_task_then_passes(tmp_path: Path) -> None:
    script = [
        *write("a"),
        *write("b"),
        bad("نام‌ها ساختگی‌اند"),
        *write(FIX_TASK_PREFIX),  # the fix task writes final-fix.txt ... and checks b.txt
        "unused",
    ]
    rig = Rig(tmp_path, script[:-1], plan())
    # the fix task's model-judged criterion needs a verdict
    verdict = json.dumps({"criteria": [{"criterion": 0, "passed": True, "reason": "fixed"}]})
    rig.provider.push(verdict, OK)
    report = rig.run()
    assert report.status is ProjectStatus.DONE, report
    p = rig.store.load_plan(PID)
    assert p.version == 2 and p.tasks[-1].id == FIX_TASK_PREFIX
    fix = p.tasks[-1]
    assert fix.depends_on == ["a", "b"] and fix.role == "writer"
    assert "نام‌ها ساختگی‌اند" in fix.description and "ask_user" in fix.description
    assert fix.title == "رفع ایرادهای بازبینی نهایی محصول"
    assert any(c.check is not None and c.check.path == "b.txt" for c in fix.acceptance_criteria)
    assert rig.state(FIX_TASK_PREFIX).status is TaskStatus.PASSED
    rev = load_review(rig.store.project_dir(PID))
    assert rev is not None and rev.passed and rev.plan_version == 2 and rev.fix_round == 1
    assert len(review_requests(rig)) == 2


def test_second_failure_is_shown_not_looped(tmp_path: Path) -> None:
    verdict = json.dumps({"criteria": [{"criterion": 0, "passed": True, "reason": "ok"}]})
    script = [
        *write("a"),
        *write("b"),
        bad("problem one"),
        *write(FIX_TASK_PREFIX),
        verdict,
        bad("still a problem"),
    ]
    rig = Rig(tmp_path, script, plan(goal="make b"))
    report = rig.run()
    assert report.status is ProjectStatus.DONE
    assert rig.store.load_plan(PID).version == 2  # no second fix task
    d = build_delivery(rig.store, PID)
    assert d is not None and d.state is DeliveryState.PARTIAL
    assert d.reviewed is False and d.review_issues == ("still a problem",)
    md = render_markdown(d, "en")
    assert "Final review: still a problem" in md
    assert rig.store.load_plan(PID).tasks[-1].title == "Fix the final review issues"


def test_unusable_reply_does_not_block(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [*write("a"), *write("b"), "no json here", "still nothing"], plan())
    assert rig.run().status is ProjectStatus.DONE
    rev = load_review(rig.store.project_dir(PID))
    assert rev is not None and rev.unavailable
    d = build_delivery(rig.store, PID)
    assert d is not None and d.state is DeliveryState.READY and d.reviewed is None


def test_budget_stop_during_review_pauses_and_retries_later(tmp_path: Path) -> None:
    from automa.budget import BudgetExceeded
    from automa.budget.guard import BudgetScope

    rig = Rig(tmp_path, [*write("a"), *write("b")], plan())
    rig.provider.push(
        BudgetExceeded(BudgetScope.PROJECT, limit_nusd=1, spent_nusd=1, requested_nusd=1)
    )
    report = rig.run()
    assert report.status is ProjectStatus.PAUSED and report.reason is StopReason.BUDGET
    assert load_review(rig.store.project_dir(PID)) is None
    rig.provider.push(OK)
    assert rig.run().status is ProjectStatus.DONE


def test_no_review_without_deliverable_or_when_off(tmp_path: Path) -> None:
    from test_orchestrator import make_plan

    rig = Rig(tmp_path, [*write("a")], make_plan(task_dict("a")))
    assert rig.run().status is ProjectStatus.DONE
    assert review_requests(rig) == []
    rig2 = Rig(tmp_path / "x", [*write("a"), *write("b")], plan())
    assert rig2.run(final_review=False).status is ProjectStatus.DONE
    assert review_requests(rig2) == []


def test_parse_reply_and_message() -> None:
    assert parse_reply('```json\n{"passed": true}\n```').passed
    r = parse_reply('text {"passed": false, "issues": ["  a  b ", "", "c"]} end')
    assert r.issues == ["a b", "c"]
    with pytest.raises(ValueError):
        parse_reply('{"passed": false, "issues": []}')
    with pytest.raises(ValueError):
        parse_reply("nothing")
    msg = review_message(plan(), [("b.txt", "hello")], ["Q: name?\nA: Ali"])
    assert "Ali" in msg and "<file>\nhello\n</file>" in msg
    assert "(nothing beyond the goal)" in review_message(plan(), [], [])
