"""Tests for Plan/Task/TaskResult/Verdict (automa.plan) and plan.json in ProjectStore.

Local only: no network, no model calls. FakeProvider is used to feed model-like JSON.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from automa.budget import BudgetGuard, CostLedger
from automa.config import AutomaConfig, Tier
from automa.llm import CallContext, ChatResponse, FakeProvider, LLMGateway, Message, Usage
from automa.plan import (
    PLAN_SCHEMA_VERSION,
    AcceptanceCriterion,
    CheckResult,
    CommandCheck,
    FileContainsCheck,
    FileExistsCheck,
    Issue,
    Plan,
    PlanError,
    ResultStatus,
    Task,
    TaskResult,
    TaskStatus,
    Verdict,
    VerdictOutcome,
    parse_plan,
)
from automa.project import ProjectConflict, ProjectStore, ProjectStoreError

T0 = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)


def crit(desc: str = "works", check: Any = None) -> dict[str, Any]:
    d: dict[str, Any] = {"description": desc}
    if check is not None:
        d["check"] = check
    return d


def task(tid: str, *deps: str, **kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": tid,
        "title": f"Task {tid}",
        "role": "developer",
        "depends_on": list(deps),
        "acceptance_criteria": [crit()],
    }
    d.update(kw)
    return d


def plan_dict(*tasks: dict[str, Any], **kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"goal": "build a CLI", "tasks": list(tasks)}
    d.update(kw)
    return d


def make_plan(*tasks: dict[str, Any], **kw: Any) -> Plan:
    return Plan.model_validate(plan_dict(*tasks, **kw))


# --- tasks -------------------------------------------------------------------


@pytest.mark.parametrize("tid", ["a", "t1", "setup-db", "step_2", "x" * 64, "0"])
def test_task_ids_accepted(tid: str) -> None:
    assert Task.model_validate(task(tid)).id == tid


@pytest.mark.parametrize(
    "tid",
    ["", "T1", "-a", "a-", "_a", "a_", "../x", "a/b", "a b", ".", "..", "x" * 65, "تست", "a\x00"],
)
def test_task_ids_rejected(tid: str) -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(task(tid))


@pytest.mark.parametrize("role", ["Developer", "1dev", "", "dev/../x", "a" * 33, "dev role"])
def test_bad_role_rejected(role: str) -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(task("a", role=role))


def test_task_defaults_and_tier() -> None:
    t = Task.model_validate(task("a"))
    assert t.tier is None and t.inputs == {} and t.description == ""
    assert Task.model_validate(task("a", tier="large")).tier is Tier.LARGE
    with pytest.raises(ValidationError):
        Task.model_validate(task("a", tier="huge"))


@pytest.mark.parametrize("title", ["", "   ", "x" * 201])
def test_bad_title_rejected(title: str) -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(task("a", title=title))


def test_task_needs_acceptance_criteria() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(task("a", acceptance_criteria=[]))
    with pytest.raises(ValidationError):
        Task.model_validate({k: v for k, v in task("a").items() if k != "acceptance_criteria"})


def test_task_self_and_duplicate_dependency_rejected() -> None:
    with pytest.raises(ValidationError, match="itself"):
        Task.model_validate(task("a", "a"))
    with pytest.raises(ValidationError, match="duplicate"):
        Task.model_validate(task("a", "b", "b"))


def test_task_extra_fields_rejected_and_frozen() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(task("a", budget_usd=1000))
    t = Task.model_validate(task("a"))
    with pytest.raises(ValidationError):
        t.role = "admin"  # type: ignore[misc]


def test_task_inputs_keys_validated() -> None:
    assert Task.model_validate(task("a", inputs={"lang": "python"})).inputs == {"lang": "python"}
    for bad in ({"": "x"}, {"a-b": "x"}, {"1a": "x"}):
        with pytest.raises(ValidationError):
            Task.model_validate(task("a", inputs=bad))
    with pytest.raises(ValidationError):
        Task.model_validate(task("a", inputs={"k": 5}))


# --- checks ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "norm"),
    [("a.txt", "a.txt"), ("./src/main.py", "src/main.py"), ("a//b/./c", "a/b/c"), ("d/", "d")],
)
def test_check_paths_normalized(raw: str, norm: str) -> None:
    assert FileExistsCheck(path=raw).path == norm


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "./",
        "..",
        "../x",
        "a/../../x",
        "a/..",
        "/etc/passwd",
        "C:/x",
        "c:x",
        "a\\..\\b",
        "a\x00b",
        "x" * 256,
    ],
)
def test_check_paths_escaping_rejected(path: str) -> None:
    with pytest.raises(ValidationError):
        FileExistsCheck(path=path)
    with pytest.raises(ValidationError):
        FileContainsCheck(path=path, text="x")


def test_check_discriminated_union() -> None:
    c = AcceptanceCriterion.model_validate(
        crit("tests pass", {"type": "command", "command": "pytest -q"})
    )
    assert isinstance(c.check, CommandCheck)
    assert c.check.exit_code == 0 and c.check.timeout_s == 60 and c.deterministic
    c2 = AcceptanceCriterion.model_validate(
        crit("has readme", {"type": "file_contains", "path": "README.md", "text": "Usage"})
    )
    assert isinstance(c2.check, FileContainsCheck)
    assert not AcceptanceCriterion.model_validate(crit("readable")).deterministic
    for bad in ({"type": "http_get", "url": "x"}, {"path": "a"}, {"type": "file_exists"}):
        with pytest.raises(ValidationError):
            AcceptanceCriterion.model_validate(crit("x", bad))


@pytest.mark.parametrize(
    "kw",
    [
        {"command": ""},
        {"command": "   "},
        {"command": "ls\nrm -rf ~"},
        {"command": "ls\r"},
        {"command": "a\x00"},
        {"command": "x" * 2001},
        {"command": "ls", "exit_code": -1},
        {"command": "ls", "exit_code": 256},
        {"command": "ls", "timeout_s": 0},
        {"command": "ls", "timeout_s": 601},
    ],
)
def test_command_check_limits(kw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CommandCheck(**kw)


def test_file_contains_needs_text() -> None:
    with pytest.raises(ValidationError):
        FileContainsCheck(path="a", text="")


# --- plan graph --------------------------------------------------------------


def test_plan_valid_and_defaults() -> None:
    p = make_plan(task("a"), task("b", "a"), task("c", "a", "b"))
    assert p.schema_version == PLAN_SCHEMA_VERSION and p.version == 1
    assert p.task_ids == ["a", "b", "c"]
    assert p.task("b").depends_on == ["a"]
    with pytest.raises(KeyError):
        p.task("zz")


def test_plan_requires_tasks_and_goal() -> None:
    with pytest.raises(ValidationError):
        make_plan()
    with pytest.raises(ValidationError):
        make_plan(task("a"), goal="  ")
    with pytest.raises(ValidationError):
        make_plan(*[task(f"t{i}") for i in range(101)])
    assert len(make_plan(*[task(f"t{i}") for i in range(100)]).tasks) == 100


def test_plan_duplicate_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate task ids: a"):
        make_plan(task("a"), task("b"), task("a"))


def test_plan_unknown_dependency_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown task"):
        make_plan(task("a"), task("b", "ghost"))


@pytest.mark.parametrize(
    "tasks",
    [
        [task("a", "b"), task("b", "a")],
        [task("a", "c"), task("b", "a"), task("c", "b")],
        [task("x"), task("a", "x", "d"), task("b", "a"), task("c", "b"), task("d", "c")],
    ],
)
def test_plan_cycles_rejected(tasks: list[dict[str, Any]]) -> None:
    with pytest.raises(ValidationError, match="cycle"):
        make_plan(*tasks)


def test_plan_cycle_message_names_cycle() -> None:
    with pytest.raises(ValidationError) as ei:
        make_plan(task("a", "b"), task("b", "a"))
    assert "a -> b -> a" in str(ei.value) or "b -> a -> b" in str(ei.value)


def test_long_chain_no_recursion_limit() -> None:
    tasks = [task("t0")] + [task(f"t{i}", f"t{i - 1}") for i in range(1, 100)]
    p = make_plan(*tasks)
    assert [t.id for t in p.topological_order()] == [f"t{i}" for i in range(100)]
    # Chain closed into a cycle is still detected.
    cyc = [task("t0", "t99")] + tasks[1:]
    with pytest.raises(ValidationError, match="cycle"):
        make_plan(*cyc)


def test_topological_order_respects_deps_and_plan_order() -> None:
    p = make_plan(task("d", "b", "c"), task("c", "a"), task("b", "a"), task("a"), task("e"))
    order = [t.id for t in p.topological_order()]
    assert order == ["a", "e", "c", "b", "d"]
    pos = {tid: i for i, tid in enumerate(order)}
    for t in p.tasks:
        assert all(pos[d] < pos[t.id] for d in t.depends_on)


def test_ready_tasks_and_dependents() -> None:
    p = make_plan(task("a"), task("b", "a"), task("c", "a"), task("d", "b", "c"), task("e"))
    assert [t.id for t in p.ready_tasks(set())] == ["a", "e"]
    assert [t.id for t in p.ready_tasks({"a"})] == ["b", "c", "e"]
    assert [t.id for t in p.ready_tasks({"a", "b", "c", "e"})] == ["d"]
    assert p.ready_tasks(set(p.task_ids)) == []
    with pytest.raises(KeyError):
        p.ready_tasks({"ghost"})
    assert [t.id for t in p.dependents("a")] == ["b", "c", "d"]
    assert [t.id for t in p.dependents("b")] == ["d"]
    assert p.dependents("e") == []
    with pytest.raises(KeyError):
        p.dependents("ghost")


def test_plan_unknown_schema_version_rejected() -> None:
    with pytest.raises(ValidationError, match="schema version"):
        make_plan(task("a"), schema_version=2)
    with pytest.raises(ValidationError):
        make_plan(task("a"), version=0)


def test_plan_roundtrip_json() -> None:
    p = make_plan(
        task(
            "a",
            acceptance_criteria=[
                crit("file", {"type": "file_exists", "path": "./out/a.txt"}),
                crit("خوانا باشه"),
            ],
            inputs={"lang": "fa"},
            tier="small",
        ),
        task("b", "a"),
        goal="یک ابزار CLI ساده",
    )
    again = Plan.model_validate_json(p.model_dump_json())
    assert again == p
    assert again.tasks[0].acceptance_criteria[0].check == FileExistsCheck(path="out/a.txt")


def test_plan_json_schema_exports() -> None:
    schema = Plan.model_json_schema()
    assert "tasks" in schema["properties"]
    json.dumps(schema)  # serializable, can be embedded in a prompt


# --- parse_plan (model output) -----------------------------------------------


def test_parse_plan_plain_and_fenced() -> None:
    raw = json.dumps(plan_dict(task("a"), task("b", "a")), ensure_ascii=False)
    assert parse_plan(raw).task_ids == ["a", "b"]
    assert parse_plan(f"```json\n{raw}\n```").task_ids == ["a", "b"]
    assert parse_plan(f"  \n```\n{raw}\n```\n").task_ids == ["a", "b"]


@pytest.mark.parametrize(
    "text",
    ["", "not json", "{", "[]", "null", "42", '"plan"', "Here is the plan: {}", "```json\n{\n```"],
)
def test_parse_plan_invalid_json(text: str) -> None:
    with pytest.raises(PlanError) as ei:
        parse_plan(text)
    assert ei.value.details.startswith("- <root>")


def test_parse_plan_validation_details_are_model_readable() -> None:
    bad = plan_dict(task("A"), task("b", "ghost"))
    with pytest.raises(PlanError) as ei:
        parse_plan(json.dumps(bad))
    details = ei.value.details
    assert "tasks.0.id" in details
    assert details.count("\n") < 20
    # Inputs are never echoed back (they may be large or contain anything).
    assert "Task A" not in details


def test_parse_plan_graph_error_detail() -> None:
    with pytest.raises(PlanError) as ei:
        parse_plan(json.dumps(plan_dict(task("a", "b"), task("b", "a"))))
    assert "cycle" in ei.value.details


def test_parse_plan_caps_error_count() -> None:
    bad = plan_dict(*[task(f"T{i}") for i in range(60)])
    with pytest.raises(PlanError) as ei:
        parse_plan(json.dumps(bad))
    assert len(ei.value.details.splitlines()) == 20


# --- TaskResult --------------------------------------------------------------


def result(**kw: Any) -> TaskResult:
    d: dict[str, Any] = {
        "task_id": "a",
        "attempt": 1,
        "status": "completed",
        "tier": "small",
        "model": "fake-small",
        "started_at": T0,
        "finished_at": T0 + timedelta(seconds=5),
    }
    d.update(kw)
    return TaskResult.model_validate(d)


def test_task_result_valid() -> None:
    r = result(files=["./src/a.py", "README.md"], steps=3, cost_nusd=1234, summary="ok")
    assert r.files == ["src/a.py", "README.md"]
    assert r.status is ResultStatus.COMPLETED
    assert TaskResult.model_validate_json(r.model_dump_json()) == r


def test_task_result_timestamps_normalized_to_utc() -> None:
    tz = timezone(timedelta(hours=3, minutes=30))
    r = result(started_at=T0.astimezone(tz), finished_at=T0.astimezone(tz))
    assert r.started_at.tzinfo is UTC and r.started_at == T0


@pytest.mark.parametrize(
    "kw",
    [
        {"attempt": 0},
        {"steps": -1},
        {"cost_nusd": -1},
        {"cost_nusd": 1.5},
        {"model": ""},
        {"files": ["../x"]},
        {"files": ["/etc/passwd"]},
        {"files": ["a", "./a"]},
        {"status": "done"},
        {"status": "failed"},
        {"status": "failed", "error": "  "},
        {"finished_at": T0 - timedelta(seconds=1)},
        {"started_at": datetime(2026, 11, 9, 12, 0)},
        {"task_id": "../a"},
        {"extra": 1},
    ],
)
def test_task_result_rejects(kw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        result(**kw)


def test_failed_result_with_error_ok() -> None:
    assert result(status="failed", error="tool crashed").error == "tool crashed"
    assert result(status="step_limit").status is ResultStatus.STEP_LIMIT


# --- Verdict -----------------------------------------------------------------


def verdict(**kw: Any) -> Verdict:
    d: dict[str, Any] = {"task_id": "a", "attempt": 1, "outcome": "pass", "created_at": T0}
    d.update(kw)
    return Verdict.model_validate(d)


def test_verdict_pass() -> None:
    v = verdict(checks=[{"criterion": 0, "passed": True}], model_evaluated=True)
    assert v.passed and v.outcome is VerdictOutcome.PASS
    assert Verdict.model_validate_json(v.model_dump_json()) == v


def test_verdict_fail_needs_reason() -> None:
    with pytest.raises(ValidationError, match="needs"):
        verdict(outcome="fail")
    with pytest.raises(ValidationError, match="needs"):
        verdict(outcome="fail", checks=[{"criterion": 0, "passed": True}])
    v = verdict(outcome="fail", checks=[{"criterion": 0, "passed": False, "detail": "missing"}])
    assert not v.passed and not v.model_evaluated
    v2 = verdict(outcome="fail", issues=[{"description": "wrong output", "criterion": 1}])
    assert v2.issues == [Issue(description="wrong output", criterion=1)]


def test_verdict_pass_cannot_hide_failures() -> None:
    with pytest.raises(ValidationError, match="failed check"):
        verdict(checks=[{"criterion": 0, "passed": True}, {"criterion": 1, "passed": False}])
    with pytest.raises(ValidationError, match="issues"):
        verdict(issues=[{"description": "meh"}])


@pytest.mark.parametrize(
    "kw",
    [
        {"outcome": "maybe"},
        {"attempt": 0},
        {"checks": [{"criterion": -1, "passed": True}]},
        {"checks": [{"criterion": 0, "passed": True}, {"criterion": 0, "passed": True}]},
        {"outcome": "fail", "issues": [{"description": " "}]},
        {"outcome": "fail", "issues": [{"description": "x", "criterion": -1}]},
        {"created_at": datetime(2026, 11, 9)},
        {"checks": [{"criterion": 0, "passed": "yes please"}]},
    ],
)
def test_verdict_rejects(kw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        verdict(**kw)


def test_verdict_matches_task_and_result() -> None:
    t = Task.model_validate(task("a", acceptance_criteria=[crit("x"), crit("y")]))
    r = result()
    assert verdict(checks=[{"criterion": 1, "passed": True}]).matches(t, r)
    assert not verdict(checks=[{"criterion": 2, "passed": True}]).matches(t, r)
    assert not verdict(outcome="fail", issues=[{"description": "x", "criterion": 5}]).matches(t, r)
    assert not verdict(attempt=2).matches(t, r)
    assert not verdict(task_id="b").matches(t, r)
    assert not verdict().matches(t, result(task_id="b"))


def test_task_status_values() -> None:
    assert {s.value for s in TaskStatus} == {
        "pending",
        "running",
        "verifying",
        "passed",
        "failed",
        "blocked",
    }


def test_check_result_model() -> None:
    assert CheckResult(criterion=0, passed=False).detail == ""


# --- ProjectStore: plan.json -------------------------------------------------


class FixedClock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def store(tmp_path: Path) -> ProjectStore:
    return ProjectStore.for_workspace(tmp_path / "ws", clock=FixedClock())


def test_save_and_load_plan(store: ProjectStore) -> None:
    pr = store.create("goal", project_id="p1")
    assert not store.has_plan(pr.id)
    with pytest.raises(ProjectStoreError, match="no plan"):
        store.load_plan(pr.id)
    p = make_plan(task("a"), task("b", "a"), goal="یک هدف")
    store.save_plan(pr.id, p)
    assert store.has_plan(pr.id)
    assert store.load_plan(pr.id) == p
    text = store.plan_file(pr.id).read_text(encoding="utf-8")
    assert "یک هدف" in text  # readable, not \u-escaped
    ev = [e for e in store.events(pr.id).read() if e.type == "plan.saved"]
    assert len(ev) == 1 and ev[0].data == {"version": 1, "tasks": 2}
    assert ev[0].project_id == "p1"


def test_save_plan_requires_higher_version(store: ProjectStore) -> None:
    store.create("goal", project_id="p1")
    p1 = make_plan(task("a"))
    store.save_plan("p1", p1)
    with pytest.raises(ProjectConflict):
        store.save_plan("p1", p1)
    with pytest.raises(ProjectConflict):
        store.save_plan("p1", make_plan(task("z"), version=1))
    p2 = make_plan(task("a"), task("b", "a"), version=2)
    store.save_plan("p1", p2)
    assert store.load_plan("p1").version == 2
    with pytest.raises(ProjectConflict):
        store.save_plan("p1", p1)
    assert store.load_plan("p1") == p2


def test_save_plan_revalidates_unvalidated_copy(store: ProjectStore) -> None:
    store.create("goal", project_id="p1")
    p = make_plan(task("a"), task("b", "a"))
    # model_copy skips validation: try to smuggle a cycle in.
    a = p.tasks[0].model_copy(update={"depends_on": ["b"]})
    bad = p.model_copy(update={"tasks": [a, p.tasks[1]]})
    with pytest.raises(ProjectStoreError, match="invalid plan"):
        store.save_plan("p1", bad)
    assert not store.has_plan("p1")
    bad2 = Plan.model_construct(goal="g", tasks=[])
    with pytest.raises(ProjectStoreError):
        store.save_plan("p1", bad2)
    assert not store.has_plan("p1")


def test_plan_needs_existing_project(store: ProjectStore) -> None:
    with pytest.raises(ProjectStoreError):
        store.save_plan("ghost", make_plan(task("a")))
    with pytest.raises(ProjectStoreError):
        store.load_plan("ghost")
    with pytest.raises(ProjectStoreError):
        store.save_plan("../x", make_plan(task("a")))


@pytest.mark.parametrize(
    "content",
    [
        "{",
        "[]",
        json.dumps({"goal": "g", "tasks": []}),
        json.dumps(plan_dict(task("a", "b"), task("b", "a"))),
        json.dumps(plan_dict(task("a"), schema_version=99)),
    ],
)
def test_corrupt_plan_file_raises(store: ProjectStore, content: str) -> None:
    store.create("goal", project_id="p1")
    store.plan_file("p1").write_text(content, encoding="utf-8")
    with pytest.raises(ProjectStoreError):
        store.load_plan("p1")
    # A corrupt stored plan is never silently replaced.
    with pytest.raises(ProjectStoreError):
        store.save_plan("p1", make_plan(task("a"), version=5))


def test_symlinked_plan_file_rejected(store: ProjectStore, tmp_path: Path) -> None:
    store.create("goal", project_id="p1")
    outside = tmp_path / "outside.json"
    outside.write_text(make_plan(task("a")).model_dump_json(), encoding="utf-8")
    store.plan_file("p1").symlink_to(outside)
    with pytest.raises(ProjectStoreError, match="no plan"):
        store.load_plan("p1")
    with pytest.raises(ProjectStoreError, match="symlink"):
        store.save_plan("p1", make_plan(task("a"), version=2))
    assert json.loads(outside.read_text(encoding="utf-8"))["version"] == 1


def test_plan_write_is_atomic(store: ProjectStore, monkeypatch: pytest.MonkeyPatch) -> None:
    import automa.project.store as store_mod

    store.create("goal", project_id="p1")
    p1 = make_plan(task("a"))
    store.save_plan("p1", p1)

    def boom(*a: Any, **k: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store_mod.os, "replace", boom)
    with pytest.raises(OSError):
        store.save_plan("p1", make_plan(task("a"), task("b"), version=2))
    monkeypatch.undo()
    assert store.load_plan("p1") == p1
    leftovers = [f.name for f in store.project_dir("p1").iterdir() if f.name.endswith(".tmp")]
    assert leftovers == []


# --- integration: plan JSON from a (fake) model through the guarded gateway ---


def test_plan_from_fake_model_through_guarded_gateway(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path / "ws", clock=FixedClock())
    pr = store.create("build a CLI", project_id="p1")
    cfg = AutomaConfig.model_validate(
        {
            "tiers": {"small": "fake-small"},
            "prices": {"fake-small": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}},
        }
    )
    plan_json = json.dumps(plan_dict(task("a"), task("b", "a")))
    provider = FakeProvider(
        [
            ChatResponse(
                message=Message.assistant(f"```json\n{plan_json}\n```"),
                model="fake-small",
                usage=Usage(input_tokens=100, output_tokens=200),
            )
        ]
    )
    guard = BudgetGuard(cfg.limits, CostLedger(tmp_path / "ledger.sqlite3"))
    gw = LLMGateway(cfg, provider, guard=guard)
    out = gw.complete(
        [Message.user("plan it")],
        tier=Tier.SMALL,
        max_tokens=1000,
        context=CallContext(project_id=pr.id),
    )
    plan = parse_plan(out.text)
    store.save_plan(pr.id, plan)
    assert store.load_plan(pr.id).task_ids == ["a", "b"]
    assert len(provider.requests) == 1
