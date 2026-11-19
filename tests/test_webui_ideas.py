"""Tests for "New idea" in the web UI (``automa web --allow-actions``, D124).

Goal → clarifying questions → answers → plan and cost range, from the browser. The
same Host / same-origin / CSRF checks as approvals; model calls only through a fake
planner or the real CLI planner with a ``FakeProvider`` (guarded gateway, real
Ledger). No network and no real model calls.
"""

from __future__ import annotations

import http.client
import json
import re
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

from automa import cli
from automa.config import AutomaConfig, load_config
from automa.llm import FakeProvider, ProviderError
from automa.plan import Clarification
from automa.plan.models import MAX_ANSWER_CHARS, MAX_CLARIFICATIONS
from automa.project import ProjectStatus, ProjectStore
from automa.project.store import MAX_GOAL_CHARS, ProjectConflict, ProjectStoreError
from automa.webui import actions as wa
from automa.webui import ideas as wi
from automa.webui import server as ws
from automa.webui.render import ActionForms, render_clarify, render_new_idea

LEDGER = "ledger.sqlite3"
PORT = 8765
HOST = f"127.0.0.1:{PORT}"
ORIGIN = f"http://{HOST}"
TOKEN = "k" * 43
FORM = "application/x-www-form-urlencoded"
EVIL = '<script>alert(1)</script> "q" ‮'


def _settings(token: str = TOKEN) -> wa.ActionSettings:
    hosts = ws.allowed_hosts("127.0.0.1", PORT)
    return wa.ActionSettings(csrf_token=token, origins=wa.allowed_origins(hosts))


class FakePlanner:
    """Scripted planner: records calls, writes a minimal plan on ``plan``."""

    def __init__(self, store: ProjectStore) -> None:
        self.store = store
        self.questions: list[str] = ["Who is it for?", EVIL]
        self.calls: list[tuple[str, Any]] = []
        self.fail: dict[str, wi.IdeaError] = {}
        self.range = "about $0.05 to $0.30"

    def prepare(self) -> None:
        self.calls.append(("prepare", None))
        if "prepare" in self.fail:
            raise self.fail["prepare"]

    def clarify(self, project_id: str) -> list[str]:
        self.calls.append(("clarify", project_id))
        if "clarify" in self.fail:
            raise self.fail["clarify"]
        return list(self.questions)

    def plan(self, project_id: str, clarifications: Sequence[Clarification]) -> None:
        self.calls.append(("plan", (project_id, list(clarifications))))
        if "plan" in self.fail:
            raise self.fail["plan"]
        from automa.plan import Plan

        plan = Plan.model_validate(
            {
                "goal": "g",
                "tasks": [
                    {
                        "id": "build",
                        "title": "Build",
                        "role": "developer",
                        "acceptance_criteria": [{"description": "works"}],
                    }
                ],
            }
        )
        self.store.save_plan(project_id, plan)

    def cost_range(self, project_id: str) -> str:
        self.calls.append(("cost_range", project_id))
        return self.range

    def kinds(self) -> list[str]:
        return [c[0] for c in self.calls]


class Env:
    def __init__(self, root: Path, *, planner: bool = True, actions: bool = True) -> None:
        self.root = root
        self.store = ProjectStore.for_workspace(root)
        self.planner = FakePlanner(self.store)
        hosts = ws.allowed_hosts("127.0.0.1", PORT)
        self.app = ws.WebApp(
            root,
            root / LEDGER,
            hosts,
            actions=_settings() if actions else None,
            planner=self.planner if planner else None,
        )

    def get(self, path: str) -> ws.Response:
        return self.app.handle("GET", path, HOST)

    def post(
        self,
        path: str = "/new",
        fields: dict[str, str] | None = None,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> ws.Response:
        data = body if body is not None else urlencode(fields or {}).encode("ascii")
        hdrs = {"Origin": ORIGIN, "Content-Type": FORM, "Sec-Fetch-Site": "same-origin"}
        hdrs.update(headers or {})
        return self.app.handle("POST", path, HOST, list(hdrs.items()), data)

    def new(self, goal: str = "Build a site", ask: bool = True, **extra: str) -> ws.Response:
        fields = {"csrf": TOKEN, "goal": goal, **extra}
        if ask:
            fields["ask"] = "yes"
        return self.post("/new", fields)


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def _location(r: ws.Response) -> str:
    return dict(r.headers)["Location"]


def _body(r: ws.Response) -> str:
    return r.body.decode("utf-8")


# -- availability ---------------------------------------------------------------


def test_read_only_server_has_no_idea_pages(tmp_path: Path) -> None:
    e = Env(tmp_path, actions=False)
    assert e.get("/new").status == 404
    assert e.post("/new", {"csrf": TOKEN, "goal": "x"}).status == 405
    assert "New idea" not in _body(e.get("/"))
    assert not e.app.ideas_enabled
    assert e.planner.calls == []
    assert e.store.list_ids() == []


def test_action_server_without_planner_has_no_idea_pages(tmp_path: Path) -> None:
    e = Env(tmp_path, planner=False)
    assert e.get("/new").status == 404
    assert e.post("/new", {"csrf": TOKEN, "goal": "x"}).status == 404
    assert "New idea" not in _body(e.get("/"))
    assert e.store.list_ids() == []


def test_overview_links_new_idea(env: Env) -> None:
    body = _body(env.get("/"))
    assert 'href="/new"' in body


def test_new_idea_form(env: Env) -> None:
    r = env.get("/new")
    assert r.status == 200
    body = _body(r)
    assert f'name="csrf" value="{TOKEN}"' in body
    assert 'action="/new"' in body and 'method="post"' in body
    assert f'maxlength="{MAX_GOAL_CHARS}"' in body
    assert "form-action 'self'" in dict(r.headers)["Content-Security-Policy"]
    assert "<script" not in body
    assert env.planner.calls == []


def test_head_new_has_no_side_effects(env: Env) -> None:
    assert env.app.handle("HEAD", "/new", HOST).status == 200
    assert env.store.list_ids() == []


# -- request checks before anything happens ---------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Content-Type": "text/plain"},
        {"Content-Type": "multipart/form-data"},
    ],
)
def test_cross_origin_or_odd_posts_are_refused(env: Env, headers: dict[str, str]) -> None:
    r = env.post("/new", {"csrf": TOKEN, "goal": "x", "ask": "yes"}, headers=headers)
    assert r.status in (403, 415)
    assert env.store.list_ids() == []
    assert env.planner.calls == []


def test_post_without_origin_or_referer_is_refused(env: Env) -> None:
    data = urlencode({"csrf": TOKEN, "goal": "x"}).encode()
    r = env.app.handle("POST", "/new", HOST, [("Content-Type", FORM)], data)
    assert r.status == 403
    assert env.store.list_ids() == []


def test_wrong_host_is_refused(env: Env) -> None:
    data = urlencode({"csrf": TOKEN, "goal": "x"}).encode()
    hdrs = [("Origin", ORIGIN), ("Content-Type", FORM)]
    assert env.app.handle("POST", "/new", "evil.example:8765", hdrs, data).status == 421
    assert env.store.list_ids() == []


@pytest.mark.parametrize("token", ["", "wrong", TOKEN + "x"])
def test_wrong_token_is_refused(env: Env, token: str) -> None:
    r = env.post("/new", {"csrf": token, "goal": "x"})
    assert r.status == 403
    assert env.store.list_ids() == []
    assert env.planner.calls == []


def test_duplicate_origin_header_is_refused(env: Env) -> None:
    data = urlencode({"csrf": TOKEN, "goal": "x"}).encode()
    hdrs = [("Origin", ORIGIN), ("Origin", "http://evil"), ("Content-Type", FORM)]
    assert env.app.handle("POST", "/new", HOST, hdrs, data).status == 400
    assert env.store.list_ids() == []


@pytest.mark.parametrize(
    "fields",
    [
        {"csrf": TOKEN},
        {"csrf": TOKEN, "goal": "   "},
        {"csrf": TOKEN, "goal": "x" * (MAX_GOAL_CHARS + 1)},
        {"csrf": TOKEN, "goal": "x", "ask": "maybe"},
        {"csrf": TOKEN, "goal": "x", "fingerprint": "f"},
        {"csrf": TOKEN, "goal": "x", "a0": "answer"},
    ],
)
def test_bad_idea_forms_are_refused(env: Env, fields: dict[str, str]) -> None:
    assert env.post("/new", fields).status == 400
    assert env.store.list_ids() == []
    assert env.planner.calls == []


def test_repeated_field_is_refused(env: Env) -> None:
    r = env.post("/new", body=f"csrf={TOKEN}&goal=a&goal=b".encode())
    assert r.status == 400
    assert env.store.list_ids() == []


def test_query_string_on_post_is_not_an_idea(env: Env) -> None:
    r = env.post("/new?x=1", {"csrf": TOKEN, "goal": "x"})
    assert r.status == 404
    assert env.store.list_ids() == []


def test_body_limits_depend_on_path(env: Env, tmp_path: Path) -> None:
    assert env.app.max_body_bytes("/new") == wi.MAX_IDEA_FORM_BYTES
    assert env.app.max_body_bytes("/projects/p1/clarify") == wi.MAX_IDEA_FORM_BYTES
    assert env.app.max_body_bytes("/projects/p1/approvals/a/approve") == wa.MAX_FORM_BYTES
    ro = Env(tmp_path / "ro", actions=False)
    assert ro.app.max_body_bytes("/new") == wa.MAX_FORM_BYTES
    nop = Env(tmp_path / "nop", planner=False)
    assert nop.app.max_body_bytes("/new") == wa.MAX_FORM_BYTES


def test_oversized_idea_form_is_refused(env: Env) -> None:
    body = b"csrf=" + TOKEN.encode() + b"&goal=" + b"x" * wi.MAX_IDEA_FORM_BYTES
    assert env.post("/new", body=body).status == 413
    assert env.store.list_ids() == []


def test_persian_goal_near_the_limit_fits(env: Env) -> None:
    goal = "س" * MAX_GOAL_CHARS
    r = env.new(goal, ask=False)
    assert r.status == 303
    (pid,) = env.store.list_ids()
    assert env.store.load(pid).goal == goal


# -- the flow -------------------------------------------------------------------------


def test_idea_with_questions_then_answers_makes_plan(env: Env) -> None:
    r = env.new("  Build a site  ")
    assert r.status == 303
    (pid,) = env.store.list_ids()
    assert _location(r) == f"/projects/{pid}/clarify"
    assert env.store.load(pid).goal == "Build a site"
    assert env.store.load_questions(pid) == ["Who is it for?", EVIL]
    assert not env.store.has_clarifications(pid)
    assert not env.store.has_plan(pid)
    assert env.planner.kinds() == ["prepare", "clarify"]

    page = env.get(f"/projects/{pid}")
    assert "answer them" in _body(page)

    form = env.get(f"/projects/{pid}/clarify")
    assert form.status == 200
    body = _body(form)
    assert "<script>alert" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert "\\u202e" in body and "‮" not in body
    assert 'name="a0"' in body and 'name="a1"' in body and 'name="a2"' not in body
    assert f'name="csrf" value="{TOKEN}"' in body

    r = env.post(f"/projects/{pid}/clarify", {"csrf": TOKEN, "a0": " کاربران ", "a1": ""})
    assert r.status == 303
    assert _location(r) == f"/projects/{pid}"
    saved = env.store.load_clarifications(pid)
    assert [(c.question, c.answer) for c in saved] == [
        ("Who is it for?", "کاربران"),
        (EVIL, ""),
    ]
    assert env.planner.kinds() == ["prepare", "clarify", "prepare", "plan"]
    assert env.planner.calls[-1][1] == (pid, saved)
    assert env.store.has_plan(pid)

    page = _body(env.get(f"/projects/{pid}"))
    assert "Cost range:" in page and "about $0.05 to $0.30" in page
    assert "answer them" not in page
    assert env.get(f"/projects/{pid}/clarify").status == 409


def test_answers_twice_is_refused(env: Env) -> None:
    env.new()
    (pid,) = env.store.list_ids()
    assert env.post(f"/projects/{pid}/clarify", {"csrf": TOKEN, "a0": "x"}).status == 303
    r = env.post(f"/projects/{pid}/clarify", {"csrf": TOKEN, "a0": "y"})
    assert r.status == 409
    assert env.planner.kinds().count("plan") == 1


def test_no_ask_plans_directly(env: Env) -> None:
    r = env.new(ask=False)
    (pid,) = env.store.list_ids()
    assert _location(r) == f"/projects/{pid}"
    assert env.planner.kinds() == ["prepare", "plan"]
    assert env.planner.calls[-1][1] == (pid, [])
    assert env.store.load_questions(pid) is None


def test_no_questions_plans_directly(env: Env) -> None:
    env.planner.questions = []
    r = env.new()
    (pid,) = env.store.list_ids()
    assert _location(r) == f"/projects/{pid}"
    assert env.planner.kinds() == ["prepare", "clarify", "plan"]
    assert env.store.load_questions(pid) is None
    assert env.get(f"/projects/{pid}/clarify").status == 409


@pytest.mark.parametrize(
    "fields",
    [
        {"csrf": TOKEN, "a0": "x", "a2": "unknown question"},
        {"csrf": TOKEN, "a0": "x" * (MAX_ANSWER_CHARS + 1)},
        {"csrf": TOKEN, "goal": "x"},
    ],
)
def test_bad_answer_forms_are_refused(env: Env, fields: dict[str, str]) -> None:
    env.new()
    (pid,) = env.store.list_ids()
    assert env.post(f"/projects/{pid}/clarify", fields).status == 400
    assert not env.store.has_clarifications(pid)
    assert "plan" not in env.planner.kinds()


def test_answer_with_wrong_token_or_origin_is_refused(env: Env) -> None:
    env.new()
    (pid,) = env.store.list_ids()
    path = f"/projects/{pid}/clarify"
    assert env.post(path, {"csrf": "no", "a0": "x"}).status == 403
    r = env.post(path, {"csrf": TOKEN, "a0": "x"}, headers={"Origin": "http://evil"})
    assert r.status == 403
    assert not env.store.has_clarifications(pid)
    assert "plan" not in env.planner.kinds()


def test_clarify_for_unknown_or_bad_project(env: Env) -> None:
    assert env.get("/projects/nope/clarify").status == 404
    assert env.post("/projects/nope/clarify", {"csrf": TOKEN}).status == 404
    assert env.post("/projects/Bad..Id/clarify", {"csrf": TOKEN}).status == 404
    assert env.get("/projects/Bad..Id/clarify").status == 404


def test_clarify_for_project_without_questions(env: Env) -> None:
    env.store.create("from cli", project_id="cli-made")
    assert env.get("/projects/cli-made/clarify").status == 409
    assert env.post("/projects/cli-made/clarify", {"csrf": TOKEN}).status == 409
    assert "answer them" not in _body(env.get("/projects/cli-made"))


def test_planned_cli_project_shows_cost_range(env: Env) -> None:
    env.store.create("from cli", project_id="cli-made")
    env.planner.plan("cli-made", [])
    body = _body(env.get("/projects/cli-made"))
    assert "Cost range:" in body
    assert ("cost_range", "cli-made") in env.planner.calls


def test_read_only_page_never_asks_planner(tmp_path: Path) -> None:
    e = Env(tmp_path, actions=False)
    e.store.create("g", project_id="p1")
    e.planner.plan("p1", [])
    e.planner.calls.clear()
    body = _body(e.get("/projects/p1"))
    assert "Cost range:" not in body
    assert e.planner.calls == []


def test_cost_range_is_escaped(env: Env) -> None:
    env.planner.range = EVIL
    env.new(ask=False)
    (pid,) = env.store.list_ids()
    body = _body(env.get(f"/projects/{pid}"))
    assert "<script>alert" not in body


# -- failures --------------------------------------------------------------------------


def test_prepare_failure_creates_nothing(env: Env) -> None:
    env.planner.fail["prepare"] = wi.IdeaError("config", "configuration: no AUTOMA_API_KEY")
    r = env.new()
    assert r.status == 500
    assert "AUTOMA_API_KEY" in _body(r) and "nothing was created" in _body(r)
    assert env.store.list_ids() == []
    assert env.planner.kinds() == ["prepare"]


@pytest.mark.parametrize(
    ("kind", "status", "project_status"),
    [
        ("budget", 409, ProjectStatus.PAUSED),
        ("planning", 422, ProjectStatus.FAILED),
        ("provider", 502, ProjectStatus.NEW),
    ],
)
def test_clarify_failure_statuses(
    env: Env, kind: str, status: int, project_status: ProjectStatus
) -> None:
    env.planner.fail["clarify"] = wi.IdeaError(kind, f"{kind} <b>stop</b>")
    r = env.new()
    assert r.status == status
    (pid,) = env.store.list_ids()
    assert pid in _body(r)
    assert "<b>stop" not in _body(r)
    assert env.store.load(pid).status is project_status
    assert env.store.load_questions(pid) is None
    assert "plan" not in env.planner.kinds()


def test_plan_failure_after_answers(env: Env) -> None:
    env.new()
    (pid,) = env.store.list_ids()
    env.planner.fail["plan"] = wi.IdeaError("budget", "budget stop (project cap)")
    r = env.post(f"/projects/{pid}/clarify", {"csrf": TOKEN, "a0": "x"})
    assert r.status == 409
    assert "budget stop" in _body(r) and pid in _body(r)
    assert env.store.has_clarifications(pid)
    assert not env.store.has_plan(pid)


def test_prepare_failure_before_answers_keeps_questions(env: Env) -> None:
    env.new()
    (pid,) = env.store.list_ids()
    env.planner.fail["prepare"] = wi.IdeaError("config", "configuration: broken")
    r = env.post(f"/projects/{pid}/clarify", {"csrf": TOKEN, "a0": "x"})
    assert r.status == 500
    assert not env.store.has_clarifications(pid)
    assert env.get(f"/projects/{pid}/clarify").status == 200


def test_one_executive_call_at_a_time(env: Env) -> None:
    flow = env.app._ideas
    assert flow is not None
    flow._busy.acquire()
    try:
        r = env.new()
        assert r.status == 409
        assert "already working" in _body(r)
        assert env.store.list_ids() == []
    finally:
        flow._busy.release()
    assert env.new().status == 303


def test_lock_released_after_failure(env: Env) -> None:
    env.planner.fail["prepare"] = wi.IdeaError("config", "x")
    assert env.new().status == 500
    del env.planner.fail["prepare"]
    assert env.new().status == 303


def test_idea_error_kind_is_checked() -> None:
    with pytest.raises(ValueError):
        wi.IdeaError("other", "x")


# -- pure helpers ----------------------------------------------------------------------


def test_paths() -> None:
    assert wi.parse_clarify_path("/projects/p1/clarify") == "p1"
    assert wi.parse_clarify_path("/projects/p1/clarify/") is None
    assert wi.parse_clarify_path("/projects/../clarify") is None
    assert wi.parse_clarify_path("/projects/a/b/clarify") is None
    assert wi.is_idea_path("/new")
    assert not wi.is_idea_path("/new/")
    assert wi.clarify_path("p1") == "/projects/p1/clarify"


def test_parse_answers_counts() -> None:
    items = wi.parse_answers({"csrf": "t"}, ["q1", "q2"])
    assert [(c.question, c.answer) for c in items] == [("q1", ""), ("q2", "")]
    assert len(wi.ANSWER_FIELDS) == MAX_CLARIFICATIONS + 1


def test_render_pages_are_safe() -> None:
    forms = ActionForms('t"><x')
    new = render_new_idea(forms)
    assert 't"><x' not in new and "t&quot;&gt;&lt;x" in new
    clar = render_clarify("p1", EVIL, [EVIL], forms)
    assert "<script>alert" not in clar


# -- store: pending questions -------------------------------------------------------------


def test_store_questions_once(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("g", project_id="p1")
    assert store.load_questions("p1") is None
    assert store.save_questions("p1", [" a ", "b"]) == ["a", "b"]
    assert store.load_questions("p1") == ["a", "b"]
    with pytest.raises(ProjectConflict):
        store.save_questions("p1", ["c"])
    events = [json.loads(line)["type"] for line in store.events("p1").path.read_text().splitlines()]
    assert "questions.saved" in events


def test_store_questions_refused_after_answers(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("g", project_id="p1")
    store.save_clarifications("p1", [Clarification(question="q", answer="a")])
    with pytest.raises(ProjectConflict):
        store.save_questions("p1", ["q"])


@pytest.mark.parametrize("questions", [["", "b"], ["x" * 501], ["q"] * (MAX_CLARIFICATIONS + 1)])
def test_store_questions_invalid(tmp_path: Path, questions: list[str]) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("g", project_id="p1")
    with pytest.raises(ProjectStoreError):
        store.save_questions("p1", questions)
    assert store.load_questions("p1") is None


def test_store_questions_bad_files(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("g", project_id="p1")
    path = store.questions_file("p1")
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ProjectStoreError):
        store.load_questions("p1")
    path.write_text(json.dumps({"schema_version": 99, "questions": []}), encoding="utf-8")
    with pytest.raises(ProjectStoreError):
        store.load_questions("p1")
    path.unlink()
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({"questions": ["x"]}), encoding="utf-8")
    path.symlink_to(target)
    with pytest.raises(ProjectStoreError):
        store.load_questions("p1")
    with pytest.raises(ProjectStoreError):
        store.save_questions("p1", ["y"])
    assert json.loads(target.read_text())["questions"] == ["x"]


def test_unreadable_questions_file_shows_error_page(env: Env) -> None:
    env.store.create("g", project_id="p1")
    env.store.questions_file("p1").write_text("{", encoding="utf-8")
    assert env.get("/projects/p1/clarify").status == 500
    assert env.get("/projects/p1").status == 200


# -- real CLI planner with a FakeProvider (guarded gateway, real Ledger) ---------------------

TOML = """
workspace = "{ws}"
[tiers]
small = "fake-small"
medium = "fake-medium"
large = "fake-large"
[prices."fake-small"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[prices."fake-medium"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[prices."fake-large"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[limits]
project_usd = {project}
daily_usd = {daily}
per_call_usd = {per_call}
"""

PLAN = json.dumps(
    {
        "goal": "model goal",
        "tasks": [
            {
                "id": "build",
                "title": "Build",
                "role": "developer",
                "acceptance_criteria": [{"description": "works"}],
            }
        ],
    }
)


class RealEnv:
    def __init__(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, project: float = 5.0
    ) -> None:
        self.ws = tmp_path / "ws"
        cfg = tmp_path / "automa.toml"
        text = TOML.format(
            ws=self.ws.as_posix(),
            project=project,
            daily=max(project, 2.0) if project >= 1 else project,
            per_call=min(project, 0.25),
        )
        cfg.write_text(text, encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        self.config = load_config(cfg)
        self.provider = FakeProvider()
        self.factory_calls = 0

        def factory(config: AutomaConfig) -> FakeProvider:
            self.factory_calls += 1
            return self.provider

        monkeypatch.setattr(cli, "make_provider", factory)
        self.store = ProjectStore.for_workspace(self.ws)
        hosts = ws.allowed_hosts("127.0.0.1", PORT)
        self.app = ws.WebApp(
            self.ws,
            self.ws / LEDGER,
            hosts,
            actions=_settings(),
            planner=cli._WebIdeaPlanner(self.config),
        )

    def post(self, path: str, fields: dict[str, str]) -> ws.Response:
        hdrs = [("Origin", ORIGIN), ("Content-Type", FORM)]
        return self.app.handle("POST", path, HOST, hdrs, urlencode(fields).encode())


def test_real_planner_full_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    e = RealEnv(tmp_path, monkeypatch)
    e.provider.push(json.dumps({"questions": ["Which language?"]}), PLAN)
    r = e.post("/new", {"csrf": TOKEN, "goal": "Build a site", "ask": "yes"})
    assert r.status == 303, r.body
    (pid,) = e.store.list_ids()
    assert e.store.load_questions(pid) == ["Which language?"]
    r = e.post(f"/projects/{pid}/clarify", {"csrf": TOKEN, "a0": "فارسی"})
    assert r.status == 303, r.body
    assert e.store.has_plan(pid)
    assert e.store.load_clarifications(pid)[0].answer == "فارسی"
    assert e.provider.remaining == 0
    # Both calls were recorded in the Ledger through the guarded gateway.
    from automa.budget.ledger import CostLedger

    with CostLedger(e.ws / LEDGER, read_only=True) as ledger:
        assert len([x for x in ledger.entries() if x.project_id == pid]) == 2
    body = e.app.handle("GET", f"/projects/{pid}", HOST).body.decode()
    assert "Cost range:" in body and "project safety cap $" in body
    # D131: no terminal commands and no theoretical worst case on the web page.
    assert "automa run" not in body and "worst case" not in body


def test_real_planner_missing_key_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automa.llm.factory import provider_from_config

    e = RealEnv(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "make_provider", provider_from_config)
    r = e.post("/new", {"csrf": TOKEN, "goal": "x", "ask": "yes"})
    assert r.status == 500
    assert "AUTOMA_API_KEY" in r.body.decode()
    assert e.store.list_ids() == []


def test_real_planner_provider_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    e = RealEnv(tmp_path, monkeypatch)
    e.provider.push(ProviderError("HTTP 500 upstream"))
    r = e.post("/new", {"csrf": TOKEN, "goal": "x", "ask": "yes"})
    assert r.status == 502
    assert "provider error" in r.body.decode()


def test_real_planner_budget_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    e = RealEnv(tmp_path, monkeypatch, project=0.001)
    e.provider.push(json.dumps({"questions": []}), PLAN)
    r = e.post("/new", {"csrf": TOKEN, "goal": "x", "ask": "yes"})
    assert r.status == 409, r.body
    assert "budget stop" in r.body.decode()
    (pid,) = e.store.list_ids()
    assert e.store.load(pid).status is ProjectStatus.PAUSED
    assert not e.store.has_plan(pid)
    assert e.provider.requests == []


def test_real_planner_bad_plan_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    e = RealEnv(tmp_path, monkeypatch)
    e.provider.push(*(["not json"] * 10))
    r = e.post("/new", {"csrf": TOKEN, "goal": "x"})
    assert r.status == 422, r.body
    (pid,) = e.store.list_ids()
    assert e.store.load(pid).status is ProjectStatus.FAILED


# -- real socket ---------------------------------------------------------------------------


@pytest.fixture
def running(tmp_path: Path) -> Iterator[tuple[int, FakePlanner, ProjectStore]]:
    store = ProjectStore.for_workspace(tmp_path)
    planner = FakePlanner(store)
    server = ws.create_server(
        tmp_path, tmp_path / LEDGER, port=0, allow_actions=True, planner=planner
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1]), planner, store
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _req(
    port: int, method: str, path: str, headers: dict[str, str], body: bytes = b""
) -> tuple[int, dict[str, str], str]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request(method, path, body=body or None, headers=headers)
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read().decode("utf-8")
    finally:
        conn.close()


def test_socket_idea_flow(running: tuple[int, FakePlanner, ProjectStore]) -> None:
    port, planner, store = running
    host = {"Host": f"127.0.0.1:{port}"}
    status, _, page = _req(port, "GET", "/new", host)
    assert status == 200
    m = re.search(r'name="csrf" value="([^"]+)"', page)
    assert m is not None
    token = m.group(1)
    post = {**host, "Origin": f"http://127.0.0.1:{port}", "Content-Type": FORM}
    body = urlencode({"csrf": token, "goal": "سایت فروشگاهی " * 300, "ask": "yes"}).encode()
    assert len(body) > wa.MAX_FORM_BYTES
    status, headers, _ = _req(port, "POST", "/new", post, body)
    assert status == 303
    (pid,) = store.list_ids()
    # The real server plans in the background behind a "thinking" page (D137).
    assert headers["Location"] == f"/projects/{pid}/thinking"
    location = _follow_thinking(port, host, pid)
    assert location == f"/projects/{pid}/clarify"
    answers = urlencode({"csrf": token}).encode()
    status, headers, _ = _req(port, "POST", location, post, answers)
    assert status == 303 and headers["Location"] == f"/projects/{pid}/thinking"
    assert _follow_thinking(port, host, pid) == f"/projects/{pid}"
    assert store.has_plan(pid)


def _follow_thinking(port: int, host: dict[str, str], pid: str) -> str:
    import time

    for _ in range(100):
        status, headers, _ = _req(port, "GET", f"/projects/{pid}/thinking", host)
        if status == 303:
            return headers["Location"]
        assert status == 200
        time.sleep(0.05)
    raise AssertionError("the background step did not finish")


def test_socket_cross_origin_idea_refused(running: tuple[int, FakePlanner, ProjectStore]) -> None:
    port, planner, store = running
    post = {"Host": f"127.0.0.1:{port}", "Origin": "http://evil.example", "Content-Type": FORM}
    status, _, _ = _req(port, "POST", "/new", post, b"csrf=x&goal=y")
    assert status == 403
    assert store.list_ids() == []
    assert planner.calls == []


# -- background mode: a "thinking" page instead of a hanging request (D137) -------


class SlowPlanner(FakePlanner):
    """Blocks in clarify/plan until released, like a real Executive call."""

    def __init__(self, store: ProjectStore) -> None:
        super().__init__(store)
        import threading

        self.gate = threading.Event()

    def clarify(self, project_id: str) -> list[str]:
        self.gate.wait(5)
        return super().clarify(project_id)

    def plan(self, project_id: str, clarifications: Sequence[Clarification]) -> None:
        self.gate.wait(5)
        super().plan(project_id, clarifications)


def _bg_env(root: Path, planner_cls: type[FakePlanner] = SlowPlanner) -> Env:
    env = Env.__new__(Env)
    env.root = root
    env.store = ProjectStore.for_workspace(root)
    env.planner = planner_cls(env.store)
    env.app = ws.WebApp(
        root,
        root / LEDGER,
        ws.allowed_hosts("127.0.0.1", PORT),
        actions=_settings(),
        planner=env.planner,
        background_ideas=True,
    )
    return env


def test_background_idea_returns_at_once_and_waits_on_thinking_page(tmp_path: Path) -> None:
    env = _bg_env(tmp_path)
    r = env.new("Build a site")
    assert r.status == 303
    (pid,) = env.store.list_ids()
    assert _location(r) == f"/projects/{pid}/thinking"
    waiting = env.get(f"/projects/{pid}/thinking")
    assert waiting.status == 200
    body = _body(waiting)
    assert '<meta http-equiv="refresh" content="3">' in body and "Thinking" in body
    # A second idea while the Executive is busy is refused.
    assert env.new("Another").status == 409
    env.planner.gate.set()
    env.app._ideas.wait(pid)  # type: ignore[union-attr]
    done = env.get(f"/projects/{pid}/thinking")
    assert done.status == 303 and _location(done) == f"/projects/{pid}/clarify"

    r = env.post(f"/projects/{pid}/clarify", {"csrf": TOKEN, "a0": "me", "a1": ""})
    assert r.status == 303 and _location(r) == f"/projects/{pid}/thinking"
    env.app._ideas.wait(pid)  # type: ignore[union-attr]
    done = env.get(f"/projects/{pid}/thinking")
    assert done.status == 303 and _location(done) == f"/projects/{pid}"
    assert env.store.has_plan(pid)
    # The lock is free again.
    assert env.new("Next idea").status == 303


def test_background_failure_is_shown_and_frees_the_lock(tmp_path: Path) -> None:
    env = _bg_env(tmp_path, FakePlanner)
    env.planner.fail["clarify"] = wi.IdeaError("budget", "daily cap reached")
    r = env.new("Build a site")
    (pid,) = env.store.list_ids()
    env.app._ideas.wait(pid)  # type: ignore[union-attr]
    page = env.get(_location(r))
    assert page.status == 409 and "daily cap reached" in _body(page)
    assert env.new("Retry").status == 303


def test_thinking_without_job_goes_to_project(tmp_path: Path) -> None:
    env = _bg_env(tmp_path, FakePlanner)
    env.store.create("g", project_id="p1")
    r = env.get("/projects/p1/thinking")
    assert r.status == 303 and _location(r) == "/projects/p1"
    assert env.get("/projects/..%2Fx/thinking").status == 404


def test_read_only_server_has_no_thinking_page(tmp_path: Path) -> None:
    env = Env(tmp_path, actions=False)
    env.store.create("g", project_id="p1")
    assert env.get("/projects/p1/thinking").status == 404


def test_serve_uses_background_ideas() -> None:
    import inspect

    assert "background_ideas=True" in inspect.getsource(ws.create_server)
