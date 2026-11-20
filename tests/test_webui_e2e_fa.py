"""End-to-end check of the Persian web interface (D149).

One workspace with projects in every state, served by one in-process ``WebApp`` in
action mode with a fake planner and fake background runs. Every page a person can
reach (home, each project state, run panel, approvals, notice, forms, files, error
pages and refused forms) is requested in ``fa`` and must have no English interface
text. Recorded data stays as written and is removed before the check: ids, goals,
tool/role/risk names, event types, arguments, file names and technical details.

No model, no network, temporary directories only.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote, urlencode

import pytest

import test_finance_compare as fc
from automa.budget import CostLedger
from automa.events import Event
from automa.plan import TaskState, TaskStatus
from automa.project import ProjectStatus, ProjectStore
from automa.project.store import TASK_STATE_FILE
from automa.tools.base import RiskLevel
from automa.webui import runs as wr
from automa.webui import server as ws
from test_ledger import FixedClock, make_pair
from test_webui_actions import FORM, ORIGIN, PORT, TOKEN, _settings
from test_webui_i18n import LATIN_WORD, _strip_data
from test_webui_ideas import FakePlanner
from test_webui_runs import FakePreflight, FakeProcess, _plan

HOST = f"127.0.0.1:{PORT}"
LEDGER = "ledger.sqlite3"
RATE = 323_000.0
NOW = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)

# Recorded values shown as written (tool, role, risk, task id, argument keys).
RECORDED = ("fetch_url", "developer", "external", "build", "url", "m-small")
# Key names are printed on the keyboard in Latin letters, in Persian text too.
KEYS = ("Ctrl+C",)
# Event types (``project.created``, ``web.run_started``) are recorded data too.
EVENT_TYPE = re.compile(r"\b[a-z]+(?:_[a-z]+)*(?:\.[a-z]+(?:_[a-z]+)*)+\b")


@dataclass
class Site:
    root: Path
    store: ProjectStore
    app: ws.WebApp
    ids: dict[str, str]
    request_id: str
    procs: list[FakeProcess]

    def get(self, target: str) -> ws.Response:
        return self.app.handle("GET", target, HOST)

    def post(self, target: str, fields: dict[str, str]) -> ws.Response:
        headers = [("Content-Type", FORM), ("Origin", ORIGIN), ("Sec-Fetch-Site", "same-origin")]
        return self.app.handle("POST", target, HOST, headers, urlencode(fields).encode())


def _state(store: ProjectStore, pid: str, status: TaskStatus) -> None:
    tdir = store.task_dir(pid, "build")
    tdir.mkdir(parents=True, exist_ok=True)
    attempts = 0 if status is TaskStatus.PENDING else 1
    error = "compile error: g++ exited 1" if status is TaskStatus.FAILED else None
    st = TaskState(
        task_id="build",
        status=status,
        attempts=attempts,
        model="m-small",
        error=error,
        updated_at=NOW,
    )
    (tdir / TASK_STATE_FILE).write_text(st.model_dump_json(), encoding="utf-8")


def _site(root: Path, lang: str = "fa") -> Site:
    store = ProjectStore.for_workspace(root)
    ids: dict[str, str] = {}
    states: Sequence[tuple[str, ProjectStatus | None, bool, TaskStatus | None]] = (
        ("new", None, False, None),
        ("planned", None, True, None),
        ("running", ProjectStatus.RUNNING, True, TaskStatus.RUNNING),
        ("paused", ProjectStatus.PAUSED, True, TaskStatus.PENDING),
        ("done", ProjectStatus.DONE, True, TaskStatus.PASSED),
        ("failed", ProjectStatus.FAILED, True, TaskStatus.FAILED),
    )
    for name, status, plan, task in states:
        pid = store.create(f"یک صفحهٔ ساده ({name})").id
        ids[name] = pid
        if plan:
            store.save_plan(pid, _plan())
            store.events(pid).emit(
                Event(ts=NOW, type="finance.estimate", project_id=pid, data=fc.data())
            )
        if status is not None:
            store.set_status(pid, status)
        if task is not None:
            _state(store, pid, task)
    store.save_questions(ids["new"], ["برای چه کسیه؟", "چه رنگی؟"])
    req = store.approvals(ids["running"]).request(
        tool="fetch_url",
        risk=RiskLevel.EXTERNAL,
        arguments={"url": "http://1.2.3.4:8080/a?x=1&y=2"},
        reason="خواندن مستندات",
        task_id="build",
    )
    files = store.files_dir(ids["done"])
    (files / "sub").mkdir(parents=True)
    (files / "index.html").write_text("<p>سلام</p>", encoding="utf-8")
    (files / "sub" / "notes.txt").write_text("hello", encoding="utf-8")
    (files / "data.bin").write_bytes(b"\x00\x01\x02")
    with CostLedger(root / LEDGER, clock=FixedClock(NOW)) as led:
        for name in ("running", "done", "failed"):
            led.record(*make_pair(0.12, project=ids[name], task="build"))
            led.record(*make_pair(0.03, project=ids[name], task=None))
    procs: list[FakeProcess] = []

    def spawn(argv: Sequence[str], log_path: Path) -> FakeProcess:
        log_path.write_text("project: x\n", encoding="utf-8")
        proc = FakeProcess()
        procs.append(proc)
        return proc

    app = ws.WebApp(
        root,
        root / LEDGER,
        ws.allowed_hosts("127.0.0.1", PORT),
        actions=_settings(),
        planner=FakePlanner(store),
        runs=wr.BackgroundRuns(store, FakePreflight(), spawn=spawn),
        toman_per_usd=RATE,
        language=lang,  # type: ignore[arg-type]
        clock=lambda: NOW + timedelta(minutes=10),
    )
    return Site(root, store, app, ids, req.id, procs)


def _interface(doc: str, site: Site) -> str:
    """Interface text only: recorded data and technical details removed."""
    doc = re.sub(r"<title>.*?</title>", "", doc, flags=re.S)
    doc = re.sub(r"<(pre|code|bdi)\b[^>]*>.*?</\1>", "", doc, flags=re.S)
    doc = re.sub(r'<(a|span|h1|h2|li|p|td|div)\b[^>]*dir="auto"[^>]*>.*?</\1>', "", doc, flags=re.S)
    for value in (*site.ids.values(), site.request_id):
        doc = doc.replace(value, "")
    doc = EVENT_TYPE.sub("", doc)
    doc = _strip_data(doc)
    for word in (*RECORDED, *KEYS):
        doc = re.sub(rf"(?<![\w-]){re.escape(word)}(?![\w-])", "", doc)
    return doc


def _english(r: ws.Response, site: Site) -> list[str]:
    return LATIN_WORD.findall(_interface(r.body.decode("utf-8"), site))


@pytest.fixture
def site(tmp_path: Path) -> Site:
    return _site(tmp_path)


def _pages(site: Site) -> dict[str, tuple[str, int]]:
    """Every GET page a person can reach: label → (target, expected status)."""
    ids = site.ids
    done = quote(ids["done"])
    pages: dict[str, tuple[str, int]] = {
        "home": ("/", 200),
        "home with tests": ("/?tests=1", 200),
        "new idea": ("/new", 200),
        "questions": (f"/projects/{ids['new']}/clarify", 200),
        "notice": (f"/projects/{ids['running']}?decided={site.request_id}", 200),
        "files": (f"/projects/{done}/files", 200),
        "sub folder": (f"/projects/{done}/files?path=sub", 200),
        "text file": (f"/projects/{done}/files/view?path=sub%2Fnotes.txt", 200),
        "html file": (f"/projects/{done}/files/view?path=index.html", 200),
        "binary file": (f"/projects/{done}/files/view?path=data.bin", 200),
        "missing file": (f"/projects/{done}/files/view?path=nope.txt", 404),
        "files of a project without files": (f"/projects/{ids['new']}/files", 200),
        "unknown project": ("/projects/p-unknown", 404),
        "bad project id": ("/projects/BAD!", 404),
        "unknown page": ("/nope", 404),
        "questions of a planned project": (f"/projects/{ids['planned']}/clarify", 409),
    }
    for name, pid in ids.items():
        pages[f"project {name}"] = (f"/projects/{pid}", 200)
    return pages


def test_every_persian_page_has_no_english_interface(site: Site) -> None:
    site.store.approvals(site.ids["running"])  # pending request stays pending for "project"
    found: dict[str, list[str]] = {}
    for label, (target, status) in _pages(site).items():
        r = site.get(target)
        assert r.status == status, (label, r.status)
        body = r.body.decode("utf-8")
        if "text/html" in dict(r.headers).get("Content-Type", ""):
            assert '<html lang="fa" dir="rtl">' in body, label
            if words := _english(r, site):
                found[label] = words
    assert found == {}


def test_notice_page_shows_the_decision_in_persian(site: Site) -> None:
    pid = site.ids["running"]
    req = site.store.approvals(pid).get(site.request_id)
    fields = {"csrf": TOKEN, "fingerprint": req.fingerprint, "confirm": "yes"}
    r = site.post(f"/projects/{pid}/approvals/{req.id}/approve", fields)
    assert r.status == 303
    page = site.get(f"/projects/{pid}?decided={req.id}")
    assert "تأیید شد" in page.body.decode() and _english(page, site) == []


def test_run_panel_states_are_persian(site: Site) -> None:
    pid = site.ids["planned"]
    assert site.post(f"/projects/{pid}/start", {"csrf": TOKEN}).status == 303
    running = site.get(f"/projects/{pid}")
    assert _english(running, site) == []
    assert site.post(f"/projects/{pid}/stop", {"csrf": TOKEN}).status == 303
    assert _english(site.get(f"/projects/{pid}"), site) == []
    for code in (3,):
        site.procs[-1].finish(code)
    for _ in range(100):
        if "web.run_finished" in site.get(f"/projects/{pid}").body.decode():
            break
        time.sleep(0.02)
    finished = site.get(f"/projects/{pid}")
    assert "web.run_finished" in finished.body.decode()
    assert _english(finished, site) == []


@pytest.mark.parametrize(
    "label,target,fields,status",
    [
        ("wrong token", "approve", {"csrf": "x" * 43}, 403),
        ("other request shown", "approve", {"csrf": TOKEN, "fingerprint": "f"}, 409),
        ("stop without a run", "stop", {"csrf": TOKEN}, 409),
        ("start without a plan", "start-new", {"csrf": TOKEN}, 409),
        ("empty goal", "new", {"csrf": TOKEN, "goal": ""}, 400),
        ("wrong answers", "clarify", {"csrf": TOKEN, "a7": "x"}, 400),
    ],
)
def test_refused_forms_are_persian(
    site: Site, label: str, target: str, fields: dict[str, str], status: int
) -> None:
    ids = site.ids
    paths = {
        "approve": f"/projects/{ids['running']}/approvals/{site.request_id}/approve",
        "stop": f"/projects/{ids['done']}/stop",
        "start-new": f"/projects/{ids['new']}/start",
        "new": "/new",
        "clarify": f"/projects/{ids['new']}/clarify",
    }
    r = site.post(paths[target], fields)
    assert r.status == status, (label, r.status, r.body[:300])
    assert '<html lang="fa" dir="rtl">' in r.body.decode()
    assert _english(r, site) == [], label


def test_the_check_is_not_empty_in_english(tmp_path: Path) -> None:
    """The same filter on the English site finds English words on every page."""
    site = _site(tmp_path, lang="en")
    for label, (target, _status) in _pages(site).items():
        r = site.get(target)
        if "text/html" in dict(r.headers).get("Content-Type", ""):
            assert _english(r, site), label


def test_recorded_data_is_shown_as_written(site: Site) -> None:
    body = site.get(f"/projects/{site.ids['running']}").body.decode()
    assert "fetch_url" in body and "http://1.2.3.4:8080/a?x=1&amp;y=2" in body
    assert "یک صفحهٔ ساده (running)" in body
    listing = site.get(f"/projects/{quote(site.ids['done'])}/files").body.decode()
    assert "index.html" in listing and "data.bin" in listing


def test_ids_in_persian_refusals_are_isolated_left_to_right(site: Site) -> None:
    body = site.get("/projects/p-unknown").body.decode()
    assert 'پروژهٔ <bdi dir="ltr">p-unknown</bdi> وجود نداره' in body
    english = _site(site.root / "en", lang="en").get("/projects/p-unknown").body.decode()
    assert "no project p-unknown" in english and "<bdi" not in english


def test_isolated_parameters_are_escaped(site: Site) -> None:
    html = ws._coded_sentence("refusal.no_project", "fa", {"id": '<b>"x"</b>'})
    assert html == 'پروژهٔ <bdi dir="ltr">&lt;b&gt;&quot;x&quot;&lt;/b&gt;</bdi> وجود نداره'
    assert ws._coded_sentence("refusal.no_project", "en", {"id": "<b>"}) == "no project &lt;b&gt;"
