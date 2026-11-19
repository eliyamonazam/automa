"""Tests for "Start run" in the web UI (``automa web --allow-actions``, D125).

The run is started as a child process (``automa run``); here the child is a fake
process (or, in a few tests, a harmless real Python child that only prints). The
same Host / same-origin / CSRF checks as approvals. No model and no network.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import pytest

from automa import cli
from automa.config import AutomaConfig
from automa.plan import Plan
from automa.project import ProjectStatus, ProjectStore
from automa.webui import actions as wa
from automa.webui import runs as wr
from automa.webui import server as ws
from automa.webui.render import ActionForms, RunPanel, render_project
from automa.webui.views import build_project_view

LEDGER = "ledger.sqlite3"
PORT = 8765
HOST = f"127.0.0.1:{PORT}"
ORIGIN = f"http://{HOST}"
TOKEN = "t" * 43
FORM = "application/x-www-form-urlencoded"
EVIL = '<script>alert(1)</script> "q" ‮'


def _settings() -> wa.ActionSettings:
    hosts = ws.allowed_hosts("127.0.0.1", PORT)
    return wa.ActionSettings(csrf_token=TOKEN, origins=wa.allowed_origins(hosts))


def _plan() -> Plan:
    return Plan.model_validate(
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


class FakeProcess:
    _next_pid = 1000

    def __init__(self) -> None:
        FakeProcess._next_pid += 1
        self._pid = FakeProcess._next_pid
        self.code: int | None = None
        self.done = threading.Event()

    @property
    def pid(self) -> int:
        return self._pid

    def finish(self, code: int) -> None:
        self.code = code
        self.done.set()

    def poll(self) -> int | None:
        return self.code

    def wait(self) -> int:
        self.done.wait(30)
        return self.code if self.code is not None else -15


class FakePreflight:
    def __init__(self) -> None:
        self.confirm = False
        self.commands: list[tuple[str, bool]] = []

    def needs_confirm(self, project_id: str) -> bool:
        return self.confirm

    def command(self, project_id: str, *, yes: bool) -> list[str]:
        self.commands.append((project_id, yes))
        return ["automa", "run", project_id, *(["--yes"] if yes else [])]


class Env:
    def __init__(self, root: Path, *, actions: bool = True, runs: bool = True) -> None:
        self.root = root
        self.store = ProjectStore.for_workspace(root)
        self.preflight = FakePreflight()
        self.spawned: list[tuple[list[str], Path, FakeProcess]] = []
        self.spawn_error: OSError | None = None
        self.background = wr.BackgroundRuns(self.store, self.preflight, spawn=self._spawn)
        self.app = ws.WebApp(
            root,
            root / LEDGER,
            ws.allowed_hosts("127.0.0.1", PORT),
            actions=_settings() if actions else None,
            runs=self.background if runs else None,
        )

    def _spawn(self, argv: Sequence[str], log_path: Path) -> FakeProcess:
        if self.spawn_error is not None:
            raise self.spawn_error
        log_path.write_text("project: x\nestimate: about $0.05\n", encoding="utf-8")
        proc = FakeProcess()
        self.spawned.append((list(argv), log_path, proc))
        return proc

    def project(self, *, plan: bool = True, status: ProjectStatus | None = None) -> str:
        pid = self.store.create("Build a site " + EVIL).id
        if plan:
            self.store.save_plan(pid, _plan())
        if status is not None:
            self.store.set_status(pid, status)
        return pid

    def get(self, path: str) -> ws.Response:
        return self.app.handle("GET", path, HOST)

    def post(
        self,
        path: str,
        fields: dict[str, str] | None = None,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> ws.Response:
        data = body if body is not None else urlencode(fields or {}).encode("ascii")
        hdrs = {"Origin": ORIGIN, "Content-Type": FORM, "Sec-Fetch-Site": "same-origin"}
        hdrs.update(headers or {})
        return self.app.handle("POST", path, HOST, list(hdrs.items()), data)

    def start(self, pid: str, **extra: str) -> ws.Response:
        return self.post(f"/projects/{pid}/start", {"csrf": TOKEN, **extra})

    def events(self, pid: str) -> list[dict[str, object]]:
        path = self.store.project_dir(pid) / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def wait_event(self, pid: str, kind: str) -> dict[str, object]:
        for _ in range(200):
            found = [e for e in self.events(pid) if e["type"] == kind]
            if found:
                return found[-1]
            time.sleep(0.01)
        raise AssertionError(f"no {kind} event")


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    e = Env(tmp_path)
    yield e
    for _, _, proc in e.spawned:  # let every watcher thread end
        if proc.code is None:
            proc.finish(0)


def _body(r: ws.Response) -> str:
    return r.body.decode("utf-8")


# -- availability -------------------------------------------------------------------


def test_read_only_server_never_starts(tmp_path: Path) -> None:
    e = Env(tmp_path, actions=False)
    pid = e.project()
    assert e.start(pid).status == 405
    assert "Start run" not in _body(e.get(f"/projects/{pid}"))
    assert not e.app.runs_enabled
    assert e.spawned == []


def test_action_server_without_runs_has_no_start(tmp_path: Path) -> None:
    e = Env(tmp_path, runs=False)
    pid = e.project()
    assert e.start(pid).status == 404
    assert "Start run" not in _body(e.get(f"/projects/{pid}"))
    assert e.spawned == []


def test_get_start_path_is_not_an_action(env: Env) -> None:
    pid = env.project()
    assert env.get(f"/projects/{pid}/start").status == 404
    assert env.app.handle("HEAD", f"/projects/{pid}/start", HOST).status == 404
    assert env.spawned == []


def test_project_page_shows_start_button(env: Env) -> None:
    pid = env.project()
    body = _body(env.get(f"/projects/{pid}"))
    assert f'action="/projects/{pid}/start"' in body
    assert f'name="csrf" value="{TOKEN}"' in body
    assert 'name="confirm"' not in body
    assert "<script" not in body
    assert env.spawned == []


def test_project_page_shows_confirm_box_above_threshold(env: Env) -> None:
    env.preflight.confirm = True
    pid = env.project()
    body = _body(env.get(f"/projects/{pid}"))
    assert 'name="confirm" value="yes" required' in body


@pytest.mark.parametrize("status", [ProjectStatus.RUNNING, ProjectStatus.DONE])
def test_no_start_button_when_running_or_done(env: Env, status: ProjectStatus) -> None:
    pid = env.project(status=status)
    body = _body(env.get(f"/projects/{pid}"))
    assert "Start run" not in body
    assert "Not startable now" in body


def test_no_start_button_without_plan(env: Env) -> None:
    pid = env.project(plan=False)
    assert "Start run" not in _body(env.get(f"/projects/{pid}"))


# -- request checks before anything happens ----------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Content-Type": "text/plain"},
    ],
)
def test_cross_origin_or_bad_type_refused(env: Env, headers: dict[str, str]) -> None:
    pid = env.project()
    r = env.post(f"/projects/{pid}/start", {"csrf": TOKEN}, headers=headers)
    assert r.status in (400, 403, 415)
    assert env.spawned == []
    assert env.preflight.commands == []


def test_missing_origin_and_referer_refused(env: Env) -> None:
    pid = env.project()
    hdrs = [("Content-Type", FORM)]
    data = urlencode({"csrf": TOKEN}).encode()
    r = env.app.handle("POST", f"/projects/{pid}/start", HOST, hdrs, data)
    assert r.status == 403
    assert env.spawned == []


def test_wrong_host_refused(env: Env) -> None:
    pid = env.project()
    data = urlencode({"csrf": TOKEN}).encode()
    hdrs = [("Origin", ORIGIN), ("Content-Type", FORM)]
    r = env.app.handle("POST", f"/projects/{pid}/start", "evil.example:8765", hdrs, data)
    assert r.status == 421
    assert env.spawned == []


@pytest.mark.parametrize("token", ["", "x" * 43, TOKEN + "x"])
def test_wrong_token_refused(env: Env, token: str) -> None:
    pid = env.project()
    r = env.post(f"/projects/{pid}/start", {"csrf": token})
    assert r.status == 403
    assert env.spawned == []


def test_unknown_field_refused(env: Env) -> None:
    pid = env.project()
    r = env.post(f"/projects/{pid}/start", {"csrf": TOKEN, "argv": "rm -rf /"})
    assert r.status == 400
    assert env.spawned == []


def test_duplicate_field_refused(env: Env) -> None:
    pid = env.project()
    r = env.post(f"/projects/{pid}/start", body=f"csrf={TOKEN}&csrf={TOKEN}".encode())
    assert r.status == 400
    assert env.spawned == []


def test_duplicate_origin_header_refused(env: Env) -> None:
    pid = env.project()
    hdrs = [("Origin", ORIGIN), ("Origin", "http://evil.example"), ("Content-Type", FORM)]
    data = urlencode({"csrf": TOKEN}).encode()
    r = env.app.handle("POST", f"/projects/{pid}/start", HOST, hdrs, data)
    assert r.status == 400
    assert env.spawned == []


def test_bad_confirm_value_refused(env: Env) -> None:
    env.preflight.confirm = True
    pid = env.project()
    assert env.start(pid, confirm="on").status == 400
    assert env.spawned == []


@pytest.mark.parametrize("path", ["/projects/NOPE/start", "/projects/../start", "/projects//start"])
def test_bad_project_path(env: Env, path: str) -> None:
    assert env.post(path, {"csrf": TOKEN}).status == 404
    assert env.spawned == []


def test_unknown_project(env: Env) -> None:
    assert env.start("p-nothere").status == 404
    assert env.spawned == []


def test_query_string_not_accepted(env: Env) -> None:
    pid = env.project()
    assert env.post(f"/projects/{pid}/start?x=1", {"csrf": TOKEN}).status == 404
    assert env.spawned == []


# -- starting -------------------------------------------------------------------------


def test_start_below_threshold_without_yes(env: Env) -> None:
    pid = env.project()
    r = env.start(pid)
    assert r.status == 303
    assert dict(r.headers)["Location"] == f"/projects/{pid}"
    assert len(env.spawned) == 1
    argv, log, _ = env.spawned[0]
    # Below the threshold no --yes: if the range grew meanwhile, the child stops (exit 11).
    assert argv == ["automa", "run", pid]
    assert env.preflight.commands == [(pid, False)]
    assert log.parent == env.store.project_dir(pid) / wr.RUNS_DIR
    assert log.name.startswith("run-") and log.suffix == ".log"
    started = env.wait_event(pid, wr.STARTED_EVENT)
    assert started["data"] == {"confirmed": False, "log": log.name}


def test_start_above_threshold_needs_box(env: Env) -> None:
    env.preflight.confirm = True
    pid = env.project()
    r = env.start(pid)
    assert r.status == 409
    assert "tick the box" in _body(r)
    assert env.spawned == []
    assert env.preflight.commands == []
    assert not [e for e in env.events(pid) if str(e["type"]).startswith("web.")]


def test_start_above_threshold_with_box_passes_yes(env: Env) -> None:
    env.preflight.confirm = True
    pid = env.project()
    assert env.start(pid, confirm="yes").status == 303
    assert env.spawned[0][0] == ["automa", "run", pid, "--yes"]
    assert env.wait_event(pid, wr.STARTED_EVENT)["data"]["confirmed"] is True


def test_box_below_threshold_does_not_add_yes(env: Env) -> None:
    pid = env.project()
    assert env.start(pid, confirm="yes").status == 303
    assert env.spawned[0][0] == ["automa", "run", pid]


def test_second_start_while_running_refused(env: Env) -> None:
    pid = env.project()
    assert env.start(pid).status == 303
    r = env.start(pid)
    assert r.status == 409
    assert "already running" in _body(r)
    assert len(env.spawned) == 1
    env.spawned[0][2].finish(0)


def test_start_after_finish_allowed(env: Env) -> None:
    pid = env.project()
    assert env.start(pid).status == 303
    env.spawned[0][2].finish(7)
    env.wait_event(pid, wr.FINISHED_EVENT)
    assert env.start(pid).status == 303
    assert len(env.spawned) == 2
    assert env.spawned[0][1] != env.spawned[1][1]
    env.spawned[1][2].finish(0)


def test_two_projects_run_independently(env: Env) -> None:
    a, b = env.project(), env.project()
    assert env.start(a).status == 303
    assert env.start(b).status == 303
    assert len(env.spawned) == 2
    for _, _, p in env.spawned:
        p.finish(0)


@pytest.mark.parametrize(
    ("status", "word"), [(ProjectStatus.RUNNING, "marked running"), (ProjectStatus.DONE, "done")]
)
def test_status_refused(env: Env, status: ProjectStatus, word: str) -> None:
    pid = env.project(status=status)
    r = env.start(pid)
    assert r.status == 409
    assert word in _body(r)
    assert env.spawned == []


def test_paused_project_can_resume(env: Env) -> None:
    pid = env.project(status=ProjectStatus.PAUSED)
    assert env.start(pid).status == 303
    assert len(env.spawned) == 1


def test_no_plan_refused(env: Env) -> None:
    pid = env.project(plan=False)
    r = env.start(pid)
    assert r.status == 409
    assert "no plan" in _body(r)
    assert env.spawned == []


def test_spawn_failure_is_a_clean_500(env: Env) -> None:
    env.spawn_error = FileNotFoundError(2, "No such file or directory")
    pid = env.project()
    r = env.start(pid)
    assert r.status == 500
    assert "could not start" in _body(r)
    assert env.background.state(pid) is None
    env.spawn_error = None
    assert env.start(pid).status == 303


def test_runs_dir_symlink_refused(env: Env, tmp_path: Path) -> None:
    pid = env.project()
    target = tmp_path / "elsewhere"
    target.mkdir()
    (env.store.project_dir(pid) / wr.RUNS_DIR).symlink_to(target)
    r = env.start(pid)
    assert r.status == 500
    assert "symlink" in _body(r)
    assert env.spawned == []
    assert list(target.iterdir()) == []


def test_concurrent_starts_spawn_once(env: Env) -> None:
    pid = env.project()
    results: list[int] = []

    def go() -> None:
        results.append(env.start(pid).status)

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [303] + [409] * 7
    assert len(env.spawned) == 1
    env.spawned[0][2].finish(0)


# -- state on the page ----------------------------------------------------------------


def test_page_shows_running_state_and_log_tail(env: Env) -> None:
    pid = env.project()
    env.start(pid)
    log = env.spawned[0][1]
    log.write_text("estimate: about $0.05\n" + EVIL + "\n", encoding="utf-8")
    body = _body(env.get(f"/projects/{pid}"))
    assert "running in the background" in body
    assert f"web-runs/{log.name}" in body
    assert "Start run" not in body
    assert "<script>alert" not in body
    assert "&lt;script&gt;" in body
    assert "‮" not in body
    env.spawned[0][2].finish(0)


def test_page_shows_exit_meaning(env: Env) -> None:
    pid = env.project()
    env.start(pid)
    env.spawned[0][2].finish(3)
    env.wait_event(pid, wr.FINISHED_EVENT)
    body = _body(env.get(f"/projects/{pid}"))
    assert "exit code 3" in body and "budget stop" in body
    assert "Start run" in body  # paused/ended: can start again


def test_finished_event_records_exit_code(env: Env) -> None:
    pid = env.project()
    env.start(pid)
    env.spawned[0][2].finish(130)
    ev = env.wait_event(pid, wr.FINISHED_EVENT)
    assert ev["data"]["exit_code"] == 130


@pytest.mark.parametrize(
    ("code", "word"),
    [
        (0, "every task passed"),
        (6, "failed for good"),
        (11, "confirm"),
        (130, "interrupted"),
        (-9, "signal 9"),
        (42, "see the log"),
    ],
)
def test_exit_meanings(code: int, word: str) -> None:
    assert word in wr.exit_meaning(code)


def test_state_summary() -> None:
    now = datetime.now(UTC)
    running = wr.RunState(now, Path("x.log"), False, None)
    assert running.running and "running" in running.summary
    ended = wr.RunState(now, Path("x.log"), True, 0)
    assert not ended.running and "exit code 0" in ended.summary


def test_render_panel_without_forms_is_empty(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    pid = store.create("g").id
    view = build_project_view(store, pid, 0)
    panel = RunPanel(can_start=True, needs_confirm=False)
    assert "Start run" not in render_project(view, None, run=panel)
    assert "Start run" in render_project(view, ActionForms(TOKEN), run=panel)


# -- path parsing and log tail ---------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "pid"),
    [
        ("/projects/p-1/start", "p-1"),
        ("/projects/p-1/start/", None),
        ("/projects/P-1/start", None),
        ("/projects/p-1/approvals/a/start", None),
        ("/projects/%2e%2e/start", None),
        ("/start", None),
    ],
)
def test_parse_start_path(path: str, pid: str | None) -> None:
    assert wr.parse_start_path(path) == pid


def test_read_tail_limits(tmp_path: Path) -> None:
    log = tmp_path / "a.log"
    log.write_text("".join(f"line {i}\n" for i in range(5000)), encoding="utf-8")
    tail = wr.read_tail(log)
    assert len(tail) == wr.LOG_TAIL_LINES
    assert tail[-1] == "line 4999"
    assert wr.read_tail(tmp_path / "missing.log") == ()


def test_read_tail_refuses_symlink(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET\n")
    link = tmp_path / "a.log"
    link.symlink_to(secret)
    assert wr.read_tail(link) == ()


def test_read_tail_invalid_utf8(tmp_path: Path) -> None:
    log = tmp_path / "a.log"
    log.write_bytes(b"ok\n\xff\xfe bad\n")
    assert wr.read_tail(log)[0] == "ok"


# -- the real child process ----------------------------------------------------------


def test_default_command(tmp_path: Path) -> None:
    cfg = tmp_path / "automa.toml"
    argv = wr.default_command("p-1", yes=False, config_path=None)
    assert argv == [sys.executable, "-m", "automa", "run", "p-1"]
    argv = wr.default_command("p-1", yes=True, config_path=cfg)
    assert argv == [sys.executable, "-m", "automa", "run", "p-1", "--yes", "--config", str(cfg)]


def test_default_command_rejects_bad_id() -> None:
    with pytest.raises(Exception, match="project"):
        wr.default_command("--force", yes=False, config_path=None)


def _automa(*args: str) -> list[str]:
    return [sys.executable, "-m", "automa", "run", *args]


def test_popen_spawn_real_child(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    missing = tmp_path / "missing.toml"
    proc = wr.popen_spawn(_automa("p-1", "--config", str(missing)), log)
    assert proc.wait() == 2  # configuration error: nothing ran
    assert "config file not found" in log.read_text()
    assert (log.stat().st_mode & 0o777) == 0o600


@pytest.mark.parametrize(
    "argv",
    [
        [sys.executable, "-c", "print(1)"],
        ["python3", "-m", "automa", "run", "p-1"],
        [sys.executable, "-m", "automa", "approve", "p-1"],
        [sys.executable, "-m", "automa", "run"],
        ["sh", "-c", "automa run p-1"],
    ],
)
def test_popen_spawn_only_automa_run(tmp_path: Path, argv: list[str]) -> None:
    log = tmp_path / "run.log"
    with pytest.raises(ValueError, match="automa run"):
        wr.popen_spawn(argv, log)
    assert not log.exists()


def test_popen_spawn_never_overwrites(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("old")
    with pytest.raises(FileExistsError):
        wr.popen_spawn(_automa("--help"), log)
    assert log.read_text() == "old"


def test_popen_spawn_refuses_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    log = tmp_path / "run.log"
    log.symlink_to(target)
    with pytest.raises(OSError):
        wr.popen_spawn(_automa("--help"), log)
    assert not target.exists()


def test_background_runs_with_real_child(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    pid = store.create("g").id
    store.save_plan(pid, _plan())
    missing = tmp_path / "missing.toml"

    class Pre:
        def needs_confirm(self, project_id: str) -> bool:
            return False

        def command(self, project_id: str, *, yes: bool) -> list[str]:
            return _automa(project_id, "--config", str(missing))

    runs = wr.BackgroundRuns(store, Pre())
    runs.start(pid, {"csrf": TOKEN})
    state = None
    for _ in range(1500):
        state = runs.state(pid)
        if state is not None and not state.running:
            break
        time.sleep(0.02)
    assert state is not None and state.exit_code == 2
    assert any("config file not found" in line for line in state.tail)
    assert "configuration error" in state.summary
    assert store.load(pid).status is ProjectStatus.NEW  # nothing ran


def test_background_runs_refuses_other_commands(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    pid = store.create("g").id
    store.save_plan(pid, _plan())

    class Pre:
        def needs_confirm(self, project_id: str) -> bool:
            return False

        def command(self, project_id: str, *, yes: bool) -> list[str]:
            return ["sh", "-c", "echo pwned"]

    runs = wr.BackgroundRuns(store, Pre())
    with pytest.raises(wa.ActionRefused) as info:
        runs.start(pid, {"csrf": TOKEN})
    assert info.value.status == 500
    assert runs.state(pid) is None


def test_python_m_automa_is_the_cli() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "automa", "run", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert out.returncode == 0
    assert "--yes" in out.stdout


# -- the CLI preflight --------------------------------------------------------------


def _config(tmp_path: Path, confirm: float) -> AutomaConfig:
    price = {"input_per_mtok": 1.0, "output_per_mtok": 2.0}
    return AutomaConfig.model_validate(
        {
            "workspace": str(tmp_path / "ws"),
            "tiers": {"small": "m-s", "medium": "m-m", "large": "m-l"},
            "prices": {"m-s": price, "m-m": price, "m-l": price},
            "limits": {"confirm_above_usd": confirm},
        }
    )


@pytest.mark.parametrize(("confirm", "expected"), [(0.0, True), (1000.0, False)])
def test_cli_preflight_threshold(tmp_path: Path, confirm: float, expected: bool) -> None:
    path = tmp_path / "automa.toml"
    config = _config(tmp_path, confirm)
    store = ProjectStore.for_workspace(config.workspace)
    pid = store.create("g").id
    store.save_plan(pid, _plan())
    pre = cli._WebRunPreflight(config, path)
    assert pre.needs_confirm(pid) is expected
    argv = pre.command(pid, yes=True)
    assert argv[-2:] == ["--config", str(path.resolve())]
    assert "--yes" in argv


def test_cli_preflight_unavailable_estimate_asks(tmp_path: Path) -> None:
    config = _config(tmp_path, 1000.0)
    store = ProjectStore.for_workspace(config.workspace)
    pid = store.create("g").id  # no plan: estimate unavailable
    assert cli._WebRunPreflight(config, None).needs_confirm(pid) is True


@pytest.mark.parametrize("allow", [True, False])
def test_create_server_wires_runs(tmp_path: Path, allow: bool) -> None:
    import http.client

    store = ProjectStore.for_workspace(tmp_path)
    pid = store.create("g").id
    store.save_plan(pid, _plan())
    server = ws.create_server(
        tmp_path, tmp_path / LEDGER, port=0, allow_actions=allow, runs=FakePreflight()
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", f"/projects/{pid}", headers={"Host": f"127.0.0.1:{port}"})
        body = conn.getresponse().read().decode("utf-8")
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
    assert ("Start run" in body) is allow
