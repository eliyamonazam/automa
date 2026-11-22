"""Web UI: the product card (D157), the "try again" button (D156) and the zip (D158).

Projects are produced by the real orchestrator with a scripted FakeProvider (Rig);
the web side uses fake child processes (test_webui_runs). No network, no real calls.
"""

from __future__ import annotations

import io
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from automa.plan import Plan, TaskStatus
from automa.project import ProjectStatus
from automa.retry import RETRY_GRANTED, grant_retry
from automa.webui import bundle as wb
from automa.webui import server as ws
from test_orchestrator import PID, Rig, config, task_dict, write
from test_webui_runs import LEDGER, PORT, TOKEN, Env, FakePreflight, _settings

DELIVERABLE = {
    "summary": "A text file that says ok",
    "main_files": ["b.txt"],
    "how_to_use": "Open b.txt in any editor.",
}


class RetryPreflight(FakePreflight):
    """FakePreflight plus the optional grant (as the CLI's preflight does it)."""

    def __init__(self) -> None:
        super().__init__()
        self.granted: list[str] = []

    store: Any

    def grant_retry(self, project_id: str) -> list[str]:
        ids = grant_retry(self.store, project_id, config(), via="web")
        self.granted += ids
        return ids


def _plan(**deliverable: Any) -> Plan:
    data: dict[str, Any] = {"goal": "g", "tasks": [task_dict("a"), task_dict("b", deps=["a"])]}
    data["deliverable"] = deliverable or DELIVERABLE
    return Plan.model_validate(data)


def _rig(tmp_path: Path, script: list[object], plan: Plan | None = None) -> Rig:
    rig = Rig(tmp_path, script, plan or _plan())
    rig.run(max_attempts=2)
    return rig


def _env(tmp_path: Path, *, retry: bool = True, **kw: bool) -> Env:
    env = Env(tmp_path / "ws", **kw)
    if retry:
        env.preflight.__class__ = RetryPreflight
        env.preflight.granted = []  # type: ignore[attr-defined]
        env.preflight.store = env.store  # type: ignore[attr-defined]
    return env


@pytest.fixture
def done(tmp_path: Path) -> Iterator[Env]:
    _rig(tmp_path, [*write("a"), *write("b"), '{"passed": true, "issues": []}'])
    yield _env(tmp_path)


@pytest.fixture
def failed(tmp_path: Path) -> Iterator[Env]:
    _rig(tmp_path, [*write("a", "bad"), *write("a", "bad")])
    env = _env(tmp_path)
    yield env
    for _, _, proc in env.spawned:
        if proc.code is None:
            proc.finish(0)


def _page(env: Env, pid: str = PID) -> str:
    r = env.get(f"/projects/{pid}")
    assert r.status == 200
    return r.body.decode("utf-8")


# --- product card -------------------------------------------------------------------


def test_ready_product_card_comes_first_with_open_view_download(done: Env) -> None:
    body = _page(done)
    card = body[body.index('class="card deliver ready"') :].split("</section>")[0]
    assert "Your product is ready" in card
    assert "A text file that says ok" in card and "Open b.txt in any editor." in card
    assert f"/projects/{PID}/files/view?path=b.txt" in card
    assert f"/projects/{PID}/files/download?path=b.txt" in card
    assert f'href="/projects/{PID}/download.zip"' in card
    assert "2 of 2 steps passed" in card
    assert body.index("deliver ready") < body.index("<h2>Plan")
    assert "/retry" not in body
    # every task card lists its checks, all passed
    assert body.count('class="c-ok"') >= 2


def test_html_main_file_gets_an_open_button(tmp_path: Path) -> None:
    plan = Plan.model_validate(
        {
            "goal": "g",
            "tasks": [
                {
                    **task_dict("a"),
                    "acceptance_criteria": [
                        {
                            "description": "page exists",
                            "check": {"type": "file_exists", "path": "index.html"},
                        }
                    ],
                }
            ],
            "deliverable": {"summary": "a page", "main_files": ["index.html"]},
        }
    )
    rig = Rig(tmp_path, [], plan)
    rig.store.files_dir(PID).mkdir(parents=True, exist_ok=True)
    (rig.store.files_dir(PID) / "index.html").write_text("<p>hi</p>", encoding="utf-8")
    env = _env(tmp_path)
    body = _page(env)
    assert "/preview/index.html" in body and 'target="_blank"' in body


def test_promise_card_before_work(tmp_path: Path) -> None:
    Rig(tmp_path, [], _plan())
    body = _page(_env(tmp_path))
    card = body[body.index('class="card deliver promise"') :].split("</section>")[0]
    assert "What you will get" in card and "not made yet" in card
    assert "download.zip" not in card and "steps passed" not in card


def test_partial_card_explains_and_offers_retry(failed: Env) -> None:
    body = _page(failed)
    card = body[body.index('class="card deliver partial"') :].split("</section>")[0]
    assert "The product is not complete yet" in card
    assert "What is still missing" in card and "waits for an earlier step" in card
    assert f'action="/projects/{PID}/retry"' in card
    assert 'name="confirm" value="yes" required' in card
    assert f'name="csrf" value="{TOKEN}"' in card
    assert "not made" in card
    assert 'class="c-bad"' in body  # the failed criterion is marked


def test_read_only_partial_card_shows_the_command(tmp_path: Path) -> None:
    _rig(tmp_path, [*write("a", "bad"), *write("a", "bad")])
    env = _env(tmp_path, retry=False, actions=False, runs=False)
    body = _page(env)
    assert "/retry" not in body
    assert f"automa retry {PID}" in body


def test_no_retry_button_without_retry_preflight(tmp_path: Path) -> None:
    _rig(tmp_path, [*write("a", "bad"), *write("a", "bad")])
    env = _env(tmp_path, retry=False)
    assert '/retry"' not in _page(env)
    r = env.post(f"/projects/{PID}/retry", {"csrf": TOKEN, "confirm": "yes"})
    assert r.status == 404


def test_persian_card(done: Env) -> None:
    app = ws.WebApp(
        done.root,
        done.root / LEDGER,
        ws.allowed_hosts("127.0.0.1", PORT),
        actions=_settings(),
        runs=done.background,
        language="fa",
    )
    body = app.handle("GET", f"/projects/{PID}", f"127.0.0.1:{PORT}").body.decode("utf-8")
    assert "محصولت آماده‌ست" in body and "دانلود همهٔ فایل‌ها" in body
    assert "معیارهای پذیرش" in body


# --- retry from the web ---------------------------------------------------------------


def test_post_retry_grants_and_starts_with_yes(failed: Env) -> None:
    r = failed.post(f"/projects/{PID}/retry", {"csrf": TOKEN, "confirm": "yes"})
    assert r.status == 303, r.body
    assert failed.preflight.granted == ["a"]  # type: ignore[attr-defined]
    assert failed.preflight.commands == [(PID, True)]
    assert len(failed.spawned) == 1
    assert failed.store.task_states(PID)["a"].status is TaskStatus.PENDING
    grants = [e for e in failed.events(PID) if e["type"] == RETRY_GRANTED]
    assert grants and grants[0]["data"]["via"] == "web"  # type: ignore[index]
    # while running: no second retry, no button
    r = failed.post(f"/projects/{PID}/retry", {"csrf": TOKEN, "confirm": "yes"})
    assert r.status == 409 and len(failed.spawned) == 1


def test_retry_needs_the_confirm_box(failed: Env) -> None:
    r = failed.post(f"/projects/{PID}/retry", {"csrf": TOKEN})
    assert r.status == 409 and failed.spawned == []
    assert failed.store.load(PID).status is ProjectStatus.FAILED


@pytest.mark.parametrize(
    ("fields", "headers"),
    [
        ({"csrf": "x" * 43, "confirm": "yes"}, None),
        ({"csrf": TOKEN, "confirm": "yes", "extra": "1"}, None),
        ({"csrf": TOKEN, "confirm": "yes"}, {"Origin": "http://evil.example"}),
    ],
)
def test_retry_checks_csrf_fields_and_origin(
    failed: Env, fields: dict[str, str], headers: dict[str, str] | None
) -> None:
    r = failed.post(f"/projects/{PID}/retry", fields, headers=headers)
    assert r.status in (400, 403)
    assert failed.spawned == [] and failed.store.load(PID).status is ProjectStatus.FAILED


def test_retry_refused_for_done_or_unknown(done: Env) -> None:
    r = done.post(f"/projects/{PID}/retry", {"csrf": TOKEN, "confirm": "yes"})
    assert r.status == 409 and done.spawned == []
    r = done.post("/projects/p-nope/retry", {"csrf": TOKEN, "confirm": "yes"})
    assert r.status == 404
    r = done.post(f"/projects/{PID}/retry?x=1", {"csrf": TOKEN, "confirm": "yes"})
    assert r.status in (400, 404, 405)
    assert done.get(f"/projects/{PID}/retry").status in (404, 405)


def test_read_only_server_has_no_retry_endpoint(tmp_path: Path) -> None:
    _rig(tmp_path, [*write("a", "bad"), *write("a", "bad")])
    env = _env(tmp_path, retry=False, actions=False, runs=False)
    r = env.post(f"/projects/{PID}/retry", {"csrf": TOKEN, "confirm": "yes"})
    assert r.status in (403, 404, 405)
    assert env.store.load(PID).status is ProjectStatus.FAILED


# --- zip --------------------------------------------------------------------------------


def _zip(env: Env, pid: str = PID) -> zipfile.ZipFile:
    r = env.get(f"/projects/{pid}/download.zip")
    assert r.status == 200, r.body
    headers = dict(r.headers)
    assert headers["Content-Type"] == "application/zip"
    assert "attachment" in headers["Content-Disposition"]
    return zipfile.ZipFile(io.BytesIO(r.body))


def test_zip_has_files_and_delivery_report(done: Env) -> None:
    files = done.store.files_dir(PID)
    (files / "sub").mkdir()
    (files / "sub" / "n.txt").write_text("x", encoding="utf-8")
    zf = _zip(done)
    assert sorted(zf.namelist()) == ["DELIVERY.md", "a.txt", "b.txt", "sub/n.txt"]
    assert zf.read("b.txt") == b"ok"
    report = zf.read("DELIVERY.md").decode("utf-8")
    assert "The product is ready" in report and "`b.txt`" in report


def test_zip_skips_symlinks_and_keeps_a_user_delivery_file(done: Env, tmp_path: Path) -> None:
    files = done.store.files_dir(PID)
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    os.symlink(secret, files / "link.txt")
    os.symlink(tmp_path, files / "dirlink")
    (files / "DELIVERY.md").write_text("mine", encoding="utf-8")
    zf = _zip(done)
    names = zf.namelist()
    assert "link.txt" not in names and not any(n.startswith("dirlink") for n in names)
    assert zf.read("DELIVERY.md") == b"mine"
    assert "DELIVERY-automa.md" in names
    assert b"secret" not in b"".join(zf.read(n) for n in names)


def test_zip_refusals(done: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    assert done.get("/projects/p-nope/download.zip").status == 404
    assert done.get("/projects/BAD!/download.zip").status == 404
    monkeypatch.setattr(wb, "MAX_BUNDLE_BYTES", 1)
    r = done.get(f"/projects/{PID}/download.zip")
    assert r.status == 413
    monkeypatch.setattr(wb, "MAX_BUNDLE_BYTES", 10**6)
    monkeypatch.setattr(wb, "MAX_BUNDLE_FILES", 1)
    assert done.get(f"/projects/{PID}/download.zip").status == 413


def test_zip_of_project_without_files_or_plan(tmp_path: Path) -> None:
    env = _env(tmp_path)
    pid = env.store.create("goal").id
    zf = _zip(env, pid)
    assert zf.namelist() == []
    assert wb.parse_bundle_path(f"/projects/{pid}/download.zip") == pid
    assert wb.parse_bundle_path(f"/projects/{pid}/download.zip/x") is None
    assert wb.bundle_name(pid) == f"{pid}.zip"
