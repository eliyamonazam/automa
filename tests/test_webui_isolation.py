# ruff: noqa: S106 - test passwords
"""Tests for keeping accounts apart in the web UI (M15, D182): every project has an
owner, a member sees and touches only their own projects (the same 404 as a missing
one everywhere under ``/projects/``), an owner account sees everything.

Only in-process calls; no network, no model calls.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from automa.budget import CostLedger
from automa.project import ProjectStore
from automa.project.store import Project, ProjectStoreError
from automa.users import UserRole
from automa.webui import server as ws
from automa.webui import views as wv
from test_ledger import FixedClock, make_pair
from test_webui_ideas import FakePlanner
from test_webui_login import HOST, LEDGER, PW, PW_KID, Env, cookie_of

# Every GET route that takes a project id; each must be refused for another's project.
GET_ROUTES = (
    "/projects/{pid}",
    "/projects/{pid}/files",
    "/projects/{pid}/files/view?path=out.txt",
    "/projects/{pid}/files/raw?path=out.txt",
    "/projects/{pid}/files/download?path=out.txt",
    "/projects/{pid}/preview/page.html",
    "/projects/{pid}/download.zip",
    "/projects/{pid}/thinking",
    "/projects/{pid}/clarify",
)
POST_ROUTES = (
    "/projects/{pid}/start",
    "/projects/{pid}/stop",
    "/projects/{pid}/brake",
    "/projects/{pid}/retry",
    "/projects/{pid}/change",
    "/projects/{pid}/clarify",
    "/projects/{pid}/requests/req-0123456789ab/answer",
    "/projects/{pid}/requests/req-0123456789ab/decline",
    "/projects/{pid}/approvals/{rid}/approve",
    "/projects/{pid}/approvals/{rid}/reject",
)


class IsoEnv(Env):
    """``p1`` (no owner: made from the CLI), ``pdad`` (dad's), ``pkid`` (kid's)."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.users.add("kid", "kid-pass-123", UserRole.MEMBER)
        self.store.create("dad goal: bourse screen", project_id="pdad", owner="dad")
        self.store.create("kid goal: my web page", project_id="pkid", owner="kid")
        for pid in ("p1", "pdad", "pkid"):
            files = self.store.files_dir(pid)
            (files / "out.txt").write_text(f"secret of {pid}")
            (files / "page.html").write_text(f"<p>page of {pid}</p>")
        self.kid_req = self.store.approvals("pkid").request(
            tool="publish",
            risk=self.req.risk,
            arguments={"text": "kid"},
            reason="needs approval",
            task_id="a",
        )


@pytest.fixture
def env(tmp_path: Path) -> IsoEnv:
    return IsoEnv(tmp_path)


def _body(r: ws.Response) -> str:
    return r.body.decode("utf-8", "replace")


# Random per-session tokens (CSRF of the sign-out form) are base64url text: a short
# project id such as "p1" can occur inside one by chance, so id checks skip them.
_TOKEN_VALUE = re.compile(r'value="[A-Za-z0-9_-]{32,}"')


def _shown(r: ws.Response) -> str:
    """The page without random token values (for "this id is not on the page")."""
    return _TOKEN_VALUE.sub('value=""', _body(r))


# -- the store keeps the owner ------------------------------------------------------------


def test_owner_is_stored_and_old_files_stay_without_the_key(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("a", project_id="pa", owner="dad")
    store.create("b", project_id="pb")
    assert store.load("pa").owner == "dad"
    assert store.load("pb").owner is None
    raw_b = json.loads(store.project_file("pb").read_text())
    assert "owner" not in raw_b  # unchanged format for projects made from the CLI
    assert json.loads(store.project_file("pa").read_text())["owner"] == "dad"


def test_owner_survives_saves_and_cannot_change(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    p = store.create("a", project_id="pa", owner="dad")
    saved = store.save(p.model_copy(update={"goal": "a2"}))
    assert saved.owner == "dad" and store.load("pa").owner == "dad"
    with pytest.raises(ProjectStoreError, match="owner"):
        store.save(saved.model_copy(update={"owner": "kid"}))
    with pytest.raises(ProjectStoreError, match="owner"):
        store.save(saved.model_copy(update={"owner": None}))
    assert store.load("pa").owner == "dad"


@pytest.mark.parametrize("bad", ["", "Dad", "../x", "a", "dad\n", "x" * 40, "د"])
def test_bad_owner_names_are_refused(tmp_path: Path, bad: str) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    with pytest.raises(ProjectStoreError):
        store.create("a", project_id="pa", owner=bad)
    assert not store.exists("pa")


def test_tampered_owner_in_project_json_is_unreadable(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("a", project_id="pa", owner="dad")
    path = store.project_file("pa")
    data = json.loads(path.read_text())
    data["owner"] = "../../eli"
    path.write_text(json.dumps(data))
    with pytest.raises(ProjectStoreError):
        store.load("pa")


def test_project_model_rejects_unknown_fields_still() -> None:
    with pytest.raises(ValueError):
        Project.model_validate(
            {
                "id": "pa",
                "goal": "g",
                "created_at": "2026-11-23T00:00:00Z",
                "updated_at": "2026-11-23T00:00:00Z",
                "owners": ["dad"],
            }
        )


# -- the list on the home page -----------------------------------------------------------


def test_member_list_shows_only_their_projects(env: IsoEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    body = _shown(env.get("/", dad))
    assert "bourse screen" in body
    assert "my web page" not in body and "secret family goal" not in body
    assert "pkid" not in body


def test_owner_account_sees_every_project(env: IsoEnv) -> None:
    eli = env.signed_in("eli", PW)
    body = _body(env.get("/", eli))
    for goal in ("bourse screen", "my web page", "secret family goal"):
        assert goal in body


def test_no_accounts_shows_every_project(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("dad goal", project_id="pdad", owner="dad")
    store.create("cli goal", project_id="p1")
    app = ws.WebApp(tmp_path, tmp_path / LEDGER, ws.allowed_hosts("127.0.0.1", 8765))
    body = _body(app.handle("GET", "/", HOST))
    assert "dad goal" in body and "cli goal" in body
    assert app.handle("GET", "/projects/pdad", HOST).status == 200


def test_member_list_hides_unreadable_projects_of_others(env: IsoEnv) -> None:
    env.store.project_file("pkid").write_text("{broken")
    dad = env.signed_in("dad", PW_KID)
    body = _shown(env.get("/", dad))
    assert "pkid" not in body and "bourse screen" in body
    eli = env.signed_in("eli", PW)
    assert "pkid" in _body(env.get("/", eli))  # the owner account still sees the error


def test_build_overview_owner_filter(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("a", project_id="pa", owner="dad")
    store.create("b", project_id="pb", owner="kid")
    store.create("c", project_id="bench-x")
    assert [r.id for r in wv.build_overview(store, {}, owner="dad").projects] == ["pa"]
    view = wv.build_overview(store, {}, include_tests=False, owner="dad")
    assert view.hidden_tests == 0
    assert {r.id for r in wv.build_overview(store, {}).projects} == {"pa", "pb", "bench-x"}


# -- every project route -----------------------------------------------------------------


@pytest.mark.parametrize("route", GET_ROUTES)
@pytest.mark.parametrize("pid", ["pkid", "p1", "nope"])
def test_member_gets_404_for_projects_not_theirs(env: IsoEnv, route: str, pid: str) -> None:
    dad = env.signed_in("dad", PW_KID)
    r = env.get(route.format(pid=pid), dad)
    assert r.status == 404
    body = _body(r)
    assert "secret of" not in body and "page of" not in body
    assert "my web page" not in body and "secret family goal" not in body
    # The same page as a missing project: nothing tells "exists, not yours" apart.
    assert pid not in _shown(r)


@pytest.mark.parametrize("route", POST_ROUTES)
@pytest.mark.parametrize("pid", ["pkid", "p1"])
def test_member_posts_to_others_projects_are_404(env: IsoEnv, route: str, pid: str) -> None:
    dad = env.signed_in("dad", PW_KID)
    csrf = env.csrf_of(dad)
    rid = env.kid_req.id if pid == "pkid" else env.req.id
    r = env.post(route.format(pid=pid, rid=rid), {"csrf": csrf}, cookie=dad)
    assert r.status == 404
    # Nothing happened: approvals still wait, no stop was requested.
    assert env.store.approvals("pkid").get(env.kid_req.id).status.value == "pending"
    assert env.store.approvals("p1").get(env.req.id).status.value == "pending"
    assert not env.store.stop_requested(pid)


def test_member_head_is_refused_too(env: IsoEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    headers = [("Cookie", f"automa_session={dad}")]
    assert env.app.handle("HEAD", "/projects/pkid", HOST, headers).status == 404
    assert env.app.handle("HEAD", "/projects/pdad", HOST, headers).status == 200


@pytest.mark.parametrize(
    "path",
    [
        "/projects/pkid%2F..%2Fpdad",
        "/projects/%70kid",
        "/projects/pkid/../pdad",
        "/projects/PKID",
        "/projects/pkid%00",
        "/projects//pkid",
        "/projects/",
        "/projects",
        "/projects/pdad/../pkid/files/raw?path=out.txt",
    ],
)
def test_member_odd_spellings_fail_closed(env: IsoEnv, path: str) -> None:
    dad = env.signed_in("dad", PW_KID)
    r = env.get(path, dad)
    body = _body(r)
    assert "secret of pkid" not in body and "my web page" not in body
    if not path.startswith("/projects/pdad"):
        assert r.status == 404


def test_member_reaches_their_own_project_everywhere(env: IsoEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    assert env.get("/projects/pdad", dad).status == 200
    files = env.get("/projects/pdad/files/view?path=out.txt", dad)
    assert files.status == 200 and b"secret of pdad" in files.body
    assert env.get("/projects/pdad/download.zip", dad).status == 200


def test_owner_account_reaches_every_project(env: IsoEnv) -> None:
    eli = env.signed_in("eli", PW)
    for pid in ("p1", "pdad", "pkid"):
        assert env.get(f"/projects/{pid}", eli).status == 200
        raw = env.get(f"/projects/{pid}/files/view?path=out.txt", eli)
        assert raw.status == 200 and f"secret of {pid}".encode() in raw.body


def test_owner_account_may_still_decide_any_approval(env: IsoEnv) -> None:
    eli = env.signed_in("eli", PW)
    csrf = env.csrf_of(eli)
    form = {"csrf": csrf, "fingerprint": env.kid_req.fingerprint}
    r = env.post(f"/projects/pkid/approvals/{env.kid_req.id}/reject", form, cookie=eli)
    assert r.status == 303, _body(r)
    assert env.store.approvals("pkid").get(env.kid_req.id).status.value == "rejected"


def test_ownership_is_read_on_every_request(env: IsoEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    assert env.get("/projects/pdad", dad).status == 200
    env.store.project_file("pdad").write_text("{broken")
    assert env.get("/projects/pdad", dad).status == 404  # unreadable: fail closed


def test_kid_cannot_see_dads_project_either(env: IsoEnv) -> None:
    kid = cookie_of(env.login("kid", "kid-pass-123"))
    assert env.get("/projects/pdad", kid).status == 404
    assert env.get("/projects/pkid", kid).status == 200
    assert "bourse screen" not in _body(env.get("/", kid))


# -- spending and the owner's notices -----------------------------------------------------


def _spend(root: Path, pid: str | None, usd: float, when) -> None:  # type: ignore[no-untyped-def]
    with CostLedger(root / LEDGER, clock=FixedClock(when)) as led:
        led.record(*make_pair(usd, project=pid))


def _month(body: str) -> str:
    m = re.search(r'<header class="topbar">.*?</header>', body, re.S)
    assert m
    return m.group(0)


def test_member_month_total_counts_only_their_projects(env: IsoEnv) -> None:
    when = env.clock.now
    _spend(env.root, "pdad", 0.25, when)
    _spend(env.root, "pkid", 4.00, when)
    _spend(env.root, None, 7.00, when)  # not a project's (e.g. a benchmark tool call)
    dad = env.signed_in("dad", PW_KID)
    bar = _month(_body(env.get("/", dad)))
    assert "0.25" in bar and "4.25" not in bar and "11.25" not in bar
    eli = env.signed_in("eli", PW)
    assert "11.25" in _month(_body(env.get("/", eli)))


# -- a new idea belongs to whoever is signed in --------------------------------------------


def _ideas_app(env: IsoEnv) -> tuple[ws.WebApp, FakePlanner]:
    planner = FakePlanner(env.store)
    app = ws.WebApp(
        env.root,
        env.root / LEDGER,
        ws.allowed_hosts("127.0.0.1", 8765),
        actions=env.app._base_actions,
        planner=planner,
        auth=env.auth,
    )
    return app, planner


def test_new_idea_of_a_member_is_theirs(env: IsoEnv) -> None:
    env.app, _ = _ideas_app(env)
    dad = env.signed_in("dad", PW_KID)
    csrf = env.csrf_of(dad)
    before = set(env.store.list_ids())
    r = env.post("/new", {"csrf": csrf, "goal": "compare two symbols"}, cookie=dad)
    assert r.status == 303, _body(r)
    (new,) = set(env.store.list_ids()) - before
    assert env.store.load(new).owner == "dad"
    assert env.get(f"/projects/{new}", dad).status == 200
    kid = cookie_of(env.login("kid", "kid-pass-123"))
    assert env.get(f"/projects/{new}", kid).status == 404
    assert "compare two symbols" not in _body(env.get("/", kid))


def test_new_idea_of_the_owner_account_is_theirs(env: IsoEnv) -> None:
    env.app, _ = _ideas_app(env)
    eli = env.signed_in("eli", PW)
    csrf = env.csrf_of(eli)
    before = set(env.store.list_ids())
    assert env.post("/new", {"csrf": csrf, "goal": "owner idea"}, cookie=eli).status == 303
    (new,) = set(env.store.list_ids()) - before
    assert env.store.load(new).owner == "eli"
    dad = env.signed_in("dad", PW_KID)
    assert env.get(f"/projects/{new}", dad).status == 404


def test_new_idea_without_accounts_has_no_owner(tmp_path: Path) -> None:
    from automa.webui import actions as wa

    store = ProjectStore.for_workspace(tmp_path)
    hosts = ws.allowed_hosts("127.0.0.1", 8765)
    token = "t" * 43
    app = ws.WebApp(
        tmp_path,
        tmp_path / LEDGER,
        hosts,
        actions=wa.ActionSettings(csrf_token=token, origins=wa.allowed_origins(hosts)),
        planner=FakePlanner(store),
    )
    headers = [
        ("Origin", "http://" + HOST),
        ("Content-Type", "application/x-www-form-urlencoded"),
        ("Sec-Fetch-Site", "same-origin"),
    ]
    r = app.handle("POST", "/new", HOST, headers, f"csrf={token}&goal=plain".encode())
    assert r.status == 303, _body(r)
    (pid,) = store.list_ids()
    assert store.load(pid).owner is None
