# ruff: noqa: S106 - test passwords
"""Tests for the owner's approval queue (M15, D186): a member's project shows its
approval requests as "waiting for the owner of Automa" with no decision form, a
member's approve/reject POST is refused (nothing is decided), and the owner sees
those requests on their own home page (with whose project asks) and decides them on
the project page; the decision event keeps who decided.

Only in-process calls; no network, no model calls.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from automa.users import UserRole
from automa.webui import server as ws
from automa.webui import views as wv
from automa.webui.cards import ICONS, MESSAGES, CardState, card_hint, card_label, card_state
from automa.webui.render import render_project
from test_webui_isolation import IsoEnv
from test_webui_login import PW, PW_KID, Env

APPROVAL_FORM = re.compile(r'action="/projects/[^"]+/approvals/')


class QueueEnv(IsoEnv):
    """``IsoEnv`` plus a request in dad's project and a project of the owner's own."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.dad_req = self.store.approvals("pdad").request(
            tool="run_command",
            risk=self.req.risk,
            arguments={"program": "python3", "args": ["screen.py"]},
            reason="run the screen script",
            task_id="t1",
        )
        self.store.create("eli goal: own tool", project_id="peli", owner="eli")
        self.eli_req = self.store.approvals("peli").request(
            tool="publish",
            risk=self.req.risk,
            arguments={"text": "eli"},
            reason="needs approval",
            task_id="a",
        )


@pytest.fixture
def env(tmp_path: Path) -> QueueEnv:
    return QueueEnv(tmp_path)


def _body(r: ws.Response) -> str:
    return r.body.decode("utf-8", "replace")


def _status(env: QueueEnv, pid: str, rid: str) -> str:
    return env.store.approvals(pid).get(rid).status.value


# -- pure card state -----------------------------------------------------------------------


def test_owner_approvals_alone_mean_with_owner() -> None:
    assert card_state("paused", "not_started", owner_approvals=1) is CardState.WITH_OWNER
    assert card_state("running", None, owner_approvals=2) is CardState.WITH_OWNER


@pytest.mark.parametrize(
    "kw",
    [{"open_asks": 1}, {"questions_waiting": True}, {"pending_approvals": 1}],
)
def test_own_answers_come_before_the_owner(kw: dict[str, object]) -> None:
    assert card_state("paused", None, owner_approvals=1, **kw) is CardState.WAITING  # type: ignore[arg-type]


def test_no_owner_approvals_keeps_the_old_states() -> None:
    assert card_state("done", "ready") is CardState.READY
    assert card_state("paused", None, pending_approvals=1) is CardState.WAITING


@pytest.mark.parametrize("lang", ["en", "fa"])
def test_with_owner_state_has_icon_label_and_hint(lang: str) -> None:
    assert "<svg" in ICONS[CardState.WITH_OWNER]
    assert card_label(CardState.WITH_OWNER, lang)  # type: ignore[arg-type]
    assert card_hint(CardState.WITH_OWNER, lang)  # type: ignore[arg-type]
    assert "صاحب Automa" in MESSAGES["card.with_owner"]["fa"]


def _row(**kw: object) -> wv.ProjectRow:
    base: dict[str, object] = {
        "id": "p",
        "status": "paused",
        "goal_preview": "g",
        "created_at": "2026-11-25T00:00:00+00:00",
        "cost_usd": 0.0,
        "pending_approvals": 1,
    }
    base.update(kw)
    return wv.ProjectRow.model_validate(base)


def test_project_row_card_follows_owner_decides() -> None:
    assert _row().card is CardState.WAITING
    assert _row(owner_decides=True).card is CardState.WITH_OWNER
    assert _row(owner_decides=True, open_asks=1).card is CardState.WAITING
    assert _row(owner_decides=True, pending_approvals=0).card is not CardState.WITH_OWNER


def test_build_overview_marks_members_and_owner_decides(env: QueueEnv) -> None:
    member_view = wv.build_overview(env.store, {}, owner="dad")
    assert [r.id for r in member_view.projects] == ["pdad"]
    assert member_view.projects[0].owner_decides is True
    assert member_view.projects[0].member is None
    owner_view = wv.build_overview(env.store, {}, members=frozenset({"dad", "kid"}))
    rows = {r.id: r for r in owner_view.projects}
    assert rows["pdad"].member == "dad" and rows["pkid"].member == "kid"
    assert rows["peli"].member is None and rows["p1"].member is None
    assert not any(r.owner_decides for r in rows.values())


# -- the member's pages --------------------------------------------------------------------


def test_member_project_page_shows_the_request_without_forms(env: QueueEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    body = _body(env.get("/projects/pdad", dad))
    assert env.dad_req.id in body and "screen.py" in body and "run the screen script" in body
    assert not APPROVAL_FORM.search(body)
    assert 'name="fingerprint"' not in body and 'name="confirm"' not in body
    assert 'class="owner-decides"' in body and "Waiting for the owner of Automa" in body
    assert "belongs to the account" not in body  # that lead is the owner's


def test_member_home_card_is_with_owner_not_waiting_for_them(env: QueueEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    body = _body(env.get("/", dad))
    assert "card project cs-with_owner" in body
    assert 'class="waiting"' not in body  # no "waiting for your answer" section
    assert "request(s) for the owner of Automa to approve" in body
    assert "request(s) to approve" not in body.replace("for the owner of Automa to approve", "")


def test_member_question_still_puts_the_project_in_their_waiting_list(env: QueueEnv) -> None:
    env.store.user_requests("pdad").request("t1", "which symbols?")
    dad = env.signed_in("dad", PW_KID)
    body = _body(env.get("/", dad))
    assert 'class="waiting"' in body and "cs-waiting" in body
    assert "question(s) from the work" in body
    assert "request(s) for the owner of Automa to approve" in body


@pytest.mark.parametrize("kind", ["approve", "reject"])
def test_member_cannot_decide_their_own_request(env: QueueEnv, kind: str) -> None:
    dad = env.signed_in("dad", PW_KID)
    form = {
        "csrf": env.csrf_of(dad),
        "fingerprint": env.dad_req.fingerprint,
        "confirm": "yes",
        "note": "ok",
    }
    r = env.post(f"/projects/pdad/approvals/{env.dad_req.id}/{kind}", form, cookie=dad)
    assert r.status == 403, _body(r)
    assert "only the owner of Automa" in _body(r)
    assert _status(env, "pdad", env.dad_req.id) == "pending"
    events = [e for e in env.store.events("pdad").read() if e.type == "approval.decided"]
    assert events == []


def test_member_refusal_is_after_the_csrf_check(env: QueueEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    form = {"csrf": "x" * 43, "fingerprint": env.dad_req.fingerprint}
    r = env.post(f"/projects/pdad/approvals/{env.dad_req.id}/reject", form, cookie=dad)
    assert r.status == 403
    assert _status(env, "pdad", env.dad_req.id) == "pending"


def test_member_still_answers_their_own_questions(env: QueueEnv) -> None:
    rid = env.store.user_requests("pdad").request("t1", "which symbols?").id
    dad = env.signed_in("dad", PW_KID)
    form = {"csrf": env.csrf_of(dad), "reason": "I do not know"}
    r = env.post(f"/projects/pdad/requests/{rid}/decline", form, cookie=dad)
    assert r.status == 303, _body(r)
    assert not env.store.user_requests("pdad").pending()


# -- the owner's queue ---------------------------------------------------------------------


def test_owner_home_lists_member_requests_with_whose_project(env: QueueEnv) -> None:
    eli = env.signed_in("eli", PW)
    body = _body(env.get("/", eli))
    waiting = body.split('class="waiting"', 1)[1].split("</section>", 1)[0]
    assert "bourse screen" in waiting and "from dad" in waiting
    assert "my web page" in waiting and "from kid" in waiting
    assert "own tool" in waiting
    own_row = waiting.split("own tool", 1)[1].split("</li>", 1)[0]
    assert "from " not in own_row  # the owner's own project names nobody
    assert "card project cs-with_owner" not in body


def test_owner_project_page_of_a_member_has_forms_and_says_whose(env: QueueEnv) -> None:
    eli = env.signed_in("eli", PW)
    body = _body(env.get("/projects/pdad", eli))
    assert APPROVAL_FORM.search(body)
    assert "belongs to the account dad" in body
    assert 'class="owner-decides"' not in body


def test_owner_own_project_page_has_no_member_line(env: QueueEnv) -> None:
    eli = env.signed_in("eli", PW)
    body = _body(env.get("/projects/peli", eli))
    assert APPROVAL_FORM.search(body)
    assert "belongs to the account" not in body


def test_owner_decision_records_who_decided(env: QueueEnv) -> None:
    eli = env.signed_in("eli", PW)
    form = {"csrf": env.csrf_of(eli), "fingerprint": env.dad_req.fingerprint, "confirm": "yes"}
    r = env.post(f"/projects/pdad/approvals/{env.dad_req.id}/approve", form, cookie=eli)
    assert r.status == 303, _body(r)
    assert _status(env, "pdad", env.dad_req.id) == "approved"
    events = [e for e in env.store.events("pdad").read() if e.type == "approval.decided"]
    assert len(events) == 1
    assert events[0].data["by"] == "eli" and events[0].data["via"] == "web"
    # Afterwards the member's page shows no waiting request any more.
    dad = env.signed_in("dad", PW_KID)
    page = _body(env.get("/projects/pdad", dad))
    assert 'id="approvals"' not in page and 'class="owner-decides"' not in page
    assert "card project cs-with_owner" not in _body(env.get("/", dad))


def test_disabled_member_project_still_names_the_member(env: QueueEnv) -> None:
    env.users.disable("dad")
    eli = env.signed_in("eli", PW)
    assert "from dad" in _body(env.get("/", eli))


def test_project_of_a_removed_account_names_nobody(env: QueueEnv) -> None:
    env.store.create("ghost goal", project_id="pghost", owner="ghost")
    env.store.approvals("pghost").request(
        tool="publish", risk=env.req.risk, arguments={}, reason="r", task_id="a"
    )
    eli = env.signed_in("eli", PW)
    body = _body(env.get("/", eli))
    row = body.split("ghost goal", 1)[1].split("</li>", 1)[0]
    assert "from " not in row


def test_a_second_owner_account_also_decides(env: QueueEnv) -> None:
    env.users.add("mom", "mom-pass-1234", UserRole.OWNER)
    mom = env.signed_in("mom", "mom-pass-1234")
    form = {"csrf": env.csrf_of(mom), "fingerprint": env.dad_req.fingerprint}
    r = env.post(f"/projects/pdad/approvals/{env.dad_req.id}/reject", form, cookie=mom)
    assert r.status == 303, _body(r)
    events = [e for e in env.store.events("pdad").read() if e.type == "approval.decided"]
    assert events[0].data["by"] == "mom"


# -- without accounts nothing changes ------------------------------------------------------


def test_no_accounts_keeps_forms_and_event_shape(tmp_path: Path) -> None:
    e = Env(tmp_path)
    (tmp_path / "users.json").unlink()
    body = _body(e.get("/projects/p1"))
    assert APPROVAL_FORM.search(body) and 'class="owner-decides"' not in body
    form = {"csrf": "s" * 43, "fingerprint": e.req.fingerprint}
    r = e.post(f"/projects/p1/approvals/{e.req.id}/reject", form)
    assert r.status == 303, _body(r)
    events = [x for x in e.store.events("p1").read() if x.type == "approval.decided"]
    assert "by" not in events[0].data and events[0].data["via"] == "web"


# -- Persian -------------------------------------------------------------------------------


def test_persian_member_page_says_waiting_for_the_owner(tmp_path: Path) -> None:
    env = QueueEnv(tmp_path)
    view = wv.build_project_view(env.store, "pdad", 0, lang="fa")
    body = render_project(view, None, lang="fa", owner_decides=True)
    assert "منتظر تأیید صاحب Automa" in body
    assert "در ترمینال تصمیم بگیر" not in body
    owner_body = render_project(view, None, lang="fa", member="dad")
    assert "این پروژه مال حساب dad است" in owner_body


# -- the "now" sentence and the home-page line ----------------------------------------------


def test_member_project_now_line_waits_for_the_owner(env: QueueEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    body = _body(env.get("/projects/pdad", dad))
    assert "Waiting for the owner of Automa: 1 request(s)" in body
    assert "need your OK" not in body


def test_owner_project_now_line_still_asks_the_owner(env: QueueEnv) -> None:
    eli = env.signed_in("eli", PW)
    body = _body(env.get("/projects/pdad", eli))
    assert "need your OK" in body
    assert "Waiting for the owner of Automa:" not in body


def test_member_questions_come_before_the_owner_in_the_now_line(env: QueueEnv) -> None:
    from automa.webui.journey import build_journey

    view = wv.build_project_view(env.store, "pdad", 0)
    assert view.pending
    member = build_journey(view, questions_waiting=True, owner_decides=True).now
    assert "questions" in member and "owner" not in member
    owner = build_journey(view, questions_waiting=True).now
    assert "need your OK" in owner


def test_member_home_line_says_requests_go_to_the_owner(env: QueueEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    body = _body(env.get("/", dad))
    assert "requests to approve go to the owner of Automa" in body
    assert "approve or reject requests on each project page" not in body
    eli = env.signed_in("eli", PW)
    assert "approve or reject requests on each project page" in _body(env.get("/", eli))
