# ruff: noqa: S106 - test passwords
"""Tests for the three-step first-visit card of a member (M15, D191).

Only in-process calls; no network, no model calls.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path

import pytest

from automa.users import User, UserRole, UserStore
from automa.webui import guide
from automa.webui import server as ws
from test_webui_ideas import FakePlanner
from test_webui_login import LEDGER, PW, PW_KID, Env, N

CARD = 'class="card first-steps"'


class FirstEnv(Env):
    """Action mode with a planner (new ideas allowed): where the card's steps hold."""

    def __init__(self, root: Path, *, lang: str = "en", planner: bool = True) -> None:
        super().__init__(root, lang=lang)
        if planner:
            self.app = ws.WebApp(
                root,
                root / LEDGER,
                ws.allowed_hosts("127.0.0.1", 8765),
                actions=self.app._base_actions,
                planner=FakePlanner(self.store),
                auth=self.auth,
                language=lang,  # type: ignore[arg-type]
            )

    def done(self, cookie: str, csrf: str | None = None, **kw: object) -> ws.Response:
        token = self.csrf_of(cookie) if csrf is None else csrf
        return self.post(guide.FIRST_STEPS_PATH, {"csrf": token}, cookie=cookie, **kw)  # type: ignore[arg-type]


@pytest.fixture
def env(tmp_path: Path) -> FirstEnv:
    return FirstEnv(tmp_path)


def _body(r: ws.Response) -> str:
    return r.body.decode("utf-8", "replace")


# -- the account field --------------------------------------------------------------------


def test_new_accounts_are_not_welcomed_and_old_files_still_load(tmp_path: Path) -> None:
    store = UserStore(tmp_path, scrypt_n=N)
    store.add("eli", PW, UserRole.OWNER)
    assert store.get("eli").welcomed_at is None  # type: ignore[union-attr]
    path = tmp_path / "users.json"
    data = json.loads(path.read_text())
    for u in data["users"]:
        u.pop("welcomed_at", None)  # a file written before D191
    path.write_text(json.dumps(data))
    assert store.get("eli").welcomed_at is None  # type: ignore[union-attr]


def test_mark_welcomed_is_kept_and_idempotent(env: FirstEnv) -> None:
    before = env.users.get("dad")
    assert before is not None
    first = env.users.mark_welcomed("dad")
    assert first.welcomed_at == env.clock.now
    env.clock.advance(timedelta(days=1))
    again = env.users.mark_welcomed("dad")
    assert again.welcomed_at == first.welcomed_at
    stored = env.users.get("dad")
    assert stored is not None and stored.welcomed_at == first.welcomed_at
    # Nothing that ends a session changes.
    assert stored.model_dump(exclude={"welcomed_at"}) == before.model_dump(exclude={"welcomed_at"})


def test_mark_welcomed_unknown_name_fails(env: FirstEnv) -> None:
    from automa.users import UsersError

    with pytest.raises(UsersError):
        env.users.mark_welcomed("nobody")


def test_naive_welcomed_at_is_refused() -> None:
    from datetime import datetime

    from automa.users import hash_password

    with pytest.raises(ValueError):
        User(
            name="dad",
            role=UserRole.MEMBER,
            password=hash_password(PW, n=N),
            created_at=datetime(2026, 1, 1).astimezone(),
            password_changed_at=datetime(2026, 1, 1).astimezone(),
            welcomed_at=datetime(2026, 1, 1),  # noqa: DTZ001 - the point of the test
        )


# -- who sees the card ----------------------------------------------------------------------


def test_member_sees_the_card_until_got_it(env: FirstEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    page = _body(env.get("/", dad))
    assert CARD in page
    assert page.count('<li><span class="num"') == 3
    assert f'action="{guide.FIRST_STEPS_PATH}"' in page and 'href="/guide"' in page
    r = env.done(dad)
    assert r.status == 303 and dict(r.headers)["Location"] == "/"
    assert CARD not in _body(env.get("/", dad))
    # The session is still alive and the choice survives a new sign-in.
    assert env.get("/", dad).status == 200
    again = env.signed_in("dad", PW_KID)
    assert CARD not in _body(env.get("/", again))


def test_card_only_on_the_home_page(env: FirstEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    for target in ("/new", "/guide", "/spending"):
        r = env.get(target, dad)
        assert CARD not in _body(r), target


def test_owner_never_sees_the_card(env: FirstEnv) -> None:
    eli = env.signed_in("eli", PW)
    assert CARD not in _body(env.get("/", eli))
    r = env.done(eli)
    assert r.status == 404
    owner = env.users.get("eli")
    assert owner is not None and owner.welcomed_at is None


def test_no_card_without_new_ideas(tmp_path: Path) -> None:
    env = FirstEnv(tmp_path, planner=False)  # action mode, no planner
    dad = env.signed_in("dad", PW_KID)
    assert CARD not in _body(env.get("/", dad))


def test_no_card_on_a_read_only_server(tmp_path: Path) -> None:
    env = Env(tmp_path, actions=False)
    dad = env.signed_in("dad", PW_KID)
    assert CARD not in _body(env.get("/", dad))
    r = env.post(guide.FIRST_STEPS_PATH, {"csrf": "x" * 43}, cookie=dad)
    assert r.status == 405
    member = env.users.get("dad")
    assert member is not None and member.welcomed_at is None


def test_card_shows_again_for_a_member_never_welcomed(env: FirstEnv) -> None:
    env.users.mark_welcomed("dad")
    env.users.add("kid", "kid-pass-123", UserRole.MEMBER)
    kid = env.signed_in("kid", "kid-pass-123")
    assert CARD in _body(env.get("/", kid))
    dad = env.signed_in("dad", PW_KID)
    assert CARD not in _body(env.get("/", dad))


def test_card_with_projects_too(env: FirstEnv) -> None:
    env.store.create("dad goal", project_id="pdad", owner="dad")
    dad = env.signed_in("dad", PW_KID)
    page = _body(env.get("/", dad))
    assert CARD in page and page.index(CARD) < page.index('class="cards"')


# -- "Got it" is a checked action -----------------------------------------------------------


def test_got_it_needs_a_session(env: FirstEnv) -> None:
    r = env.post(guide.FIRST_STEPS_PATH, {"csrf": "x" * 43})
    assert r.status == 403
    member = env.users.get("dad")
    assert member is not None and member.welcomed_at is None


def test_got_it_needs_the_session_token(env: FirstEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    for bad in ("", "x" * 43, env.csrf_of(env.signed_in("eli", PW))):
        r = env.done(dad, csrf=bad)
        assert r.status == 403, bad
    member = env.users.get("dad")
    assert member is not None and member.welcomed_at is None


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Origin": None, "Referer": None},
        {"Sec-Fetch-Site": "cross-site"},
        {"Content-Type": "text/plain"},
    ],
)
def test_got_it_refuses_cross_site_and_odd_forms(
    env: FirstEnv, headers: dict[str, str | None]
) -> None:
    dad = env.signed_in("dad", PW_KID)
    r = env.done(dad, headers=headers)
    assert r.status in (400, 403, 415), _body(r)
    member = env.users.get("dad")
    assert member is not None and member.welcomed_at is None


def test_got_it_refuses_extra_fields_and_query(env: FirstEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    csrf = env.csrf_of(dad)
    r = env.post(guide.FIRST_STEPS_PATH, {"csrf": csrf, "name": "eli"}, cookie=dad)
    assert r.status in (400, 403)
    r = env.post(guide.FIRST_STEPS_PATH + "?x=1", {"csrf": csrf}, cookie=dad)
    assert r.status == 404
    member = env.users.get("dad")
    assert member is not None and member.welcomed_at is None


def test_got_it_for_a_disabled_member_ends_the_session(env: FirstEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    csrf = env.csrf_of(dad)
    env.users.disable("dad")
    r = env.done(dad, csrf=csrf)
    assert r.status == 403
    member = env.users.get("dad")
    assert member is not None and member.welcomed_at is None


def test_get_welcome_is_not_a_page(env: FirstEnv) -> None:
    dad = env.signed_in("dad", PW_KID)
    assert env.get(guide.FIRST_STEPS_PATH, dad).status == 404


# -- text ----------------------------------------------------------------------------------


@pytest.mark.parametrize("lang", ["fa", "en"])
def test_card_text_uses_the_real_button_names(lang: str) -> None:
    html = guide.render_first_steps(lang, "t" * 43)  # type: ignore[arg-type]
    names = [
        guide.label("ui", "nav.new", lang),  # type: ignore[arg-type]
        guide.label("ui", "run.start.button", lang),  # type: ignore[arg-type]
        guide.label("card", "card.ready", lang),  # type: ignore[arg-type]
    ]
    for name in names:
        assert '<b class="ui">' in html and name in html
    assert 'value="' + "t" * 43 + '"' in html
    assert "{" not in html and "}" not in html


def test_card_fa_has_persian_numbers_and_no_latin_step_digits() -> None:
    html = guide.render_first_steps("fa", "t" * 43)
    nums = re.findall(r'<span class="num" aria-hidden="true">([^<]+)</span>', html)
    assert nums == ["۱", "۲", "۳"]
    assert "خوش اومدی" in html and "فهمیدم" in html


def test_card_escapes_the_token() -> None:
    html = guide.render_first_steps("en", '"><script>')
    assert "<script>" not in html


def test_card_unknown_language() -> None:
    with pytest.raises(ValueError):
        guide.render_first_steps("de", "t")  # type: ignore[arg-type]


def test_fa_home_page_with_the_card_has_no_english_left(tmp_path: Path) -> None:
    env = FirstEnv(tmp_path, lang="fa")
    dad = env.signed_in("dad", PW_KID)
    page = _body(env.get("/", dad))
    start = page.index(CARD)
    card = page[page.index(">", start) + 1 : page.index("</section>", start)]
    visible = re.sub(r"<[^>]+>", " ", card)
    assert not re.search(r"\b(Got it|Full guide|Welcome|steps)\b", visible)
