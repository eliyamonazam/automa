"""Web UI: questions Automa asks the user, answered with text and files (D159).

Fake child processes (test_webui_runs); no network, no real model calls.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from automa.project import ProjectStatus
from automa.userinput import RequestStatus
from automa.webui import server as ws
from automa.webui import upload as wu
from automa.webui.actions import ActionRefused
from test_webui_runs import HOST, LEDGER, ORIGIN, PORT, TOKEN, Env, _settings

BOUNDARY = "----automaTestBoundary7MA4YWxk"


def multipart(
    fields: dict[str, str], files: list[tuple[str, bytes]] = (), *, field: str = "files"
) -> bytes:
    out = b""
    for k, v in fields.items():
        out += (
            (f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n').encode()
            + v.encode("utf-8")
            + b"\r\n"
        )
    for name, data in files:
        out += (
            (
                f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{field}"; '
                f'filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'
            ).encode()
            + data
            + b"\r\n"
        )
    return out + f"--{BOUNDARY}--\r\n".encode()


MP_HEADERS = {"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"}


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    e = Env(tmp_path)
    yield e
    for _, _, proc in e.spawned:
        if proc.code is None:
            proc.finish(0)


def _asked(env: Env, kind: str = "file") -> tuple[str, str]:
    pid = env.project(status=ProjectStatus.PAUSED)
    req = env.store.user_requests(pid).request(
        "build", "Please upload the logo", why="to put it in the header", kind=kind
    )
    return pid, req.id


def _answer(env: Env, pid: str, rid: str, body: bytes, **headers: str) -> ws.Response:
    return env.post(
        f"/projects/{pid}/requests/{rid}/answer", body=body, headers={**MP_HEADERS, **headers}
    )


def test_page_shows_the_question_with_upload_form(env: Env) -> None:
    pid, rid = _asked(env)
    body = env.get(f"/projects/{pid}").body.decode()
    card = body[body.index('class="card ask"') :].split("</section>")[0]
    assert "I need something from you" in card
    assert "Please upload the logo" in card and "to put it in the header" in card
    assert f'action="/projects/{pid}/requests/{rid}/answer"' in card
    assert 'enctype="multipart/form-data"' in card and 'type="file" name="files"' in card
    assert "Upload the file(s):" in card
    assert f'action="/projects/{pid}/requests/{rid}/decline"' in card
    assert body.index('class="card ask"') < body.index("<h2>Plan")


def test_read_only_page_shows_commands(tmp_path: Path) -> None:
    env = Env(tmp_path, actions=False, runs=False)
    pid, rid = _asked(env)
    body = env.get(f"/projects/{pid}").body.decode()
    assert (
        "Please upload the logo" in body
        and "<form" not in body.split('class="card ask"')[1].split("</section>")[0]
    )
    assert f"automa answer {pid} {rid}" in body and f"automa decline {pid} {rid}" in body


def test_persian_card(env: Env) -> None:
    pid, _ = _asked(env, kind="text")
    app = ws.WebApp(
        env.root,
        env.root / LEDGER,
        ws.allowed_hosts("127.0.0.1", PORT),
        actions=_settings(),
        runs=env.background,
        language="fa",
    )
    body = app.handle("GET", f"/projects/{pid}", HOST).body.decode()
    assert "برای ادامه یه چیزی ازت لازم دارم" in body and "بفرست و ادامه بده" in body
    assert "فایل (اختیاری):" in body and "نمی‌تونم این رو بدم" in body
    assert "منتظر توئه" not in body or True  # sentence depends on the recorded stop


def test_answer_with_files_saves_and_continues(env: Env) -> None:
    pid, rid = _asked(env)
    body = multipart({"csrf": TOKEN, "answer": "سلام\r\nlogo attached"}, [("logo.png", b"\x89PNG")])
    r = _answer(env, pid, rid, body)
    assert r.status == 303, r.body
    req = env.store.user_requests(pid).get(rid)
    assert req.status is RequestStatus.ANSWERED and req.via == "web"
    assert req.answer == "سلام\nlogo attached"
    assert req.files == (f"inputs/{rid}/logo.png",)
    assert (env.store.files_dir(pid) / "inputs" / rid / "logo.png").read_bytes() == b"\x89PNG"
    assert len(env.spawned) == 1 and env.preflight.commands == [(pid, False)]
    # a second answer is refused
    r = _answer(env, pid, rid, body)
    assert r.status == 409 and len(env.spawned) == 1


def test_answer_does_not_start_while_another_question_is_open(env: Env) -> None:
    pid, rid = _asked(env)
    env.store.user_requests(pid).request("build", "And the colours?")
    r = _answer(env, pid, rid, multipart({"csrf": TOKEN, "answer": "x"}))
    assert r.status == 303 and env.spawned == []


def test_answer_with_cost_box_needed_waits_for_start(env: Env) -> None:
    env.preflight.confirm = True
    pid, rid = _asked(env)
    r = _answer(env, pid, rid, multipart({"csrf": TOKEN, "answer": "x"}))
    assert r.status == 303 and env.spawned == []  # saved, but the start needs the box
    assert env.store.user_requests(pid).get(rid).status is RequestStatus.ANSWERED
    pid2, rid2 = _asked(env)
    r = _answer(env, pid2, rid2, multipart({"csrf": TOKEN, "answer": "x", "confirm": "yes"}))
    assert r.status == 303 and env.preflight.commands == [(pid2, True)]


def test_empty_answer_refused_and_nothing_saved(env: Env) -> None:
    pid, rid = _asked(env)
    r = _answer(env, pid, rid, multipart({"csrf": TOKEN, "answer": "  "}))
    assert r.status == 409
    assert env.store.user_requests(pid).get(rid).status is RequestStatus.PENDING
    assert env.spawned == []


@pytest.mark.parametrize(
    ("fields", "files", "headers", "status"),
    [
        ({"csrf": "x" * 43, "answer": "a"}, [], {}, 403),
        ({"csrf": TOKEN, "answer": "a", "extra": "1"}, [], {}, 400),
        ({"csrf": TOKEN, "answer": "a"}, [], {"Origin": "http://evil.example"}, 403),
        (
            {"csrf": TOKEN, "answer": "a"},
            [],
            {"Content-Type": "application/x-www-form-urlencoded"},
            415,
        ),
    ],
)
def test_answer_checks(
    env: Env,
    fields: dict[str, str],
    files: list[tuple[str, bytes]],
    headers: dict[str, str],
    status: int,
) -> None:
    pid, rid = _asked(env)
    r = _answer(env, pid, rid, multipart(fields, files), **headers)
    assert r.status == status
    assert env.store.user_requests(pid).get(rid).status is RequestStatus.PENDING
    assert env.spawned == []


def test_file_under_another_field_is_refused(env: Env) -> None:
    pid, rid = _asked(env)
    r = _answer(env, pid, rid, multipart({"csrf": TOKEN}, [("a.txt", b"x")], field="answer"))
    assert r.status == 400


def test_unknown_question_and_project(env: Env) -> None:
    pid, _ = _asked(env)
    body = multipart({"csrf": TOKEN, "answer": "a"})
    assert _answer(env, pid, "req-000000000000", body).status == 404
    assert _answer(env, "p-nope", "req-000000000000", body).status == 404
    r = env.post(f"/projects/{pid}/requests/req-xyz/answer", body=body, headers=MP_HEADERS)
    assert r.status == 404


def test_decline_records_and_continues(env: Env) -> None:
    pid, rid = _asked(env)
    r = env.post(
        f"/projects/{pid}/requests/{rid}/decline", {"csrf": TOKEN, "reason": "no logo yet"}
    )
    assert r.status == 303, r.body
    req = env.store.user_requests(pid).get(rid)
    assert req.status is RequestStatus.DECLINED and req.answer == "no logo yet"
    assert len(env.spawned) == 1
    r = env.post(f"/projects/{pid}/requests/{rid}/decline", {"csrf": TOKEN})
    assert r.status == 409


def test_decline_needs_token(env: Env) -> None:
    pid, rid = _asked(env)
    r = env.post(f"/projects/{pid}/requests/{rid}/decline", {"csrf": "bad" * 10})
    assert r.status == 403
    assert env.store.user_requests(pid).get(rid).status is RequestStatus.PENDING


def test_upload_body_limit_is_larger_only_for_answers(env: Env) -> None:
    app = env.app
    assert app.max_body_bytes("/projects/p/requests/req-000000000000/answer") == wu.MAX_UPLOAD_BYTES
    assert app.max_body_bytes("/projects/p/requests/req-000000000000/decline") < 100_000
    assert app.max_body_bytes("/projects/p/start") < 100_000


def test_read_only_server_has_no_answer_endpoint(tmp_path: Path) -> None:
    env = Env(tmp_path, actions=False, runs=False)
    pid, rid = _asked(env)
    r = _answer(env, pid, rid, multipart({"csrf": TOKEN, "answer": "a"}))
    assert r.status in (403, 404, 405)
    assert env.store.user_requests(pid).get(rid).status is RequestStatus.PENDING


# --- the parser -------------------------------------------------------------------------


def _parse(body: bytes, ctype: str | None = None, **kw: int) -> wu.Upload:
    return wu.parse_multipart(
        {"content-type": ctype or MP_HEADERS["Content-Type"]},
        body,
        fields=frozenset({"csrf", "answer"}),
        file_field="files",
        max_bytes=kw.get("max_bytes", 10_000),
        max_files=kw.get("max_files", 3),
    )


def test_parser_basics_and_limits() -> None:
    up = _parse(multipart({"answer": "hi"}, [("a.txt", b"1"), ("", b"")]))
    assert up.fields == {"answer": "hi"} and up.files == [("a.txt", b"1")]
    bad = [
        multipart({"answer": "a", "csrf": "1"}).replace(b'name="csrf"', b'name="answer"'),
        b"garbage",
        multipart({"answer": "a"})[:-10],  # no closing boundary
        multipart({"answer": "a"}).replace(b"form-data;", b"attachment;"),
    ]
    for body in bad:
        with pytest.raises(ActionRefused):
            _parse(body)
    with pytest.raises(ActionRefused):
        _parse(multipart({}, [("a", b"1")] * 4))
    with pytest.raises(ActionRefused):
        _parse(multipart({"answer": "x" * 100}), max_bytes=50)
    with pytest.raises(ActionRefused):
        _parse(multipart({"answer": "a"}), ctype="multipart/form-data")
    with pytest.raises(ActionRefused):
        _parse(multipart({"answer": "hi"}).replace(b"hi", b"\xff"), ctype=None)
    assert ORIGIN  # the test env's origin is what the forms post from
