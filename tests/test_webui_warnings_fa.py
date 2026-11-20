"""Tool warnings on an approval card: structured codes, Persian in the web UI (D148)."""

from __future__ import annotations

from pathlib import Path

import pytest

from automa.tools.github import describe_github_request
from automa.tools.request_warnings import WARNING_TEXT, RequestWarning, warn
from automa.tools.web import describe_fetch_request
from automa.webui import render as rd
from automa.webui.i18n import MESSAGES, t
from automa.webui.views import approval_view, build_project_view
from test_webui_files_fa import HOST, _fa_app, _req, _store

FETCH_CASES = [
    "http://example.com/",
    "https://93.184.216.34/",
    "https://bücher.example/",
    "https://xn--bcher-kva.example/",
    "https://example.com:8443/",
    "https://example.com/?a=1&b=2&c=3",
    "https://example.com/" + "a" * 250,
    "http://10.0.0.1:8080/x?q=1",
]
GITHUB_CASES = [
    {"kind": "file", "owner": "o", "repo": "r", "path": "docs/a.md"},
    {"kind": "file", "owner": "o", "repo": "r", "path": ""},
    {"kind": "file", "owner": "o", "repo": "r", "path": "a", "ref": "main"},
    {"kind": "issue", "owner": "o", "repo": "r", "number": 3},
]


# -- the codes ------------------------------------------------------------------------


@pytest.mark.parametrize("url", FETCH_CASES)
def test_fetch_codes_match_the_english_warnings(url: str) -> None:
    v = describe_fetch_request(url)
    assert v.warning_codes, url
    assert v.warnings == tuple(w.english() for w in v.warning_codes)


@pytest.mark.parametrize("args", GITHUB_CASES)
def test_github_codes_match_the_english_warnings(args: dict[str, object]) -> None:
    v = describe_github_request(args)
    assert v.warnings == tuple(w.english() for w in v.warning_codes)


def test_english_warning_text_is_unchanged() -> None:
    """The CLI text from before D148, word for word."""
    assert describe_fetch_request("http://10.0.0.1:8080/x?q=1").warnings == (
        "not encrypted (http): anyone on the network can read or change it",
        "the host is an IP address, not a name",
        "non-default port 8080",
        "the query string (1 parameter(s)) is sent to 10.0.0.1: "
        "check it carries no secrets or project data",
    )
    assert describe_fetch_request("https://bücher.example/").warnings == (
        "non-ASCII host name; it is sent as xn--bcher-kva.example",
    )
    assert describe_github_request(GITHUB_CASES[1]).warnings == (
        "no ref: reads the default branch as it is when the task runs, which may differ from now",
        "path is the repository root: a directory listing",
    )


def test_codes_carry_technical_values() -> None:
    codes = describe_fetch_request("http://10.0.0.1:8080/x?q=1&r=2").warning_codes
    assert [w.code for w in codes] == ["fetch.http", "fetch.ip_host", "fetch.port", "fetch.query"]
    assert codes[2].params == {"port": "8080"}
    assert codes[3].params == {"count": 2, "wire_host": "10.0.0.1"}


def test_no_warnings_no_codes() -> None:
    assert describe_fetch_request("https://example.com/").warning_codes == ()
    assert describe_github_request(GITHUB_CASES[2]).warning_codes == ()
    assert describe_fetch_request("ftp://x/").warning_codes == ()


def test_warn_refuses_unknown_codes_and_wrong_parameters() -> None:
    with pytest.raises(ValueError, match="unknown"):
        warn("fetch.nope")
    with pytest.raises(ValueError, match="parameters"):
        warn("fetch.port")
    with pytest.raises(ValueError, match="parameters"):
        warn("fetch.http", extra="x")
    with pytest.raises(ValueError):
        RequestWarning(code="fetch.http", params={}, other=1)  # type: ignore[call-arg]


# -- the translation table ------------------------------------------------------------


def test_every_warning_code_has_a_translation_with_the_tool_text() -> None:
    keys = {k.removeprefix("warn.") for k in MESSAGES if k.startswith("warn.")}
    assert keys == set(WARNING_TEXT)
    for code, text in WARNING_TEXT.items():
        assert MESSAGES[f"warn.{code}"]["en"] == text


@pytest.mark.parametrize("url", FETCH_CASES)
def test_english_card_warnings_are_the_tool_text(tmp_path: Path, url: str) -> None:
    store = _store(tmp_path)
    req = _req(store, "fetch_url", url=url)
    assert approval_view(req).warnings == describe_fetch_request(url).warnings


@pytest.mark.parametrize("url", FETCH_CASES)
def test_persian_card_warnings_have_no_english_sentence(tmp_path: Path, url: str) -> None:
    store = _store(tmp_path)
    req = _req(store, "fetch_url", url=url)
    fa = approval_view(req, "fa").warnings
    assert len(fa) == len(describe_fetch_request(url).warnings)
    for line in fa:
        assert "check it" not in line and "host" not in line
        assert any("؀" <= c <= "ۿ" for c in line)


def test_persian_warning_values_stay_technical(tmp_path: Path) -> None:
    store = _store(tmp_path)
    req = _req(store, "fetch_url", url="http://10.0.0.1:8080/x?q=1&r=2")
    fa = approval_view(req, "fa").warnings
    assert fa[0] == "رمزنگاری نشده (http): هر کسی در شبکه می‌تونه ببینه یا تغییرش بده"
    assert fa[2] == "درگاه غیرمعمول 8080"  # same digits as the URL
    assert "(۲ پارامتر)" in fa[3] and "10.0.0.1" in fa[3]
    idn = approval_view(_req(store, "fetch_url", url="https://bücher.example/"), "fa")
    assert idn.warnings == (
        "نام میزبان حروف غیرلاتین داره؛ به‌شکل xn--bcher-kva.example فرستاده می‌شه",
    )


def test_persian_github_warnings(tmp_path: Path) -> None:
    store = _store(tmp_path)
    req = _req(store, "github_read", **GITHUB_CASES[1])
    fa = approval_view(req, "fa").warnings
    assert fa == (
        t("warn.github.no_ref", "fa"),
        t("warn.github.root_path", "fa"),
    )
    assert fa[1] == "مسیر، ریشهٔ مخزنه: فهرست فایل‌ها خونده می‌شه"
    assert approval_view(req).warnings == describe_github_request(GITHUB_CASES[1]).warnings


def test_project_page_shows_persian_warnings(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _req(store, "fetch_url", url="http://example.com/?a=1")
    view = build_project_view(store, "p1", 0, lang="fa")
    assert view.pending[0].warnings[0].startswith("رمزنگاری نشده (http)")
    body = _fa_app(tmp_path).handle("GET", "/projects/p1", HOST).body.decode()
    assert '<li class="warn" dir="auto">رمزنگاری نشده (http)' in body
    assert "not encrypted" not in body and "check it carries" not in body


def test_warning_values_are_escaped_on_the_page() -> None:
    store_free = RequestWarning(code="fetch.port", params={"port": "<b>1</b>"})
    line = t(f"warn.{store_free.code}", "fa", **store_free.params)
    assert "<b>" not in rd.text(line)
