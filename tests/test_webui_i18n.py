"""Tests for the web UI translation table and the Persian home page (D140).

Pure rendering plus the in-process ``WebApp``; no model, no network.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.l10n import resolve_timezone
from automa.project import ProjectStore
from automa.webui import i18n
from automa.webui import render as rd
from automa.webui import server as ws
from automa.webui.views import Overview, ProjectRow
from test_webui_actions import _config, _FakeServer

NOW = datetime(2026, 11, 19, 18, 10, tzinfo=UTC)  # 21:40 in Tehran
TEHRAN = resolve_timezone("Asia/Tehran")
RATE = 323_000.0
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


def _row(**kw: object) -> ProjectRow:
    data: dict[str, object] = {
        "id": "p-1",
        "status": "done",
        "title": "یک ابزار تبدیل ارقام",
        "goal_preview": "یک ابزار تبدیل ارقام",
        "created_at": NOW - timedelta(minutes=5),
        "cost_usd": 0.45,
        "pending_approvals": 0,
    }
    data.update(kw)
    return ProjectRow.model_validate(data)


def _strip_data(doc: str) -> str:
    """The page without its style, tags, codes and recorded data (left: interface text)."""
    doc = re.sub(r"<style>.*?</style>", "", doc, flags=re.S)
    doc = re.sub(r"<code>.*?</code>", "", doc, flags=re.S)
    doc = re.sub(r"<!doctype html>|<(meta|html|a)\b[^>]*>", "", doc)
    doc = re.sub(r"<[^>]+>", " ", doc)
    # Product and format names are written in Latin letters in Persian too.
    return doc.replace("Automa", "").replace("HTML", "")


# -- the table ---------------------------------------------------------------------


def test_every_key_has_both_languages_with_the_same_fields() -> None:
    for key, by_lang in i18n.MESSAGES.items():
        assert set(by_lang) == {"fa", "en"}, key
        assert i18n.template_fields(by_lang["fa"]) == i18n.template_fields(by_lang["en"]), key


def test_messages_have_no_control_or_bidi_characters() -> None:
    allowed = {"‌"}  # ZWNJ, part of Persian spelling
    for key, by_lang in i18n.MESSAGES.items():
        for template in by_lang.values():
            for ch in template:
                if ch in allowed:
                    continue
                assert unicodedata.category(ch) not in {"Cc", "Cf"}, (key, hex(ord(ch)))


def test_persian_templates_use_no_ascii_digits() -> None:
    for key, by_lang in i18n.MESSAGES.items():
        assert not re.search(r"[0-9]", by_lang["fa"]), key


def test_table_is_read_only() -> None:
    with pytest.raises(TypeError):
        i18n.MESSAGES["overview.title"] = {"en": "x", "fa": "y"}  # type: ignore[index]
    with pytest.raises(TypeError):
        i18n.MESSAGES["overview.title"]["en"] = "x"  # type: ignore[index]


@pytest.mark.parametrize(
    "bad",
    [
        {"k": {"en": "x"}},
        {"k": {"en": "x", "fa": "y", "de": "z"}},
        {"k": {"en": "{a}", "fa": "{b}"}},
        {"k": {"en": "{a}", "fa": "y"}},
        {"k": {"en": "", "fa": "y"}},
        {"k": {"en": "{0}", "fa": "{0}"}},
        {"k": {"en": "{a.b}", "fa": "{a.b}"}},
        {"k": {"en": "{a!r}", "fa": "{a!r}"}},
        {"k": {"en": "{a:>9}", "fa": "{a:>9}"}},
        {"k": {"en": "{}", "fa": "{}"}},
    ],
)
def test_check_messages_rejects_bad_tables(bad: dict[str, dict[str, str]]) -> None:
    with pytest.raises(ValueError):
        i18n.check_messages(bad)


def test_t_fills_and_localizes_whole_numbers() -> None:
    assert i18n.t("overview.pending", "en", count=12) == "12 pending"
    assert i18n.t("overview.pending", "fa", count=12) == "۱۲ منتظر تأیید"
    # strings are inserted as given (already localized by the caller)
    assert i18n.t("overview.cost", "fa", amount="x 1") == "هزینه: x 1"


@pytest.mark.parametrize(
    "call",
    [
        lambda: i18n.t("no.such.key", "en"),
        lambda: i18n.t("overview.title", "de"),  # type: ignore[arg-type]
        lambda: i18n.t("overview.pending", "en"),
        lambda: i18n.t("overview.pending", "en", count=1, extra=2),
        lambda: i18n.t("overview.title", "en", count=1),
    ],
)
def test_t_refuses_typos(call: object) -> None:
    with pytest.raises((KeyError, ValueError)):
        call()  # type: ignore[operator]


@pytest.mark.parametrize("value", [True, 1.5, None, b"x", ["a"]])
def test_t_refuses_non_text_parameters(value: object) -> None:
    with pytest.raises(TypeError):
        i18n.t("overview.pending", "en", count=value)


def test_parameters_are_not_formatted_again() -> None:
    assert i18n.t("overview.cost", "en", amount="{amount} {0}") == "cost: {amount} {0}"


def test_status_labels() -> None:
    assert i18n.status_label("done", "fa") == "تموم شد"
    assert i18n.status_label("running", "fa") == "در حال کار"
    assert i18n.status_label("done", "en") == "done"
    assert i18n.status_label("weird", "fa") == "weird"


# -- page shell --------------------------------------------------------------------


def test_page_shell_direction() -> None:
    assert '<html lang="fa" dir="rtl">' in rd.page("x", "", lang="fa")
    assert '<html lang="en">' in rd.page("x", "")
    with pytest.raises(ValueError):
        rd.page("x", "", lang="de")  # type: ignore[arg-type]


# -- the Persian home page -----------------------------------------------------------


def test_persian_home_page() -> None:
    view = Overview(projects=(_row(pending_approvals=3),), hidden_tests=4)
    doc = rd.render_overview(
        view, actions=True, ideas=True, rate=RATE, lang="fa", tz=TEHRAN, now=NOW
    )
    assert '<html lang="fa" dir="rtl">' in doc
    assert "<title>پروژه‌های Automa</title>" in doc and "<h1>پروژه‌های Automa</h1>" in doc
    assert 'class="button small newidea" href="/new">' in doc and "ایدهٔ جدید</a>" in doc
    assert "۴ آزمون پنهان شده" in doc and 'href="/?tests=1">نمایش آزمون‌ها<' in doc
    assert "منتظر جواب تو" in doc and "جواب بده" in doc
    assert "۳ درخواست برای تأیید" in doc
    assert "هزینه: حدود ۱۴۵ هزار تومان (۰٫۴۵ دلار)" in doc
    assert "ساخته‌شده ۵ دقیقه پیش" in doc
    assert "2026-" not in doc  # no ISO timestamp left


def test_persian_home_page_has_no_english_interface_text() -> None:
    view = Overview(projects=(_row(), _row(id="p-2", status="running", pending_approvals=1)))
    for actions in (True, False):
        doc = rd.render_overview(
            view, actions=actions, ideas=True, rate=RATE, lang="fa", tz=TEHRAN, now=NOW
        )
        assert LATIN_WORD.findall(_strip_data(doc)) == [], actions


@pytest.mark.parametrize("actions,ideas", [(True, True), (False, False), (True, False)])
def test_persian_welcome_pages(actions: bool, ideas: bool) -> None:
    doc = rd.render_overview(
        Overview(), actions=actions, ideas=ideas, lang="fa", tz=TEHRAN, now=NOW
    )
    assert '<html lang="fa" dir="rtl">' in doc
    assert LATIN_WORD.findall(_strip_data(doc)) == []
    if actions and ideas:
        assert "بیا اولین ایده‌ات رو بسازیم" in doc and "یا از یک نمونه شروع کن:" in doc
        assert doc.count('<a class="example" dir="auto"') == 3
    else:
        assert "هنوز پروژه‌ای نیست" in doc


def test_persian_showing_tests_link() -> None:
    doc = rd.render_overview(Overview(showing_tests=True), lang="fa")
    assert '<a href="/">پنهان کردن آزمون‌ها</a>' in doc


def test_persian_errors_heading() -> None:
    doc = rd.render_overview(Overview(errors=("p9: broken",)), lang="fa")
    assert "داده‌هایی که خونده نشدن" in doc and "p9: broken" in doc


def test_recorded_data_is_not_translated_and_stays_escaped() -> None:
    row = _row(title="<b>done</b>", goal_preview="new ‮ idea", status="done")
    doc = rd.render_overview(Overview(projects=(row,)), lang="fa", tz=TEHRAN, now=NOW)
    assert "&lt;b&gt;done&lt;/b&gt;" in doc and "<b>" not in doc
    assert "new \\u202e idea" in doc and "‮" not in doc


def test_english_home_page_is_unchanged_without_a_clock() -> None:
    doc = rd.render_overview(Overview(projects=(_row(status="done"),)))
    assert '<html lang="en">' in doc and "Automa projects" in doc
    assert (
        "created 2026-11-19T18:05:00+00:00" in doc
        and "Incomplete</span>" in doc  # done, but no delivery report: never "ready"
    )
    assert "cost: $0.45" in doc


def test_english_home_page_with_a_clock_is_relative_and_local() -> None:
    doc = rd.render_overview(Overview(projects=(_row(),)), lang="en", tz=TEHRAN, now=NOW)
    assert "created 5 minutes ago" in doc
    old = _row(created_at=datetime(2026, 11, 18, 18, 10, tzinfo=UTC))
    doc = rd.render_overview(Overview(projects=(old,)), lang="en", tz=TEHRAN, now=NOW)
    assert "created yesterday at 21:40" in doc


def test_persian_yesterday_uses_the_local_day() -> None:
    # 20:50 UTC on the 18th is 00:20 on the 19th in Tehran: "today", not "yesterday".
    row = _row(created_at=datetime(2026, 11, 18, 20, 50, tzinfo=UTC))
    doc = rd.render_overview(Overview(projects=(row,)), lang="fa", tz=TEHRAN, now=NOW)
    assert "ساخته‌شده امروز ساعت ۰۰:۲۰" in doc


# -- server and CLI wiring ---------------------------------------------------------


def _app(workspace: Path, **kw: object) -> ws.WebApp:
    return ws.WebApp(
        workspace, workspace / "ledger.sqlite3", ws.allowed_hosts("127.0.0.1", 8765), **kw
    )


def test_server_renders_the_home_page_in_persian(tmp_path: Path) -> None:
    store = ProjectStore.for_workspace(tmp_path)
    store.create("یک ابزار کوچک", project_id="p1")
    app = _app(tmp_path, language="fa", clock=lambda: datetime.now(UTC))
    r = app.handle("GET", "/", "127.0.0.1:8765")
    body = r.body.decode()
    assert r.status == 200 and '<html lang="fa" dir="rtl">' in body
    assert "ساخته‌شده همین الان" in body and "یک ابزار کوچک" in body


def test_server_default_language_stays_english(tmp_path: Path) -> None:
    body = _app(tmp_path).handle("GET", "/", "127.0.0.1:8765").body.decode()
    assert '<html lang="en">' in body and "Automa projects" in body


def test_server_refuses_bad_display_settings(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _app(tmp_path, language="de")
    with pytest.raises(ValueError):
        _app(tmp_path, timezone="../../etc/passwd")


def test_create_server_passes_language(tmp_path: Path) -> None:
    server = ws.create_server(
        tmp_path / "ws", tmp_path / "ws" / "ledger.sqlite3", port=0, language="fa"
    )
    server.server_close()


@pytest.mark.parametrize(
    "extra,lang,tz",
    [
        ("", "fa", "Asia/Tehran"),
        ('[display]\nlanguage = "en"\ntimezone = "Europe/Berlin"\n', "en", "Europe/Berlin"),
    ],
)
def test_cli_web_passes_display_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: str, lang: str, tz: str
) -> None:
    seen: dict[str, object] = {}

    def fake_create(workspace: Path, ledger: Path, **kw: object) -> _FakeServer:
        seen.update(kw)
        return _FakeServer()

    monkeypatch.setattr(ws, "create_server", fake_create)
    cfg = _config(tmp_path)
    cfg.write_text(cfg.read_text(encoding="utf-8") + extra, encoding="utf-8")
    result = CliRunner().invoke(cli.app, ["web", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert seen["language"] == lang and seen["timezone"] == tz


def test_persian_today_and_month_totals() -> None:
    doc = rd.render_overview(Overview(), totals=(0.45, 3.0), rate=RATE, lang="fa")
    assert "خرج امروز: حدود ۱۴۵ هزار تومان (۰٫۴۵ دلار)</span>" in doc
    assert "این ماه: <strong>" in doc and "Spent today" not in doc


def test_totals_use_the_configured_time_zone(tmp_path: Path) -> None:
    # 21:00 UTC on 30 Nov is already 1 Dec (00:30) in Tehran: a new month there.
    from automa.budget import CostLedger
    from test_ledger import FixedClock, make_pair

    with CostLedger(
        tmp_path / "ledger.sqlite3", clock=FixedClock(datetime(2026, 11, 30, 20, 0, tzinfo=UTC))
    ) as led:
        led.record(*make_pair(0.5, project="p1"))
    late = datetime(2026, 11, 30, 21, 0, tzinfo=UTC)
    tehran = _app(tmp_path, clock=lambda: late)._totals()
    utc = _app(tmp_path, timezone="UTC", clock=lambda: late)._totals()
    assert tehran is not None and utc is not None
    assert tehran[1] == 0 and utc[1] > 0
