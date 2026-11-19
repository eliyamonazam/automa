"""Language, Persian digits, Jalali dates, relative times and Persian money (D135)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from automa import l10n
from automa.config import AutomaConfig, ConfigError, DisplayConfig, load_config
from automa.l10n import (
    absolute_time,
    fa_digits,
    gregorian_to_jalali,
    localize_digits,
    relative_time,
    resolve_timezone,
)
from automa.webui.money import dollars, money, money_range, nusd_money, nusd_money_range, toman_text

TEHRAN = resolve_timezone("Asia/Tehran")
# 2026-11-19 18:10 UTC = 21:40 in Tehran (+03:30), a Thursday, 28 Aban 1405.
NOW = datetime(2026, 11, 19, 18, 10, tzinfo=UTC)


# --- digits -----------------------------------------------------------------------------


def test_fa_digits_converts_every_digit() -> None:
    assert fa_digits("0123456789") == "۰۱۲۳۴۵۶۷۸۹"


def test_fa_digits_separators_only_between_digits() -> None:
    assert fa_digits("1,234.50") == "۱٬۲۳۴٫۵۰"
    assert fa_digits("a, b. 3.") == "a, b. ۳."
    assert fa_digits("21:40") == "۲۱:۴۰"


def test_localize_digits_by_language() -> None:
    assert localize_digits("12", "fa") == "۱۲"
    assert localize_digits("12", "en") == "12"


# --- Jalali calendar -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2026, 11, 19), (1405, 8, 28)),
        (date(2025, 3, 21), (1404, 1, 1)),
        (date(2024, 3, 20), (1403, 1, 1)),
        (date(2025, 3, 20), (1403, 12, 30)),  # 1403 is a leap year
        (date(2023, 3, 21), (1402, 1, 1)),
        (date(2022, 3, 21), (1401, 1, 1)),
        (date(2000, 1, 1), (1378, 10, 11)),
        (date(2024, 9, 22), (1403, 7, 1)),
        (date(2024, 9, 21), (1403, 6, 31)),
    ],
)
def test_known_jalali_dates(day: date, expected: tuple[int, int, int]) -> None:
    assert gregorian_to_jalali(day) == expected


def test_jalali_days_advance_one_by_one() -> None:
    day = date(2019, 1, 1)
    previous = gregorian_to_jalali(day)
    for _ in range(365 * 12):
        day += timedelta(days=1)
        current = gregorian_to_jalali(day)
        py, pm, pd = previous
        cy, cm, cd = current
        if (cy, cm) == (py, pm):
            assert cd == pd + 1
        else:
            assert cd == 1
            assert (cy, cm) == ((py, pm + 1) if pm < 12 else (py + 1, 1))
            if pm <= 6:
                assert pd == 31
            elif pm <= 11:
                assert pd == 30
            else:
                assert pd in (29, 30)
        previous = current


# --- time zone -----------------------------------------------------------------------------


def test_resolve_known_zone() -> None:
    assert datetime(2026, 11, 19, 18, 10, tzinfo=UTC).astimezone(TEHRAN).hour == 21


@pytest.mark.parametrize(
    "name", ["", "Nowhere/Atlantis", "../etc/passwd", "/etc/localtime", "a b", "x" * 80, "Asia//X"]
)
def test_bad_zone_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        resolve_timezone(name)


def test_fallback_without_zone_database(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> Any:
        raise l10n.ZoneInfoNotFoundError(name)

    monkeypatch.setattr(l10n, "ZoneInfo", missing)
    tz = resolve_timezone("Asia/Tehran")
    assert tz.utcoffset(None) == timedelta(hours=3, minutes=30)
    assert resolve_timezone("UTC") is UTC
    with pytest.raises(ValueError):
        resolve_timezone("Europe/Berlin")


# --- relative and absolute times ---------------------------------------------------------


@pytest.mark.parametrize(
    ("ago", "fa", "en"),
    [
        (timedelta(seconds=0), "همین الان", "just now"),
        (timedelta(seconds=59), "همین الان", "just now"),
        (timedelta(seconds=-30), "همین الان", "just now"),
        (timedelta(minutes=1), "۱ دقیقه پیش", "1 minute ago"),
        (timedelta(minutes=5, seconds=30), "۵ دقیقه پیش", "5 minutes ago"),
        (timedelta(minutes=59), "۵۹ دقیقه پیش", "59 minutes ago"),
        (timedelta(hours=3), "امروز ساعت ۱۸:۴۰", "today at 18:40"),
        (timedelta(days=1), "دیروز ساعت ۲۱:۴۰", "yesterday at 21:40"),
        (timedelta(days=2), "سه‌شنبه ساعت ۲۱:۴۰", "Tuesday at 21:40"),
        (timedelta(days=6), "جمعه ساعت ۲۱:۴۰", "Friday at 21:40"),
        (timedelta(days=7), "۲۱ آبان ساعت ۲۱:۴۰", "12 Nov at 21:40"),
        (timedelta(days=365), "۲۸ آبان ۱۴۰۴ ساعت ۲۱:۴۰", "19 Nov 2025 at 21:40"),
    ],
)
def test_relative_times(ago: timedelta, fa: str, en: str) -> None:
    assert relative_time(NOW - ago, NOW, TEHRAN, "fa") == fa
    assert relative_time(NOW - ago, NOW, TEHRAN, "en") == en


def test_local_midnight_decides_today_and_yesterday() -> None:
    # 00:20 Tehran on the 20th is still the 19th in UTC: "today" must use local dates.
    now = datetime(2026, 11, 19, 20, 50, tzinfo=UTC)  # 00:20 on 20 Nov in Tehran
    before_midnight = datetime(2026, 11, 19, 19, 50, tzinfo=UTC)  # 23:20 on 19 Nov
    assert relative_time(before_midnight, now, TEHRAN, "fa") == "دیروز ساعت ۲۳:۲۰"
    assert relative_time(before_midnight, now, UTC, "en") == "today at 19:50"


def test_naive_times_are_read_as_utc() -> None:
    naive = datetime(2026, 11, 19, 15, 10)
    assert relative_time(naive, NOW, TEHRAN, "en") == "today at 18:40"
    assert relative_time(naive, NOW.replace(tzinfo=None), TEHRAN, "fa") == "امروز ساعت ۱۸:۴۰"


def test_far_future_shows_a_date() -> None:
    later = NOW + timedelta(days=2)
    assert relative_time(later, NOW, TEHRAN, "en") == "21 Nov at 21:40"
    assert relative_time(later, NOW, TEHRAN, "fa") == "۳۰ آبان ساعت ۲۱:۴۰"


def test_absolute_time_with_and_without_year() -> None:
    assert absolute_time(NOW, TEHRAN, "fa") == "۲۸ آبان ۱۴۰۵ ساعت ۲۱:۴۰"
    assert absolute_time(NOW, TEHRAN, "fa", now=NOW) == "۲۸ آبان ساعت ۲۱:۴۰"
    assert absolute_time(NOW, TEHRAN, "en") == "19 Nov 2026 at 21:40"
    assert absolute_time(NOW, UTC, "en", now=NOW) == "19 Nov at 18:10"


def test_jalali_year_boundary_uses_jalali_year() -> None:
    # 1 Jan 2027 is still 1405 (Dey), so the year is left out in Persian only.
    now = datetime(2027, 1, 10, 9, 0, tzinfo=UTC)
    last_year = datetime(2026, 12, 25, 9, 0, tzinfo=UTC)
    assert absolute_time(last_year, TEHRAN, "fa", now=now) == "۴ دی ساعت ۱۲:۳۰"
    assert absolute_time(last_year, TEHRAN, "en", now=now) == "25 Dec 2026 at 12:30"


def test_fixed_offset_zone_works() -> None:
    plus_two = timezone(timedelta(hours=2))
    assert relative_time(NOW - timedelta(hours=2), NOW, plus_two, "en") == "today at 18:10"


# --- Persian money -------------------------------------------------------------------------

RATE = 323_000


@pytest.mark.parametrize(
    ("usd", "rate", "expected"),
    [
        (0.45, RATE, "حدود ۱۴۵ هزار تومان (۰٫۴۵ دلار)"),
        (0, RATE, "۰ تومان (۰٫۰۰ دلار)"),
        (0.0001, RATE, "کمتر از ۱۰۰ تومان (کمتر از ۰٫۰۱ دلار)"),
        (0.002, RATE, "حدود ۶۰۰ تومان (کمتر از ۰٫۰۱ دلار)"),
        (5, RATE, "حدود ۱٫۶ میلیون تومان (۵٫۰۰ دلار)"),
        (40, RATE, "حدود ۱۳ میلیون تومان (۴۰٫۰۰ دلار)"),
        (1234.5, None, "۱٬۲۳۴٫۵۰ دلار"),
        (0.0001, None, "کمتر از ۰٫۰۱ دلار"),
    ],
)
def test_persian_money(usd: float, rate: int | None, expected: str) -> None:
    assert money(usd, rate, lang="fa") == expected


def test_persian_money_has_no_ascii_digits_or_dollar_sign() -> None:
    for usd in (0, 0.0001, 0.003, 0.45, 3.2, 123.456, 9999):
        for rate in (None, RATE):
            out = money(usd, rate, lang="fa") + money_range(usd, usd * 2 + 1, rate, lang="fa")
            assert not any(ch.isascii() and ch.isdigit() for ch in out), out
            assert "$" not in out and "about" not in out and "toman" not in out


def test_persian_ranges() -> None:
    assert money_range(0.1, 0.19, RATE, lang="fa") == "حدود ۳۲ تا ۶۱ هزار تومان (۰٫۱۰ تا ۰٫۱۹ دلار)"
    assert money_range(0.1, 0.19, None, lang="fa") == "حدود ۰٫۱۰ تا ۰٫۱۹ دلار"
    assert (
        money_range(0.0025, 0.0037, RATE, lang="fa")
        == "حدود ۸۰۰ تومان تا ۱٫۲ هزار تومان (کمتر از ۰٫۰۱ تا کمتر از ۰٫۰۱ دلار)"
    )
    assert money_range(0.45, 0.45, RATE, lang="fa") == "حدود ۱۴۵ هزار تومان (۰٫۴۵ تا ۰٫۴۵ دلار)"


def test_english_stays_the_default() -> None:
    assert money(0.45, RATE) == "about 145 thousand toman ($0.45)"
    assert dollars(0.0001) == "under $0.01"
    assert toman_text(0.0001, RATE) == "under 100 toman"
    assert money_range(0.1, 0.19, RATE) == "about 32 to 61 thousand toman ($0.10 to $0.19)"
    assert money_range(0.1, 0.19, None) == "about $0.10 to $0.19"


def test_nano_usd_helpers_take_language() -> None:
    assert nusd_money(450_000_000, RATE, lang="fa") == "حدود ۱۴۵ هزار تومان (۰٫۴۵ دلار)"
    assert nusd_money_range(100_000_000, 190_000_000, None, lang="fa") == "حدود ۰٫۱۰ تا ۰٫۱۹ دلار"
    assert money(Decimal("0.45"), RATE, lang="en") == money(0.45, RATE)


# --- configuration ------------------------------------------------------------------------


def test_display_defaults_are_persian_and_tehran() -> None:
    display = AutomaConfig().display
    assert display.language == "fa"
    assert display.timezone == "Asia/Tehran"


@pytest.mark.parametrize("language", ["fa", "en"])
def test_languages_accepted(language: str) -> None:
    assert DisplayConfig(language=language).language == language


@pytest.mark.parametrize("language", ["de", "FA", "", 1, None, "persian"])
def test_bad_language_rejected(language: Any) -> None:
    with pytest.raises(ValueError):
        DisplayConfig(language=language)


@pytest.mark.parametrize("zone", ["Nowhere/Atlantis", "", "../x", 330, None])
def test_bad_timezone_rejected(zone: Any) -> None:
    with pytest.raises(ValueError):
        DisplayConfig(timezone=zone)


def test_display_language_and_zone_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "automa.toml"
    path.write_text(
        '[display]\nlanguage = "en"\ntimezone = "Europe/Berlin"\ntoman_per_usd = 323000\n',
        encoding="utf-8",
    )
    config = load_config(path, env_file=None, environ={})
    assert config.display.language == "en"
    assert config.display.timezone == "Europe/Berlin"


def test_bad_zone_in_toml_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "automa.toml"
    path.write_text('[display]\ntimezone = "Mars/Olympus"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="display"):
        load_config(path, env_file=None, environ={})


def test_display_settings_never_change_limits(tmp_path: Path) -> None:
    a = tmp_path / "a.toml"
    a.write_text("", encoding="utf-8")
    b = tmp_path / "b.toml"
    b.write_text('[display]\nlanguage = "en"\ntimezone = "UTC"\n', encoding="utf-8")
    ca = load_config(a, env_file=None, environ={})
    cb = load_config(b, env_file=None, environ={})
    assert ca.limits == cb.limits and ca.prices == cb.prices and ca.models == cb.models


def test_example_config_documents_language_and_zone() -> None:
    text = (Path(__file__).parents[1] / "automa.toml.example").read_text(encoding="utf-8")
    assert '# language = "fa"' in text and '# timezone = "Asia/Tehran"' in text
