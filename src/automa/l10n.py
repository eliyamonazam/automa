"""Language, digits, time zone and readable dates for the web UI (D135).

Display only: nothing here changes what is recorded, spent or allowed. Recorded
timestamps stay timezone-aware UTC on disk; these helpers only turn them into what a
person reads ("۵ دقیقه پیش", "دیروز ساعت ۲۱:۴۰"). Pure functions, no I/O except
loading a time zone definition.

Persian dates use the Solar Hijri (Jalali) calendar the owner uses every day; English
dates stay Gregorian.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

Language = Literal["fa", "en"]
LANGUAGES: tuple[Language, ...] = ("fa", "en")
DEFAULT_LANGUAGE: Language = "fa"
DEFAULT_TIMEZONE = "Asia/Tehran"

# Used only when the system has no time zone database (e.g. a slim container):
# Iran has kept +03:30 all year since 2022.
_FALLBACK_ZONES: dict[str, tzinfo] = {
    "Asia/Tehran": timezone(timedelta(hours=3, minutes=30), "Asia/Tehran"),
    "UTC": UTC,
    "Etc/UTC": UTC,
}
_ZONE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*(/[A-Za-z0-9_+\-]+){0,2}")
MAX_ZONE_NAME = 64

_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
# Thousands separator and decimal point only between digits ("1,234.5" → "۱٬۲۳۴٫۵").
_NUMBER_SEPARATOR = re.compile(r"(?<=\d)([,.])(?=\d)")

FA_MONTHS = (
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
)
# datetime.weekday(): Monday is 0.
FA_WEEKDAYS = ("دوشنبه", "سه‌شنبه", "چهارشنبه", "پنج‌شنبه", "جمعه", "شنبه", "یکشنبه")
EN_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
EN_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def resolve_timezone(name: str) -> tzinfo:
    """The time zone called ``name`` (IANA, e.g. ``Asia/Tehran``).

    Raises :class:`ValueError` for a malformed or unknown name. Falls back to a fixed
    offset for ``Asia/Tehran``/``UTC`` only when the system has no zone database."""
    if not isinstance(name, str) or len(name) > MAX_ZONE_NAME or not _ZONE_NAME.fullmatch(name):
        raise ValueError(f"invalid time zone name: {name!r}")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        fallback = _FALLBACK_ZONES.get(name)
        if fallback is None:
            raise ValueError(f"unknown time zone: {name!r}") from None
        return fallback


def fa_digits(value: str) -> str:
    """ASCII digits as Persian digits; ``,``/``.`` between digits as ``٬``/``٫``."""
    separated = _NUMBER_SEPARATOR.sub(lambda m: "٬" if m.group(1) == "," else "٫", value)
    return separated.translate(_FA_DIGITS)


def localize_digits(value: str, lang: Language) -> str:
    """``value`` with Persian digits for ``fa``; unchanged for ``en``."""
    return fa_digits(value) if lang == "fa" else value


def gregorian_to_jalali(day: date) -> tuple[int, int, int]:
    """``(year, month, day)`` of ``day`` in the Solar Hijri calendar (arithmetic form)."""
    gy, gm, gd = day.year, day.month, day.day
    month_starts = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
    gy2 = gy + 1 if gm > 2 else gy
    days = (
        355666
        + 365 * gy
        + (gy2 + 3) // 4
        - (gy2 + 99) // 100
        + (gy2 + 399) // 400
        + gd
        + month_starts[gm - 1]
    )
    jy = -1595 + 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        return jy, 1 + days // 31, 1 + days % 31
    return jy, 7 + (days - 186) // 30, 1 + (days - 186) % 30


def _aware(value: datetime) -> datetime:
    """Recorded times are UTC; a naive one is read as UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _clock(local: datetime) -> str:
    return f"{local.hour:02d}:{local.minute:02d}"


def absolute_time(
    value: datetime, tz: tzinfo, lang: Language, *, now: datetime | None = None
) -> str:
    """Date and time in ``tz``: ``۲۸ آبان ۱۴۰۵ ساعت ۲۱:۴۰`` / ``19 Nov 2026 at 21:40``.

    The year is left out when it is the current one (``now`` given)."""
    local = _aware(value).astimezone(tz)
    this_year = None if now is None else _aware(now).astimezone(tz).date()
    if lang == "fa":
        jy, jm, jd = gregorian_to_jalali(local.date())
        same_year = this_year is not None and gregorian_to_jalali(this_year)[0] == jy
        day = f"{jd} {FA_MONTHS[jm - 1]}" + ("" if same_year else f" {jy}")
        return fa_digits(f"{day} ساعت {_clock(local)}")
    same_year = this_year is not None and this_year.year == local.year
    day = f"{local.day} {EN_MONTHS[local.month - 1]}" + ("" if same_year else f" {local.year}")
    return f"{day} at {_clock(local)}"


def relative_time(value: datetime, now: datetime, tz: tzinfo, lang: Language) -> str:
    """How long ago ``value`` was, as a person says it, in ``tz``.

    Under a minute (or up to a minute ahead, for clock skew): "just now"; under an
    hour: minutes ago; the same local day: "today at HH:MM"; the day before:
    "yesterday at HH:MM"; within a week: the weekday; otherwise the date
    (:func:`absolute_time`). A time further in the future is shown as a date."""
    value, now = _aware(value), _aware(now)
    seconds = (now - value).total_seconds()
    fa = lang == "fa"
    if -60 < seconds < 60:
        return "همین الان" if fa else "just now"
    if seconds < 0:
        return absolute_time(value, tz, lang, now=now)
    if seconds < 3600:
        minutes = int(seconds // 60)
        if fa:
            return fa_digits(f"{minutes} دقیقه پیش")
        return "1 minute ago" if minutes == 1 else f"{minutes} minutes ago"
    local, today = value.astimezone(tz), now.astimezone(tz).date()
    days = (today - local.date()).days
    clock = _clock(local)
    if days == 0:
        return fa_digits(f"امروز ساعت {clock}") if fa else f"today at {clock}"
    if days == 1:
        return fa_digits(f"دیروز ساعت {clock}") if fa else f"yesterday at {clock}"
    if 1 < days < 7:
        if fa:
            return fa_digits(f"{FA_WEEKDAYS[local.weekday()]} ساعت {clock}")
        return f"{EN_WEEKDAYS[local.weekday()]} at {clock}"
    return absolute_time(value, tz, lang, now=now)
