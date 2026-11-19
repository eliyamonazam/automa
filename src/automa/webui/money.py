"""Readable money for the web UI (D133).

A person in a browser thinks in toman, not in fractions of a cent: every amount on
the web pages is shown first as a rounded toman figure ("about 145 thousand toman")
when ``[display] toman_per_usd`` is set, with the dollar amount to two decimals in
parentheses ("($0.45)"). Without a rate only the dollars are shown. The figures are
display-only: BudgetGuard, the Ledger and every limit keep working in exact nano-USD.

Pure functions, no I/O. Rounding is half-up on :class:`~decimal.Decimal` so the same
amount always reads the same.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from automa.budget import NANO_PER_USD
from automa.l10n import Language, fa_digits

_THOUSAND = Decimal(1_000)
_MILLION = Decimal(1_000_000)
_CENT = Decimal("0.01")

TOMAN = "toman"
THOUSAND_TOMAN = "thousand toman"
MILLION_TOMAN = "million toman"

# Persian forms (D135); English stays the default so existing callers read as before.
_FA_UNITS = {TOMAN: "تومان", THOUSAND_TOMAN: "هزار تومان", MILLION_TOMAN: "میلیون تومان"}


def _fa_number(number: str) -> str:
    """A number from :func:`toman_parts` in Persian (``under 100`` → ``کمتر از ۱۰۰``)."""
    if number.startswith("under "):
        return fa_digits("کمتر از " + number.removeprefix("under "))
    return fa_digits(number)


def _parts(
    usd: float | int | Decimal, rate: float | int | Decimal, lang: Language
) -> tuple[str, str]:
    number, unit = toman_parts(usd, rate)
    if lang == "fa":
        return _fa_number(number), _FA_UNITS[unit]
    return number, unit


def _dec(value: float | int | Decimal) -> Decimal:
    d = value if isinstance(value, Decimal) else Decimal(str(value))
    if not d.is_finite():
        raise ValueError(f"amount must be finite, got {value!r}")
    return max(d, Decimal(0))  # amounts on the page are never negative


def _short(value: Decimal) -> str:
    """One decimal below 10, a whole number from 10 on; never a trailing ``.0``."""
    if value < 10:
        rounded = value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        if rounded < 10:
            text = f"{rounded:f}"
            return text[:-2] if text.endswith(".0") else text
    return f"{int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP)):,}"


def toman_parts(usd: float | int | Decimal, rate: float | int | Decimal) -> tuple[str, str]:
    """``(number, unit)`` of the rounded toman amount, e.g. ``("145", "thousand toman")``.

    Below 1,000 toman: the nearest 100 (``"under 100"`` for a tiny non-zero amount);
    below a million: thousands; otherwise millions (one decimal below 10)."""
    t = _dec(usd) * _dec(rate)
    if t == 0:
        return "0", TOMAN
    if t < _THOUSAND:
        hundreds = int((t / 100).quantize(Decimal(1), rounding=ROUND_HALF_UP)) * 100
        if hundreds == 0:
            return "under 100", TOMAN
        if hundreds < 1_000:
            return f"{hundreds}", TOMAN
        return "1", THOUSAND_TOMAN
    if t < _MILLION:
        k = t / _THOUSAND
        if k.quantize(Decimal(1), rounding=ROUND_HALF_UP) < 1_000:
            return _short(k), THOUSAND_TOMAN
        return "1", MILLION_TOMAN
    return _short(t / _MILLION), MILLION_TOMAN


def toman_text(
    usd: float | int | Decimal, rate: float | int | Decimal, *, lang: Language = "en"
) -> str:
    """``about 145 thousand toman`` (``0 toman`` and ``under 100 toman`` read as is).

    Persian: ``حدود ۱۴۵ هزار تومان``, ``۰ تومان``, ``کمتر از ۱۰۰ تومان``."""
    raw_number, _ = toman_parts(usd, rate)
    number, unit = _parts(usd, rate, lang)
    if raw_number == "0" or raw_number.startswith("under"):
        return f"{number} {unit}"
    return f"حدود {number} {unit}" if lang == "fa" else f"about {number} {unit}"


def _cents(usd: float | int | Decimal) -> str | None:
    """``0.45``/``1,234.50``, or ``None`` for a non-zero amount below half a cent."""
    d = _dec(usd)
    cents = d.quantize(_CENT, rounding=ROUND_HALF_UP)
    if d > 0 and cents == 0:
        return None
    return f"{cents:,.2f}"


def dollars(usd: float | int | Decimal, *, lang: Language = "en") -> str:
    """Two decimals (``$0.45``); a non-zero amount below half a cent is ``under $0.01``.

    Persian: ``۰٫۴۵ دلار`` and ``کمتر از ۰٫۰۱ دلار`` (no ``$`` sign, which would sit on
    the wrong side of the number in right-to-left text)."""
    cents = _cents(usd)
    if lang == "fa":
        return fa_digits("کمتر از 0.01 دلار" if cents is None else f"{cents} دلار")
    return "under $0.01" if cents is None else f"${cents}"


def money(
    usd: float | int | Decimal, rate: float | int | Decimal | None, *, lang: Language = "en"
) -> str:
    """``about 145 thousand toman ($0.45)``; just ``$0.45`` without a rate."""
    if rate is None:
        return dollars(usd, lang=lang)
    return f"{toman_text(usd, rate, lang=lang)} ({dollars(usd, lang=lang)})"


def _dollar_range(low: float | int | Decimal, high: float | int | Decimal, lang: Language) -> str:
    if lang == "en":
        return f"{dollars(low)} to {dollars(high)}"
    low_c, high_c = _cents(low), _cents(high)
    low_text = "کمتر از 0.01" if low_c is None else low_c
    high_text = "کمتر از 0.01" if high_c is None else high_c
    return fa_digits(f"{low_text} تا {high_text} دلار")


def money_range(
    low_usd: float | int | Decimal,
    high_usd: float | int | Decimal,
    rate: float | int | Decimal | None,
    *,
    lang: Language = "en",
) -> str:
    """``about 32 to 60 thousand toman ($0.10 to $0.19)``; dollars only without a rate.

    The unit is written once when both ends share it; otherwise each end keeps its own
    (``about 800 toman to 1.2 thousand toman``). Persian:
    ``حدود ۳۲ تا ۶۰ هزار تومان (۰٫۱۰ تا ۰٫۱۹ دلار)``."""
    usd = _dollar_range(low_usd, high_usd, lang)
    about, to = ("حدود", "تا") if lang == "fa" else ("about", "to")
    if rate is None:
        return f"{about} {usd}"
    raw_low = toman_parts(low_usd, rate)
    (ln, lu), (hn, hu) = _parts(low_usd, rate, lang), _parts(high_usd, rate, lang)
    if raw_low == toman_parts(high_usd, rate):
        toman = toman_text(high_usd, rate, lang=lang)
    elif lu == hu and not raw_low[0].startswith("under"):
        toman = f"{about} {ln} {to} {hn} {hu}"
    else:
        toman = f"{toman_text(low_usd, rate, lang=lang)} {to} {hn} {hu}"
    return f"{toman} ({usd})"


def nusd_money(nusd: int, rate: float | int | Decimal | None, *, lang: Language = "en") -> str:
    """:func:`money` of an exact Ledger amount in nano-USD."""
    return money(Decimal(nusd) / NANO_PER_USD, rate, lang=lang)


def nusd_money_range(
    low_nusd: int, high_nusd: int, rate: float | int | Decimal | None, *, lang: Language = "en"
) -> str:
    """:func:`money_range` of exact Ledger amounts in nano-USD."""
    return money_range(
        Decimal(low_nusd) / NANO_PER_USD, Decimal(high_nusd) / NANO_PER_USD, rate, lang=lang
    )
