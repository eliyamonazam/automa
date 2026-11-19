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

_THOUSAND = Decimal(1_000)
_MILLION = Decimal(1_000_000)
_CENT = Decimal("0.01")

TOMAN = "toman"
THOUSAND_TOMAN = "thousand toman"
MILLION_TOMAN = "million toman"


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


def toman_text(usd: float | int | Decimal, rate: float | int | Decimal) -> str:
    """``about 145 thousand toman`` (``0 toman`` and ``under 100 toman`` read as is)."""
    number, unit = toman_parts(usd, rate)
    if number == "0" or number.startswith("under"):
        return f"{number} {unit}"
    return f"about {number} {unit}"


def dollars(usd: float | int | Decimal) -> str:
    """Two decimals (``$0.45``); a non-zero amount below half a cent is ``under $0.01``."""
    d = _dec(usd)
    cents = d.quantize(_CENT, rounding=ROUND_HALF_UP)
    if d > 0 and cents == 0:
        return "under $0.01"
    return f"${cents:,.2f}"


def money(usd: float | int | Decimal, rate: float | int | Decimal | None) -> str:
    """``about 145 thousand toman ($0.45)``; just ``$0.45`` without a rate."""
    if rate is None:
        return dollars(usd)
    return f"{toman_text(usd, rate)} ({dollars(usd)})"


def money_range(
    low_usd: float | int | Decimal,
    high_usd: float | int | Decimal,
    rate: float | int | Decimal | None,
) -> str:
    """``about 32 to 60 thousand toman ($0.10 to $0.19)``; dollars only without a rate.

    The unit is written once when both ends share it; otherwise each end keeps its own
    (``about 800 toman to 1.2 thousand toman``)."""
    usd = f"{dollars(low_usd)} to {dollars(high_usd)}"
    if rate is None:
        return f"about {usd}"
    (ln, lu), (hn, hu) = toman_parts(low_usd, rate), toman_parts(high_usd, rate)
    if (ln, lu) == (hn, hu):
        toman = toman_text(high_usd, rate)
    elif lu == hu and not ln.startswith("under"):
        toman = f"about {ln} to {hn} {hu}"
    else:
        toman = f"{toman_text(low_usd, rate)} to {hn} {hu}"
    return f"{toman} ({usd})"


def nusd_money(nusd: int, rate: float | int | Decimal | None) -> str:
    """:func:`money` of an exact Ledger amount in nano-USD."""
    return money(Decimal(nusd) / NANO_PER_USD, rate)


def nusd_money_range(low_nusd: int, high_nusd: int, rate: float | int | Decimal | None) -> str:
    """:func:`money_range` of exact Ledger amounts in nano-USD."""
    return money_range(Decimal(low_nusd) / NANO_PER_USD, Decimal(high_nusd) / NANO_PER_USD, rate)
