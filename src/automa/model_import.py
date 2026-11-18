"""Turn a gateway's text price list into ``[models."<id>"]`` catalog entries (D95).

Pure functions: no model call, no network, no file written. ``automa models import``
reads the file, calls :func:`parse_1xai_price_list` and prints :func:`render_catalog_toml`;
the owner reviews the printed entries and pastes them into ``automa.toml`` by hand.

The 1xAi format is one model per line: the model id, then the input and the output
price in toman per 1,000 tokens. Fields are separated by ``|``, tabs, ``;``, commas
(when the line has exactly three comma-separated fields) or whitespace. Persian and
Arabic-Indic digits and the thousands separators ``,``/``٬``/``_`` are accepted inside
numbers; ``٫`` is a decimal point. Lines starting with ``#`` are comments, and the first
content line may be a header (no digits in its price fields). Anything else that does
not parse is an error with its line number, never a guess.

Prices are converted to USD per one million tokens with exact decimal arithmetic and
rounded **up** (``PRICE_DECIMALS`` places), so the BudgetGuard never under-estimates a
call. The family comes from the id's prefix and the capabilities are a conservative
default; both are marked in the output as guesses to review.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation

from pydantic import ValidationError

from automa.config import MAX_CATALOG_MODELS, MODEL_ID_RE, Capability, ModelInfo

PRICE_DECIMALS = 6
MAX_PRICE_LIST_BYTES = 1_000_000
UNKNOWN_FAMILY = "unknown"
# Largest accepted toman price per 1K tokens / rate (sanity bounds, not policy).
MAX_TOMAN_PER_KTOK = Decimal("1000000000")
MAX_RATE = Decimal("1000000000")

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_NUMBER_RE = re.compile(r"^[0-9]+(\.[0-9]+)?$")
_ANY_DIGIT_RE = re.compile(r"[0-9۰-۹٠-٩]")

# Id prefix (after an optional "vendor/") -> family. Longest prefix wins.
_FAMILY_PREFIXES: dict[str, str] = {
    "claude": "anthropic",
    "gpt": "openai",
    "chatgpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "dall-e": "openai",
    "codex": "openai",
    "gemini": "google",
    "gemma": "google",
    "imagen": "google",
    "deepseek": "deepseek",
    "grok": "xai",
    "llama": "meta",
    "mistral": "mistral",
    "mixtral": "mistral",
    "codestral": "mistral",
    "magistral": "mistral",
    "qwen": "alibaba",
    "qwq": "alibaba",
    "kimi": "moonshot",
    "moonshot": "moonshot",
    "glm": "zhipu",
    "flux": "bfl",
}
# A "vendor/" prefix as some gateways spell it -> family.
_VENDOR_FAMILIES: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "deepseek": "deepseek",
    "x-ai": "xai",
    "xai": "xai",
    "meta-llama": "meta",
    "meta": "meta",
    "mistralai": "mistral",
    "mistral": "mistral",
    "qwen": "alibaba",
    "moonshotai": "moonshot",
    "z-ai": "zhipu",
    "black-forest-labs": "bfl",
}
_IMAGE_MARKERS = ("dall-e", "gpt-image", "imagen", "flux", "-image")
_CODE_FAMILIES = frozenset(
    {"anthropic", "openai", "google", "deepseek", "xai", "mistral", "alibaba", "moonshot", "zhipu"}
)


class PriceListError(ValueError):
    """The price list (or the rate) is not usable; nothing is printed."""


@dataclass(frozen=True)
class ImportedModel:
    """One parsed line: the catalog entry plus what was guessed about it."""

    model_id: str
    info: ModelInfo
    line: int
    input_toman_per_ktok: Decimal
    output_toman_per_ktok: Decimal
    family_guessed: bool  # True when the family is UNKNOWN_FAMILY (no prefix matched)


def guess_family(model_id: str) -> str:
    """The family for ``model_id`` from its prefix, or ``UNKNOWN_FAMILY``."""
    lower = model_id.lower()
    if "/" in lower:
        vendor, _, rest = lower.partition("/")
        if vendor in _VENDOR_FAMILIES:
            return _VENDOR_FAMILIES[vendor]
        lower = rest
    best = ""
    for prefix in _FAMILY_PREFIXES:
        if lower.startswith(prefix) and len(prefix) > len(best):
            nxt = lower[len(prefix) : len(prefix) + 1]
            # "o1" must not match "o100x"; letters after a short prefix are another name.
            if len(prefix) <= 2 and nxt and nxt not in "-._:":
                continue
            best = prefix
    return _FAMILY_PREFIXES[best] if best else UNKNOWN_FAMILY


def guess_capabilities(model_id: str, family: str) -> tuple[Capability, ...]:
    """Conservative default capabilities: image models produce images only; chat models
    of well-known families get ``text`` and ``code``; anything else just ``text``."""
    lower = model_id.lower()
    if any(m in lower for m in _IMAGE_MARKERS):
        return (Capability.IMAGE_OUT,)
    if family in _CODE_FAMILIES:
        return (Capability.TEXT, Capability.CODE)
    return (Capability.TEXT,)


def parse_rate(text: str) -> Decimal:
    """``--rate`` (toman per USD): a positive finite number."""
    value = _number(text)
    if value is None or value <= 0 or value > MAX_RATE:
        raise PriceListError(f"rate must be a positive number of toman per USD, got {text!r}")
    return value


def toman_per_ktok_to_usd_per_mtok(toman: Decimal, rate: Decimal) -> Decimal:
    """Exact conversion, rounded up to ``PRICE_DECIMALS`` places (never under-priced)."""
    usd = toman * 1000 / rate
    return usd.quantize(Decimal(1).scaleb(-PRICE_DECIMALS), rounding=ROUND_CEILING)


def _number(field: str) -> Decimal | None:
    s = field.strip().translate(_DIGITS).replace("٫", ".")
    for sep in (",", "٬", "_", "٬"):
        s = s.replace(sep, "")
    if not _NUMBER_RE.fullmatch(s):
        return None
    try:
        return Decimal(s)
    except InvalidOperation:  # pragma: no cover - the regex already excludes this
        return None


def _split(line: str) -> list[str]:
    for sep in ("|", "\t", ";"):
        if sep in line:
            return [f.strip() for f in line.strip().strip("|").split(sep) if f.strip()]
    if "," in line:
        parts = [f.strip() for f in line.split(",")]
        if len(parts) == 3 and all(parts):
            return parts
    return line.split()


def parse_1xai_price_list(text: str, *, rate: Decimal) -> list[ImportedModel]:
    """Parse a 1xAi text price list into catalog entries (see the module docstring).

    Raises :class:`PriceListError` for any unparseable line (with its number), a
    repeated model id, an empty list or more than ``MAX_CATALOG_MODELS`` models.
    """
    if rate <= 0:
        raise PriceListError("rate must be positive")
    if len(text.encode("utf-8")) > MAX_PRICE_LIST_BYTES:
        raise PriceListError(f"price list is larger than {MAX_PRICE_LIST_BYTES} bytes")
    models: list[ImportedModel] = []
    seen: dict[str, int] = {}
    errors: list[str] = []
    first_content = True
    for lineno, raw in enumerate(text.lstrip("﻿").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = _split(line)
        is_first = first_content
        first_content = False
        if len(fields) != 3:
            if is_first and not any(_ANY_DIGIT_RE.search(f) for f in fields[1:]):
                continue  # header line
            errors.append(f"line {lineno}: expected 3 fields (id, input, output)")
            continue
        model_id, in_raw, out_raw = fields
        in_t, out_t = _number(in_raw), _number(out_raw)
        if in_t is None or out_t is None:
            if is_first and not _ANY_DIGIT_RE.search(in_raw + out_raw):
                continue  # header line
            errors.append(f"line {lineno}: prices must be non-negative numbers")
            continue
        if not MODEL_ID_RE.fullmatch(model_id):
            errors.append(f"line {lineno}: invalid model id {model_id!r}")
            continue
        if in_t > MAX_TOMAN_PER_KTOK or out_t > MAX_TOMAN_PER_KTOK:
            errors.append(f"line {lineno}: price is implausibly large")
            continue
        if model_id in seen:
            errors.append(f"line {lineno}: model {model_id} repeats line {seen[model_id]}")
            continue
        seen[model_id] = lineno
        family = guess_family(model_id)
        try:
            info = ModelInfo(
                family=family,
                capabilities=guess_capabilities(model_id, family),
                input_per_mtok=float(toman_per_ktok_to_usd_per_mtok(in_t, rate)),
                output_per_mtok=float(toman_per_ktok_to_usd_per_mtok(out_t, rate)),
            )
        except ValidationError:  # pragma: no cover - inputs are validated above
            errors.append(f"line {lineno}: not a valid catalog entry")
            continue
        models.append(
            ImportedModel(
                model_id=model_id,
                info=info,
                line=lineno,
                input_toman_per_ktok=in_t,
                output_toman_per_ktok=out_t,
                family_guessed=family == UNKNOWN_FAMILY,
            )
        )
    if errors:
        shown = errors[:20]
        more = f" (and {len(errors) - 20} more)" if len(errors) > 20 else ""
        raise PriceListError("; ".join(shown) + more)
    if not models:
        raise PriceListError("no models found in the price list")
    if len(models) > MAX_CATALOG_MODELS:
        raise PriceListError(f"at most {MAX_CATALOG_MODELS} models in the catalog")
    return models


def _fmt(value: float) -> str:
    return _fmt_dec(Decimal(str(value)))


def _fmt_dec(value: Decimal) -> str:
    text = f"{value:f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def render_catalog_toml(models: list[ImportedModel], *, rate: Decimal) -> str:
    """TOML ``[models."<id>"]`` sections for ``models`` (sorted by id), with comments
    marking the guessed fields. The result is re-parsed and re-validated before it is
    returned, so what is printed always loads (fail closed)."""
    lines = [
        f"# Imported from a 1xAi price list at {_fmt_dec(rate)} toman per USD;",
        "# prices are USD per 1M tokens, rounded up. Review family and capabilities",
        "# (guessed from the id) before pasting into automa.toml.",
    ]
    for m in sorted(models, key=lambda x: x.model_id):
        caps = ", ".join(f'"{c.value}"' for c in m.info.capabilities)
        lines.append("")
        lines.append(f'[models."{m.model_id}"]')
        note = "  # unknown maker: set the family" if m.family_guessed else ""
        lines.append(f'family = "{m.info.family}"{note}')
        lines.append(f"capabilities = [{caps}]")
        lines.append(
            f"input_per_mtok = {_fmt(m.info.input_per_mtok)}"
            f"  # {_fmt_dec(m.input_toman_per_ktok)} toman / 1K"
        )
        lines.append(
            f"output_per_mtok = {_fmt(m.info.output_per_mtok)}"
            f"  # {_fmt_dec(m.output_toman_per_ktok)} toman / 1K"
        )
    out = "\n".join(lines) + "\n"
    parsed = tomllib.loads(out)["models"]
    for m in models:
        if ModelInfo.model_validate(parsed[m.model_id]) != m.info:
            raise PriceListError(f"internal error: {m.model_id} does not round-trip")
    return out
