"""Tests for ``automa.model_import`` (1xAi price list -> catalog entries; D95). Pure."""

from __future__ import annotations

import tomllib
from decimal import Decimal

import pytest

from automa.config import AutomaConfig, Capability, ModelInfo
from automa.model_import import (
    MAX_PRICE_LIST_BYTES,
    UNKNOWN_FAMILY,
    PriceListError,
    guess_capabilities,
    guess_family,
    parse_1xai_price_list,
    parse_rate,
    render_catalog_toml,
    toman_per_ktok_to_usd_per_mtok,
)

RATE = Decimal(323000)


def _one(text: str, rate: Decimal = RATE) -> ModelInfo:
    (m,) = parse_1xai_price_list(text, rate=rate)
    return m.info


# --- conversion -----------------------------------------------------------------


def test_exact_conversion() -> None:
    # 969 toman / 1K tokens at 323,000 toman / USD = 3 USD / 1M tokens exactly.
    assert toman_per_ktok_to_usd_per_mtok(Decimal(969), RATE) == Decimal("3")


def test_conversion_rounds_up() -> None:
    # 400 * 1000 / 323000 = 1.2383900928... -> 1.238391 (ceiling, never below).
    v = toman_per_ktok_to_usd_per_mtok(Decimal(400), RATE)
    assert v == Decimal("1.238391")
    assert v >= Decimal(400) * 1000 / RATE


def test_tiny_price_rounds_up_to_smallest_unit() -> None:
    assert toman_per_ktok_to_usd_per_mtok(Decimal("0.0001"), RATE) == Decimal("0.000001")


def test_zero_price_stays_zero() -> None:
    assert _one("free-model 0 0").input_per_mtok == 0


@pytest.mark.parametrize("toman", ["400", "1", "7", "3200", "0.3", "123.456", "99999"])
def test_stored_price_not_below_exact_price(toman: str) -> None:
    info = _one(f"gpt-5 {toman} {toman}")
    exact = Decimal(toman) * 1000 / RATE
    # The stored float is the rounded-up 6-decimal price (as printed and re-read).
    stored = Decimal(str(info.input_per_mtok))
    assert stored >= exact
    assert stored - exact < Decimal("0.000001")
    assert info.input_per_mtok == float(toman_per_ktok_to_usd_per_mtok(Decimal(toman), RATE))


# --- rate -----------------------------------------------------------------------


@pytest.mark.parametrize("text", ["323000", "323,000", "۳۲۳۰۰۰", "1.5"])
def test_rate_ok(text: str) -> None:
    assert parse_rate(text) > 0


@pytest.mark.parametrize("text", ["0", "-5", "abc", "", "1e5", "inf", "nan", "10000000000"])
def test_rate_rejected(text: str) -> None:
    with pytest.raises(PriceListError):
        parse_rate(text)


def test_parse_rejects_non_positive_rate() -> None:
    with pytest.raises(PriceListError):
        parse_1xai_price_list("gpt-5 1 1", rate=Decimal(0))


# --- line formats ---------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "gpt-5 400 3200",
        "gpt-5\t400\t3200",
        "gpt-5 | 400 | 3200",
        "| gpt-5 | 400 | 3200 |",
        "gpt-5;400;3200",
        "gpt-5,400,3200",
        "gpt-5 | ۴۰۰ | ۳۲۰۰",
        "gpt-5 | ٤٠٠ | ٣٢٠٠",
        "gpt-5 | 400 | 3,200",
        "gpt-5 | 400 | 3٬200",
        "gpt-5    400    3_200",
    ],
)
def test_line_formats(line: str) -> None:
    info = _one(line)
    assert info.input_per_mtok == pytest.approx(1.238391)
    assert info.output_per_mtok == pytest.approx(9.907121)


def test_persian_decimal_point() -> None:
    assert _one("gpt-5 | ۳۲۳٫۰ | 1").input_per_mtok == 1.0


def test_comments_blank_lines_bom_and_header() -> None:
    text = "﻿# 1xAi prices\n\nmodel | input | output\n# note\ngpt-5 | 400 | 3200\n"
    (m,) = parse_1xai_price_list(text, rate=RATE)
    assert m.model_id == "gpt-5"
    assert m.line == 5
    assert m.input_toman_per_ktok == Decimal(400)


def test_header_only_allowed_as_first_content_line() -> None:
    with pytest.raises(PriceListError, match="line 2"):
        parse_1xai_price_list("gpt-5 1 1\nmodel input output\n", rate=RATE)


def test_header_with_extra_words_is_skipped() -> None:
    models = parse_1xai_price_list("Model name | Input price\ngpt-5 1 1\n", rate=RATE)
    assert [m.model_id for m in models] == ["gpt-5"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("gpt-5 400", "line 1: expected 3 fields"),
        ("gpt-5 1 2 3", "line 1: expected 3 fields"),
        ("gpt-5 1 two\n", "line 1: prices must be"),
        ("gpt-5 -1 2", "prices must be"),
        ("gpt-5 1e3 2", "prices must be"),
        ("gpt-5 1.2.3 2", "prices must be"),
        ("gpt 5 1 2", "expected 3 fields"),
        ('bad"id 1 2', "invalid model id"),
        ("-lead 1 2", "invalid model id"),
        ("gpt-5 9999999999 1", "implausibly large"),
        ("gpt-5 1 1\ngpt-5 2 2", "line 2: model gpt-5 repeats line 1"),
        ("# only comments\n\n", "no models found"),
        ("", "no models found"),
    ],
)
def test_bad_lists_rejected(text: str, message: str) -> None:
    with pytest.raises(PriceListError, match=message):
        parse_1xai_price_list(text, rate=RATE)


def test_all_errors_reported_with_line_numbers() -> None:
    text = "gpt-5 1 1\nx\ny\ngpt-4 1 1\nz 1\n"
    with pytest.raises(PriceListError) as e:
        parse_1xai_price_list(text, rate=RATE)
    assert "line 2" in str(e.value) and "line 3" in str(e.value) and "line 5" in str(e.value)


def test_too_many_errors_are_summarised() -> None:
    text = "gpt-5 1 1\n" + "bad\n" * 30
    with pytest.raises(PriceListError, match="and 10 more"):
        parse_1xai_price_list(text, rate=RATE)


def test_too_large_list_rejected() -> None:
    text = "#" * (MAX_PRICE_LIST_BYTES + 1)
    with pytest.raises(PriceListError, match="larger than"):
        parse_1xai_price_list(text, rate=RATE)


def test_too_many_models_rejected() -> None:
    text = "\n".join(f"m-{i} 1 1" for i in range(501))
    with pytest.raises(PriceListError, match="at most 500"):
        parse_1xai_price_list(text, rate=RATE)


# --- guesses --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "family"),
    [
        ("claude-sonnet-5-5", "anthropic"),
        ("anthropic/claude-opus-5-5", "anthropic"),
        ("gpt-5", "openai"),
        ("gpt-image-1", "openai"),
        ("o3-mini", "openai"),
        ("o1", "openai"),
        ("o100x", UNKNOWN_FAMILY),
        ("gemini-2.5-pro", "google"),
        ("deepseek-chat", "deepseek"),
        ("x-ai/grok-4", "xai"),
        ("grok-4", "xai"),
        ("llama-4-maverick", "meta"),
        ("Qwen3-coder", "alibaba"),
        ("kimi-k2", "moonshot"),
        ("glm-4.6", "zhipu"),
        ("mystery", UNKNOWN_FAMILY),
        ("someone/gpt-5", "openai"),
    ],
)
def test_guess_family(model_id: str, family: str) -> None:
    assert guess_family(model_id) == family


def test_guess_capabilities() -> None:
    assert guess_capabilities("gpt-image-1", "openai") == (Capability.IMAGE_OUT,)
    assert guess_capabilities("dall-e-3", "openai") == (Capability.IMAGE_OUT,)
    assert guess_capabilities("gemini-2.5-flash-image", "google") == (Capability.IMAGE_OUT,)
    assert guess_capabilities("claude-sonnet-5-5", "anthropic") == (
        Capability.TEXT,
        Capability.CODE,
    )
    assert guess_capabilities("mystery", UNKNOWN_FAMILY) == (Capability.TEXT,)
    # Guesses never claim vision: that must be set by hand.
    assert Capability.VISION not in guess_capabilities("gpt-4o", "openai")


def test_unknown_family_is_flagged() -> None:
    (m,) = parse_1xai_price_list("mystery 1 1", rate=RATE)
    assert m.family_guessed and m.info.family == UNKNOWN_FAMILY


# --- rendering ------------------------------------------------------------------

LIST = """model | input | output
gpt-5 | 400 | 3200
claude-sonnet-5-5 | 969 | 4845
mystery-x | 10 | 20
gpt-image-1 | 1000 | 1000
"""


def test_render_round_trips_and_sorts() -> None:
    models = parse_1xai_price_list(LIST, rate=RATE)
    out = render_catalog_toml(models, rate=RATE)
    data = tomllib.loads(out)
    assert list(data["models"]) == sorted(m.model_id for m in models)
    for m in models:
        assert ModelInfo.model_validate(data["models"][m.model_id]) == m.info
    assert "323000 toman per USD" in out
    assert "unknown maker: set the family" in out
    assert "# 969 toman / 1K" in out


def test_rendered_entries_load_into_a_config() -> None:
    models = parse_1xai_price_list(LIST, rate=RATE)
    data = tomllib.loads(render_catalog_toml(models, rate=RATE))
    config = AutomaConfig.model_validate(
        {"tiers": {"small": "gpt-5", "large": "claude-sonnet-5-5"}, **data}
    )
    assert config.price_for("claude-sonnet-5-5").input_per_mtok == 3
    assert config.family_of("gpt-5") == "openai"


def test_render_ids_with_special_chars_are_quoted() -> None:
    models = parse_1xai_price_list("openai/gpt-5:beta@x+1 1 1", rate=RATE)
    data = tomllib.loads(render_catalog_toml(models, rate=RATE))
    assert "openai/gpt-5:beta@x+1" in data["models"]
