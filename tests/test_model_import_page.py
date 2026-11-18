"""Importing the 1xAi pricing page as copied from a browser (one cell per line).

The first real import failed: the owner copied the page, which puts every table
cell on its own line, and the line-based parser rejected all of it. The excerpt
below is verbatim from that copy (2026-11-18).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from automa.config import Capability
from automa.model_import import (
    PriceListError,
    looks_like_page_copy,
    parse_1xai_page_copy,
    render_catalog_toml,
)

PAGE = """خانوادهٔ GPT-5
پرچم‌دارِ OpenAI با استدلالِ درون‌مدلی — پیش از پاسخ، برای مسئله «فکر» می‌کند.

۵۰ مدل
شناسهٔ مدل
ورودی، هر ۱K توکن
خروجی، هر ۱K توکن
پنجره
gpt-5.5
۱٬۵۴۸
۹٬۲۹۱
۴۰۰K
آزمایشگاه
کد
gpt-5-search-api
۴۰۴
۳٬۲۳۲
۲۷۲K
آزمایشگاه
کد
Claude Sonnet
نقطهٔ تعادلِ خانوادهٔ Claude — کیفیتِ نزدیک به Opus با هزینهٔ متوسط.

۱۸ مدل
شناسهٔ مدل
ورودی، هر ۱K توکن
خروجی، هر ۱K توکن
پنجره
claude-sonnet-5-5
۶۴۶
۳٬۲۳۲
۱M
آزمایشگاه
کد
DeepSeek
مدل‌های DeepSeek با قیمتِ توکنِ بسیار پایین.

deepseek-v4.1-flash
۹۷
۳۸۸
—
آزمایشگاه
کد
gemini-3-pro-image
۶۱۹
۳۷٬۱۶۳
۱M
آزمایشگاه
کد
gemini-1.5-flash
۲۴
—
۸K
آزمایشگاه
کد
text-embedding-3-large
۴۲
—
۸K
آزمایشگاه
کد
gemini-2.5-flash-preview-tts
۱۶۲
۳٬۲۳۲
—
آزمایشگاه
کد
"""

RATE = Decimal(323000)


def test_page_copy_is_detected() -> None:
    assert looks_like_page_copy(PAGE)
    assert not looks_like_page_copy("gpt-5.5 1548 9291\nclaude-sonnet-5-5 646 3232\n")


def test_page_copy_imports_chat_models_with_exact_prices() -> None:
    models, skipped = parse_1xai_page_copy(PAGE, rate=RATE)
    by_id = {m.model_id: m for m in models}
    assert set(by_id) == {
        "gpt-5.5",
        "claude-sonnet-5-5",
        "deepseek-v4.1-flash",
        "gemini-3-pro-image",
    }
    sonnet = by_id["claude-sonnet-5-5"]
    assert sonnet.info.family == "anthropic"
    assert sonnet.input_toman_per_ktok == 646 and sonnet.output_toman_per_ktok == 3232
    assert sonnet.info.input_per_mtok == 2.0
    assert sonnet.info.output_per_mtok >= 3232 * 1000 / 323000  # rounded up, never down
    assert by_id["gemini-3-pro-image"].info.capabilities == (Capability.IMAGE_OUT,)
    assert by_id["deepseek-v4.1-flash"].info.family == "deepseek"
    reasons = "\n".join(skipped)
    assert "gpt-5-search-api: not a chat model" in reasons
    assert "gemini-1.5-flash: no output price" in reasons
    assert "text-embedding-3-large: no output price" in reasons
    assert "gemini-2.5-flash-preview-tts: not a chat model" in reasons


def test_headings_and_windows_are_never_models() -> None:
    models, _ = parse_1xai_page_copy(PAGE, rate=RATE)
    ids = {m.model_id for m in models}
    assert not ids & {"DeepSeek", "400K", "1M", "8K", "Claude", "GPT-5"}


def test_rendered_page_import_loads() -> None:
    models, _ = parse_1xai_page_copy(PAGE, rate=RATE)
    out = render_catalog_toml(models, rate=RATE)
    assert '[models."claude-sonnet-5-5"]' in out


def test_page_copy_without_models_is_an_error() -> None:
    with pytest.raises(PriceListError):
        parse_1xai_page_copy("عنوان\nتوضیح\nشناسهٔ مدل\nورودی\nخروجی\nپنجره\n", rate=RATE)


def test_repeated_ids_keep_the_first() -> None:
    page = "gpt-5.5\n۱\n۲\nx\ngpt-5.5\n۹\n۹\ny\nz\nw\n"
    models, skipped = parse_1xai_page_copy(page, rate=RATE)
    assert [m.input_toman_per_ktok for m in models] == [1]
    assert any("repeated" in s for s in skipped)


def test_more_families_are_recognised() -> None:
    from automa.model_import import guess_family

    assert guess_family("MiniMax-M3") == "minimax"
    assert guess_family("ft:gpt-4o-mini-2024-07-18") == "openai"
