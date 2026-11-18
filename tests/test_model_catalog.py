"""Tests for the model catalog (``[models."<id>"]`` in automa.toml; D94). No model calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from automa.budget import BudgetGuard, CostLedger
from automa.config import (
    AutomaConfig,
    Capability,
    ConfigError,
    ModelPrice,
    Tier,
    load_config,
)
from automa.llm import CallContext, ChatResponse, FakeProvider, LLMGateway, Message, Usage

CATALOG_TOML = """
base_url = "https://gateway.example/v1"
[tiers]
small = "claude-haiku-5-5"
medium = "gpt-5"
large = "m-legacy"
[prices.m-legacy]
input_per_mtok = 3
output_per_mtok = 9
[models."claude-haiku-5-5"]
family = "anthropic"
capabilities = ["text", "code"]
input_per_mtok = 1
output_per_mtok = 5
[models."gpt-5"]
family = "openai"
capabilities = ["text", "code", "vision"]
input_per_mtok = 1.25
output_per_mtok = 10
[models."gpt-image-1"]
family = "openai"
capabilities = ["image_out"]
input_per_mtok = 5
output_per_mtok = 40
"""


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _load(cwd: Path, text: str) -> AutomaConfig:
    (cwd / "automa.toml").write_text(text, encoding="utf-8")
    return load_config(env_file=None, environ={})


def _cfg(models: dict[str, object], **extra: object) -> AutomaConfig:
    return AutomaConfig.model_validate({"models": models, **extra})


def _entry(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "family": "anthropic",
        "capabilities": ["text"],
        "input_per_mtok": 1,
        "output_per_mtok": 2,
    }
    base.update(over)
    return base


def test_catalog_loads_from_toml(cwd: Path) -> None:
    cfg = _load(cwd, CATALOG_TOML)
    assert set(cfg.models) == {"claude-haiku-5-5", "gpt-5", "gpt-image-1"}
    info = cfg.model_info("gpt-5")
    assert info is not None
    assert info.family == "openai"
    assert info.capabilities == (Capability.TEXT, Capability.CODE, Capability.VISION)
    assert cfg.family_of("claude-haiku-5-5") == "anthropic"
    assert cfg.family_of("m-legacy") is None
    assert cfg.model_info("unknown") is None


def test_tier_models_priced_from_catalog_or_prices(cwd: Path) -> None:
    cfg = _load(cwd, CATALOG_TOML)
    assert cfg.price_for(cfg.model_for(Tier.SMALL)) == ModelPrice(
        input_per_mtok=1, output_per_mtok=5
    )
    assert cfg.price_for("gpt-5").output_per_mtok == 10
    assert cfg.price_for("m-legacy").input_per_mtok == 3
    with pytest.raises(ConfigError, match="no price configured for model 'nope'"):
        cfg.price_for("nope")


def test_no_catalog_is_still_valid_and_empty() -> None:
    cfg = AutomaConfig()
    assert cfg.models == {}
    assert cfg.catalog() == []


def test_catalog_filters(cwd: Path) -> None:
    cfg = _load(cwd, CATALOG_TOML)
    assert [m for m, _ in cfg.catalog()] == ["claude-haiku-5-5", "gpt-5", "gpt-image-1"]
    assert [m for m, _ in cfg.catalog(capability=Capability.VISION)] == ["gpt-5"]
    assert [m for m, _ in cfg.catalog(capability=Capability.IMAGE_OUT)] == ["gpt-image-1"]
    assert [m for m, _ in cfg.catalog(family="openai")] == ["gpt-5", "gpt-image-1"]
    assert [m for m, _ in cfg.catalog(family="openai", capability=Capability.CODE)] == ["gpt-5"]
    assert cfg.catalog(family="google") == []


def test_tier_model_without_any_price_is_rejected() -> None:
    with pytest.raises(ValueError, match="no price configured for tier model"):
        _cfg({"a": _entry()}, tiers={"small": "b"})


def test_model_priced_twice_is_rejected(cwd: Path) -> None:
    text = CATALOG_TOML + '\n[prices."gpt-5"]\ninput_per_mtok = 1\noutput_per_mtok = 1\n'
    with pytest.raises(ConfigError, match=r"priced both in \[prices\] and \[models\]: gpt-5"):
        _load(cwd, text)


def test_model_priced_twice_with_equal_price_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="priced both"):
        _cfg({"a": _entry()}, prices={"a": {"input_per_mtok": 1, "output_per_mtok": 2}})


def test_tier_model_without_text_capability_is_rejected() -> None:
    with pytest.raises(ValueError, match="without the 'text' capability: img"):
        _cfg({"img": _entry(capabilities=["image_out"])}, tiers={"large": "img"})


def test_non_tier_image_only_model_is_fine() -> None:
    cfg = _cfg({"img": _entry(capabilities=["image_out"])})
    assert cfg.model_info("img") is not None


@pytest.mark.parametrize(
    "field",
    ["family", "capabilities", "input_per_mtok", "output_per_mtok"],
)
def test_missing_field_is_rejected(field: str) -> None:
    entry = _entry()
    del entry[field]
    with pytest.raises(ValueError, match=field):
        _cfg({"a": entry})


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        _cfg({"a": _entry(tier="large")})


@pytest.mark.parametrize(
    "caps",
    [[], "text", ["text", "text"], ["telepathy"], ["TEXT"], [1]],
)
def test_bad_capabilities_are_rejected(caps: object) -> None:
    with pytest.raises(ValueError):
        _cfg({"a": _entry(capabilities=caps)})


@pytest.mark.parametrize("family", ["", "Anthropic", "1ai", "a b", "x" * 33, "an/thropic"])
def test_bad_family_is_rejected(family: str) -> None:
    with pytest.raises(ValueError, match="family"):
        _cfg({"a": _entry(family=family)})


@pytest.mark.parametrize(
    "price",
    [-1, True, "1", float("nan"), float("inf"), None],
)
def test_bad_price_is_rejected(price: object) -> None:
    with pytest.raises(ValueError):
        _cfg({"a": _entry(input_per_mtok=price)})
    with pytest.raises(ValueError):
        _cfg({"a": _entry(output_per_mtok=price)})


def test_zero_price_is_allowed() -> None:
    cfg = _cfg({"free": _entry(input_per_mtok=0, output_per_mtok=0.0)})
    assert cfg.price_for("free") == ModelPrice(input_per_mtok=0, output_per_mtok=0)


@pytest.mark.parametrize(
    "model_id",
    ["", " gpt", "gpt 5", "-gpt", 'g"pt', "gpt\n5", "مدل", "a" * 201, "gpt‮5"],
)
def test_bad_model_id_is_rejected(model_id: str) -> None:
    with pytest.raises(ValueError, match="invalid model ids"):
        _cfg({model_id: _entry()})


@pytest.mark.parametrize(
    "model_id",
    ["claude-sonnet-5-5", "openai/gpt-5", "gemini-2.5-pro", "deepseek-v3.1:free", "x@1+2"],
)
def test_good_model_ids(model_id: str) -> None:
    assert _cfg({model_id: _entry()}).model_info(model_id) is not None


def test_too_many_models_are_rejected() -> None:
    from automa.config import MAX_CATALOG_MODELS

    models = {f"m{i}": _entry() for i in range(MAX_CATALOG_MODELS + 1)}
    with pytest.raises(ValueError, match="at most"):
        _cfg(models)


def test_catalog_is_immutable(cwd: Path) -> None:
    cfg = _load(cwd, CATALOG_TOML)
    info = cfg.model_info("gpt-5")
    assert info is not None
    with pytest.raises(ValueError):
        info.family = "x"  # type: ignore[misc]


def test_secret_key_inside_models_is_rejected(cwd: Path) -> None:
    text = CATALOG_TOML + 'api_key = "sk-not-here"\n'
    with pytest.raises(ConfigError, match="secrets are not allowed"):
        _load(cwd, text)


def test_error_message_names_the_model(cwd: Path) -> None:
    text = CATALOG_TOML.replace(
        'family = "openai"\ncapabilities = ["text", "code", "vision"]',
        'family = "OpenAI"\ncapabilities = ["text", "code", "vision"]',
    )
    with pytest.raises(ConfigError, match=r"models\.gpt-5\.family"):
        _load(cwd, text)


def test_gateway_charges_catalog_price(cwd: Path, tmp_path: Path) -> None:
    """A tier model priced only by the catalog is metered by the BudgetGuard as usual."""
    cfg = _load(cwd, CATALOG_TOML)
    ledger = CostLedger(tmp_path / "ledger.sqlite3")
    guard = BudgetGuard(cfg.limits, ledger)
    reply = ChatResponse(
        model="claude-haiku-5-5",
        message=Message(role="assistant", content="ok"),
        usage=Usage(input_tokens=1000, output_tokens=1000),
    )
    provider = FakeProvider([reply])
    gateway = LLMGateway(cfg, provider, guard=guard)
    result = gateway.complete(
        [Message(role="user", content="hi")],
        tier=Tier.SMALL,
        max_tokens=100,
        context=CallContext(project_id="p1"),
    )
    assert provider.requests[0].model == "claude-haiku-5-5"
    assert result is not None
    # 1000 * 1 / 1e6 + 1000 * 5 / 1e6 USD = 0.006 USD = 6_000_000 nano-USD
    assert ledger.spent_nusd(project_id="p1") == 6_000_000
