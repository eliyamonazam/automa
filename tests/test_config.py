"""Tests for configuration loading (no network, no real secrets)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from automa.config import (
    AutomaConfig,
    BudgetLimits,
    ConfigError,
    Tier,
    load_config,
    parse_env_file,
)

ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "fake-test-key-123"

VALID_TOML = """
base_url = "https://gateway.example/v1"
[tiers]
small = "m-small"
large = "m-large"
[prices.m-small]
input_per_mtok = 0.1
output_per_mtok = 0.2
[prices.m-large]
input_per_mtok = 3
output_per_mtok = 9
[limits]
project_usd = 10
daily_usd = 3
per_call_usd = 0.5
"""


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_defaults_without_files(cwd: Path) -> None:
    cfg = load_config(environ={})
    assert cfg.api_key is None
    assert cfg.limits == BudgetLimits()
    assert cfg.workspace == Path("workspace")
    with pytest.raises(ConfigError, match="AUTOMA_API_KEY"):
        cfg.require_api_key()
    with pytest.raises(ConfigError, match="tier 'small'"):
        cfg.model_for(Tier.SMALL)


def test_default_limits_are_finite_and_positive() -> None:
    limits = BudgetLimits()
    assert 0 < limits.per_call_usd <= limits.daily_usd <= limits.project_usd


def test_loads_toml(cwd: Path) -> None:
    (cwd / "automa.toml").write_text(VALID_TOML, encoding="utf-8")
    cfg = load_config(environ={})
    assert cfg.base_url == "https://gateway.example/v1"
    assert cfg.model_for(Tier.LARGE) == "m-large"
    assert cfg.price_for("m-small").output_per_mtok == 0.2
    assert cfg.limits.daily_usd == 3
    with pytest.raises(ConfigError, match="tier 'medium'"):
        cfg.model_for(Tier.MEDIUM)
    with pytest.raises(ConfigError, match="no price"):
        cfg.price_for("unknown")


def test_env_file_and_environ_precedence(cwd: Path) -> None:
    (cwd / "automa.toml").write_text(VALID_TOML, encoding="utf-8")
    (cwd / ".env").write_text(
        f"AUTOMA_API_KEY={FAKE_KEY}\nAUTOMA_BASE_URL=https://from-dotenv/v1\n",
        encoding="utf-8",
    )
    cfg = load_config(environ={})
    assert cfg.base_url == "https://from-dotenv/v1"
    assert cfg.require_api_key() == FAKE_KEY

    cfg = load_config(environ={"AUTOMA_BASE_URL": "https://from-env/v1"})
    assert cfg.base_url == "https://from-env/v1"
    assert cfg.require_api_key() == FAKE_KEY


def test_env_file_can_be_disabled(cwd: Path) -> None:
    (cwd / ".env").write_text(f"AUTOMA_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    assert load_config(env_file=None, environ={}).api_key is None


def test_api_key_is_not_leaked_in_repr(cwd: Path) -> None:
    cfg = load_config(environ={"AUTOMA_API_KEY": FAKE_KEY})
    assert FAKE_KEY not in repr(cfg)
    assert FAKE_KEY not in str(cfg)


def test_api_key_not_leaked_in_validation_error(cwd: Path) -> None:
    with pytest.raises(ConfigError) as info:
        load_config(environ={"AUTOMA_API_KEY": FAKE_KEY, "AUTOMA_BASE_URL": "ftp://bad"})
    assert FAKE_KEY not in str(info.value)
    assert "base_url" in str(info.value)


@pytest.mark.parametrize(
    "snippet",
    ['api_key = "x"', '[gateway]\ntoken = "x"', '[limits]\nsecret = "x"'],
)
def test_secrets_in_toml_are_rejected(cwd: Path, snippet: str) -> None:
    (cwd / "automa.toml").write_text(snippet, encoding="utf-8")
    with pytest.raises(ConfigError, match="secrets are not allowed"):
        load_config(environ={})


def test_tier_model_without_price_is_rejected(cwd: Path) -> None:
    (cwd / "automa.toml").write_text('[tiers]\nsmall = "unpriced"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="no price"):
        load_config(environ={})


@pytest.mark.parametrize(
    "limits",
    [
        "project_usd = 0",
        "daily_usd = -1",
        'per_call_usd = "inf"',
        "per_call_usd = 3\ndaily_usd = 2",
        "per_call_usd = 6\nproject_usd = 5\ndaily_usd = 10",
        "unknown_cap = 1",
    ],
)
def test_invalid_limits_are_rejected(cwd: Path, limits: str) -> None:
    (cwd / "automa.toml").write_text(f"[limits]\n{limits}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(environ={})


def test_infinite_limit_is_rejected() -> None:
    with pytest.raises(ValueError):
        BudgetLimits(project_usd=float("inf"))


def test_unknown_top_level_key_is_rejected(cwd: Path) -> None:
    (cwd / "automa.toml").write_text("bogus = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="bogus"):
        load_config(environ={})


def test_invalid_toml_is_reported(cwd: Path) -> None:
    (cwd / "automa.toml").write_text("not = = toml", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(environ={})


def test_explicit_missing_config_path_is_error(cwd: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("missing.toml", environ={})
    with pytest.raises(ConfigError, match="not found"):
        load_config(environ={"AUTOMA_CONFIG": "missing.toml"})


def test_config_path_from_env(cwd: Path) -> None:
    (cwd / "custom.toml").write_text(VALID_TOML, encoding="utf-8")
    cfg = load_config(environ={"AUTOMA_CONFIG": "custom.toml"})
    assert cfg.model_for(Tier.SMALL) == "m-small"


def test_config_is_immutable(cwd: Path) -> None:
    cfg = load_config(environ={})
    with pytest.raises(ValueError):
        cfg.limits.daily_usd = 1000  # type: ignore[misc]


def test_example_toml_is_valid(cwd: Path) -> None:
    shutil.copy(ROOT / "automa.toml.example", cwd / "automa.toml")
    cfg = load_config(environ={})
    assert isinstance(cfg, AutomaConfig)
    assert set(cfg.tiers) == set(Tier)


def test_parse_env_file() -> None:
    text = "# comment\n\nA=1\nexport B = two\nC=\"quoted # value\"\nD='single'\nE=x # trailing\n"
    assert parse_env_file(text) == {
        "A": "1",
        "B": "two",
        "C": "quoted # value",
        "D": "single",
        "E": "x",
    }


@pytest.mark.parametrize("line", ["NOEQUALS", "BAD-KEY=1"])
def test_parse_env_file_rejects_malformed(line: str) -> None:
    with pytest.raises(ConfigError):
        parse_env_file(line)


def test_nan_price_is_rejected() -> None:
    from automa.config import ModelPrice

    with pytest.raises(ValueError):
        ModelPrice(input_per_mtok=float("nan"), output_per_mtok=1)
