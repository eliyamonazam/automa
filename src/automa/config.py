"""Configuration loading.

Sources, from lowest to highest precedence:

1. Built-in defaults (conservative, finite budget limits).
2. ``automa.toml`` (non-secret settings: tiers, prices, limits, workspace).
3. ``.env`` file in the working directory.
4. Process environment variables.

Secrets (the API key) are accepted **only** from the environment or ``.env``.
A secret found in ``automa.toml`` is rejected, so the TOML file stays safe to commit.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)

DEFAULT_CONFIG_FILE = "automa.toml"
DEFAULT_ENV_FILE = ".env"

ENV_BASE_URL = "AUTOMA_BASE_URL"
ENV_API_KEY = "AUTOMA_API_KEY"
ENV_CONFIG = "AUTOMA_CONFIG"
ENV_WORKSPACE = "AUTOMA_WORKSPACE"

# Keys that must never appear in the TOML file (it may be committed).
_FORBIDDEN_TOML_KEYS = frozenset({"api_key", "apikey", "key", "token", "secret", "password"})


class ConfigError(Exception):
    """Raised when configuration is missing, malformed or unsafe."""


class Tier(StrEnum):
    """Model capability tiers. Roles request a tier, never a model name."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ModelPrice(_Strict):
    """Price in USD per one million tokens."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class BudgetLimits(_Strict):
    """Hard spending caps in USD. All caps are finite and strictly positive."""

    project_usd: float = Field(default=5.0, gt=0)
    daily_usd: float = Field(default=2.0, gt=0)
    per_call_usd: float = Field(default=0.25, gt=0)

    @model_validator(mode="after")
    def _per_call_within_caps(self) -> BudgetLimits:
        if self.per_call_usd > self.daily_usd or self.per_call_usd > self.project_usd:
            raise ValueError("per_call_usd must not exceed daily_usd or project_usd")
        return self


class AutomaConfig(_Strict):
    """Fully resolved, validated configuration."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: SecretStr | None = None
    tiers: dict[Tier, str] = Field(default_factory=dict)
    prices: dict[str, ModelPrice] = Field(default_factory=dict)
    limits: BudgetLimits = Field(default_factory=BudgetLimits)
    workspace: Path = Path("workspace")

    @model_validator(mode="after")
    def _check(self) -> AutomaConfig:
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("base_url must start with http:// or https://")
        missing = sorted({m for m in self.tiers.values() if m not in self.prices})
        if missing:
            # Without a price the budget layer cannot bound spending.
            raise ValueError(f"no price configured for tier model(s): {', '.join(missing)}")
        return self

    def model_for(self, tier: Tier) -> str:
        """Return the model name mapped to ``tier``."""
        try:
            return self.tiers[tier]
        except KeyError:
            raise ConfigError(f"no model configured for tier '{tier.value}'") from None

    def price_for(self, model: str) -> ModelPrice:
        """Return the configured price for ``model``."""
        try:
            return self.prices[model]
        except KeyError:
            raise ConfigError(f"no price configured for model '{model}'") from None

    def require_api_key(self) -> str:
        """Return the API key, or raise if it is not configured."""
        if self.api_key is None or not self.api_key.get_secret_value():
            raise ConfigError(f"{ENV_API_KEY} is not set (put it in .env or the environment)")
        return self.api_key.get_secret_value()


def parse_env_file(text: str) -> dict[str, str]:
    """Parse a minimal ``.env`` format: ``KEY=value`` lines, ``#`` comments, optional quotes."""
    result: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            raise ConfigError(f".env line {lineno}: expected KEY=value")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key.replace("_", "").isalnum():
            raise ConfigError(f".env line {lineno}: invalid key {key!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        result[key] = value
    return result


def _find_forbidden_keys(data: Mapping[str, Any], prefix: str = "") -> list[str]:
    found: list[str] = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        if key.lower() in _FORBIDDEN_TOML_KEYS:
            found.append(path)
        if isinstance(value, Mapping):
            found.extend(_find_forbidden_keys(value, f"{path}."))
    return found


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    forbidden = _find_forbidden_keys(data)
    if forbidden:
        raise ConfigError(
            f"{path}: secrets are not allowed in the config file ({', '.join(forbidden)}); "
            f"use {ENV_API_KEY} in .env instead"
        )
    return data


def load_config(
    config_path: Path | str | None = None,
    *,
    env_file: Path | str | None = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> AutomaConfig:
    """Load and validate configuration.

    ``environ`` defaults to ``os.environ``; tests pass an explicit mapping.
    ``env_file=None`` disables ``.env`` loading. A missing ``.env`` is not an error.
    An explicitly given ``config_path`` (argument or ``AUTOMA_CONFIG``) must exist;
    the default ``automa.toml`` is optional.
    """
    env: dict[str, str] = {}
    if env_file is not None:
        env_path = Path(env_file)
        if env_path.is_file():
            env.update(parse_env_file(env_path.read_text(encoding="utf-8")))
    env.update(os.environ if environ is None else environ)

    explicit = config_path if config_path is not None else env.get(ENV_CONFIG)
    toml_path = Path(explicit) if explicit else Path(DEFAULT_CONFIG_FILE)
    data: dict[str, Any] = {}
    if toml_path.is_file():
        data = _read_toml(toml_path)
    elif explicit:
        raise ConfigError(f"config file not found: {toml_path}")

    if env.get(ENV_BASE_URL):
        data["base_url"] = env[ENV_BASE_URL]
    if env.get(ENV_WORKSPACE):
        data["workspace"] = env[ENV_WORKSPACE]
    if env.get(ENV_API_KEY):
        data["api_key"] = env[ENV_API_KEY]

    try:
        return AutomaConfig.model_validate(data)
    except ValidationError as exc:
        # Build the message without input values: they may contain the API key.
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'config'}: {err['msg']}"
            for err in exc.errors(include_input=False, include_url=False)
        )
        raise ConfigError(f"invalid configuration: {details}") from None
