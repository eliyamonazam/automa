"""Configuration loading.

Sources, from lowest to highest precedence:

1. Built-in defaults (conservative, finite budget limits).
2. ``automa.toml`` (non-secret settings: tiers, prices, model catalog, limits,
   workspace, sandbox, orchestrator, extra skills for built-in roles, memory in
   prompts, router).
3. ``.env`` file in the working directory.
4. Process environment variables.

Secrets (the API key) are accepted **only** from the environment or ``.env``.
A secret found in ``automa.toml`` is rejected, so the TOML file stays safe to commit.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from automa.tools.sandbox import DockerSandbox

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


class Capability(StrEnum):
    """What a catalogued model can do (``capabilities`` in ``[models."<id>"]``)."""

    TEXT = "text"
    CODE = "code"
    VISION = "vision"
    IMAGE_OUT = "image_out"


# A model id as the gateway knows it: printable ASCII, no whitespace or quotes.
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")
MODEL_ID_RE = _MODEL_ID_RE  # public name for importers (D95)
_FAMILY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
MAX_CATALOG_MODELS = 500


def _price_number(v: Any) -> Any:
    # TOML booleans and strings are not prices (pydantic would coerce some of them).
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ValueError("price must be a number")
    return v


class ModelInfo(_Strict):
    """One entry of the model catalog: ``[models."<id>"]`` in automa.toml (D94).

    Every catalogued model carries its price (USD per one million tokens), so the
    BudgetGuard can bound spending on any model the catalog offers. ``family`` (the
    model's maker, e.g. ``anthropic``) and ``capabilities`` are data for model
    selection; they never grant a tool, risk level or budget.
    """

    family: str
    capabilities: tuple[Capability, ...] = Field(min_length=1)
    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)

    @field_validator("family")
    @classmethod
    def _family(cls, v: str) -> str:
        if not _FAMILY_RE.fullmatch(v):
            raise ValueError(
                "family must be lowercase letters, digits, '_' or '-' (e.g. anthropic)"
            )
        return v

    @field_validator("capabilities", mode="before")
    @classmethod
    def _capability_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            raise ValueError("capabilities must be a list, not a string")
        if isinstance(v, list | tuple) and len(set(map(str, v))) != len(v):
            raise ValueError("capabilities must not repeat")
        return v

    @field_validator("input_per_mtok", "output_per_mtok", mode="before")
    @classmethod
    def _price(cls, v: Any) -> Any:
        return _price_number(v)

    @property
    def price(self) -> ModelPrice:
        return ModelPrice(input_per_mtok=self.input_per_mtok, output_per_mtok=self.output_per_mtok)

    def has(self, capability: Capability) -> bool:
        return capability in self.capabilities


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


class SandboxMode(StrEnum):
    """Where ``run_command`` runs: on the host (``local``) or in Docker (``docker``)."""

    LOCAL = "local"
    DOCKER = "docker"


def _docker_sandbox_cls() -> type[DockerSandbox]:
    # Imported lazily: automa.tools imports automa.plan, which imports this module.
    from automa.tools.sandbox import DockerSandbox

    return DockerSandbox


class SandboxConfig(_Strict):
    """Where ``run_command`` runs (``[sandbox]`` in automa.toml; D33).

    ``mode = "local"`` (default) runs commands on the host with the user's privileges.
    ``mode = "docker"`` runs each call in a fresh, isolated container (D32); if Docker
    is unavailable, nothing runs (fail closed, no host fallback). Docker settings
    given with ``mode = "local"`` are rejected: they would suggest an isolation that
    is not there.
    """

    mode: SandboxMode = SandboxMode.LOCAL
    image: str | None = None
    docker: str | None = None
    memory_mb: int | None = None
    cpus: float | None = None
    pids_limit: int | None = None

    def _docker_fields(self) -> dict[str, Any]:
        names = ("image", "docker", "memory_mb", "cpus", "pids_limit")
        return {n: getattr(self, n) for n in names if getattr(self, n) is not None}

    @model_validator(mode="after")
    def _check(self) -> SandboxConfig:
        fields = self._docker_fields()
        if self.mode is SandboxMode.LOCAL and fields:
            raise ValueError(f'docker settings ({", ".join(sorted(fields))}) need mode = "docker"')
        if self.mode is SandboxMode.DOCKER:
            _docker_sandbox_cls().model_validate(fields)  # validate now, not at the first command
        return self

    def docker_sandbox(self) -> DockerSandbox | None:
        """The Docker sandbox to use, or ``None`` for ``mode = "local"``."""
        if self.mode is SandboxMode.LOCAL:
            return None
        return _docker_sandbox_cls().model_validate(self._docker_fields())

    def describe(self) -> str:
        """One human-readable line, e.g. for ``automa status``."""
        sb = self.docker_sandbox()
        if sb is None:
            return "local (run_command runs on this machine, not isolated)"
        return (
            f"docker (image {sb.image}, memory {sb.memory_mb}m, cpus {sb.cpus:g}, "
            f"pids {sb.pids_limit}, no network)"
        )


# Upper bound of concurrently driven tasks (each holds model calls in flight).
MAX_PARALLEL = 8


MAX_PROVIDER_RETRIES = 6
MAX_RETRY_DELAY_S = 120.0


class ProviderConfig(_Strict):
    """``[provider]`` in automa.toml: retrying transient gateway failures (D91).

    Only failures the provider marks retryable (timeouts, network errors, 408/409/
    425/429/5xx) are retried, each retry is a new call authorized by the BudgetGuard,
    and ``max_retries = 0`` turns retrying off. ``retry_delays_s[i]`` is the wait
    before retry ``i + 1`` (the last value repeats); a gateway ``Retry-After`` hint
    is honoured up to ``MAX_RETRY_DELAY_S``.
    """

    max_retries: int = Field(default=3, ge=0, le=MAX_PROVIDER_RETRIES)
    retry_delays_s: tuple[float, ...] = Field(
        default=(3.0, 10.0, 30.0), min_length=1, max_length=10
    )
    timeout_s: float = Field(default=180.0, gt=0, le=900)

    @field_validator("retry_delays_s")
    @classmethod
    def _delays(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if any(not (0 <= d <= MAX_RETRY_DELAY_S) for d in v):
            raise ValueError(f"retry delays must be between 0 and {MAX_RETRY_DELAY_S:g} seconds")
        return v

    def delay(self, retry: int, hint: float | None = None) -> float:
        """Seconds to wait before retry number ``retry`` (1-based)."""
        base = self.retry_delays_s[min(retry, len(self.retry_delays_s)) - 1]
        if hint is not None and hint > base:
            return min(hint, MAX_RETRY_DELAY_S)
        return base


class WebConfig(_Strict):
    """``[web]`` in automa.toml (D93).

    ``auto_approve_reads``: run ``fetch_url`` without asking when the URL cannot
    plausibly carry project data out (https, no query string, short plain path).
    Off by default; anything that does not qualify still needs approval.
    ``max_auto_reads_per_task``: automatic reads allowed per task attempt.
    """

    auto_approve_reads: bool = False
    max_auto_reads_per_task: int = Field(default=20, ge=1, le=100)


class OrchestratorConfig(_Strict):
    """How ``automa run`` schedules tasks (``[orchestrator]`` in automa.toml; D36).

    Spending is bounded by the BudgetGuard whatever this says; with a tight cap,
    parallel runs effectively become sequential.
    """

    # Ready tasks driven at the same time; 1 keeps the sequential loop (default).
    max_parallel: int = Field(default=1, ge=1, le=MAX_PARALLEL)

    def describe(self) -> str:
        """One-line description for ``automa run`` / ``automa status``."""
        return describe_parallel(self.max_parallel)


def describe_parallel(max_parallel: int) -> str:
    """Human description of a ``max_parallel`` value."""
    if max_parallel == 1:
        return "1 (sequential)"
    return (
        f"up to {max_parallel} tasks at a time (they share files/; a file being "
        "written by one running task cannot be written by another; "
        "run_command writes are not checked)"
    )


# Same as automa.plan.models.ROLE_PATTERN (not imported: automa.plan imports this module).
_ROLE_KEY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class RoleSettings(_Strict):
    """``[roles.<name>]`` in automa.toml: skills added to a built-in role (D47).

    Only ``skills`` is accepted: the file cannot change a role's tools, risk level,
    tier, steps or instructions (``extra="forbid"``). Skill names, the per-role cap
    and the existence of each skill are checked when the roles are loaded.
    """

    skills: tuple[str, ...] = ()

    @field_validator("skills", mode="before")
    @classmethod
    def _skill_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            raise ValueError("skills must be a list of names, not a string")
        return v


class MemoryConfig(_Strict):
    """``[memory]`` in automa.toml (D51).

    ``in_prompts``: whether ``automa new``/``automa run`` add the relevant memory
    (user preferences, the project's own memory, lessons) to the Executive's and
    Worker's system prompts. Default on; ``--no-memory`` turns it off for one command.
    Memory is data only either way: it never grants a tool, risk level or budget.
    """

    in_prompts: StrictBool = True


class RouterConfig(_Strict):
    """``[router]`` in automa.toml: the data router (D58).

    ``enabled``: whether ``automa run`` lets the router pick a task's starting tier
    from the quality log (off by default). A tier *qualifies* for a role when its
    model has at least ``min_attempts`` judged attempts for that role with a pass
    rate of at least ``min_pass_rate``. The router only changes which configured
    tier (and so which priced model) a task starts on: every call still goes
    through the gateway and the BudgetGuard, and escalation works as before.
    """

    enabled: StrictBool = False
    min_attempts: StrictInt = Field(default=5, ge=1, le=100_000)
    # A TOML integer (``1``) is accepted; booleans and strings are not.
    min_pass_rate: StrictFloat | StrictInt = Field(default=0.8, gt=0, le=1)

    @field_validator("min_pass_rate", mode="before")
    @classmethod
    def _no_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("min_pass_rate must be a number, not a boolean")
        return v


class AutomaConfig(_Strict):
    """Fully resolved, validated configuration."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: SecretStr | None = None
    tiers: dict[Tier, str] = Field(default_factory=dict)
    prices: dict[str, ModelPrice] = Field(default_factory=dict)
    models: dict[str, ModelInfo] = Field(default_factory=dict)
    limits: BudgetLimits = Field(default_factory=BudgetLimits)
    workspace: Path = Path("workspace")
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    roles: dict[str, RoleSettings] = Field(default_factory=dict)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)

    @field_validator("roles")
    @classmethod
    def _role_keys(cls, v: dict[str, RoleSettings]) -> dict[str, RoleSettings]:
        bad = sorted(repr(k) for k in v if not _ROLE_KEY_RE.fullmatch(k))
        if bad:
            raise ValueError(f"invalid role names: {', '.join(bad)}")
        return v

    @field_validator("models")
    @classmethod
    def _model_ids(cls, v: dict[str, ModelInfo]) -> dict[str, ModelInfo]:
        if len(v) > MAX_CATALOG_MODELS:
            raise ValueError(f"at most {MAX_CATALOG_MODELS} models in the catalog")
        bad = sorted(repr(k) for k in v if not _MODEL_ID_RE.fullmatch(k))
        if bad:
            raise ValueError(f"invalid model ids: {', '.join(bad)}")
        return v

    def role_skills(self) -> dict[str, tuple[str, ...]]:
        """Extra skills per role from ``[roles.<name>]`` (roles with none are left out)."""
        return {name: rs.skills for name, rs in sorted(self.roles.items()) if rs.skills}

    @model_validator(mode="after")
    def _check(self) -> AutomaConfig:
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("base_url must start with http:// or https://")
        twice = sorted(set(self.prices) & set(self.models))
        if twice:
            # Two prices for one model would make the effective cap ambiguous.
            raise ValueError(
                f"model(s) priced both in [prices] and [models]: {', '.join(twice)} "
                "(keep the price in one place)"
            )
        missing = sorted(
            {m for m in self.tiers.values() if m not in self.prices and m not in self.models}
        )
        if missing:
            # Without a price the budget layer cannot bound spending.
            raise ValueError(f"no price configured for tier model(s): {', '.join(missing)}")
        textless = sorted(
            {
                m
                for m in self.tiers.values()
                if m in self.models and not self.models[m].has(Capability.TEXT)
            }
        )
        if textless:
            # Roles talk to their tier's model; a model without text output cannot serve one.
            raise ValueError(f"tier model(s) without the 'text' capability: {', '.join(textless)}")
        return self

    def model_for(self, tier: Tier) -> str:
        """Return the model name mapped to ``tier``."""
        try:
            return self.tiers[tier]
        except KeyError:
            raise ConfigError(f"no model configured for tier '{tier.value}'") from None

    def price_for(self, model: str) -> ModelPrice:
        """Return the configured price for ``model`` (from ``[prices]`` or the catalog)."""
        if model in self.prices:
            return self.prices[model]
        if model in self.models:
            return self.models[model].price
        raise ConfigError(f"no price configured for model '{model}'")

    def model_info(self, model: str) -> ModelInfo | None:
        """The catalog entry for ``model``, or ``None`` if it is not catalogued."""
        return self.models.get(model)

    def family_of(self, model: str) -> str | None:
        """The catalogued family of ``model``, or ``None`` if it is not catalogued."""
        info = self.models.get(model)
        return None if info is None else info.family

    def catalog(
        self, *, capability: Capability | None = None, family: str | None = None
    ) -> list[tuple[str, ModelInfo]]:
        """Catalogued models (sorted by id), optionally filtered by capability/family."""
        return [
            (mid, info)
            for mid, info in sorted(self.models.items())
            if (capability is None or info.has(capability))
            and (family is None or info.family == family)
        ]

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
