"""``Router``: the data router — the cheapest tier that is good enough (D58).

For a task type (its role) and the tier it asks for, the router looks at the
quality log statistics of the model behind every *configured* tier and picks the
starting tier:

* a tier **qualifies** when its model has at least ``min_attempts`` judged
  attempts for that role and a pass rate of at least ``min_pass_rate``;
* a tier is **proven weak** when it has at least ``min_attempts`` attempts and a
  pass rate *below* ``min_pass_rate``;
* if the requested tier is not proven weak, the router may only move to a
  qualified tier that is **not more expensive** (by configured price) than the
  requested one — it never spends more on a hunch. The requested tier wins ties;
* if the requested tier is proven weak, the cheapest qualified tier is chosen,
  even a more expensive one (the data says the requested one usually fails);
* otherwise the requested tier is kept.

The router is pure: it reads statistics and the configuration, calls no model,
writes nothing and never authorizes anything. Spending stays bounded by the
gateway and the BudgetGuard whatever tier is chosen, and the orchestrator still
escalates after rejections.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from automa.config import AutomaConfig, ConfigError, RouterConfig, Tier
from automa.plan.models import ROLE_PATTERN
from automa.quality import QualityLog, QualityStats

_ROLE_RE = re.compile(ROLE_PATTERN)
_TIER_INDEX = {t: i for i, t in enumerate(Tier)}


class RouteReason(StrEnum):
    """Why the router chose the tier it chose."""

    REQUESTED = "requested"  # kept the requested tier
    CHEAPER = "cheaper"  # a qualified tier no more expensive than the requested one
    REPLACED = "replaced"  # the requested tier is proven weak; cheapest qualified tier


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TierEvidence(_Frozen):
    """What the quality log says about one configured tier for one role."""

    tier: Tier
    model: str
    price_per_mtok: float = Field(ge=0)  # input + output price, USD per 1M tokens
    attempts: int = Field(ge=0)
    passed: int = Field(ge=0)
    qualified: bool
    proven_weak: bool

    @property
    def pass_rate(self) -> float | None:
        return self.passed / self.attempts if self.attempts else None


class RouteDecision(_Frozen):
    """The starting tier for one task type, with the evidence behind it."""

    role: str
    requested: Tier
    tier: Tier
    model: str
    reason: RouteReason
    evidence: list[TierEvidence]  # every configured tier, lowest first

    @property
    def changed(self) -> bool:
        return self.tier is not self.requested


class Router:
    """Picks a task's starting tier from per-(role, model) quality statistics."""

    def __init__(
        self,
        config: AutomaConfig,
        stats: Iterable[QualityStats],
        settings: RouterConfig | None = None,
    ):
        self._config = config
        self._settings = settings if settings is not None else config.router
        self._stats: dict[tuple[str, str], QualityStats] = {}
        for s in stats:
            key = (s.role, s.model)
            if key in self._stats:
                raise ValueError(f"duplicate statistics for role {s.role!r} model {s.model!r}")
            self._stats[key] = s

    @classmethod
    def from_log(
        cls, config: AutomaConfig, log: QualityLog, settings: RouterConfig | None = None
    ) -> Router:
        """A router over the current statistics of ``log`` (raises ``QualityError``)."""
        return cls(config, log.stats(), settings)

    @property
    def settings(self) -> RouterConfig:
        return self._settings

    def evidence(self, role: str) -> list[TierEvidence]:
        """Evidence for every configured tier (lowest tier first)."""
        if not _ROLE_RE.fullmatch(role):
            raise ValueError(f"invalid role name {role!r}")
        out: list[TierEvidence] = []
        for tier in sorted(self._config.tiers, key=_TIER_INDEX.__getitem__):
            model = self._config.model_for(tier)
            price = self._config.price_for(model)
            st = self._stats.get((role, model))
            attempts = st.attempts if st is not None else 0
            passed = st.passed if st is not None else 0
            enough = attempts >= self._settings.min_attempts
            # Compare passed/attempts >= rate without float division error at the edge.
            good = enough and passed >= self._settings.min_pass_rate * attempts - 1e-9
            out.append(
                TierEvidence(
                    tier=tier,
                    model=model,
                    price_per_mtok=price.input_per_mtok + price.output_per_mtok,
                    attempts=attempts,
                    passed=passed,
                    qualified=good,
                    proven_weak=enough and not good,
                )
            )
        return out

    def route(self, role: str, requested: Tier) -> RouteDecision:
        """The starting tier for a ``role`` task that asks for ``requested``.

        Raises ``ConfigError`` if ``requested`` has no configured model (the same
        error the gateway would raise), ``ValueError`` for an invalid role name.
        """
        evidence = self.evidence(role)
        by_tier = {e.tier: e for e in evidence}
        if requested not in by_tier:
            raise ConfigError(f"no model configured for tier '{requested.value}'")
        req = by_tier[requested]
        qualified = [e for e in evidence if e.qualified]
        if not req.proven_weak:
            qualified = [e for e in qualified if e.price_per_mtok <= req.price_per_mtok]

        def key(e: TierEvidence) -> tuple[float, int, int]:
            return (e.price_per_mtok, 0 if e.tier is requested else 1, _TIER_INDEX[e.tier])

        if not qualified:
            chosen, reason = req, RouteReason.REQUESTED
        else:
            chosen = min(qualified, key=key)
            if chosen.tier is requested:
                reason = RouteReason.REQUESTED
            elif req.proven_weak:
                reason = RouteReason.REPLACED
            else:
                reason = RouteReason.CHEAPER
        return RouteDecision(
            role=role,
            requested=requested,
            tier=chosen.tier,
            model=chosen.model,
            reason=reason,
            evidence=evidence,
        )
