"""Permission Policy: which tools a role may use, and up to which risk level.

Rules (enforced in code, never in a prompt):

1. A tool not in ``allowed_tools`` is denied.
2. A tool whose risk is above ``max_risk`` is denied.
3. ``external`` tools are never allowed directly: a listed ``external`` tool
   always needs the Approval Gate (``needs_approval``), whatever ``max_risk`` is;
   an unlisted one is denied (an approval can never widen a role's tool list).
4. An ``irreversible`` tool that passes rules 1-2 needs the Approval Gate too,
   at any ``max_risk``; the flag never makes a tool *more* available (an
   approval cannot lift a risk-level denial).

The policy is immutable; there is no way to widen it after construction.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator

from automa.tools.base import TOOL_NAME_RE, RiskLevel


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    NEEDS_APPROVAL = "needs_approval"


class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Decision
    reason: str

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW


class PermissionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_tools: frozenset[str]
    max_risk: RiskLevel = RiskLevel.READ

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def _names(cls, v: Iterable[str]) -> frozenset[str]:
        if isinstance(v, str):
            raise ValueError("allowed_tools must be a collection of names, not a string")
        names = frozenset(v)
        bad = sorted(n for n in names if not isinstance(n, str) or not TOOL_NAME_RE.fullmatch(n))
        if bad:
            raise ValueError(f"invalid tool names: {bad}")
        return names

    @field_validator("max_risk")
    @classmethod
    def _not_external(cls, v: RiskLevel) -> RiskLevel:
        if v is RiskLevel.EXTERNAL:
            raise ValueError("max_risk cannot be 'external': external tools need approval")
        return v

    def decide(
        self, tool_name: str, risk: RiskLevel, *, irreversible: bool = False
    ) -> PolicyDecision:
        if not isinstance(risk, RiskLevel):
            return PolicyDecision(decision=Decision.DENY, reason="unknown risk level")
        if type(irreversible) is not bool:
            return PolicyDecision(decision=Decision.DENY, reason="invalid irreversible flag")
        if tool_name not in self.allowed_tools:
            return PolicyDecision(
                decision=Decision.DENY, reason=f"tool {tool_name!r} is not allowed for this role"
            )
        if risk is RiskLevel.EXTERNAL:
            return PolicyDecision(
                decision=Decision.NEEDS_APPROVAL,
                reason=f"{tool_name!r} is an external action and needs user approval",
            )
        if risk.rank > self.max_risk.rank:
            return PolicyDecision(
                decision=Decision.DENY,
                reason=f"tool {tool_name!r} has risk {risk.value!r}, "
                f"above this role's limit {self.max_risk.value!r}",
            )
        if irreversible:
            return PolicyDecision(
                decision=Decision.NEEDS_APPROVAL,
                reason=f"{tool_name!r} is irreversible and needs user approval",
            )
        return PolicyDecision(decision=Decision.ALLOW, reason="allowed")
