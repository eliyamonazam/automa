"""Financial Controller: advisory cost analysis on top of the budget layer (D62).

Nothing here calls a model, writes anything or authorizes spending: the
``BudgetGuard`` remains the only authority over money.
"""

from automa.finance.anomaly import (
    ANOMALY_EVENT,
    Anomaly,
    AnomalyReport,
    AnomalySettings,
    CostSample,
    SampleKind,
    attempt_samples,
    call_samples,
    detect_anomalies,
)
from automa.finance.compare import (
    ComparisonOutcome,
    EstimateComparison,
    EstimateRecord,
    compare_estimate,
)
from automa.finance.estimate import (
    DEFAULT_TOKEN_PROFILES,
    BudgetFit,
    EstimateBasis,
    EstimateSettings,
    PlanEstimate,
    RangeBasis,
    RoleTokenProfile,
    TaskEstimate,
    TaskProgress,
    TokenAllowance,
    attempt_costs,
    estimate_plan,
    percentile_nusd,
    profile_for,
    tokens_nusd,
)
from automa.finance.report import SpendGroup, SpendingReport, spending_report

__all__ = [
    "DEFAULT_TOKEN_PROFILES",
    "RangeBasis",
    "RoleTokenProfile",
    "TokenAllowance",
    "attempt_costs",
    "percentile_nusd",
    "profile_for",
    "tokens_nusd",
    "ANOMALY_EVENT",
    "Anomaly",
    "AnomalyReport",
    "AnomalySettings",
    "CostSample",
    "SampleKind",
    "attempt_samples",
    "call_samples",
    "detect_anomalies",
    "BudgetFit",
    "ComparisonOutcome",
    "EstimateComparison",
    "EstimateRecord",
    "EstimateBasis",
    "EstimateSettings",
    "PlanEstimate",
    "SpendGroup",
    "SpendingReport",
    "TaskEstimate",
    "TaskProgress",
    "compare_estimate",
    "estimate_plan",
    "spending_report",
]
