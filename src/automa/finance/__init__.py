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
    BudgetFit,
    EstimateBasis,
    EstimateSettings,
    PlanEstimate,
    TaskEstimate,
    TaskProgress,
    estimate_plan,
)
from automa.finance.report import SpendGroup, SpendingReport, spending_report

__all__ = [
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
