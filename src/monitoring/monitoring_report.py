"""Build a transparent monitoring status object from drift and performance evidence.

The report distinguishes labelled evaluation periods from the post-23-July 2016
label-free diagnostic period. It never calculates outcome-based metrics when labels are
not trusted.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from src.monitoring.drift_metrics import DriftThresholds, numeric_feature_drift, prediction_drift
from src.monitoring.performance_monitoring import evaluate_labelled_period


def overall_drift_status(feature_drift: pd.DataFrame, score_drift: dict) -> str:
    severities = set(feature_drift["severity"].astype(str))
    severities.add(str(score_drift["severity"]))
    if "escalate" in severities:
        return "escalate"
    if "watch" in severities:
        return "watch"
    return "ok"


def build_monitoring_status(
    *,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    reference_scores: pd.Series,
    current_scores: pd.Series,
    monitoring_period: str,
    labels_reliable: bool,
    threshold: float,
    current_labels: pd.Series | None = None,
    thresholds: DriftThresholds | None = None,
) -> dict:
    thresholds = thresholds or DriftThresholds()
    feature_drift = numeric_feature_drift(reference, current, features, thresholds)
    score_drift = prediction_drift(reference_scores, current_scores, thresholds)

    result = {
        "monitoring_period": monitoring_period,
        "monitoring_mode": "labelled" if labels_reliable else "label_free",
        "outcome_labels_treated_as_ground_truth": bool(labels_reliable),
        "explanation": (
            "Outcome-based performance monitoring is enabled because reliable labels are available."
            if labels_reliable
            else "Only label-free drift monitoring is performed. The post-23-July 2016 "
                 "all-successful outcome regime is unexplained and is not treated as ground truth."
        ),
        "thresholds": asdict(thresholds),
        "overall_drift_status": overall_drift_status(feature_drift, score_drift),
        "feature_drift": feature_drift.to_dict(orient="records"),
        "prediction_drift": score_drift,
        "performance": None,
    }

    if labels_reliable:
        if current_labels is None:
            raise ValueError("Reliable labels were declared available but current_labels is None.")
        result["performance"] = evaluate_labelled_period(
            current_labels,
            current_scores,
            threshold,
            period_name=monitoring_period,
            labels_reliable=True,
        )
    return result


def write_monitoring_status(status: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
