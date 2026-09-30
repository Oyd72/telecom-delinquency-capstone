"""Outcome-based monitoring metrics.

Performance metrics are only valid when reliable outcome labels exist. The post-23-July
2016 diagnostic block is explicitly excluded from outcome-based evaluation because its
all-successful label regime is unexplained and is not treated as ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

POST_23_JULY_DIAGNOSTIC = "post_23_july_2016_diagnostic"


def expected_calibration_error(y_true, probabilities, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    if total == 0:
        return float("nan")
    result = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.any():
            result += (mask.sum() / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(result)


def top20_capture(y_true, probabilities) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    n = max(1, int(np.ceil(len(y) * 0.20)))
    order = np.argsort(-p)
    return float(y[order[:n]].sum() / positives)


def assert_labels_permitted(period_name: str, labels_reliable: bool) -> None:
    if period_name == POST_23_JULY_DIAGNOSTIC or not labels_reliable:
        raise ValueError(
            "Outcome-based monitoring is not permitted for this period. "
            "Post-23-July 2016 outcomes are not treated as reliable ground truth; "
            "use label-free drift monitoring only."
        )


def evaluate_labelled_period(
    y_true: pd.Series,
    probabilities: pd.Series,
    threshold: float,
    *,
    period_name: str,
    labels_reliable: bool,
) -> dict:
    """Calculate performance only for periods with reliable matured labels."""
    assert_labels_permitted(period_name, labels_reliable)

    y = pd.Series(y_true).astype(int).to_numpy()
    p = pd.Series(probabilities).astype(float).to_numpy()
    pred = (p >= threshold).astype(int)

    return {
        "period_name": period_name,
        "rows": int(len(y)),
        "observed_delinquency_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size > 1 else float("nan"),
        "average_precision": (
            float(average_precision_score(y, p)) if np.unique(y).size > 1 else float("nan")
        ),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10bin": expected_calibration_error(y, p),
        "top20_capture": top20_capture(y, p),
        "operating_threshold": float(threshold),
        "labels_reliable": True,
    }
