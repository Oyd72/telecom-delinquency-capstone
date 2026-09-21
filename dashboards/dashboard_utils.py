"""Helpers for the Module 5 Streamlit stakeholder dashboard.

This module only reads committed Module 4 evidence and the frozen packaged model.
It does not retrain, recalibrate, or replace the Module 4 deliverables.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "selected_random_forest_isotonic.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "selected_model_metadata.json"
CLASSIFICATION_METRICS_PATH = PROJECT_ROOT / "reports" / "tables" / "module4_classification_metrics.csv"
SHAP_SUMMARY_PATH = PROJECT_ROOT / "reports" / "tables" / "module4_shap_summary.json"
COUNTERFACTUAL_SUMMARY_PATH = PROJECT_ROOT / "reports" / "tables" / "module4_counterfactual_summary.json"
FAIRNESS_SUMMARY_PATH = PROJECT_ROOT / "reports" / "tables" / "module4_fairness_robustness_summary.json"
ROBUSTNESS_SEGMENTS_PATH = PROJECT_ROOT / "reports" / "tables" / "module4_operational_robustness_segments.csv"
THRESHOLD_TRADEOFF_PATH = PROJECT_ROOT / "reports" / "tables" / "module5_threshold_tradeoff.csv"

FEATURE_LABELS = {
    "cnt_ma_rech90": "Recharge count — 90 days",
    "daily_decr30": "Daily decrement measure — 30 days",
    "last_rech_date_ma": "Days associated with last main-account recharge",
    "sumamnt_ma_rech90": "Total main-account recharge amount — 90 days",
    "aon": "Account / network tenure",
    "last_rech_amt_ma": "Most recent main-account recharge amount",
    "daily_decr90": "Daily decrement measure — 90 days",
    "sumamnt_ma_rech30": "Total main-account recharge amount — 30 days",
    "medianamnt_ma_rech30": "Median main-account recharge amount — 30 days",
    "medianmarechprebal90": "Median pre-recharge balance — 90 days",
    "rental30": "Rental / activity measure — 30 days",
    "cnt_ma_rech30": "Recharge count — 30 days",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_evidence() -> dict:
    metrics = pd.read_csv(CLASSIFICATION_METRICS_PATH)
    selected = metrics.loc[metrics["evaluation"] == "selected_random_forest_isotonic"].iloc[0]

    return {
        "metadata": read_json(MODEL_METADATA_PATH),
        "classification": selected.to_dict(),
        "shap": read_json(SHAP_SUMMARY_PATH),
        "counterfactual": read_json(COUNTERFACTUAL_SUMMARY_PATH),
        "fairness": read_json(FAIRNESS_SUMMARY_PATH),
        "robustness": pd.read_csv(ROBUSTNESS_SEGMENTS_PATH),
        "threshold_tradeoff": pd.read_csv(THRESHOLD_TRADEOFF_PATH),
    }


def load_model_package() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Frozen model artefact is not available at {MODEL_PATH}. "
            "The dashboard can still show committed Module 4 evidence, "
            "but live scoring requires this exact artefact."
        )
    return joblib.load(MODEL_PATH)


def case_feature_values(shap_summary: dict, case_name: str) -> dict[str, float]:
    case = next(
        item for item in shap_summary["representative_cases"] if item["case"] == case_name
    )
    return {
        contributor["feature"]: float(contributor["feature_value"])
        for contributor in case["top_contributors"]
    }


def local_median_sensitivity(df: pd.DataFrame, package: dict) -> pd.DataFrame:
    """Measure one-feature-at-a-time sensitivity against the fitted imputer median.

    This is a contrastive diagnostic, not a causal explanation and not a SHAP value.
    """
    from src.inference.predict_selected_model import predict_frame

    features = package["features"]
    baseline = float(
        predict_frame(df[features], package).iloc[0]["calibrated_delinquency_probability"]
    )
    medians = package["imputer"].statistics_
    rows = []

    for idx, feature in enumerate(features):
        original = df.iloc[0][feature]
        if pd.isna(original):
            continue

        changed = df[features].copy()
        changed.loc[changed.index[0], feature] = float(medians[idx])
        changed_score = float(
            predict_frame(changed, package).iloc[0]["calibrated_delinquency_probability"]
        )
        rows.append(
            {
                "feature": feature,
                "original_value": float(original),
                "reference_value": float(medians[idx]),
                "baseline_risk": baseline,
                "reference_risk": changed_score,
                "risk_change": baseline - changed_score,
                "abs_risk_change": abs(baseline - changed_score),
            }
        )

    columns = [
        "feature",
        "original_value",
        "reference_value",
        "baseline_risk",
        "reference_risk",
        "risk_change",
        "abs_risk_change",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "abs_risk_change", ascending=False
    )
