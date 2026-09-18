"""Assess fairness feasibility and operational robustness for the selected Module 4 model.

This analysis deliberately does NOT manufacture protected demographic groups.

Fairness scope:
- confirm that the selected modelling data contain no direct protected-group fields;
- document that group-fairness metrics are not supportable from the available data;
- keep proxy concerns as an interpretation/governance limitation rather than infer protected
  traits from unrelated variables.

Robustness scope:
- evaluate the selected tuned Random Forest + isotonic calibration across defensible
  operational segments based on observed account tenure and recharge activity;
- run controlled feature-sensitivity checks on the most influential SHAP features.

Operational segments are NOT treated as fairness groups. They are used only to test whether
model performance is excessively dependent on a narrow part of the observed population.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"

TABLE_DIR = PROJECT_ROOT / "reports/tables"
FIGURE_DIR = PROJECT_ROOT / "reports/figures/module4"

SEGMENT_TABLE = TABLE_DIR / "module4_operational_robustness_segments.csv"
SENSITIVITY_TABLE = TABLE_DIR / "module4_feature_sensitivity.csv"
SUMMARY_JSON = TABLE_DIR / "module4_fairness_robustness_summary.json"

SEGMENT_FIGURE = FIGURE_DIR / "operational_segment_roc_auc.png"
SENSITIVITY_FIGURE = FIGURE_DIR / "feature_sensitivity_p95.png"

TARGET = "delinquent_5d"
DATE = "pdate"

TRAIN_END = pd.Timestamp("2016-07-06")
CALIBRATION_START = pd.Timestamp("2016-07-07")
CALIBRATION_END = pd.Timestamp("2016-07-13")
HOLDOUT_START = pd.Timestamp("2016-07-14")
HOLDOUT_END = pd.Timestamp("2016-07-23")

RANDOM_STATE = 42
SENSITIVITY_SAMPLE = 2_000

FEATURES = [
    "cnt_ma_rech90",
    "daily_decr30",
    "last_rech_date_ma",
    "sumamnt_ma_rech90",
    "aon",
    "last_rech_amt_ma",
    "daily_decr90",
    "sumamnt_ma_rech30",
    "medianamnt_ma_rech30",
    "medianmarechprebal90",
    "rental30",
    "cnt_ma_rech30",
]

TOP_SENSITIVITY_FEATURES = [
    "cnt_ma_rech90",
    "sumamnt_ma_rech90",
    "sumamnt_ma_rech30",
    "cnt_ma_rech30",
    "daily_decr30",
]

DIRECT_PROTECTED_TERMS = {
    "age",
    "gender",
    "sex",
    "race",
    "ethnicity",
    "religion",
    "disability",
    "nationality",
    "marital",
    "pregnancy",
}


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if not mask.any():
            continue
        ece += (mask.sum() / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def top20_capture(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    n = max(1, int(np.ceil(len(y) * 0.20)))
    order = np.argsort(-p)
    return float(y[order[:n]].sum() / positives)


def metric_row(group_type: str, group_value: str, y: pd.Series, p: np.ndarray) -> dict:
    if y.nunique() < 2:
        roc_auc = np.nan
        ap = np.nan
    else:
        roc_auc = float(roc_auc_score(y, p))
        ap = float(average_precision_score(y, p))

    return {
        "group_type": group_type,
        "group_value": group_value,
        "rows": int(len(y)),
        "delinquency_rate": float(y.mean()),
        "mean_predicted_risk": float(np.mean(p)),
        "roc_auc": roc_auc,
        "average_precision": ap,
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": expected_calibration_error(y.to_numpy(), p),
        "top20_capture": top20_capture(y.to_numpy(), p),
    }


def direct_protected_name_hits(columns: list[str]) -> dict[str, list[str]]:
    hits = {}
    for column in columns:
        lowered = column.lower()
        matched = sorted(term for term in DIRECT_PROTECTED_TERMS if term in lowered)
        if matched:
            hits[column] = matched
    return hits


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df[DATE] = pd.to_datetime(df[DATE], errors="raise")

    protected_hits = direct_protected_name_hits(list(df.columns))

    train = df.loc[df[DATE] <= TRAIN_END].copy()
    calibration = df.loc[
        (df[DATE] >= CALIBRATION_START) & (df[DATE] <= CALIBRATION_END)
    ].copy()
    holdout = df.loc[
        (df[DATE] >= HOLDOUT_START) & (df[DATE] <= HOLDOUT_END)
    ].copy()

    imputer = SimpleImputer(strategy="median")
    x_train = pd.DataFrame(
        imputer.fit_transform(train[FEATURES]), columns=FEATURES, index=train.index
    )
    x_cal = pd.DataFrame(
        imputer.transform(calibration[FEATURES]), columns=FEATURES, index=calibration.index
    )
    x_holdout = pd.DataFrame(
        imputer.transform(holdout[FEATURES]), columns=FEATURES, index=holdout.index
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, train[TARGET])

    raw_cal = model.predict_proba(x_cal)[:, 1]
    raw_holdout = model.predict_proba(x_holdout)[:, 1]

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_cal, calibration[TARGET].to_numpy())
    p_holdout = np.asarray(isotonic.predict(raw_holdout), dtype=float)

    # Operational robustness segments.
    holdout_eval = holdout.copy()
    holdout_eval["score"] = p_holdout

    # qcut may merge duplicate boundaries; duplicates='drop' keeps segmentation valid.
    holdout_eval["tenure_band"] = pd.qcut(
        holdout_eval["aon"], q=4, duplicates="drop"
    ).astype(str)
    holdout_eval["recharge90_band"] = pd.qcut(
        holdout_eval["cnt_ma_rech90"], q=4, duplicates="drop"
    ).astype(str)

    segment_rows = [
        metric_row(
            "overall",
            "all_holdout",
            holdout_eval[TARGET],
            holdout_eval["score"].to_numpy(),
        )
    ]

    for group_type, col in [
        ("account_tenure_quartile", "tenure_band"),
        ("recharge90_quartile", "recharge90_band"),
    ]:
        for group_value, group in holdout_eval.groupby(col, observed=True):
            segment_rows.append(
                metric_row(
                    group_type,
                    str(group_value),
                    group[TARGET],
                    group["score"].to_numpy(),
                )
            )

    segments = pd.DataFrame(segment_rows)

    # Controlled feature sensitivity: +/- 0.10 IQR, clipped to training min/max.
    rng = np.random.default_rng(RANDOM_STATE)
    if len(x_holdout) > SENSITIVITY_SAMPLE:
        positions = np.sort(
            rng.choice(len(x_holdout), size=SENSITIVITY_SAMPLE, replace=False)
        )
        x_sample = x_holdout.iloc[positions].copy()
    else:
        x_sample = x_holdout.copy()

    base_raw = model.predict_proba(x_sample)[:, 1]
    base_cal = np.asarray(isotonic.predict(base_raw), dtype=float)

    sensitivity_rows = []

    for feature in TOP_SENSITIVITY_FEATURES:
        train_feature = x_train[feature]
        q1 = float(train_feature.quantile(0.25))
        q3 = float(train_feature.quantile(0.75))
        iqr = q3 - q1
        delta = 0.10 * iqr

        train_min = float(train_feature.min())
        train_max = float(train_feature.max())

        for direction, signed_delta in [("decrease", -delta), ("increase", delta)]:
            x_perturbed = x_sample.copy()
            x_perturbed[feature] = np.clip(
                x_perturbed[feature] + signed_delta,
                train_min,
                train_max,
            )

            perturbed_raw = model.predict_proba(x_perturbed)[:, 1]
            perturbed_cal = np.asarray(isotonic.predict(perturbed_raw), dtype=float)
            change = perturbed_cal - base_cal
            abs_change = np.abs(change)

            sensitivity_rows.append(
                {
                    "feature": feature,
                    "direction": direction,
                    "delta_in_feature_units": float(signed_delta),
                    "sample_rows": int(len(x_sample)),
                    "mean_probability_change": float(change.mean()),
                    "median_probability_change": float(np.median(change)),
                    "median_abs_probability_change": float(np.median(abs_change)),
                    "p95_abs_probability_change": float(np.quantile(abs_change, 0.95)),
                    "max_abs_probability_change": float(abs_change.max()),
                    "share_abs_change_ge_5pp": float(np.mean(abs_change >= 0.05)),
                    "share_abs_change_ge_10pp": float(np.mean(abs_change >= 0.10)),
                }
            )

    sensitivity = pd.DataFrame(sensitivity_rows)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    segments.to_csv(SEGMENT_TABLE, index=False)
    sensitivity.to_csv(SENSITIVITY_TABLE, index=False)

    # Visual 1: subgroup ROC-AUC, excluding overall row.
    plot_segments = segments.loc[
        (segments["group_type"] != "overall") & segments["roc_auc"].notna()
    ].copy()
    plot_segments["label"] = (
        plot_segments["group_type"].str.replace("_", " ", regex=False)
        + "\n"
        + plot_segments["group_value"]
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(plot_segments["label"], plot_segments["roc_auc"])
    ax.axhline(
        segments.loc[segments["group_type"] == "overall", "roc_auc"].iloc[0],
        linestyle="--",
        linewidth=1.5,
        label="Overall holdout ROC-AUC",
    )
    ax.set_title("Operational robustness across observed holdout segments")
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SEGMENT_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Visual 2: p95 absolute calibrated-risk movement under controlled perturbation.
    sens_plot = sensitivity.copy()
    sens_plot["label"] = sens_plot["feature"] + " / " + sens_plot["direction"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sens_plot["label"], sens_plot["p95_abs_probability_change"] * 100)
    ax.set_title("Selected-model sensitivity to modest feature perturbations")
    ax.set_xlabel("95th percentile absolute change in calibrated risk (percentage points)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(SENSITIVITY_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    subgroup_roc = plot_segments["roc_auc"].dropna()
    max_roc_spread = (
        float(subgroup_roc.max() - subgroup_roc.min()) if len(subgroup_roc) else None
    )

    max_p95_sensitivity = float(sensitivity["p95_abs_probability_change"].max())
    max_share_10pp = float(sensitivity["share_abs_change_ge_10pp"].max())

    summary = {
        "fairness_scope": {
            "direct_protected_fields_available": bool(protected_hits),
            "direct_protected_name_hits": protected_hits,
            "group_fairness_metrics_supported": False,
            "reason": (
                "The modelling dataset does not contain attributes that directly identify "
                "protected demographic groups or other clearly defensible fairness groups. "
                "No protected groups are inferred or fabricated."
            ),
            "proxy_note": (
                "Potential indirect proxy effects cannot be ruled out solely from column "
                "names or model explainability and remain a governance limitation."
            ),
        },
        "robustness_scope": {
            "operational_segments_are_fairness_groups": False,
            "segment_types": [
                "account_tenure_quartile",
                "recharge90_quartile",
            ],
            "overall_holdout_roc_auc": float(
                segments.loc[segments["group_type"] == "overall", "roc_auc"].iloc[0]
            ),
            "max_operational_segment_roc_auc_spread": max_roc_spread,
            "sensitivity_features": TOP_SENSITIVITY_FEATURES,
            "sensitivity_perturbation": "plus/minus 0.10 training IQR, clipped to training range",
            "max_p95_abs_probability_change": max_p95_sensitivity,
            "max_share_abs_change_ge_10pp": max_share_10pp,
        },
        "interpretation_caution": (
            "Operational subgroup stability is not evidence of demographic fairness. "
            "Sensitivity analysis tests local model stability under modest input changes, "
            "not causal responses or policy appropriateness."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Module 4 fairness-feasibility and robustness assessment completed.")
    print()
    print("Fairness feasibility")
    print("Direct protected fields available:", bool(protected_hits))
    print("Group fairness metrics supported: False")
    print()
    print("Operational robustness segments")
    print(
        segments[
            [
                "group_type",
                "group_value",
                "rows",
                "delinquency_rate",
                "roc_auc",
                "brier",
                "top20_capture",
            ]
        ].to_string(index=False)
    )
    print()
    print("Feature sensitivity")
    print(
        sensitivity[
            [
                "feature",
                "direction",
                "delta_in_feature_units",
                "median_abs_probability_change",
                "p95_abs_probability_change",
                "share_abs_change_ge_5pp",
                "share_abs_change_ge_10pp",
            ]
        ].to_string(index=False)
    )
    print()
    print("Max operational-segment ROC-AUC spread:", max_roc_spread)
    print("Max p95 absolute probability change:", max_p95_sensitivity)
    print("Max share with >=10pp change:", max_share_10pp)
    print()
    print(f"Segment table written to: {SEGMENT_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Sensitivity table written to: {SENSITIVITY_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Summary written to: {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Segment figure written to: {SEGMENT_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Sensitivity figure written to: {SENSITIVITY_FIGURE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
