"""Incremental add-back analysis for the five predictors excluded from the 7-feature model.

Base specification:
- 7 recharge-behaviour features.

Candidate add-backs:
- aon
- last_rech_date_ma
- daily_decr30
- daily_decr90
- rental30

The experiment evaluates:
1. each predictor added back individually;
2. tenure/recency pair;
3. activity trio;
4. all five together (the original 12-feature model).

Random Forest hyperparameters, isotonic calibration, and chronological development folds
remain fixed. The final holdout is not used.

This is designed to identify whether the five extra predictors add value individually,
in small thematic groups, or mainly through interactions.
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"

TABLE_DIR = PROJECT_ROOT / "reports/tables"
FIGURE_DIR = PROJECT_ROOT / "reports/figures/module4"

FOLD_TABLE = TABLE_DIR / "module4_incremental_addback_fold_metrics.csv"
SUMMARY_TABLE = TABLE_DIR / "module4_incremental_addback_summary.csv"
SUMMARY_JSON = TABLE_DIR / "module4_incremental_addback_summary.json"

AP_FIGURE = FIGURE_DIR / "incremental_addback_average_precision.png"
Brier_FIGURE = FIGURE_DIR / "incremental_addback_brier.png"
ROC_FIGURE = FIGURE_DIR / "incremental_addback_roc_auc.png"

TARGET = "delinquent_5d"
DATE = "pdate"
CALIBRATION_DAYS = 7
RANDOM_STATE = 42

BASE_7 = [
    "cnt_ma_rech90",
    "sumamnt_ma_rech90",
    "last_rech_amt_ma",
    "sumamnt_ma_rech30",
    "medianamnt_ma_rech30",
    "medianmarechprebal90",
    "cnt_ma_rech30",
]

EXTRA_5 = [
    "aon",
    "last_rech_date_ma",
    "daily_decr30",
    "daily_decr90",
    "rental30",
]

VARIANTS = {
    "base_7": BASE_7,
    "base_plus_aon": BASE_7 + ["aon"],
    "base_plus_last_rech_date_ma": BASE_7 + ["last_rech_date_ma"],
    "base_plus_daily_decr30": BASE_7 + ["daily_decr30"],
    "base_plus_daily_decr90": BASE_7 + ["daily_decr90"],
    "base_plus_rental30": BASE_7 + ["rental30"],
    "base_plus_tenure_recency": BASE_7 + ["aon", "last_rech_date_ma"],
    "base_plus_activity_trio": BASE_7 + ["daily_decr30", "daily_decr90", "rental30"],
    "full_12": BASE_7 + EXTRA_5,
}

EVALUATION_FOLDS = [
    {"fold": "late_june", "eval_start": "2016-06-14", "eval_end": "2016-06-23"},
    {"fold": "turn_of_month", "eval_start": "2016-06-24", "eval_end": "2016-07-03"},
    {"fold": "early_july", "eval_start": "2016-07-04", "eval_end": "2016-07-13"},
]


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if not mask.any():
            continue
        ece += (mask.sum() / len(y)) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def top20_capture(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    n_top = max(1, int(np.ceil(len(y) * 0.20)))
    order = np.argsort(-p)
    return float(y[order[:n_top]].sum() / positives)


def fit_and_score(train, calibration, evaluation, features):
    imputer = SimpleImputer(strategy="median")
    x_train = pd.DataFrame(imputer.fit_transform(train[features]), columns=features, index=train.index)
    x_cal = pd.DataFrame(imputer.transform(calibration[features]), columns=features, index=calibration.index)
    x_eval = pd.DataFrame(imputer.transform(evaluation[features]), columns=features, index=evaluation.index)

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

    p_cal_raw = model.predict_proba(x_cal)[:, 1]
    p_eval_raw = model.predict_proba(x_eval)[:, 1]

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p_cal_raw, calibration[TARGET].to_numpy())
    p_eval = np.asarray(isotonic.predict(p_eval_raw), dtype=float)

    y = evaluation[TARGET].to_numpy()

    return {
        "roc_auc": float(roc_auc_score(y, p_eval)),
        "average_precision": float(average_precision_score(y, p_eval)),
        "brier": float(brier_score_loss(y, p_eval)),
        "ece_10bin": expected_calibration_error(y, p_eval),
        "top20_capture": top20_capture(y, p_eval),
        "mean_predicted_risk": float(p_eval.mean()),
        "observed_delinquency_rate": float(y.mean()),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df[DATE] = pd.to_datetime(df[DATE], errors="raise")
    development = df.loc[df[DATE] <= pd.Timestamp("2016-07-13")].copy()

    rows = []

    for fold in EVALUATION_FOLDS:
        eval_start = pd.Timestamp(fold["eval_start"])
        eval_end = pd.Timestamp(fold["eval_end"])
        calibration_end = eval_start - pd.Timedelta(days=1)
        calibration_start = calibration_end - pd.Timedelta(days=CALIBRATION_DAYS - 1)
        train_end = calibration_start - pd.Timedelta(days=1)

        train = development.loc[development[DATE] <= train_end].copy()
        calibration = development.loc[
            (development[DATE] >= calibration_start) &
            (development[DATE] <= calibration_end)
        ].copy()
        evaluation = development.loc[
            (development[DATE] >= eval_start) &
            (development[DATE] <= eval_end)
        ].copy()

        for variant, features in VARIANTS.items():
            metrics = fit_and_score(train, calibration, evaluation, features)
            rows.append({
                "fold": fold["fold"],
                "variant": variant,
                "feature_count": len(features),
                "added_features": ";".join([f for f in features if f not in BASE_7]),
                **metrics,
            })

    fold_metrics = pd.DataFrame(rows)

    summary = (
        fold_metrics.groupby("variant", as_index=False)
        .agg(
            feature_count=("feature_count", "first"),
            added_features=("added_features", "first"),
            mean_roc_auc=("roc_auc", "mean"),
            min_roc_auc=("roc_auc", "min"),
            mean_average_precision=("average_precision", "mean"),
            min_average_precision=("average_precision", "min"),
            mean_brier=("brier", "mean"),
            mean_ece=("ece_10bin", "mean"),
            mean_top20_capture=("top20_capture", "mean"),
            min_top20_capture=("top20_capture", "min"),
        )
        .reset_index(drop=True)
    )

    base = summary.loc[summary["variant"] == "base_7"].iloc[0]
    for metric in [
        "mean_roc_auc",
        "mean_average_precision",
        "mean_brier",
        "mean_ece",
        "mean_top20_capture",
    ]:
        summary[f"{metric}_change_vs_base7"] = summary[metric] - base[metric]

    # Rank mainly for descriptive convenience, not model selection.
    summary = summary.sort_values(
        ["mean_average_precision", "mean_roc_auc", "mean_top20_capture"],
        ascending=False,
    ).reset_index(drop=True)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fold_metrics.to_csv(FOLD_TABLE, index=False)
    summary.to_csv(SUMMARY_TABLE, index=False)

    payload = {
        "analysis_role": "development_only_incremental_addback",
        "final_holdout_used": False,
        "base_features": BASE_7,
        "candidate_addbacks": EXTRA_5,
        "variants": VARIANTS,
        "folds": EVALUATION_FOLDS,
        "summary": summary.to_dict(orient="records"),
        "interpretation_note": (
            "The experiment is diagnostic. It identifies where the five additional predictors "
            "add value relative to the seven-feature recharge base, but it does not by itself "
            "justify replacing the formally selected model."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Average precision figure
    ap_plot = summary.set_index("variant")["mean_average_precision"].sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(ap_plot.index, ap_plot.values)
    ax.set_title("Incremental add-back: mean average precision")
    ax.set_xlabel("Mean average precision")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(AP_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Brier figure
    brier_plot = summary.set_index("variant")["mean_brier"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(brier_plot.index, brier_plot.values)
    ax.set_title("Incremental add-back: mean Brier score")
    ax.set_xlabel("Mean Brier score (lower is better)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(Brier_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ROC-AUC figure
    roc_plot = summary.set_index("variant")["mean_roc_auc"].sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(roc_plot.index, roc_plot.values)
    ax.set_title("Incremental add-back: mean ROC-AUC")
    ax.set_xlabel("Mean ROC-AUC")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(ROC_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Module 4 incremental add-back analysis completed.")
    print("Final holdout used: False")
    print()
    print("Summary")
    print(
        summary[
            [
                "variant",
                "feature_count",
                "added_features",
                "mean_roc_auc",
                "mean_average_precision",
                "mean_brier",
                "mean_ece",
                "mean_top20_capture",
                "mean_roc_auc_change_vs_base7",
                "mean_average_precision_change_vs_base7",
                "mean_brier_change_vs_base7",
                "mean_top20_capture_change_vs_base7",
            ]
        ].to_string(index=False)
    )
    print()
    print("Fold-level metrics")
    print(
        fold_metrics[
            [
                "fold",
                "variant",
                "roc_auc",
                "average_precision",
                "brier",
                "ece_10bin",
                "top20_capture",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Fold metrics written to: {FOLD_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Summary table written to: {SUMMARY_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Summary JSON written to: {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Average precision figure: {AP_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Brier figure: {Brier_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"ROC-AUC figure: {ROC_FIGURE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
