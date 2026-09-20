"""Test whether daily_decr30 and daily_decr90 are redundant when added to the 7-feature recharge base.

Development-only chronological experiment.

Variants:
- base_7
- base_plus_daily_decr30
- base_plus_daily_decr90
- base_plus_both_decr
- full_12 (reference only)

Random Forest hyperparameters and isotonic calibration remain fixed.
The final 14-23 July holdout is not used.
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

FOLD_TABLE = TABLE_DIR / "module4_decr30_decr90_redundancy_fold_metrics.csv"
SUMMARY_TABLE = TABLE_DIR / "module4_decr30_decr90_redundancy_summary.csv"
SUMMARY_JSON = TABLE_DIR / "module4_decr30_decr90_redundancy_summary.json"

AP_FIGURE = FIGURE_DIR / "decr30_decr90_average_precision.png"
ROC_FIGURE = FIGURE_DIR / "decr30_decr90_roc_auc.png"
BRIER_FIGURE = FIGURE_DIR / "decr30_decr90_brier.png"

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

FULL_12 = [
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

VARIANTS = {
    "base_7": BASE_7,
    "base_plus_daily_decr30": BASE_7 + ["daily_decr30"],
    "base_plus_daily_decr90": BASE_7 + ["daily_decr90"],
    "base_plus_both_decr": BASE_7 + ["daily_decr30", "daily_decr90"],
    "full_12_reference": FULL_12,
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
        ece += (mask.sum() / len(y)) * abs(
            float(y[mask].mean()) - float(p[mask].mean())
        )

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
    x_train = pd.DataFrame(
        imputer.fit_transform(train[features]),
        columns=features,
        index=train.index,
    )
    x_cal = pd.DataFrame(
        imputer.transform(calibration[features]),
        columns=features,
        index=calibration.index,
    )
    x_eval = pd.DataFrame(
        imputer.transform(evaluation[features]),
        columns=features,
        index=evaluation.index,
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
            (development[DATE] >= calibration_start)
            & (development[DATE] <= calibration_end)
        ].copy()
        evaluation = development.loc[
            (development[DATE] >= eval_start)
            & (development[DATE] <= eval_end)
        ].copy()

        for variant, features in VARIANTS.items():
            metrics = fit_and_score(train, calibration, evaluation, features)
            rows.append(
                {
                    "fold": fold["fold"],
                    "variant": variant,
                    "feature_count": len(features),
                    **metrics,
                }
            )

    fold_metrics = pd.DataFrame(rows)

    summary = (
        fold_metrics.groupby("variant", as_index=False)
        .agg(
            feature_count=("feature_count", "first"),
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
    both = summary.loc[summary["variant"] == "base_plus_both_decr"].iloc[0]
    single30 = summary.loc[summary["variant"] == "base_plus_daily_decr30"].iloc[0]
    single90 = summary.loc[summary["variant"] == "base_plus_daily_decr90"].iloc[0]

    for metric in [
        "mean_roc_auc",
        "mean_average_precision",
        "mean_brier",
        "mean_ece",
        "mean_top20_capture",
    ]:
        summary[f"{metric}_change_vs_base7"] = summary[metric] - base[metric]

    incremental_both_vs_best_single = {
        "mean_roc_auc": float(both["mean_roc_auc"] - max(single30["mean_roc_auc"], single90["mean_roc_auc"])),
        "mean_average_precision": float(
            both["mean_average_precision"]
            - max(single30["mean_average_precision"], single90["mean_average_precision"])
        ),
        "mean_brier": float(
            both["mean_brier"]
            - min(single30["mean_brier"], single90["mean_brier"])
        ),
        "mean_ece": float(
            both["mean_ece"]
            - min(single30["mean_ece"], single90["mean_ece"])
        ),
        "mean_top20_capture": float(
            both["mean_top20_capture"]
            - max(single30["mean_top20_capture"], single90["mean_top20_capture"])
        ),
    }

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fold_metrics.to_csv(FOLD_TABLE, index=False)
    summary.to_csv(SUMMARY_TABLE, index=False)

    payload = {
        "analysis_role": "development_only_decrement_redundancy_check",
        "final_holdout_used": False,
        "variants": VARIANTS,
        "folds": EVALUATION_FOLDS,
        "summary": summary.to_dict(orient="records"),
        "incremental_both_vs_best_single": incremental_both_vs_best_single,
        "interpretation_note": (
            "If adding both decrement variables produces little or no gain over the best "
            "single-variable add-back, they are functionally redundant for this model setup. "
            "If both materially improve multiple metrics, they carry complementary signal."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for metric, title, xlabel, path in [
        ("mean_average_precision", "daily_decr30 vs daily_decr90: mean average precision", "Mean average precision", AP_FIGURE),
        ("mean_roc_auc", "daily_decr30 vs daily_decr90: mean ROC-AUC", "Mean ROC-AUC", ROC_FIGURE),
        ("mean_brier", "daily_decr30 vs daily_decr90: mean Brier score", "Mean Brier score (lower is better)", BRIER_FIGURE),
    ]:
        s = summary.set_index("variant")[metric].sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(s.index, s.values)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print("Module 4 daily_decr30/daily_decr90 redundancy analysis completed.")
    print("Final holdout used: False")
    print()
    print("Summary")
    print(
        summary[
            [
                "variant",
                "feature_count",
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
    print("Incremental value of BOTH vs best SINGLE")
    for k, v in incremental_both_vs_best_single.items():
        print(f"{k}: {v:.6f}")
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
    print(f"ROC-AUC figure: {ROC_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Brier figure: {BRIER_FIGURE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
