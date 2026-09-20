"""Compare the 8-feature challenger directly with the 12-feature selected model.

The 8-feature challenger consists of:
- the seven-feature recharge-behaviour core;
- daily_decr30 as the single retained decrement feature.

The comparison has two parts:
1. development-only chronological folds (primary comparison);
2. the already-opened 14-23 July holdout as sensitivity evidence only.

Random Forest hyperparameters and isotonic calibration remain fixed.
No tuning is performed in this script.
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

DEV_FOLD_TABLE = TABLE_DIR / "module4_8_vs_12_dev_fold_metrics.csv"
DEV_SUMMARY_TABLE = TABLE_DIR / "module4_8_vs_12_dev_summary.csv"
HOLDOUT_TABLE = TABLE_DIR / "module4_8_vs_12_holdout_sensitivity.csv"
SUMMARY_JSON = TABLE_DIR / "module4_8_vs_12_summary.json"

DEV_FIGURE = FIGURE_DIR / "model_8_vs_12_development.png"
HOLDOUT_FIGURE = FIGURE_DIR / "model_8_vs_12_holdout.png"
TRADEOFF_FIGURE = FIGURE_DIR / "model_8_vs_12_tradeoff.png"

TARGET = "delinquent_5d"
DATE = "pdate"
CALIBRATION_DAYS = 7
RANDOM_STATE = 42

TRAIN_END = pd.Timestamp("2016-07-06")
CALIBRATION_START = pd.Timestamp("2016-07-07")
CALIBRATION_END = pd.Timestamp("2016-07-13")
HOLDOUT_START = pd.Timestamp("2016-07-14")
HOLDOUT_END = pd.Timestamp("2016-07-23")

BASE_7 = [
    "cnt_ma_rech90",
    "sumamnt_ma_rech90",
    "last_rech_amt_ma",
    "sumamnt_ma_rech30",
    "medianamnt_ma_rech30",
    "medianmarechprebal90",
    "cnt_ma_rech30",
]

CHALLENGER_8 = BASE_7 + ["daily_decr30"]

CURRENT_12 = [
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
    "challenger_8": CHALLENGER_8,
    "current_12": CURRENT_12,
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


def fit_score(train, calibration, evaluation, features):
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

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal_raw, calibration[TARGET].to_numpy())
    p_eval = np.asarray(iso.predict(p_eval_raw), dtype=float)

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

    dev_rows = []
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
            metrics = fit_score(train, calibration, evaluation, features)
            dev_rows.append({
                "fold": fold["fold"],
                "variant": variant,
                "feature_count": len(features),
                **metrics,
            })

    dev_metrics = pd.DataFrame(dev_rows)

    dev_summary = (
        dev_metrics.groupby("variant", as_index=False)
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
    )

    # Holdout sensitivity using the same train/calibration chronology as prior calibrated holdout work.
    train = df.loc[df[DATE] <= TRAIN_END].copy()
    calibration = df.loc[
        (df[DATE] >= CALIBRATION_START)
        & (df[DATE] <= CALIBRATION_END)
    ].copy()
    holdout = df.loc[
        (df[DATE] >= HOLDOUT_START)
        & (df[DATE] <= HOLDOUT_END)
    ].copy()

    holdout_rows = []
    for variant, features in VARIANTS.items():
        metrics = fit_score(train, calibration, holdout, features)
        holdout_rows.append({
            "variant": variant,
            "feature_count": len(features),
            **metrics,
        })

    holdout_metrics = pd.DataFrame(holdout_rows)

    # Deltas from current 12-feature model.
    dev_ref = dev_summary.loc[dev_summary["variant"] == "current_12"].iloc[0]
    hold_ref = holdout_metrics.loc[holdout_metrics["variant"] == "current_12"].iloc[0]

    for metric in [
        "mean_roc_auc",
        "mean_average_precision",
        "mean_brier",
        "mean_ece",
        "mean_top20_capture",
    ]:
        dev_summary[f"{metric}_change_vs_current12"] = dev_summary[metric] - dev_ref[metric]

    for metric in [
        "roc_auc",
        "average_precision",
        "brier",
        "ece_10bin",
        "top20_capture",
    ]:
        holdout_metrics[f"{metric}_change_vs_current12"] = holdout_metrics[metric] - hold_ref[metric]

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    dev_metrics.to_csv(DEV_FOLD_TABLE, index=False)
    dev_summary.to_csv(DEV_SUMMARY_TABLE, index=False)
    holdout_metrics.to_csv(HOLDOUT_TABLE, index=False)

    payload = {
        "analysis_role": "direct_8_vs_12_comparison",
        "development_final_holdout_used": False,
        "holdout_sensitivity_independent_validation": False,
        "model_hyperparameters_changed": False,
        "calibration_method_changed": False,
        "variants": VARIANTS,
        "development_summary": dev_summary.to_dict(orient="records"),
        "holdout_sensitivity": holdout_metrics.to_dict(orient="records"),
        "interpretation_note": (
            "Development folds provide the main comparison. The already-opened holdout "
            "is used only as sensitivity evidence and cannot reset the independent validation."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Development visual.
    dev_plot = dev_summary.set_index("variant")[
        ["mean_roc_auc", "mean_average_precision", "mean_top20_capture"]
    ]
    ax = dev_plot.plot(kind="bar", figsize=(10, 6))
    ax.set_title("8-feature challenger vs 12-feature model: development")
    ax.set_ylabel("Metric value")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(["Mean ROC-AUC", "Mean average precision", "Mean top-20% capture"])
    ax.figure.tight_layout()
    ax.figure.savefig(DEV_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)

    # Holdout sensitivity visual.
    hold_plot = holdout_metrics.set_index("variant")[
        ["roc_auc", "average_precision", "top20_capture"]
    ]
    ax = hold_plot.plot(kind="bar", figsize=(10, 6))
    ax.set_title("8-feature challenger vs 12-feature model: holdout sensitivity")
    ax.set_ylabel("Metric value")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(["ROC-AUC", "Average precision", "Top-20% capture"])
    ax.figure.tight_layout()
    ax.figure.savefig(HOLDOUT_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)

    # Trade-off visual: AP vs Brier on development.
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in dev_summary.iterrows():
        ax.scatter(row["mean_average_precision"], row["mean_brier"], s=80)
        ax.annotate(
            row["variant"],
            (row["mean_average_precision"], row["mean_brier"]),
            xytext=(6, 4),
            textcoords="offset points",
        )
    ax.set_title("Development trade-off: precision vs probability error")
    ax.set_xlabel("Mean average precision (higher is better)")
    ax.set_ylabel("Mean Brier score (lower is better)")
    fig.tight_layout()
    fig.savefig(TRADEOFF_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Module 4 direct 8-vs-12 model comparison completed.")
    print("Development final holdout used: False")
    print("Holdout sensitivity is independent validation: False")
    print()
    print("Development summary")
    print(dev_summary.to_string(index=False))
    print()
    print("Holdout sensitivity")
    print(holdout_metrics.to_string(index=False))
    print()
    print("Development fold metrics")
    print(
        dev_metrics[
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
    print(f"Development fold table: {DEV_FOLD_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Development summary table: {DEV_SUMMARY_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Holdout sensitivity table: {HOLDOUT_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Summary JSON: {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Development figure: {DEV_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Holdout figure: {HOLDOUT_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Trade-off figure: {TRADEOFF_FIGURE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
