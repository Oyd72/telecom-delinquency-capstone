"""Compare the 12-feature and 7-feature Random Forest variants on development-only chronology.

This experiment responds to the post-hoc holdout sensitivity result where the 7-feature
drift-reduced model retained nearly all discrimination and improved capture/calibration.

To avoid choosing a model based only on an already-opened holdout, this script returns to
the development period and compares the two fixed feature sets on the same chronological
folds. Random Forest hyperparameters and isotonic calibration remain fixed.

Variants:
- current_12: selected 12-feature specification
- drift_reduced_7: recharge-behaviour-only specification

Chronological design for each fold:
- train on all earlier observations;
- fit isotonic calibration on the immediately preceding 7 calendar days;
- evaluate on a later development block.

The final 14-23 July holdout is not used here.
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

FOLD_TABLE = TABLE_DIR / "module4_dev_12_vs_7_fold_metrics.csv"
SUMMARY_TABLE = TABLE_DIR / "module4_dev_12_vs_7_summary.csv"
SUMMARY_JSON = TABLE_DIR / "module4_dev_12_vs_7_summary.json"

ROC_FIGURE = FIGURE_DIR / "development_12_vs_7_roc_auc.png"
CAPTURE_FIGURE = FIGURE_DIR / "development_12_vs_7_capture.png"
CALIBRATION_FIGURE = FIGURE_DIR / "development_12_vs_7_calibration.png"

TARGET = "delinquent_5d"
DATE = "pdate"
CALIBRATION_DAYS = 7
RANDOM_STATE = 42

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

DRIFT_REDUCED_7 = [
    "cnt_ma_rech90",
    "sumamnt_ma_rech90",
    "last_rech_amt_ma",
    "sumamnt_ma_rech30",
    "medianamnt_ma_rech30",
    "medianmarechprebal90",
    "cnt_ma_rech30",
]

VARIANTS = {
    "current_12": CURRENT_12,
    "drift_reduced_7": DRIFT_REDUCED_7,
}

EVALUATION_FOLDS = [
    {
        "fold": "late_june",
        "eval_start": "2016-06-14",
        "eval_end": "2016-06-23",
    },
    {
        "fold": "turn_of_month",
        "eval_start": "2016-06-24",
        "eval_end": "2016-07-03",
    },
    {
        "fold": "early_july",
        "eval_start": "2016-07-04",
        "eval_end": "2016-07-13",
    },
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


def fit_and_score(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
) -> dict:
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

    y_eval = evaluation[TARGET].to_numpy()

    return {
        "roc_auc": float(roc_auc_score(y_eval, p_eval)),
        "average_precision": float(average_precision_score(y_eval, p_eval)),
        "brier": float(brier_score_loss(y_eval, p_eval)),
        "ece_10bin": expected_calibration_error(y_eval, p_eval),
        "top20_capture": top20_capture(y_eval, p_eval),
        "mean_predicted_risk": float(p_eval.mean()),
        "observed_delinquency_rate": float(y_eval.mean()),
    }


def main() -> None:
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

        if train.empty or calibration.empty or evaluation.empty:
            raise ValueError(f"Empty split for fold {fold['fold']}")

        for variant, features in VARIANTS.items():
            metrics = fit_and_score(train, calibration, evaluation, features)
            rows.append(
                {
                    "fold": fold["fold"],
                    "variant": variant,
                    "feature_count": len(features),
                    "train_rows": int(len(train)),
                    "calibration_rows": int(len(calibration)),
                    "evaluation_rows": int(len(evaluation)),
                    "train_end": str(train_end.date()),
                    "calibration_start": str(calibration_start.date()),
                    "calibration_end": str(calibration_end.date()),
                    "eval_start": str(eval_start.date()),
                    "eval_end": str(eval_end.date()),
                    **metrics,
                }
            )

    fold_metrics = pd.DataFrame(rows)

    summary = (
        fold_metrics.groupby("variant", as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            min_roc_auc=("roc_auc", "min"),
            mean_average_precision=("average_precision", "mean"),
            mean_brier=("brier", "mean"),
            mean_ece=("ece_10bin", "mean"),
            mean_top20_capture=("top20_capture", "mean"),
            min_top20_capture=("top20_capture", "min"),
        )
        .sort_values("mean_roc_auc", ascending=False)
        .reset_index(drop=True)
    )

    baseline = summary.loc[summary["variant"] == "current_12"].iloc[0]
    summary["mean_roc_auc_change_vs_current"] = (
        summary["mean_roc_auc"] - baseline["mean_roc_auc"]
    )
    summary["mean_top20_capture_change_vs_current"] = (
        summary["mean_top20_capture"] - baseline["mean_top20_capture"]
    )
    summary["mean_brier_change_vs_current"] = (
        summary["mean_brier"] - baseline["mean_brier"]
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fold_metrics.to_csv(FOLD_TABLE, index=False)
    summary.to_csv(SUMMARY_TABLE, index=False)

    payload = {
        "analysis_role": "development_only_feature_set_sensitivity",
        "final_holdout_used": False,
        "model_hyperparameters_changed": False,
        "calibration_method_changed": False,
        "calibration_days": CALIBRATION_DAYS,
        "variants": VARIANTS,
        "folds": EVALUATION_FOLDS,
        "summary": summary.to_dict(orient="records"),
        "interpretation_note": (
            "This comparison revisits the 12-feature and 7-feature variants using only "
            "development-period chronological folds. It is intended to determine whether "
            "the holdout result for the smaller model is supported by earlier time windows."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Figure 1: mean/min ROC-AUC.
    roc_plot = summary.set_index("variant")[["mean_roc_auc", "min_roc_auc"]]
    ax = roc_plot.plot(kind="bar", figsize=(9, 6))
    ax.set_title("Development-only ROC-AUC: 12 vs 7 features")
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(["Mean ROC-AUC", "Minimum fold ROC-AUC"])
    ax.figure.tight_layout()
    ax.figure.savefig(ROC_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)

    # Figure 2: capture.
    capture_plot = summary.set_index("variant")[
        ["mean_top20_capture", "min_top20_capture"]
    ]
    ax = capture_plot.plot(kind="bar", figsize=(9, 6))
    ax.set_title("Development-only top-20% capture: 12 vs 7 features")
    ax.set_ylabel("Capture rate")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(["Mean capture", "Minimum fold capture"])
    ax.figure.tight_layout()
    ax.figure.savefig(CAPTURE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)

    # Figure 3: calibration quality.
    calib_plot = summary.set_index("variant")[["mean_brier", "mean_ece"]]
    ax = calib_plot.plot(kind="bar", figsize=(9, 6))
    ax.set_title("Development-only calibration: 12 vs 7 features")
    ax.set_ylabel("Error (lower is better)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(["Mean Brier", "Mean ECE"])
    ax.figure.tight_layout()
    ax.figure.savefig(CALIBRATION_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)

    print("Module 4 development-only 12-vs-7 comparison completed.")
    print("Final holdout used: False")
    print("Model hyperparameters changed: False")
    print("Calibration method changed: False")
    print()
    print("Summary")
    print(summary.to_string(index=False))
    print()
    print("Fold metrics")
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
                "mean_predicted_risk",
                "observed_delinquency_rate",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Fold metrics written to: {FOLD_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Summary table written to: {SUMMARY_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Summary JSON written to: {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"ROC-AUC figure: {ROC_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Capture figure: {CAPTURE_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Calibration figure: {CALIBRATION_FIGURE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
