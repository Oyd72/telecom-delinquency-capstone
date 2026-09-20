"""Generate conventional classification artefacts for Module 4.

The project is primarily a risk-ranking model, so ROC-AUC, average precision,
calibration, and top-risk capture remain the main evaluation measures. This script
adds the classification artefacts explicitly requested by the assignment:

- accuracy, precision, recall, F1;
- a majority-class baseline;
- a confusion matrix;
- a ROC curve.

The classification threshold is selected using development-only chronological
predictions. The final holdout is not used to choose the threshold.

Threshold selection rule:
1. require recall >= 0.50 and precision >= 0.45 where feasible;
2. among qualifying thresholds, choose the one with the highest F1;
3. if none qualify, choose the threshold with highest F1 and record that the
   original precision/recall criterion was not feasible.
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
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"

TABLE_PATH = PROJECT_ROOT / "reports/tables/module4_classification_metrics.csv"
THRESHOLD_PATH = PROJECT_ROOT / "reports/tables/module4_classification_threshold.json"
CM_PATH = PROJECT_ROOT / "reports/figures/module4/final_holdout_confusion_matrix.png"
ROC_PATH = PROJECT_ROOT / "reports/figures/module4/final_holdout_roc_curve.png"

TARGET = "delinquent_5d"
DATE = "pdate"

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

DEV_FOLDS = [
    ("late_june", "2016-06-14", "2016-06-23"),
    ("turn_of_month", "2016-06-24", "2016-07-03"),
    ("early_july", "2016-07-04", "2016-07-13"),
]

CALIBRATION_DAYS = 7
HOLDOUT_START = pd.Timestamp("2016-07-14")
HOLDOUT_END = pd.Timestamp("2016-07-23")


def make_components():
    imputer = SimpleImputer(strategy="median")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    return imputer, model


def fit_predict_calibrated(train, calibration, evaluation):
    imputer, model = make_components()

    x_train = imputer.fit_transform(train[FEATURES])
    x_cal = imputer.transform(calibration[FEATURES])
    x_eval = imputer.transform(evaluation[FEATURES])

    model.fit(x_train, train[TARGET])

    p_cal_raw = model.predict_proba(x_cal)[:, 1]
    p_eval_raw = model.predict_proba(x_eval)[:, 1]

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p_cal_raw, calibration[TARGET].to_numpy())

    return np.asarray(isotonic.predict(p_eval_raw), dtype=float)


def development_predictions(df):
    parts = []

    for fold_name, eval_start_s, eval_end_s in DEV_FOLDS:
        eval_start = pd.Timestamp(eval_start_s)
        eval_end = pd.Timestamp(eval_end_s)
        cal_end = eval_start - pd.Timedelta(days=1)
        cal_start = cal_end - pd.Timedelta(days=CALIBRATION_DAYS - 1)
        train_end = cal_start - pd.Timedelta(days=1)

        train = df.loc[df[DATE] <= train_end].copy()
        calibration = df.loc[
            (df[DATE] >= cal_start) & (df[DATE] <= cal_end)
        ].copy()
        evaluation = df.loc[
            (df[DATE] >= eval_start) & (df[DATE] <= eval_end)
        ].copy()

        p = fit_predict_calibrated(train, calibration, evaluation)

        parts.append(
            pd.DataFrame(
                {
                    "fold": fold_name,
                    "y_true": evaluation[TARGET].to_numpy(dtype=int),
                    "probability": p,
                }
            )
        )

    return pd.concat(parts, ignore_index=True)


def choose_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)

    rows = []
    for i, threshold in enumerate(thresholds):
        p = float(precision[i])
        r = float(recall[i])
        f1 = 0.0 if (p + r) == 0 else float(2 * p * r / (p + r))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": p,
                "recall": r,
                "f1": f1,
                "meets_precision_recall_target": bool(p >= 0.45 and r >= 0.50),
            }
        )

    candidates = pd.DataFrame(rows)
    qualifying = candidates.loc[candidates["meets_precision_recall_target"]]

    if not qualifying.empty:
        selected = qualifying.sort_values(
            ["f1", "precision", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        rule_status = "precision_recall_target_met"
    else:
        selected = candidates.sort_values(
            ["f1", "recall", "precision"],
            ascending=[False, False, False],
        ).iloc[0]
        rule_status = "fallback_max_f1"

    return selected.to_dict(), rule_status


def metric_row(label, y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)

    return {
        "evaluation": label,
        "rows": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "predicted_positive_rate": float(pred.mean()),
        "observed_positive_rate": float(np.mean(y_true)),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df[DATE] = pd.to_datetime(df[DATE], errors="raise")

    dev_pred = development_predictions(df)
    threshold_info, threshold_status = choose_threshold(
        dev_pred["y_true"].to_numpy(),
        dev_pred["probability"].to_numpy(),
    )
    threshold = float(threshold_info["threshold"])

    train = df.loc[df[DATE] <= pd.Timestamp("2016-07-06")].copy()
    calibration = df.loc[
        (df[DATE] >= pd.Timestamp("2016-07-07"))
        & (df[DATE] <= pd.Timestamp("2016-07-13"))
    ].copy()
    holdout = df.loc[
        (df[DATE] >= HOLDOUT_START) & (df[DATE] <= HOLDOUT_END)
    ].copy()

    p_holdout = fit_predict_calibrated(train, calibration, holdout)
    y_holdout = holdout[TARGET].to_numpy(dtype=int)
    holdout_pred = (p_holdout >= threshold).astype(int)

    model_metrics = metric_row(
        "selected_random_forest_isotonic",
        y_holdout,
        p_holdout,
        threshold,
    )

    majority_pred = np.zeros_like(y_holdout)
    majority_metrics = {
        "evaluation": "majority_class_baseline",
        "rows": int(len(y_holdout)),
        "threshold": np.nan,
        "accuracy": float(accuracy_score(y_holdout, majority_pred)),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": 0.5,
        "average_precision": float(np.mean(y_holdout)),
        "predicted_positive_rate": 0.0,
        "observed_positive_rate": float(np.mean(y_holdout)),
    }

    metrics = pd.DataFrame([model_metrics, majority_metrics])
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLE_PATH, index=False)

    tn, fp, fn, tp = confusion_matrix(y_holdout, holdout_pred).ravel()

    payload = {
        "threshold_selected_on_final_holdout": False,
        "development_folds": [name for name, _, _ in DEV_FOLDS],
        "threshold_selection_rule": (
            "Require recall >= 0.50 and precision >= 0.45 where feasible; "
            "then maximize F1 among qualifying thresholds."
        ),
        "threshold_selection_status": threshold_status,
        "selected_threshold": threshold,
        "development_precision_at_selected_threshold": float(
            threshold_info["precision"]
        ),
        "development_recall_at_selected_threshold": float(
            threshold_info["recall"]
        ),
        "development_f1_at_selected_threshold": float(threshold_info["f1"]),
        "holdout_confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "methodological_note": (
            "Classification metrics are supplementary to the project's ranking "
            "and calibration metrics. The threshold was frozen using development-only "
            "chronological predictions before being applied to the final holdout."
        ),
    }
    THRESHOLD_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Predicted on-time", "Predicted delinquent"])
    ax.set_yticks([0, 1], labels=["Observed on-time", "Observed delinquent"])
    ax.set_title(f"Final holdout confusion matrix\nthreshold = {threshold:.3f}")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Observed")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    CM_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CM_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_holdout, p_holdout)
    roc_auc = roc_auc_score(y_holdout, p_holdout)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(fpr, tpr, label=f"Selected model (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="No-skill reference")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Final holdout ROC curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROC_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Module 4 conventional classification artefacts completed.")
    print("Threshold selected on final holdout: False")
    print(f"Threshold selection status: {threshold_status}")
    print(f"Selected threshold: {threshold:.6f}")
    print()
    print("Final holdout metrics")
    print(metrics.to_string(index=False))
    print()
    print(
        f"Confusion matrix: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}"
    )
    print()
    print(f"Metrics written to: {TABLE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Threshold summary: {THRESHOLD_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Confusion matrix figure: {CM_PATH.relative_to(PROJECT_ROOT)}")
    print(f"ROC figure: {ROC_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
