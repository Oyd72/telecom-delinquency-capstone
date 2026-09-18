"""Confirm the frozen isotonic calibration approach on the already-opened final holdout.

This is a sensitivity/confirmatory analysis, not a second independent final validation.

Chronology is preserved:
- base model training: through 6 July 2016;
- calibration window: 7-13 July 2016;
- evaluation: 14-23 July 2016 final holdout.

For each nonlinear candidate, the script compares:
- uncalibrated probabilities from the same base model;
- isotonic-calibrated probabilities using only the 7-day pre-holdout calibration window.

No model hyperparameters or calibration method are changed using holdout outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"
TABLE_PATH = PROJECT_ROOT / "reports/tables/module4_calibrated_holdout_metrics.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports/tables/module4_calibrated_holdout_summary.json"

TARGET = "delinquent_5d"
DATE = "pdate"

TRAIN_END = pd.Timestamp("2016-07-06")
CALIBRATION_START = pd.Timestamp("2016-07-07")
CALIBRATION_END = pd.Timestamp("2016-07-13")
HOLDOUT_START = pd.Timestamp("2016-07-14")
HOLDOUT_END = pd.Timestamp("2016-07-23")

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


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
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


def top20_capture(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    n_top = max(1, int(np.ceil(len(y) * 0.20)))
    order = np.argsort(-p)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    return float(y[order[:n_top]].sum() / positives)


def tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                FEATURES,
            )
        ],
        remainder="drop",
    )


def make_random_forest() -> Pipeline:
    return Pipeline(
        [
            ("prep", tree_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=20,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def make_xgboost(y_train: pd.Series) -> Pipeline:
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    return Pipeline(
        [
            ("prep", tree_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    min_child_weight=5,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=negatives / max(positives, 1),
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def metric_row(
    model_name: str,
    method: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    prevalence_brier: float,
) -> dict:
    observed = float(np.mean(y_true))
    predicted = float(np.mean(probabilities))
    brier = float(brier_score_loss(y_true, probabilities))

    return {
        "model": model_name,
        "method": method,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier": brier,
        "prevalence_baseline_brier": prevalence_brier,
        "brier_better_than_prevalence_baseline": bool(brier < prevalence_brier),
        "ece_10bin": expected_calibration_error(y_true, probabilities),
        "top20_capture": top20_capture(y_true, probabilities),
        "mean_predicted_risk": predicted,
        "observed_delinquency_rate": observed,
        "calibration_in_the_large_error": float(predicted - observed),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df[DATE] = pd.to_datetime(df[DATE], errors="raise")

    train = df.loc[df[DATE] <= TRAIN_END].copy()
    calibration = df.loc[
        (df[DATE] >= CALIBRATION_START) & (df[DATE] <= CALIBRATION_END)
    ].copy()
    holdout = df.loc[
        (df[DATE] >= HOLDOUT_START) & (df[DATE] <= HOLDOUT_END)
    ].copy()

    if train.empty or calibration.empty or holdout.empty:
        raise ValueError("Train, calibration, or holdout partition is empty.")
    if calibration[TARGET].nunique() < 2:
        raise ValueError("Calibration partition contains only one target class.")

    x_train = train[FEATURES]
    y_train = train[TARGET]
    x_cal = calibration[FEATURES]
    y_cal = calibration[TARGET].to_numpy()
    x_holdout = holdout[FEATURES]
    y_holdout = holdout[TARGET].to_numpy()

    prevalence_probability = float(y_train.mean())
    prevalence_brier = float(
        brier_score_loss(
            y_holdout,
            np.full(len(y_holdout), prevalence_probability),
        )
    )

    mlflow_db = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
    mlflow.set_experiment("module4_calibrated_holdout_sensitivity")

    models = {
        "random_forest": make_random_forest(),
        "xgboost": make_xgboost(y_train),
    }

    rows = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)

        p_cal_raw = np.clip(model.predict_proba(x_cal)[:, 1], 1e-6, 1 - 1e-6)
        p_holdout_raw = np.clip(
            model.predict_proba(x_holdout)[:, 1],
            1e-6,
            1 - 1e-6,
        )

        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(p_cal_raw, y_cal)
        p_holdout_iso = np.clip(
            isotonic.predict(p_holdout_raw),
            1e-6,
            1 - 1e-6,
        )

        for method, probabilities in {
            "uncalibrated_same_split": p_holdout_raw,
            "isotonic": p_holdout_iso,
        }.items():
            row = metric_row(
                model_name=model_name,
                method=method,
                y_true=y_holdout,
                probabilities=probabilities,
                prevalence_brier=prevalence_brier,
            )
            rows.append(row)

            with mlflow.start_run(run_name=f"{model_name}_{method}_holdout"):
                mlflow.log_params(
                    {
                        "model": model_name,
                        "method": method,
                        "feature_count": len(FEATURES),
                        "train_end": str(TRAIN_END.date()),
                        "calibration_start": str(CALIBRATION_START.date()),
                        "calibration_end": str(CALIBRATION_END.date()),
                        "holdout_start": str(HOLDOUT_START.date()),
                        "holdout_end": str(HOLDOUT_END.date()),
                        "confirmatory_only": True,
                    }
                )
                mlflow.log_metrics(
                    {
                        k: float(v)
                        for k, v in row.items()
                        if isinstance(v, (float, int))
                        and not isinstance(v, bool)
                    }
                )

    metrics = pd.DataFrame(rows)
    metrics["abs_calibration_gap"] = metrics[
        "calibration_in_the_large_error"
    ].abs()

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLE_PATH, index=False)

    payload = {
        "analysis_role": "confirmatory_sensitivity",
        "independent_final_validation": False,
        "calibration_method_frozen_before_this_run": "isotonic",
        "train_end": str(TRAIN_END.date()),
        "calibration_start": str(CALIBRATION_START.date()),
        "calibration_end": str(CALIBRATION_END.date()),
        "holdout_start": str(HOLDOUT_START.date()),
        "holdout_end": str(HOLDOUT_END.date()),
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "holdout_rows": int(len(holdout)),
        "results": metrics.to_dict(orient="records"),
        "methodological_note": (
            "The final holdout had already been opened before this sensitivity analysis. "
            "The isotonic method itself was selected using development-only evidence. "
            "These results therefore assess whether the frozen calibration approach behaves "
            "as expected on the holdout, but they do not constitute a second independent "
            "final validation."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Module 4 calibrated-holdout sensitivity analysis completed.")
    print("Independent final validation: False")
    print("Calibration method frozen before this run: isotonic")
    print()
    print(
        metrics[
            [
                "model",
                "method",
                "roc_auc",
                "average_precision",
                "brier",
                "ece_10bin",
                "top20_capture",
                "mean_predicted_risk",
                "observed_delinquency_rate",
                "abs_calibration_gap",
                "brier_better_than_prevalence_baseline",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Metrics written to: {TABLE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Summary written to: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"MLflow tracking database: {mlflow_db.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
