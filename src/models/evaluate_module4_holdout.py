"""Evaluate the frozen Module 4 candidate models on the untouched final holdout.

This script opens the 14-23 July 2016 holdout for the first time after:
- the modelling population was frozen;
- the 12-feature specification passed leakage review;
- candidate models were benchmarked using development-only chronological folds;
- random forest and XGBoost were tuned using development-only chronological folds.

Frozen candidates:
- Logistic regression baseline
- Tuned random forest: max_depth=8, min_samples_leaf=20, max_features='sqrt'
- Tuned XGBoost: max_depth=5, learning_rate=0.05, min_child_weight=5

The script evaluates discrimination, ranking, calibration and operational capture.
It does not retune any model or parameter on the holdout.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"
TABLE_PATH = PROJECT_ROOT / "reports/tables/module4_final_holdout_metrics.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports/tables/module4_final_holdout_summary.json"

TARGET = "delinquent_5d"
DATE = "pdate"
DEVELOPMENT_END = pd.Timestamp("2016-07-13")
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

DEVELOPMENT_CV_REFERENCE = {
    "logistic_regression": {
        "mean_roc_auc": 0.801738,
        "mean_top20_capture": 0.602002,
    },
    "random_forest": {
        "mean_roc_auc": 0.871091,
        "mean_top20_capture": 0.664079,
    },
    "xgboost": {
        "mean_roc_auc": 0.868756,
        "mean_top20_capture": 0.649633,
    },
}

ACCEPTANCE = {
    "minimum_roc_auc": 0.80,
    "minimum_top20_capture": 0.50,
    "maximum_roc_auc_drop_vs_development": 0.03,
}


def top_fraction_capture(y_true: np.ndarray, proba: np.ndarray, fraction: float = 0.20) -> float:
    n = max(1, int(np.ceil(len(proba) * fraction)))
    order = np.argsort(-proba)
    positives = np.asarray(y_true).sum()
    if positives == 0:
        return float("nan")
    return float(np.asarray(y_true)[order[:n]].sum() / positives)


def make_scaled_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURES,
            )
        ],
        remainder="drop",
    )


def make_tree_preprocessor() -> ColumnTransformer:
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


def make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("prep", make_scaled_preprocessor()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def make_random_forest() -> Pipeline:
    return Pipeline(
        [
            ("prep", make_tree_preprocessor()),
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
            ("prep", make_tree_preprocessor()),
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


def evaluate(
    name: str,
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_holdout: pd.DataFrame,
    y_holdout: pd.Series,
) -> dict:
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_holdout)[:, 1]

    roc_auc = float(roc_auc_score(y_holdout, proba))
    ap = float(average_precision_score(y_holdout, proba))
    brier = float(brier_score_loss(y_holdout, proba))
    capture = top_fraction_capture(y_holdout.to_numpy(), proba)
    mean_risk = float(np.mean(proba))
    observed_rate = float(np.mean(y_holdout))

    prevalence_brier = float(
        brier_score_loss(
            y_holdout,
            np.full(len(y_holdout), float(np.mean(y_train))),
        )
    )

    reference = DEVELOPMENT_CV_REFERENCE[name]
    roc_auc_drop = float(reference["mean_roc_auc"] - roc_auc)

    return {
        "model": name,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "brier": brier,
        "prevalence_baseline_brier": prevalence_brier,
        "brier_better_than_prevalence_baseline": bool(brier < prevalence_brier),
        "top20_capture": capture,
        "mean_predicted_risk": mean_risk,
        "observed_delinquency_rate": observed_rate,
        "calibration_in_the_large_error": float(mean_risk - observed_rate),
        "development_mean_roc_auc": reference["mean_roc_auc"],
        "roc_auc_drop_vs_development": roc_auc_drop,
        "passes_roc_auc": bool(roc_auc >= ACCEPTANCE["minimum_roc_auc"]),
        "passes_top20_capture": bool(
            capture >= ACCEPTANCE["minimum_top20_capture"]
        ),
        "passes_stability": bool(
            roc_auc_drop <= ACCEPTANCE["maximum_roc_auc_drop_vs_development"]
        ),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df[DATE] = pd.to_datetime(df[DATE], errors="raise")

    development = df.loc[df[DATE] <= DEVELOPMENT_END].copy()
    holdout = df.loc[
        (df[DATE] >= HOLDOUT_START) & (df[DATE] <= HOLDOUT_END)
    ].copy()

    if len(development) != 122_119:
        raise AssertionError(
            f"Unexpected development row count: {len(development):,}"
        )
    if len(holdout) != 28_648:
        raise AssertionError(f"Unexpected holdout row count: {len(holdout):,}")

    x_train = development[FEATURES]
    y_train = development[TARGET]
    x_holdout = holdout[FEATURES]
    y_holdout = holdout[TARGET]

    mlflow_db = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
    mlflow.set_experiment("module4_final_holdout_evaluation")

    models = {
        "logistic_regression": make_logistic(),
        "random_forest": make_random_forest(),
        "xgboost": make_xgboost(y_train),
    }

    rows = []
    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_final_holdout"):
            result = evaluate(
                name,
                model,
                x_train,
                y_train,
                x_holdout,
                y_holdout,
            )
            rows.append(result)

            mlflow.log_params(
                {
                    "model": name,
                    "feature_count": len(FEATURES),
                    "development_end": str(DEVELOPMENT_END.date()),
                    "holdout_start": str(HOLDOUT_START.date()),
                    "holdout_end": str(HOLDOUT_END.date()),
                    "final_holdout": True,
                }
            )
            mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in result.items()
                    if isinstance(v, (float, int))
                    and not isinstance(v, bool)
                    and k != "model"
                }
            )

    metrics = pd.DataFrame(rows)
    metrics["passes_core_acceptance"] = (
        metrics["passes_roc_auc"]
        & metrics["passes_top20_capture"]
        & metrics["passes_stability"]
        & metrics["brier_better_than_prevalence_baseline"]
    )

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLE_PATH, index=False)

    payload = {
        "holdout_opened": True,
        "holdout_opened_for_tuning": False,
        "development_rows": int(len(development)),
        "holdout_rows": int(len(holdout)),
        "development_delinquency_rate": float(y_train.mean()),
        "holdout_delinquency_rate": float(y_holdout.mean()),
        "features": FEATURES,
        "acceptance": ACCEPTANCE,
        "results": metrics.to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Module 4 final holdout evaluation completed.")
    print("Final holdout was used for evaluation only; no tuning was performed.")
    print()
    display_cols = [
        "model",
        "roc_auc",
        "average_precision",
        "brier",
        "prevalence_baseline_brier",
        "top20_capture",
        "mean_predicted_risk",
        "observed_delinquency_rate",
        "roc_auc_drop_vs_development",
        "passes_core_acceptance",
    ]
    print(metrics[display_cols].to_string(index=False))
    print()
    print(f"Metrics written to: {TABLE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Summary written to: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"MLflow tracking database: {mlflow_db.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
