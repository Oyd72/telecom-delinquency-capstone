"""Benchmark three Module 4 candidate classifiers using development data only.

The final 14-23 July holdout is deliberately not opened here.

Models:
- Logistic regression: transparent baseline.
- Random forest: nonlinear tree ensemble with moderate complexity.
- XGBoost: stronger nonlinear challenger already supported by Module 3 evidence.

All preprocessing is fitted inside each chronological training fold. Results and
hyperparameters are logged to MLflow and written to reports/tables.
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
FOLD_PATH = PROJECT_ROOT / "reports/tables/module4_candidate_model_fold_metrics.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports/tables/module4_candidate_model_summary.csv"
JSON_PATH = PROJECT_ROOT / "reports/tables/module4_candidate_model_summary.json"

TARGET = "delinquent_5d"
DATE = "pdate"
DEVELOPMENT_END = pd.Timestamp("2016-07-13")

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

FOLDS = [
    {
        "fold": "late_june",
        "train_end": "2016-06-13",
        "eval_start": "2016-06-14",
        "eval_end": "2016-06-23",
    },
    {
        "fold": "turn_of_month",
        "train_end": "2016-06-23",
        "eval_start": "2016-06-24",
        "eval_end": "2016-07-03",
    },
    {
        "fold": "early_july",
        "train_end": "2016-07-03",
        "eval_start": "2016-07-04",
        "eval_end": "2016-07-13",
    },
]


def top_fraction_capture(y_true: np.ndarray, proba: np.ndarray, fraction: float = 0.20) -> float:
    n = max(1, int(np.ceil(len(proba) * fraction)))
    order = np.argsort(-proba)
    positives = np.asarray(y_true).sum()
    if positives == 0:
        return float("nan")
    return float(np.asarray(y_true)[order[:n]].sum() / positives)


def make_logistic() -> Pipeline:
    prep = ColumnTransformer(
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
    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    return Pipeline([("prep", prep), ("model", model)])


def make_random_forest() -> Pipeline:
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                FEATURES,
            )
        ],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("prep", prep), ("model", model)])


def make_xgboost(y_train: pd.Series) -> Pipeline:
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = negatives / max(positives, 1)

    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                FEATURES,
            )
        ],
        remainder="drop",
    )
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("prep", prep), ("model", model)])


def evaluate_model(name: str, model: Pipeline, x_train, y_train, x_eval, y_eval) -> dict:
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_eval)[:, 1]

    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_eval, proba)),
        "average_precision": float(average_precision_score(y_eval, proba)),
        "brier": float(brier_score_loss(y_eval, proba)),
        "top20_capture": top_fraction_capture(y_eval.to_numpy(), proba, 0.20),
        "mean_predicted_risk": float(np.mean(proba)),
        "observed_delinquency_rate": float(np.mean(y_eval)),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df[DATE] = pd.to_datetime(df[DATE], errors="raise")

    development = df.loc[df[DATE] <= DEVELOPMENT_END].copy()
    if development.empty:
        raise ValueError("Development population is empty.")

    required = {DATE, TARGET, *FEATURES}
    missing = sorted(required.difference(development.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment("module4_candidate_model_benchmark")

    records = []

    for fold in FOLDS:
        train_end = pd.Timestamp(fold["train_end"])
        eval_start = pd.Timestamp(fold["eval_start"])
        eval_end = pd.Timestamp(fold["eval_end"])

        train = development.loc[development[DATE] <= train_end].copy()
        evaluation = development.loc[
            (development[DATE] >= eval_start) & (development[DATE] <= eval_end)
        ].copy()

        if train.empty or evaluation.empty:
            raise ValueError(f"Empty train/evaluation population in {fold['fold']}")

        x_train = train[FEATURES]
        y_train = train[TARGET]
        x_eval = evaluation[FEATURES]
        y_eval = evaluation[TARGET]

        models = {
            "logistic_regression": make_logistic(),
            "random_forest": make_random_forest(),
            "xgboost": make_xgboost(y_train),
        }

        for model_name, model in models.items():
            with mlflow.start_run(run_name=f"{model_name}_{fold['fold']}"):
                result = evaluate_model(
                    model_name,
                    model,
                    x_train,
                    y_train,
                    x_eval,
                    y_eval,
                )

                row = {
                    "fold": fold["fold"],
                    "train_start": str(train[DATE].min().date()),
                    "train_end": str(train[DATE].max().date()),
                    "eval_start": str(evaluation[DATE].min().date()),
                    "eval_end": str(evaluation[DATE].max().date()),
                    "train_rows": int(len(train)),
                    "eval_rows": int(len(evaluation)),
                    **result,
                }
                records.append(row)

                mlflow.log_params(
                    {
                        "model": model_name,
                        "feature_count": len(FEATURES),
                        "fold": fold["fold"],
                        "train_end": row["train_end"],
                        "eval_start": row["eval_start"],
                        "eval_end": row["eval_end"],
                    }
                )
                mlflow.log_metrics(
                    {
                        "roc_auc": row["roc_auc"],
                        "average_precision": row["average_precision"],
                        "brier": row["brier"],
                        "top20_capture": row["top20_capture"],
                        "mean_predicted_risk": row["mean_predicted_risk"],
                        "observed_delinquency_rate": row["observed_delinquency_rate"],
                    }
                )

    fold_metrics = pd.DataFrame(records)
    summary = (
        fold_metrics.groupby("model", as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            min_roc_auc=("roc_auc", "min"),
            mean_average_precision=("average_precision", "mean"),
            mean_brier=("brier", "mean"),
            mean_top20_capture=("top20_capture", "mean"),
            min_top20_capture=("top20_capture", "min"),
        )
        .sort_values(["mean_roc_auc", "mean_top20_capture"], ascending=False)
        .reset_index(drop=True)
    )

    FOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(FOLD_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    payload = {
        "development_rows": int(len(development)),
        "development_start": str(development[DATE].min().date()),
        "development_end": str(development[DATE].max().date()),
        "features": FEATURES,
        "holdout_opened": False,
        "folds": FOLDS,
        "summary": summary.to_dict(orient="records"),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Module 4 candidate-model benchmark completed.")
    print("Final 14-23 July holdout opened: False")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Fold metrics written to: {FOLD_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Summary written to: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"JSON summary written to: {JSON_PATH.relative_to(PROJECT_ROOT)}")
    print(f"MLflow tracking directory: {(PROJECT_ROOT / 'mlruns').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
