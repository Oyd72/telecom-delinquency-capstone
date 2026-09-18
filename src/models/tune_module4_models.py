"""Tune the two nonlinear Module 4 challengers using chronological grid search.

The final 14-23 July holdout remains untouched.

This script evaluates a deliberately small, transparent parameter grid for:
- Random Forest
- XGBoost

Each configuration is evaluated on the same three expanding chronological folds used
for the initial benchmark. The purpose is to test whether modest tuning improves the
challengers without turning the exercise into an unnecessarily large search.

All runs are logged to the local MLflow SQLite backend.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"
FOLD_PATH = PROJECT_ROOT / "reports/tables/module4_tuning_fold_metrics.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports/tables/module4_tuning_summary.csv"
JSON_PATH = PROJECT_ROOT / "reports/tables/module4_tuning_summary.json"

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

RF_GRID = [
    {"max_depth": 8, "min_samples_leaf": 20, "max_features": "sqrt"},
    {"max_depth": 12, "min_samples_leaf": 20, "max_features": "sqrt"},
    {"max_depth": 12, "min_samples_leaf": 10, "max_features": 0.7},
    {"max_depth": None, "min_samples_leaf": 20, "max_features": "sqrt"},
]

XGB_GRID = [
    {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 10},
]


def top_fraction_capture(y_true: np.ndarray, proba: np.ndarray, fraction: float = 0.20) -> float:
    n = max(1, int(np.ceil(len(proba) * fraction)))
    order = np.argsort(-proba)
    positives = np.asarray(y_true).sum()
    if positives == 0:
        return float("nan")
    return float(np.asarray(y_true)[order[:n]].sum() / positives)


def make_preprocessor() -> ColumnTransformer:
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


def make_rf(params: dict) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
        **params,
    )
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


def make_xgb(params: dict, y_train: pd.Series) -> Pipeline:
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=300,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=negatives / max(positives, 1),
        n_jobs=-1,
        random_state=42,
        **params,
    )
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


def evaluate(model: Pipeline, x_train, y_train, x_eval, y_eval) -> dict:
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_eval)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_eval, proba)),
        "average_precision": float(average_precision_score(y_eval, proba)),
        "brier": float(brier_score_loss(y_eval, proba)),
        "top20_capture": top_fraction_capture(y_eval.to_numpy(), proba),
    }


def config_id(model_name: str, index: int) -> str:
    return f"{model_name}_{index:02d}"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df[DATE] = pd.to_datetime(df[DATE], errors="raise")
    development = df.loc[df[DATE] <= DEVELOPMENT_END].copy()

    mlflow_db = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
    mlflow.set_experiment("module4_model_tuning")

    records = []

    searches = [
        ("random_forest", RF_GRID),
        ("xgboost", XGB_GRID),
    ]

    for model_name, grid in searches:
        for idx, params in enumerate(grid, start=1):
            cid = config_id(model_name, idx)

            for fold in FOLDS:
                train_end = pd.Timestamp(fold["train_end"])
                eval_start = pd.Timestamp(fold["eval_start"])
                eval_end = pd.Timestamp(fold["eval_end"])

                train = development.loc[development[DATE] <= train_end].copy()
                evaluation = development.loc[
                    (development[DATE] >= eval_start)
                    & (development[DATE] <= eval_end)
                ].copy()

                x_train = train[FEATURES]
                y_train = train[TARGET]
                x_eval = evaluation[FEATURES]
                y_eval = evaluation[TARGET]

                if model_name == "random_forest":
                    model = make_rf(params)
                else:
                    model = make_xgb(params, y_train)

                with mlflow.start_run(run_name=f"{cid}_{fold['fold']}"):
                    metrics = evaluate(model, x_train, y_train, x_eval, y_eval)

                    row = {
                        "model": model_name,
                        "config_id": cid,
                        "fold": fold["fold"],
                        "train_rows": int(len(train)),
                        "eval_rows": int(len(evaluation)),
                        **params,
                        **metrics,
                    }
                    records.append(row)

                    mlflow.log_params(
                        {
                            "model": model_name,
                            "config_id": cid,
                            "fold": fold["fold"],
                            "feature_count": len(FEATURES),
                            **params,
                        }
                    )
                    mlflow.log_metrics(metrics)

    fold_metrics = pd.DataFrame(records)

    summary = (
        fold_metrics.groupby(["model", "config_id"], as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            min_roc_auc=("roc_auc", "min"),
            mean_average_precision=("average_precision", "mean"),
            mean_brier=("brier", "mean"),
            mean_top20_capture=("top20_capture", "mean"),
            min_top20_capture=("top20_capture", "min"),
        )
        .sort_values(
            ["mean_roc_auc", "mean_top20_capture", "mean_brier"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    param_rows = (
        fold_metrics.sort_values(["model", "config_id", "fold"])
        .groupby(["model", "config_id"], as_index=False)
        .first()
    )
    param_columns = [
        c
        for c in ["model", "config_id", "max_depth", "min_samples_leaf", "max_features", "learning_rate", "min_child_weight"]
        if c in param_rows.columns
    ]
    params_lookup = param_rows[param_columns].copy()
    summary = summary.merge(params_lookup, on=["model", "config_id"], how="left")

    FOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(FOLD_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    best_by_model = (
        summary.sort_values(
            ["model", "mean_roc_auc", "mean_top20_capture", "mean_brier"],
            ascending=[True, False, False, True],
        )
        .groupby("model", as_index=False)
        .first()
    )

    payload = {
        "development_rows": int(len(development)),
        "holdout_opened": False,
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "rf_grid": RF_GRID,
        "xgb_grid": XGB_GRID,
        "best_by_model": best_by_model.to_dict(orient="records"),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("Module 4 chronological tuning completed.")
    print("Final 14-23 July holdout opened: False")
    print()
    print("Best configuration by model")
    print(best_by_model.to_string(index=False))
    print()
    print("All configurations")
    print(summary.to_string(index=False))
    print()
    print(f"Fold metrics written to: {FOLD_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Summary written to: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"JSON summary written to: {JSON_PATH.relative_to(PROJECT_ROOT)}")
    print(f"MLflow tracking database: {mlflow_db.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
