"""Assess leakage-safe probability calibration using development data only.

This Module 4 experiment compares uncalibrated, Platt-scaled, and isotonic
probabilities for the two tuned nonlinear challengers:

- Random Forest: max_depth=8, min_samples_leaf=20, max_features='sqrt'
- XGBoost: max_depth=5, learning_rate=0.05, min_child_weight=5

For every evaluation fold:
1. the base model is fitted only on earlier observations;
2. the calibrator is fitted on the immediately preceding 7 calendar days;
3. performance is measured only on a later development-period evaluation block.

The 14-23 July final holdout is not used by this script.

Important methodological note:
The need to revisit calibration became especially visible after the final holdout
evaluation. Therefore any later re-evaluation of that same holdout with a chosen
calibration method is confirmatory/sensitivity evidence, not a second independent
final test. Calibration-method selection in this script is based only on development
data.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"
FOLD_PATH = PROJECT_ROOT / "reports/tables/module4_calibration_fold_metrics.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports/tables/module4_calibration_summary.csv"
JSON_PATH = PROJECT_ROOT / "reports/tables/module4_calibration_summary.json"

TARGET = "delinquent_5d"
DATE = "pdate"
CALIBRATION_DAYS = 7

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


def clip_probabilities(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def logit(p: np.ndarray) -> np.ndarray:
    p = clip_probabilities(p)
    return np.log(p / (1 - p))


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
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
        ece += (mask.sum() / total) * abs(
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
    fold: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    train_rows: int,
    calibration_rows: int,
    evaluation_rows: int,
    train_end: str,
    calibration_start: str,
    calibration_end: str,
    eval_start: str,
    eval_end: str,
) -> dict:
    observed = float(np.mean(y_true))
    predicted = float(np.mean(probabilities))

    return {
        "model": model_name,
        "method": method,
        "fold": fold,
        "train_rows": int(train_rows),
        "calibration_rows": int(calibration_rows),
        "evaluation_rows": int(evaluation_rows),
        "train_end": train_end,
        "calibration_start": calibration_start,
        "calibration_end": calibration_end,
        "eval_start": eval_start,
        "eval_end": eval_end,
        "observed_delinquency_rate": observed,
        "mean_predicted_risk": predicted,
        "calibration_in_the_large_error": float(predicted - observed),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "ece_10bin": expected_calibration_error(y_true, probabilities),
        "top20_capture": top20_capture(y_true, probabilities),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df[DATE] = pd.to_datetime(df[DATE], errors="raise")

    required = {DATE, TARGET, *FEATURES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Enforce development-only use.
    development = df.loc[df[DATE] <= pd.Timestamp("2016-07-13")].copy()
    assert development[DATE].max() == pd.Timestamp("2016-07-13")

    mlflow_db = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
    mlflow.set_experiment("module4_development_calibration")

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
            raise ValueError(f"Empty split in fold {fold['fold']}")
        if calibration[TARGET].nunique() < 2:
            raise ValueError(
                f"Calibration window in fold {fold['fold']} contains one class only"
            )

        x_train = train[FEATURES]
        y_train = train[TARGET]
        x_cal = calibration[FEATURES]
        y_cal = calibration[TARGET].to_numpy()
        x_eval = evaluation[FEATURES]
        y_eval = evaluation[TARGET].to_numpy()

        models = {
            "random_forest": make_random_forest(),
            "xgboost": make_xgboost(y_train),
        }

        for model_name, model in models.items():
            model.fit(x_train, y_train)

            p_cal_raw = clip_probabilities(model.predict_proba(x_cal)[:, 1])
            p_eval_raw = clip_probabilities(model.predict_proba(x_eval)[:, 1])

            platt = LogisticRegression(random_state=42, solver="lbfgs")
            platt.fit(logit(p_cal_raw).reshape(-1, 1), y_cal)
            p_eval_platt = platt.predict_proba(
                logit(p_eval_raw).reshape(-1, 1)
            )[:, 1]

            isotonic = IsotonicRegression(out_of_bounds="clip")
            isotonic.fit(p_cal_raw, y_cal)
            p_eval_isotonic = clip_probabilities(
                isotonic.predict(p_eval_raw)
            )

            common = {
                "model_name": model_name,
                "fold": fold["fold"],
                "y_true": y_eval,
                "train_rows": len(train),
                "calibration_rows": len(calibration),
                "evaluation_rows": len(evaluation),
                "train_end": str(train_end.date()),
                "calibration_start": str(calibration_start.date()),
                "calibration_end": str(calibration_end.date()),
                "eval_start": str(eval_start.date()),
                "eval_end": str(eval_end.date()),
            }

            method_probabilities = {
                "uncalibrated": p_eval_raw,
                "platt": p_eval_platt,
                "isotonic": p_eval_isotonic,
            }

            for method, probabilities in method_probabilities.items():
                row = metric_row(
                    method=method,
                    probabilities=probabilities,
                    **common,
                )
                rows.append(row)

                with mlflow.start_run(
                    run_name=f"{model_name}_{method}_{fold['fold']}"
                ):
                    mlflow.log_params(
                        {
                            "model": model_name,
                            "calibration_method": method,
                            "fold": fold["fold"],
                            "calibration_days": CALIBRATION_DAYS,
                            "feature_count": len(FEATURES),
                            "development_only": True,
                        }
                    )
                    mlflow.log_metrics(
                        {
                            "roc_auc": row["roc_auc"],
                            "average_precision": row["average_precision"],
                            "brier": row["brier"],
                            "ece_10bin": row["ece_10bin"],
                            "top20_capture": row["top20_capture"],
                            "mean_predicted_risk": row["mean_predicted_risk"],
                            "observed_delinquency_rate": row[
                                "observed_delinquency_rate"
                            ],
                            "abs_calibration_gap": abs(
                                row["calibration_in_the_large_error"]
                            ),
                        }
                    )

    fold_metrics = pd.DataFrame(rows)

    summary = (
        fold_metrics.groupby(["model", "method"], as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            min_roc_auc=("roc_auc", "min"),
            mean_average_precision=("average_precision", "mean"),
            mean_brier=("brier", "mean"),
            mean_ece=("ece_10bin", "mean"),
            mean_abs_calibration_gap=(
                "calibration_in_the_large_error",
                lambda s: float(np.abs(s).mean()),
            ),
            mean_top20_capture=("top20_capture", "mean"),
        )
        .sort_values(
            ["model", "mean_brier", "mean_ece"],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )

    best_by_model = (
        summary.sort_values(
            ["model", "mean_brier", "mean_ece", "mean_abs_calibration_gap"],
            ascending=[True, True, True, True],
        )
        .groupby("model", as_index=False)
        .first()
    )

    FOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(FOLD_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    payload = {
        "development_only": True,
        "final_holdout_used": False,
        "development_end": "2016-07-13",
        "calibration_window_days": CALIBRATION_DAYS,
        "features": FEATURES,
        "folds": EVALUATION_FOLDS,
        "best_by_model": best_by_model.to_dict(orient="records"),
        "method_summary": summary.to_dict(orient="records"),
        "method_selection_rule": (
            "Choose calibration using development-only mean Brier score as the primary "
            "criterion, with mean ECE and mean absolute calibration gap as tie-breakers. "
            "Ranking metrics are monitored to ensure calibration does not materially "
            "damage discrimination."
        ),
        "methodological_note": (
            "Calibration assessment was initiated after the final holdout exposed "
            "probability overprediction. Method selection here uses development data only. "
            "Any later evaluation on the already-opened holdout is confirmatory/sensitivity "
            "evidence and is not an independent final test."
        ),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Module 4 development-only calibration assessment completed.")
    print("Final 14-23 July holdout used: False")
    print()
    print("Best calibration method by model")
    print(best_by_model.to_string(index=False))
    print()
    print("All calibration summaries")
    print(summary.to_string(index=False))
    print()
    print(f"Fold metrics written to: {FOLD_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Summary written to: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"JSON summary written to: {JSON_PATH.relative_to(PROJECT_ROOT)}")
    print(f"MLflow tracking database: {mlflow_db.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
