"""Run Stage 3 embedded and wrapper feature-selection diagnostics.

This script complements the Stage 2 filter analysis by assessing predictors jointly with
logistic-regression-based methods across the same calendar-aware chronological folds.

Methods
-------
1. L1-regularised logistic regression (embedded selection):
   - median imputation and standardisation are fitted within each fold;
   - logistic-regression regularisation strength is selected by internal stratified CV;
   - non-zero coefficients are recorded as selected features.
2. Recursive Feature Elimination (RFE; wrapper method):
   - the same fold-local preprocessing is used;
   - a logistic-regression estimator ranks features jointly;
   - the top half of predictors is retained as a diagnostic selection target.

The script does not declare a final feature set. It reports selection frequency,
coefficient stability, and RFE stability across folds. History variables affected by
left-censoring are also summarised separately for sensitivity analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


TARGET = "delinquent_5d"
DATE_FIELD = "pdate"
RANDOM_STATE = 42
HISTORY_FEATURES = {"prior_tx_count", "is_repeat_customer"}

FOLDS = [
    ("june_early", "2016-06-01", "2016-06-13"),
    ("june_late", "2016-06-14", "2016-06-30"),
    ("july_early", "2016-07-01", "2016-07-13"),
    ("july_late", "2016-07-14", "2016-07-23"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--per-fold",
        type=Path,
        default=Path("reports/tables/embedded_wrapper_fold_metrics.csv"),
    )
    parser.add_argument(
        "--summary-table",
        type=Path,
        default=Path("reports/tables/embedded_wrapper_stability.csv"),
    )
    parser.add_argument(
        "--sensitivity-table",
        type=Path,
        default=Path("reports/tables/embedded_wrapper_without_history.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/embedded_wrapper_summary.json"),
    )
    return parser.parse_args()


def _fold_slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return df.loc[df[DATE_FIELD].between(start_ts, end_ts, inclusive="both")].copy()


def _prepare_matrix(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = frame[features]
    y = frame[TARGET].astype(int).to_numpy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled, y


def _l1_selection(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 8),
        cv=cv,
        penalty="l1",
        solver="liblinear",
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )
    model.fit(X, y)
    coef = model.coef_.ravel()
    selected = np.abs(coef) > 1e-10
    chosen_c = float(np.ravel(model.C_)[0])
    return coef, chosen_c, selected


def _rfe_selection(X: np.ndarray, y: np.ndarray, feature_count: int) -> tuple[np.ndarray, np.ndarray]:
    n_select = max(1, int(np.ceil(feature_count / 2)))
    estimator = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )
    selector = RFE(
        estimator=estimator,
        n_features_to_select=n_select,
        step=1,
    )
    selector.fit(X, y)
    return selector.support_.astype(bool), selector.ranking_.astype(int)


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.data)

    required = {DATE_FIELD, TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[DATE_FIELD] = pd.to_datetime(df[DATE_FIELD], errors="coerce")
    if df[DATE_FIELD].isna().any():
        raise ValueError("Unparseable pdate values found in model-ready data.")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must contain only 0/1 values.")

    features = [c for c in df.columns if c not in {DATE_FIELD, TARGET}]
    if not features:
        raise ValueError("No predictor columns found.")

    rows: list[dict] = []
    fold_overview: list[dict] = []

    for fold_name, start, end in FOLDS:
        fold = _fold_slice(df, start, end)
        if fold.empty:
            raise ValueError(f"Fold {fold_name} is empty.")

        X, y = _prepare_matrix(fold, features)
        coef, chosen_c, l1_selected = _l1_selection(X, y)
        rfe_selected, rfe_ranking = _rfe_selection(X, y, len(features))

        fold_overview.append(
            {
                "fold": fold_name,
                "start": start,
                "end": end,
                "rows": int(len(fold)),
                "delinquency_rate": float(fold[TARGET].mean()),
                "chosen_l1_C": chosen_c,
                "l1_selected_count": int(l1_selected.sum()),
                "rfe_selected_count": int(rfe_selected.sum()),
            }
        )

        for idx, feature in enumerate(features):
            rows.append(
                {
                    "fold": fold_name,
                    "feature": feature,
                    "l1_coefficient": float(coef[idx]),
                    "abs_l1_coefficient": float(abs(coef[idx])),
                    "l1_selected": bool(l1_selected[idx]),
                    "rfe_selected": bool(rfe_selected[idx]),
                    "rfe_rank": int(rfe_ranking[idx]),
                    "history_sensitivity_feature": bool(feature in HISTORY_FEATURES),
                }
            )

    per_fold = pd.DataFrame(rows)

    stability = (
        per_fold.groupby("feature", as_index=False)
        .agg(
            l1_selection_frequency=("l1_selected", "mean"),
            mean_abs_l1_coefficient=("abs_l1_coefficient", "mean"),
            std_abs_l1_coefficient=("abs_l1_coefficient", "std"),
            coefficient_sign_consistency=(
                "l1_coefficient",
                lambda s: float(max((s > 0).mean(), (s < 0).mean(), (s == 0).mean())),
            ),
            rfe_selection_frequency=("rfe_selected", "mean"),
            mean_rfe_rank=("rfe_rank", "mean"),
            std_rfe_rank=("rfe_rank", "std"),
        )
        .sort_values(
            ["l1_selection_frequency", "rfe_selection_frequency", "mean_abs_l1_coefficient"],
            ascending=[False, False, False],
            kind="stable",
        )
    )

    stability["history_sensitivity_feature"] = stability["feature"].isin(HISTORY_FEATURES)
    sensitivity = stability.loc[~stability["history_sensitivity_feature"]].copy()

    args.per_fold.parent.mkdir(parents=True, exist_ok=True)
    args.summary_table.parent.mkdir(parents=True, exist_ok=True)
    args.sensitivity_table.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    per_fold.to_csv(args.per_fold, index=False)
    stability.to_csv(args.summary_table, index=False)
    sensitivity.to_csv(args.sensitivity_table, index=False)

    fully_l1 = stability.loc[stability["l1_selection_frequency"] == 1.0, "feature"].tolist()
    fully_rfe = stability.loc[stability["rfe_selection_frequency"] == 1.0, "feature"].tolist()
    fully_both = stability.loc[
        (stability["l1_selection_frequency"] == 1.0)
        & (stability["rfe_selection_frequency"] == 1.0),
        "feature",
    ].tolist()

    summary = {
        "input_rows": int(len(df)),
        "feature_count": int(len(features)),
        "folds": fold_overview,
        "rfe_selection_target_per_fold": int(max(1, np.ceil(len(features) / 2))),
        "history_features_treated_as_sensitivity": sorted(HISTORY_FEATURES),
        "features_selected_by_l1_in_all_folds": fully_l1,
        "features_selected_by_rfe_in_all_folds": fully_rfe,
        "features_selected_by_both_in_all_folds": fully_both,
        "top_10_joint_stability": stability.head(10)[
            [
                "feature",
                "l1_selection_frequency",
                "mean_abs_l1_coefficient",
                "coefficient_sign_consistency",
                "rfe_selection_frequency",
                "mean_rfe_rank",
            ]
        ].to_dict(orient="records"),
        "automatic_feature_removal": False,
        "interpretation_note": (
            "L1 and RFE results are compared across the same calendar-aware folds used in Stage 2. "
            "No final feature set is declared from one method alone. History variables remain a "
            "separate sensitivity analysis because of left-censoring."
        ),
    }

    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Per-fold embedded/wrapper metrics written to {args.per_fold}")
    print(f"Stability summary written to {args.summary_table}")
    print(f"Sensitivity summary written to {args.sensitivity_table}")
    print(f"Stage 3 summary written to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
