"""Compare model performance across three candidate feature-set variants.

The comparison uses the same calendar-aware forward-chaining design introduced during
feature selection. Each evaluation trains only on earlier dates and evaluates on the
next chronological period.

Variants:
- full_non_history: all Stage 1-approved predictors except the two left-censored
  customer-history variables;
- core_plus_secondary: the consolidated core and secondary candidates emerging from
  Stages 2-4;
- compact_core: the smallest cross-method core candidate set.

Metrics include discrimination (ROC-AUC, average precision), calibration (Brier score,
expected calibration error, mean prediction minus observed rate), and the business-
oriented capture rate in the highest-risk 20% of observations.

This script is diagnostic and does not automatically declare a final model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

TARGET = "delinquent_5d"
DATE_FIELD = "pdate"
RANDOM_STATE = 42

FOLDS = [
    ("june_early", "2016-06-01", "2016-06-13"),
    ("june_late", "2016-06-14", "2016-06-30"),
    ("july_early", "2016-07-01", "2016-07-13"),
    ("july_late", "2016-07-14", "2016-07-23"),
]

CORE_FEATURES = [
    "cnt_ma_rech90",
    "daily_decr30",
    "last_rech_date_ma",
    "sumamnt_ma_rech90",
    "aon",
    "last_rech_amt_ma",
]

SECONDARY_FEATURES = [
    "daily_decr90",
    "sumamnt_ma_rech30",
    "medianamnt_ma_rech30",
    "medianmarechprebal90",
    "rental30",
    "cnt_ma_rech30",
]

HISTORY_FEATURES = {"prior_tx_count", "is_repeat_customer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("reports/tables/model_variant_fold_metrics.csv"),
    )
    parser.add_argument(
        "--summary-table",
        type=Path,
        default=Path("reports/tables/model_variant_summary.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/model_variant_comparison_summary.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("reports/figures/model_variant_comparison.png"),
    )
    return parser.parse_args()


def _fold_mask(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    return (df[DATE_FIELD] >= pd.Timestamp(start)) & (df[DATE_FIELD] <= pd.Timestamp(end))


def _build_model(y: pd.Series) -> XGBClassifier:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    scale_pos_weight = negatives / max(positives, 1)
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )


def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.digitize(probs, edges[1:-1], right=False)
    ece = 0.0
    n = len(y_true)
    for bin_id in range(bins):
        mask = ids == bin_id
        if not np.any(mask):
            continue
        observed = float(np.mean(y_true[mask]))
        predicted = float(np.mean(probs[mask]))
        ece += (int(mask.sum()) / n) * abs(observed - predicted)
    return float(ece)


def _top_fraction_capture(y_true: np.ndarray, probs: np.ndarray, fraction: float = 0.20) -> float:
    if y_true.sum() == 0:
        return 0.0
    n_top = max(1, int(np.ceil(len(y_true) * fraction)))
    order = np.argsort(-probs)
    captured = y_true[order[:n_top]].sum()
    return float(captured / y_true.sum())


def _evaluate_variant(df: pd.DataFrame, variant: str, features: list[str]) -> list[dict]:
    rows: list[dict] = []

    for eval_idx in range(1, len(FOLDS)):
        eval_name, eval_start, eval_end = FOLDS[eval_idx]
        train_end = FOLDS[eval_idx - 1][2]

        train = df[df[DATE_FIELD] <= pd.Timestamp(train_end)].copy()
        evaluation = df[_fold_mask(df, eval_start, eval_end)].copy()

        X_train = train[features]
        y_train = train[TARGET].astype(int)
        X_eval = evaluation[features]
        y_eval = evaluation[TARGET].astype(int)

        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train), columns=features, index=X_train.index
        )
        X_eval_imp = pd.DataFrame(
            imputer.transform(X_eval), columns=features, index=X_eval.index
        )

        model = _build_model(y_train)
        model.fit(X_train_imp, y_train)
        probs = model.predict_proba(X_eval_imp)[:, 1]

        y_np = y_eval.to_numpy(dtype=int)
        observed_rate = float(np.mean(y_np))
        mean_prediction = float(np.mean(probs))

        rows.append(
            {
                "variant": variant,
                "feature_count": int(len(features)),
                "evaluation_fold": eval_name,
                "train_end": train_end,
                "train_rows": int(len(train)),
                "evaluation_rows": int(len(evaluation)),
                "evaluation_delinquency_rate": observed_rate,
                "roc_auc": float(roc_auc_score(y_eval, probs)),
                "average_precision": float(average_precision_score(y_eval, probs)),
                "brier_score": float(brier_score_loss(y_eval, probs)),
                "expected_calibration_error_10bin": _expected_calibration_error(y_np, probs),
                "mean_predicted_risk": mean_prediction,
                "calibration_in_the_large_gap": float(mean_prediction - observed_rate),
                "top_20pct_capture_rate": _top_fraction_capture(y_np, probs, 0.20),
            }
        )

    return rows


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

    stage1_features = [c for c in df.columns if c not in {DATE_FIELD, TARGET}]
    full_non_history = [c for c in stage1_features if c not in HISTORY_FEATURES]
    core_plus_secondary = CORE_FEATURES + SECONDARY_FEATURES
    compact_core = CORE_FEATURES.copy()

    variants = {
        "full_non_history": full_non_history,
        "core_plus_secondary": core_plus_secondary,
        "compact_core": compact_core,
    }

    for name, features in variants.items():
        missing_features = sorted(set(features) - set(df.columns))
        if missing_features:
            raise ValueError(f"Variant {name} is missing columns: {missing_features}")

    rows: list[dict] = []
    for variant, features in variants.items():
        rows.extend(_evaluate_variant(df, variant, features))

    metrics = pd.DataFrame(rows)
    summary_table = (
        metrics.groupby("variant", as_index=False)
        .agg(
            feature_count=("feature_count", "first"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
            mean_brier_score=("brier_score", "mean"),
            std_brier_score=("brier_score", "std"),
            mean_ece=("expected_calibration_error_10bin", "mean"),
            mean_abs_calibration_gap=("calibration_in_the_large_gap", lambda s: float(np.mean(np.abs(s)))),
            mean_top20_capture=("top_20pct_capture_rate", "mean"),
            std_top20_capture=("top_20pct_capture_rate", "std"),
        )
        .sort_values(["mean_roc_auc", "mean_average_precision"], ascending=False, kind="stable")
    )

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.summary_table.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(args.metrics, index=False)
    summary_table.to_csv(args.summary_table, index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    positions = np.arange(len(summary_table))
    ax.bar(positions - 0.18, summary_table["mean_roc_auc"], width=0.36, label="ROC-AUC")
    ax.bar(positions + 0.18, summary_table["mean_average_precision"], width=0.36, label="Average precision")
    ax.set_xticks(positions)
    ax.set_xticklabels(summary_table["variant"], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean forward-chaining score")
    ax.set_title("Candidate feature-set comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_rows": int(len(df)),
        "variants": {name: features for name, features in variants.items()},
        "forward_chaining_fold_metrics": metrics.to_dict(orient="records"),
        "variant_summary": summary_table.to_dict(orient="records"),
        "automatic_final_model_choice": False,
        "interpretation_note": (
            "The three feature-set variants are compared with identical XGBoost settings and "
            "forward-chaining folds. Discrimination, calibration, stability, and top-20% capture "
            "are assessed together. History-derived variables are excluded from the primary "
            "comparison because their source window is left-censored."
        ),
    }

    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Per-fold variant metrics written to {args.metrics}")
    print(f"Variant summary table written to {args.summary_table}")
    print(f"Variant comparison figure written to {args.figure}")
    print(f"Variant comparison summary written to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
