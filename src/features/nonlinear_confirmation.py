"""Stage 4 nonlinear and model-agnostic feature confirmation.

This script evaluates the Stage 1 candidate predictors with forward-chaining temporal
validation. It trains an XGBoost classifier on earlier calendar-aware folds and
assesses feature contribution on the next chronological fold using both permutation
importance and SHAP.

The purpose is confirmation and stability analysis, not automatic feature removal.
History-derived variables are also evaluated in a sensitivity run because of the
left-censoring documented elsewhere in the project.
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
import shap
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier


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
        "--metrics",
        type=Path,
        default=Path("reports/tables/nonlinear_fold_metrics.csv"),
    )
    parser.add_argument(
        "--stability",
        type=Path,
        default=Path("reports/tables/nonlinear_importance_stability.csv"),
    )
    parser.add_argument(
        "--sensitivity",
        type=Path,
        default=Path("reports/tables/nonlinear_without_history.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/nonlinear_confirmation_summary.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("reports/figures/nonlinear_importance_stability.png"),
    )
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=3000,
        help="Maximum evaluation rows per fold used for SHAP calculation.",
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


def _evaluate_path(
    df: pd.DataFrame,
    features: list[str],
    path_name: str,
    shap_sample_size: int,
) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    fold_summaries: list[dict] = []

    for eval_idx in range(1, len(FOLDS)):
        eval_name, eval_start, eval_end = FOLDS[eval_idx]
        train_end = FOLDS[eval_idx - 1][2]

        train = df[df[DATE_FIELD] <= pd.Timestamp(train_end)].copy()
        evaluation = df[_fold_mask(df, eval_start, eval_end)].copy()

        if train.empty or evaluation.empty:
            raise ValueError(f"Empty train/evaluation set for {eval_name}")

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

        probabilities = model.predict_proba(X_eval_imp)[:, 1]
        roc_auc = float(roc_auc_score(y_eval, probabilities))
        pr_auc = float(average_precision_score(y_eval, probabilities))

        perm = permutation_importance(
            model,
            X_eval_imp,
            y_eval,
            scoring="roc_auc",
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        if len(X_eval_imp) > shap_sample_size:
            shap_sample = X_eval_imp.sample(
                n=shap_sample_size, random_state=RANDOM_STATE
            )
        else:
            shap_sample = X_eval_imp

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(shap_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        mean_abs_shap = np.abs(np.asarray(shap_values)).mean(axis=0)

        perm_rank = pd.Series(-perm.importances_mean, index=features).rank(
            method="min", ascending=True
        )
        shap_rank = pd.Series(-mean_abs_shap, index=features).rank(
            method="min", ascending=True
        )

        for idx, feature in enumerate(features):
            rows.append(
                {
                    "path": path_name,
                    "evaluation_fold": eval_name,
                    "train_end": train_end,
                    "train_rows": int(len(train)),
                    "evaluation_rows": int(len(evaluation)),
                    "feature": feature,
                    "permutation_importance_mean": float(perm.importances_mean[idx]),
                    "permutation_importance_std": float(perm.importances_std[idx]),
                    "permutation_rank": float(perm_rank[feature]),
                    "mean_abs_shap": float(mean_abs_shap[idx]),
                    "shap_rank": float(shap_rank[feature]),
                }
            )

        fold_summaries.append(
            {
                "path": path_name,
                "evaluation_fold": eval_name,
                "train_end": train_end,
                "train_rows": int(len(train)),
                "evaluation_rows": int(len(evaluation)),
                "evaluation_delinquency_rate": float(y_eval.mean()),
                "roc_auc": roc_auc,
                "average_precision": pr_auc,
            }
        )

    return pd.DataFrame(rows), fold_summaries


def _summarise_stability(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics.groupby(["path", "feature"], as_index=False)
        .agg(
            mean_permutation_importance=("permutation_importance_mean", "mean"),
            std_permutation_importance=("permutation_importance_mean", "std"),
            mean_permutation_rank=("permutation_rank", "mean"),
            std_permutation_rank=("permutation_rank", "std"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            std_abs_shap=("mean_abs_shap", "std"),
            mean_shap_rank=("shap_rank", "mean"),
            std_shap_rank=("shap_rank", "std"),
        )
    )
    grouped["joint_mean_rank"] = (
        grouped["mean_permutation_rank"] + grouped["mean_shap_rank"]
    ) / 2.0
    return grouped.sort_values(
        ["path", "joint_mean_rank", "mean_abs_shap"],
        ascending=[True, True, False],
        kind="stable",
    )


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

    all_features = [c for c in df.columns if c not in {DATE_FIELD, TARGET}]
    no_history_features = [c for c in all_features if c not in HISTORY_FEATURES]

    full_metrics, full_fold_summaries = _evaluate_path(
        df, all_features, "all_features", args.shap_sample_size
    )
    sensitivity_metrics, sensitivity_fold_summaries = _evaluate_path(
        df, no_history_features, "without_history", args.shap_sample_size
    )

    all_metrics = pd.concat([full_metrics, sensitivity_metrics], ignore_index=True)
    stability = _summarise_stability(all_metrics)
    sensitivity_stability = stability[stability["path"] == "without_history"].copy()

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.stability.parent.mkdir(parents=True, exist_ok=True)
    args.sensitivity.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)

    all_metrics.to_csv(args.metrics, index=False)
    stability.to_csv(args.stability, index=False)
    sensitivity_stability.to_csv(args.sensitivity, index=False)

    plot_data = stability[stability["path"] == "all_features"].nsmallest(
        12, "joint_mean_rank"
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    positions = np.arange(len(plot_data))
    ax.barh(positions - 0.18, plot_data["mean_permutation_rank"], height=0.36, label="Permutation rank")
    ax.barh(positions + 0.18, plot_data["mean_shap_rank"], height=0.36, label="SHAP rank")
    ax.set_yticks(positions)
    ax.set_yticklabels(plot_data["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean rank across forward-chaining evaluations (lower is better)")
    ax.set_title("Stage 4 nonlinear feature-importance stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    full_stability = stability[stability["path"] == "all_features"]
    sensitivity_top = sensitivity_stability.head(10)

    summary = {
        "input_rows": int(len(df)),
        "feature_count_all": int(len(all_features)),
        "feature_count_without_history": int(len(no_history_features)),
        "forward_chaining_evaluations": full_fold_summaries,
        "sensitivity_forward_chaining_evaluations": sensitivity_fold_summaries,
        "history_features_treated_as_sensitivity": sorted(HISTORY_FEATURES),
        "top_10_joint_importance_all_features": full_stability.head(10)[
            [
                "feature",
                "mean_permutation_importance",
                "mean_permutation_rank",
                "mean_abs_shap",
                "mean_shap_rank",
                "joint_mean_rank",
            ]
        ].to_dict(orient="records"),
        "top_10_joint_importance_without_history": sensitivity_top[
            [
                "feature",
                "mean_permutation_importance",
                "mean_permutation_rank",
                "mean_abs_shap",
                "mean_shap_rank",
                "joint_mean_rank",
            ]
        ].to_dict(orient="records"),
        "automatic_feature_removal": False,
        "interpretation_note": (
            "Stage 4 uses forward-chaining evaluation. Each model is trained only on earlier "
            "calendar periods and importance is measured on the next period using both permutation "
            "importance and SHAP. History variables are repeated in a separate sensitivity path "
            "because of left-censoring."
        ),
    }

    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Per-fold nonlinear metrics written to {args.metrics}")
    print(f"Importance stability written to {args.stability}")
    print(f"Sensitivity summary written to {args.sensitivity}")
    print(f"Importance stability figure written to {args.figure}")
    print(f"Stage 4 summary written to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
