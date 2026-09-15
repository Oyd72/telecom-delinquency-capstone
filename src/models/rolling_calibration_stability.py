"""Test rolling Platt-calibration stability across recent-window lengths.

This diagnostic follows the temporal-calibration experiment. It asks whether Platt
scaling helps or hurts depending on the length and outcome prevalence of the recent
calibration window.

For each forward evaluation fold, the base XGBoost model is trained once using data
strictly before the largest calibration window. The same base-model evaluation
probabilities are then recalibrated with nested recent windows of 3, 5, 7, and 10
calendar days. Keeping the base model fixed within a fold isolates the effect of the
calibration window rather than mixing it with a changing training sample.

The script reports:
- recent-window delinquency prevalence and its gap to the evaluation period;
- uncalibrated versus Platt Brier score, ECE, and calibration-in-the-large gap;
- whether Platt improves calibration for each fold/window combination;
- descriptive correlations between prevalence mismatch and calibration benefit.

This is a diagnostic experiment, not an automatic rule for regime-dependent
calibration.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier


TARGET = "delinquent_5d"
DATE_FIELD = "pdate"
RANDOM_STATE = 42
WINDOW_DAYS = [3, 5, 7, 10]
MAX_WINDOW_DAYS = max(WINDOW_DAYS)

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
        default=Path("reports/tables/rolling_calibration_metrics.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/rolling_calibration_summary.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("reports/figures/rolling_calibration_stability.png"),
    )
    return parser.parse_args()


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


def _clip_probabilities(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip_probabilities(p)
    return np.log(p / (1 - p))


def _expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
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
        ece += (mask.sum() / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def _top_capture(y: np.ndarray, p: np.ndarray, fraction: float = 0.20) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n_top = max(1, int(np.ceil(len(y) * fraction)))
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-p)
    return float(y[order[:n_top]].sum() / positives)


def _score(y: np.ndarray, p: np.ndarray) -> dict:
    prevalence = float(np.mean(y))
    mean_risk = float(np.mean(p))
    gap = mean_risk - prevalence
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error_10bin": _expected_calibration_error(y, p),
        "mean_predicted_risk": mean_risk,
        "calibration_in_the_large_gap": gap,
        "abs_calibration_gap": abs(gap),
        "top_20pct_capture_rate": _top_capture(y, p),
    }


def _safe_spearman(x: pd.Series, y: pd.Series) -> float | None:
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 3:
        return None
    if x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return None
    value = x[valid].corr(y[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.data)

    required = set(FEATURES) | {DATE_FIELD, TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[DATE_FIELD] = pd.to_datetime(df[DATE_FIELD], errors="coerce")
    if df[DATE_FIELD].isna().any():
        raise ValueError("Unparseable pdate values found in model-ready data.")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must contain only 0/1 values.")

    rows: list[dict] = []

    for fold_name, eval_start_text, eval_end_text in EVALUATION_FOLDS:
        eval_start = pd.Timestamp(eval_start_text)
        eval_end = pd.Timestamp(eval_end_text)
        calibration_end = eval_start - pd.Timedelta(days=1)
        common_train_end = calibration_end - pd.Timedelta(days=MAX_WINDOW_DAYS)

        train = df[df[DATE_FIELD] <= common_train_end].copy()
        evaluation = df[
            (df[DATE_FIELD] >= eval_start) & (df[DATE_FIELD] <= eval_end)
        ].copy()

        if train.empty or evaluation.empty:
            raise ValueError(f"Empty train/evaluation partition for {fold_name}")

        X_train = train[FEATURES]
        y_train = train[TARGET].astype(int)
        X_eval = evaluation[FEATURES]
        y_eval = evaluation[TARGET].astype(int).to_numpy()

        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index
        )
        X_eval_imp = pd.DataFrame(
            imputer.transform(X_eval), columns=FEATURES, index=X_eval.index
        )

        model = _build_model(y_train)
        model.fit(X_train_imp, y_train)
        p_eval_raw = _clip_probabilities(model.predict_proba(X_eval_imp)[:, 1])
        raw_metrics = _score(y_eval, p_eval_raw)

        for window_days in WINDOW_DAYS:
            calibration_start = calibration_end - pd.Timedelta(days=window_days - 1)
            calibration = df[
                (df[DATE_FIELD] >= calibration_start)
                & (df[DATE_FIELD] <= calibration_end)
            ].copy()

            if calibration.empty or calibration[TARGET].nunique() < 2:
                continue

            X_cal = calibration[FEATURES]
            y_cal = calibration[TARGET].astype(int).to_numpy()
            X_cal_imp = pd.DataFrame(
                imputer.transform(X_cal), columns=FEATURES, index=X_cal.index
            )
            p_cal_raw = _clip_probabilities(model.predict_proba(X_cal_imp)[:, 1])

            platt = LogisticRegression(random_state=RANDOM_STATE, solver="lbfgs")
            platt.fit(_logit(p_cal_raw).reshape(-1, 1), y_cal)
            p_eval_platt = platt.predict_proba(_logit(p_eval_raw).reshape(-1, 1))[:, 1]
            platt_metrics = _score(y_eval, p_eval_platt)

            calibration_rate = float(np.mean(y_cal))
            evaluation_rate = float(np.mean(y_eval))
            prevalence_gap = evaluation_rate - calibration_rate

            rows.append(
                {
                    "evaluation_fold": fold_name,
                    "window_days": int(window_days),
                    "train_end": common_train_end.date().isoformat(),
                    "train_rows": int(len(train)),
                    "calibration_start": calibration_start.date().isoformat(),
                    "calibration_end": calibration_end.date().isoformat(),
                    "calibration_rows": int(len(calibration)),
                    "calibration_delinquency_rate": calibration_rate,
                    "evaluation_rows": int(len(evaluation)),
                    "evaluation_delinquency_rate": evaluation_rate,
                    "evaluation_minus_calibration_prevalence": prevalence_gap,
                    "abs_prevalence_mismatch": abs(prevalence_gap),
                    "raw_brier_score": raw_metrics["brier_score"],
                    "platt_brier_score": platt_metrics["brier_score"],
                    "platt_minus_raw_brier": platt_metrics["brier_score"] - raw_metrics["brier_score"],
                    "raw_ece": raw_metrics["expected_calibration_error_10bin"],
                    "platt_ece": platt_metrics["expected_calibration_error_10bin"],
                    "platt_minus_raw_ece": platt_metrics["expected_calibration_error_10bin"] - raw_metrics["expected_calibration_error_10bin"],
                    "raw_abs_calibration_gap": raw_metrics["abs_calibration_gap"],
                    "platt_abs_calibration_gap": platt_metrics["abs_calibration_gap"],
                    "platt_minus_raw_abs_gap": platt_metrics["abs_calibration_gap"] - raw_metrics["abs_calibration_gap"],
                    "raw_mean_predicted_risk": raw_metrics["mean_predicted_risk"],
                    "platt_mean_predicted_risk": platt_metrics["mean_predicted_risk"],
                    "platt_probability_shift": platt_metrics["mean_predicted_risk"] - raw_metrics["mean_predicted_risk"],
                    "roc_auc": raw_metrics["roc_auc"],
                    "average_precision": raw_metrics["average_precision"],
                    "top_20pct_capture_rate": raw_metrics["top_20pct_capture_rate"],
                    "platt_improves_brier": bool(platt_metrics["brier_score"] < raw_metrics["brier_score"]),
                    "platt_improves_ece": bool(platt_metrics["expected_calibration_error_10bin"] < raw_metrics["expected_calibration_error_10bin"]),
                    "platt_improves_abs_gap": bool(platt_metrics["abs_calibration_gap"] < raw_metrics["abs_calibration_gap"]),
                }
            )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError("No valid rolling calibration comparisons were produced.")

    window_summary = (
        metrics.groupby("window_days", as_index=False)
        .agg(
            comparisons=("evaluation_fold", "count"),
            mean_abs_prevalence_mismatch=("abs_prevalence_mismatch", "mean"),
            mean_platt_minus_raw_brier=("platt_minus_raw_brier", "mean"),
            mean_platt_minus_raw_ece=("platt_minus_raw_ece", "mean"),
            mean_platt_minus_raw_abs_gap=("platt_minus_raw_abs_gap", "mean"),
            brier_improvement_share=("platt_improves_brier", "mean"),
            ece_improvement_share=("platt_improves_ece", "mean"),
            abs_gap_improvement_share=("platt_improves_abs_gap", "mean"),
        )
        .sort_values("window_days")
    )

    best_by_fold = []
    for fold_name, group in metrics.groupby("evaluation_fold"):
        best = group.sort_values(
            ["platt_brier_score", "platt_ece", "platt_abs_calibration_gap"],
            kind="stable",
        ).iloc[0]
        best_by_fold.append(
            {
                "evaluation_fold": fold_name,
                "best_window_days_by_brier_then_ece": int(best["window_days"]),
                "calibration_delinquency_rate": float(best["calibration_delinquency_rate"]),
                "evaluation_delinquency_rate": float(best["evaluation_delinquency_rate"]),
                "platt_brier_score": float(best["platt_brier_score"]),
                "platt_ece": float(best["platt_ece"]),
                "platt_abs_calibration_gap": float(best["platt_abs_calibration_gap"]),
            }
        )

    correlations = {
        "spearman_abs_prevalence_mismatch_vs_platt_minus_raw_brier": _safe_spearman(
            metrics["abs_prevalence_mismatch"], metrics["platt_minus_raw_brier"]
        ),
        "spearman_abs_prevalence_mismatch_vs_platt_minus_raw_ece": _safe_spearman(
            metrics["abs_prevalence_mismatch"], metrics["platt_minus_raw_ece"]
        ),
        "spearman_signed_prevalence_gap_vs_platt_probability_shift": _safe_spearman(
            metrics["evaluation_minus_calibration_prevalence"], metrics["platt_probability_shift"]
        ),
    }

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics, index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    for fold_name, group in metrics.groupby("evaluation_fold"):
        group = group.sort_values("window_days")
        ax.plot(
            group["window_days"],
            group["platt_minus_raw_ece"],
            marker="o",
            label=fold_name,
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Recent calibration window (days)")
    ax.set_ylabel("Platt ECE minus raw ECE (negative is improvement)")
    ax.set_title("Rolling Platt calibration stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_rows": int(len(df)),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "window_days_tested": WINDOW_DAYS,
        "base_model_training_rule": (
            "Within each evaluation fold, the base model is trained once using data strictly "
            "before the largest tested calibration window. Nested recent windows then calibrate "
            "the same base-model probabilities so window-length effects are isolated."
        ),
        "comparison_count": int(len(metrics)),
        "window_summary": window_summary.to_dict(orient="records"),
        "best_window_by_fold": best_by_fold,
        "prevalence_relationships": correlations,
        "automatic_calibration_rule": False,
        "interpretation_note": (
            "This diagnostic tests whether Platt calibration stability is associated with recent "
            "window length and prevalence mismatch. Only three evaluation folds are available, so "
            "correlations are descriptive and must not be treated as a validated regime-selection rule."
        ),
    }

    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Rolling calibration metrics written to {args.metrics}")
    print(f"Rolling calibration figure written to {args.figure}")
    print(f"Rolling calibration summary written to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
