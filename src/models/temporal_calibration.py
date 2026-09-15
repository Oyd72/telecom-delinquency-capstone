"""Compare leakage-safe temporal calibration methods for the preferred 12-feature model.

The experiment preserves chronology. For each forward evaluation period, the base
XGBoost model is fitted on earlier data, calibration is fitted on the immediately
preceding seven calendar days, and performance is measured only on the later
evaluation fold.

Methods compared:
- uncalibrated probabilities;
- Platt scaling (logistic calibration on model logits);
- isotonic regression.

The script reports discrimination, calibration, and top-20% capture. Calibration is
post-hoc only: no future evaluation labels are used to fit the model or calibrator.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier


TARGET = "delinquent_5d"
DATE_FIELD = "pdate"
RANDOM_STATE = 42
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
        default=Path("reports/tables/temporal_calibration_metrics.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/temporal_calibration_summary.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("reports/figures/temporal_calibration_comparison.png"),
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
    order = np.argsort(-p)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    return float(y[order[:n_top]].sum() / positives)


def _metric_row(
    fold: str,
    method: str,
    y: np.ndarray,
    p: np.ndarray,
    train_rows: int,
    calibration_rows: int,
    evaluation_rows: int,
    train_end: str,
    calibration_start: str,
    calibration_end: str,
) -> dict:
    return {
        "evaluation_fold": fold,
        "method": method,
        "train_rows": int(train_rows),
        "calibration_rows": int(calibration_rows),
        "evaluation_rows": int(evaluation_rows),
        "train_end": train_end,
        "calibration_start": calibration_start,
        "calibration_end": calibration_end,
        "evaluation_delinquency_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error_10bin": _expected_calibration_error(y, p),
        "mean_predicted_risk": float(np.mean(p)),
        "calibration_in_the_large_gap": float(np.mean(p) - np.mean(y)),
        "top_20pct_capture_rate": _top_capture(y, p),
    }


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
        calibration_start = calibration_end - pd.Timedelta(days=CALIBRATION_DAYS - 1)
        train_end = calibration_start - pd.Timedelta(days=1)

        train = df[df[DATE_FIELD] <= train_end].copy()
        calibration = df[
            (df[DATE_FIELD] >= calibration_start)
            & (df[DATE_FIELD] <= calibration_end)
        ].copy()
        evaluation = df[
            (df[DATE_FIELD] >= eval_start) & (df[DATE_FIELD] <= eval_end)
        ].copy()

        if train.empty or calibration.empty or evaluation.empty:
            raise ValueError(f"Empty train/calibration/evaluation partition for {fold_name}")
        if calibration[TARGET].nunique() < 2:
            raise ValueError(f"Calibration set for {fold_name} contains only one class")

        X_train = train[FEATURES]
        y_train = train[TARGET].astype(int)
        X_cal = calibration[FEATURES]
        y_cal = calibration[TARGET].astype(int).to_numpy()
        X_eval = evaluation[FEATURES]
        y_eval = evaluation[TARGET].astype(int).to_numpy()

        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index
        )
        X_cal_imp = pd.DataFrame(
            imputer.transform(X_cal), columns=FEATURES, index=X_cal.index
        )
        X_eval_imp = pd.DataFrame(
            imputer.transform(X_eval), columns=FEATURES, index=X_eval.index
        )

        model = _build_model(y_train)
        model.fit(X_train_imp, y_train)

        p_cal_raw = _clip_probabilities(model.predict_proba(X_cal_imp)[:, 1])
        p_eval_raw = _clip_probabilities(model.predict_proba(X_eval_imp)[:, 1])

        platt = LogisticRegression(random_state=RANDOM_STATE, solver="lbfgs")
        platt.fit(_logit(p_cal_raw).reshape(-1, 1), y_cal)
        p_eval_platt = platt.predict_proba(_logit(p_eval_raw).reshape(-1, 1))[:, 1]

        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(p_cal_raw, y_cal)
        p_eval_isotonic = _clip_probabilities(isotonic.predict(p_eval_raw))

        common = {
            "fold": fold_name,
            "y": y_eval,
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "evaluation_rows": len(evaluation),
            "train_end": train_end.date().isoformat(),
            "calibration_start": calibration_start.date().isoformat(),
            "calibration_end": calibration_end.date().isoformat(),
        }

        rows.append(_metric_row(method="uncalibrated", p=p_eval_raw, **common))
        rows.append(_metric_row(method="platt", p=p_eval_platt, **common))
        rows.append(_metric_row(method="isotonic", p=p_eval_isotonic, **common))

    metrics = pd.DataFrame(rows)
    summary_table = (
        metrics.groupby("method", as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_ece=("expected_calibration_error_10bin", "mean"),
            mean_abs_calibration_gap=(
                "calibration_in_the_large_gap",
                lambda s: float(np.abs(s).mean()),
            ),
            mean_top20_capture=("top_20pct_capture_rate", "mean"),
        )
        .sort_values(["mean_brier_score", "mean_ece"], kind="stable")
    )

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(args.metrics, index=False)

    summary = {
        "input_rows": int(len(df)),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "calibration_window_days": CALIBRATION_DAYS,
        "fold_metrics": metrics.to_dict(orient="records"),
        "method_summary": summary_table.to_dict(orient="records"),
        "automatic_calibration_choice": False,
        "interpretation_note": (
            "For every evaluation period, model fitting, calibration fitting, and evaluation "
            "remain chronologically separated. Platt and isotonic calibration are compared "
            "against the uncalibrated model; no method is selected automatically."
        ),
    }

    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot = metrics.copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, group in plot.groupby("method"):
        ax.plot(
            group["evaluation_fold"],
            group["expected_calibration_error_10bin"],
            marker="o",
            label=method,
        )
    ax.set_ylabel("Expected calibration error (10 bins)")
    ax.set_xlabel("Evaluation fold")
    ax.set_title("Temporal calibration comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Calibration metrics written to {args.metrics}")
    print(f"Calibration comparison figure written to {args.figure}")
    print(f"Calibration summary written to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
