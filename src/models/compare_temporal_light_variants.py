"""Compare fixed Random Forest variants with progressively less temporal dependence.

This is a post-hoc sensitivity experiment, not a new independent validation.

The selected Random Forest hyperparameters and isotonic calibration approach remain fixed.
Only the predictor set changes.

Variants:
1. current_12:
   Full selected 12-feature specification.

2. temporal_light_10:
   Removes explicit tenure/recency fields:
   - aon
   - last_rech_date_ma

3. drift_reduced_7:
   Starts from temporal_light_10 and also removes features that previously showed
   pronounced temporal/distribution instability:
   - daily_decr30
   - daily_decr90
   - rental30

All variants use the same chronology:
- base model training: through 6 July 2016
- isotonic calibration: 7-13 July 2016
- evaluation: 14-23 July 2016

The purpose is to test whether useful performance survives when the model relies less on
features that may be especially sensitive to the short observation window.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"

TABLE_DIR = PROJECT_ROOT / "reports/tables"
FIGURE_DIR = PROJECT_ROOT / "reports/figures/module4"

METRICS_TABLE = TABLE_DIR / "module4_temporal_light_model_comparison.csv"
SUMMARY_JSON = TABLE_DIR / "module4_temporal_light_model_comparison.json"

PERFORMANCE_FIGURE = FIGURE_DIR / "temporal_light_model_performance.png"
CALIBRATION_FIGURE = FIGURE_DIR / "temporal_light_model_calibration.png"

TARGET = "delinquent_5d"
DATE = "pdate"

TRAIN_END = pd.Timestamp("2016-07-06")
CALIBRATION_START = pd.Timestamp("2016-07-07")
CALIBRATION_END = pd.Timestamp("2016-07-13")
HOLDOUT_START = pd.Timestamp("2016-07-14")
HOLDOUT_END = pd.Timestamp("2016-07-23")

RANDOM_STATE = 42

CURRENT_12 = [
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

TEMPORAL_LIGHT_10 = [
    feature
    for feature in CURRENT_12
    if feature not in {"aon", "last_rech_date_ma"}
]

DRIFT_REDUCED_7 = [
    feature
    for feature in TEMPORAL_LIGHT_10
    if feature not in {"daily_decr30", "daily_decr90", "rental30"}
]

VARIANTS = {
    "current_12": CURRENT_12,
    "temporal_light_10": TEMPORAL_LIGHT_10,
    "drift_reduced_7": DRIFT_REDUCED_7,
}


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
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


def top20_capture(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")

    n_top = max(1, int(np.ceil(len(y) * 0.20)))
    order = np.argsort(-p)
    return float(y[order[:n_top]].sum() / positives)


def fit_variant(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    holdout: pd.DataFrame,
    features: list[str],
) -> dict:
    imputer = SimpleImputer(strategy="median")

    x_train = pd.DataFrame(
        imputer.fit_transform(train[features]),
        columns=features,
        index=train.index,
    )
    x_cal = pd.DataFrame(
        imputer.transform(calibration[features]),
        columns=features,
        index=calibration.index,
    )
    x_holdout = pd.DataFrame(
        imputer.transform(holdout[features]),
        columns=features,
        index=holdout.index,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, train[TARGET])

    raw_cal = model.predict_proba(x_cal)[:, 1]
    raw_holdout = model.predict_proba(x_holdout)[:, 1]

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_cal, calibration[TARGET].to_numpy())
    calibrated = np.asarray(isotonic.predict(raw_holdout), dtype=float)

    y = holdout[TARGET].to_numpy()

    return {
        "roc_auc": float(roc_auc_score(y, calibrated)),
        "average_precision": float(average_precision_score(y, calibrated)),
        "brier": float(brier_score_loss(y, calibrated)),
        "ece_10bin": expected_calibration_error(y, calibrated),
        "top20_capture": top20_capture(y, calibrated),
        "mean_predicted_risk": float(calibrated.mean()),
        "observed_delinquency_rate": float(y.mean()),
        "calibration_gap": float(calibrated.mean() - y.mean()),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df[DATE] = pd.to_datetime(df[DATE], errors="raise")

    train = df.loc[df[DATE] <= TRAIN_END].copy()
    calibration = df.loc[
        (df[DATE] >= CALIBRATION_START)
        & (df[DATE] <= CALIBRATION_END)
    ].copy()
    holdout = df.loc[
        (df[DATE] >= HOLDOUT_START)
        & (df[DATE] <= HOLDOUT_END)
    ].copy()

    rows = []
    for variant, features in VARIANTS.items():
        metrics = fit_variant(train, calibration, holdout, features)
        rows.append(
            {
                "variant": variant,
                "feature_count": len(features),
                "features": ";".join(features),
                **metrics,
            }
        )

    results = pd.DataFrame(rows)

    baseline = results.loc[results["variant"] == "current_12"].iloc[0]
    results["roc_auc_change_vs_current"] = results["roc_auc"] - baseline["roc_auc"]
    results["top20_capture_change_vs_current"] = (
        results["top20_capture"] - baseline["top20_capture"]
    )
    results["brier_change_vs_current"] = results["brier"] - baseline["brier"]

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(METRICS_TABLE, index=False)

    payload = {
        "analysis_role": "post_hoc_temporal_dependence_sensitivity",
        "independent_validation": False,
        "model_hyperparameters_changed": False,
        "calibration_method_changed": False,
        "chronology": {
            "train_end": str(TRAIN_END.date()),
            "calibration_start": str(CALIBRATION_START.date()),
            "calibration_end": str(CALIBRATION_END.date()),
            "holdout_start": str(HOLDOUT_START.date()),
            "holdout_end": str(HOLDOUT_END.date()),
        },
        "variants": {name: features for name, features in VARIANTS.items()},
        "results": results.to_dict(orient="records"),
        "interpretation_note": (
            "This experiment isolates feature-set dependence. The already-opened holdout "
            "is reused only as sensitivity evidence, so results must not be presented as "
            "a second independent final validation."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Performance visual.
    perf = results.set_index("variant")[["roc_auc", "top20_capture"]].copy()
    perf["top20_capture"] = perf["top20_capture"]  # same 0-1 scale for comparison
    ax = perf.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Performance as temporally sensitive features are removed")
    ax.set_ylabel("Metric value")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(["ROC-AUC", "Top-20% capture"])
    ax.figure.tight_layout()
    ax.figure.savefig(PERFORMANCE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)

    # Calibration visual.
    calib = results.set_index("variant")[
        ["brier", "ece_10bin"]
    ].copy()
    ax = calib.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Calibration quality across feature-set variants")
    ax.set_ylabel("Error (lower is better)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(["Brier score", "ECE (10 bins)"])
    ax.figure.tight_layout()
    ax.figure.savefig(CALIBRATION_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)

    print("Module 4 temporal-dependence sensitivity comparison completed.")
    print("Independent validation: False")
    print("Model hyperparameters changed: False")
    print("Calibration method changed: False")
    print()
    print(
        results[
            [
                "variant",
                "feature_count",
                "roc_auc",
                "average_precision",
                "brier",
                "ece_10bin",
                "top20_capture",
                "mean_predicted_risk",
                "observed_delinquency_rate",
                "roc_auc_change_vs_current",
                "top20_capture_change_vs_current",
                "brier_change_vs_current",
            ]
        ].to_string(index=False)
    )
    print()
    print("Feature sets")
    for name, features in VARIANTS.items():
        print(f"{name} ({len(features)}): {features}")
    print()
    print(f"Metrics written to: {METRICS_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Summary written to: {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Performance figure: {PERFORMANCE_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Calibration figure: {CALIBRATION_FIGURE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
