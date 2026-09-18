"""Explain the selected Module 4 Random Forest with global and local SHAP analysis.

Model:
- Tuned Random Forest selected in Module 4.
- Base-model training through 6 July 2016.
- Isotonic calibration fitted on 7-13 July 2016.
- Explanation population: 14-23 July 2016 final holdout.

SHAP explains the base Random Forest rather than the isotonic mapping. This is deliberate:
isotonic calibration is a monotonic post-processing layer that changes probability
scale, not the underlying feature logic of the Random Forest. Local case tables include
both raw and calibrated risk so the relationship remains visible.

Outputs:
- global mean absolute SHAP importance table;
- local SHAP contribution table for representative low/typical/high-risk cases;
- global bar and beeswarm figures;
- three local waterfall figures;
- JSON summary for the running experiment record.

No model fitting or parameter selection is performed on the holdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"

TABLE_DIR = PROJECT_ROOT / "reports/tables"
FIGURE_DIR = PROJECT_ROOT / "reports/figures/module4"

GLOBAL_TABLE = TABLE_DIR / "module4_shap_global_importance.csv"
LOCAL_TABLE = TABLE_DIR / "module4_shap_local_cases.csv"
SUMMARY_JSON = TABLE_DIR / "module4_shap_summary.json"

GLOBAL_BAR = FIGURE_DIR / "shap_global_importance.png"
GLOBAL_BEESWARM = FIGURE_DIR / "shap_global_beeswarm.png"

TARGET = "delinquent_5d"
DATE = "pdate"

TRAIN_END = pd.Timestamp("2016-07-06")
CALIBRATION_START = pd.Timestamp("2016-07-07")
CALIBRATION_END = pd.Timestamp("2016-07-13")
HOLDOUT_START = pd.Timestamp("2016-07-14")
HOLDOUT_END = pd.Timestamp("2016-07-23")

GLOBAL_SAMPLE_SIZE = 2_000
RANDOM_STATE = 42

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


def positive_class_shap_values(
    explainer: shap.TreeExplainer,
    x: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    values = explainer.shap_values(x)

    # SHAP versions differ for binary sklearn classifiers:
    # - list[class] of (n_samples, n_features)
    # - ndarray (n_samples, n_features, n_classes)
    # - ndarray (n_samples, n_features)
    if isinstance(values, list):
        positive = np.asarray(values[1])
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            positive = values[:, :, 1]
        elif values.ndim == 2:
            positive = values
        else:
            raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    expected = explainer.expected_value
    if isinstance(expected, (list, tuple, np.ndarray)):
        expected_arr = np.asarray(expected).reshape(-1)
        base_value = float(expected_arr[-1])
    else:
        base_value = float(expected)

    return positive, base_value


def choose_representative_cases(
    holdout: pd.DataFrame,
    calibrated_scores: np.ndarray,
) -> list[dict]:
    score_series = pd.Series(calibrated_scores, index=holdout.index)

    targets = [
        ("low_risk", 0.10),
        ("typical_risk", 0.50),
        ("high_risk", 0.90),
    ]

    cases = []
    for case_name, quantile in targets:
        target_score = float(score_series.quantile(quantile))
        idx = (score_series - target_score).abs().idxmin()
        cases.append(
            {
                "case": case_name,
                "quantile_target": quantile,
                "index": int(idx),
                "calibrated_risk": float(score_series.loc[idx]),
                "observed_outcome": int(holdout.loc[idx, TARGET]),
            }
        )
    return cases


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

    if train.empty or calibration.empty or holdout.empty:
        raise ValueError("Train, calibration, or holdout partition is empty.")

    imputer = SimpleImputer(strategy="median")
    x_train = pd.DataFrame(
        imputer.fit_transform(train[FEATURES]),
        columns=FEATURES,
        index=train.index,
    )
    x_cal = pd.DataFrame(
        imputer.transform(calibration[FEATURES]),
        columns=FEATURES,
        index=calibration.index,
    )
    x_holdout = pd.DataFrame(
        imputer.transform(holdout[FEATURES]),
        columns=FEATURES,
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
    calibrated_holdout = np.asarray(isotonic.predict(raw_holdout), dtype=float)

    rng = np.random.default_rng(RANDOM_STATE)
    if len(x_holdout) > GLOBAL_SAMPLE_SIZE:
        sample_positions = rng.choice(
            len(x_holdout),
            size=GLOBAL_SAMPLE_SIZE,
            replace=False,
        )
        x_global = x_holdout.iloc[np.sort(sample_positions)].copy()
    else:
        x_global = x_holdout.copy()

    explainer = shap.TreeExplainer(model)

    global_shap, base_value = positive_class_shap_values(explainer, x_global)

    global_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_abs_shap": np.abs(global_shap).mean(axis=0),
            "mean_shap": global_shap.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    global_importance.to_csv(GLOBAL_TABLE, index=False)

    # Global bar chart.
    plot_df = global_importance.sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"])
    ax.set_title("Random Forest global SHAP importance")
    ax.set_xlabel("Mean |SHAP value| for delinquency class")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(GLOBAL_BAR, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Global beeswarm via modern SHAP Explanation API.
    explanation = shap.Explanation(
        values=global_shap,
        base_values=np.full(len(x_global), base_value),
        data=x_global.to_numpy(),
        feature_names=FEATURES,
    )
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(explanation, max_display=len(FEATURES), show=False)
    plt.title("Random Forest SHAP distribution on final holdout sample")
    plt.tight_layout()
    plt.savefig(GLOBAL_BEESWARM, dpi=180, bbox_inches="tight")
    plt.close()

    # Representative local cases.
    cases = choose_representative_cases(holdout, calibrated_holdout)
    local_rows = []

    for case in cases:
        idx = case["index"]
        x_case = x_holdout.loc[[idx]]
        local_shap, local_base = positive_class_shap_values(explainer, x_case)
        raw_risk = float(model.predict_proba(x_case)[:, 1][0])

        contributions = pd.DataFrame(
            {
                "feature": FEATURES,
                "feature_value": x_case.iloc[0].to_numpy(),
                "shap_value": local_shap[0],
                "abs_shap": np.abs(local_shap[0]),
            }
        ).sort_values("abs_shap", ascending=False)

        for rank, row in enumerate(contributions.itertuples(index=False), start=1):
            local_rows.append(
                {
                    "case": case["case"],
                    "holdout_index": idx,
                    "quantile_target": case["quantile_target"],
                    "observed_outcome": case["observed_outcome"],
                    "raw_risk": raw_risk,
                    "calibrated_risk": case["calibrated_risk"],
                    "base_value": local_base,
                    "importance_rank": rank,
                    "feature": row.feature,
                    "feature_value": float(row.feature_value),
                    "shap_value": float(row.shap_value),
                    "abs_shap": float(row.abs_shap),
                    "direction": (
                        "increases predicted delinquency risk"
                        if row.shap_value > 0
                        else "decreases predicted delinquency risk"
                        if row.shap_value < 0
                        else "neutral"
                    ),
                }
            )

        local_explanation = shap.Explanation(
            values=local_shap[0],
            base_values=local_base,
            data=x_case.iloc[0].to_numpy(),
            feature_names=FEATURES,
        )
        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(local_explanation, max_display=12, show=False)
        plt.title(
            f"{case['case'].replace('_', ' ').title()} case — "
            f"raw={raw_risk:.1%}, calibrated={case['calibrated_risk']:.1%}"
        )
        plt.tight_layout()
        local_path = FIGURE_DIR / f"shap_local_{case['case']}.png"
        plt.savefig(local_path, dpi=180, bbox_inches="tight")
        plt.close()

    local_table = pd.DataFrame(local_rows)
    local_table.to_csv(LOCAL_TABLE, index=False)

    top_features = global_importance.head(5)["feature"].tolist()

    case_summary = []
    for case in cases:
        subset = local_table.loc[
            local_table["case"] == case["case"]
        ].sort_values("importance_rank")
        case_summary.append(
            {
                **case,
                "raw_risk": float(
                    subset["raw_risk"].iloc[0]
                ),
                "top_contributors": subset.head(5)[
                    ["feature", "feature_value", "shap_value", "direction"]
                ].to_dict(orient="records"),
            }
        )

    payload = {
        "model": "tuned_random_forest",
        "calibration": "isotonic",
        "shap_explains": "base_random_forest_positive_class_output",
        "calibration_explanation_note": (
            "SHAP values explain the Random Forest feature logic. Isotonic calibration "
            "is a monotonic post-processing layer and is reported separately through raw "
            "and calibrated risk values."
        ),
        "training_end": str(TRAIN_END.date()),
        "calibration_period": [
            str(CALIBRATION_START.date()),
            str(CALIBRATION_END.date()),
        ],
        "explanation_population": [
            str(HOLDOUT_START.date()),
            str(HOLDOUT_END.date()),
        ],
        "global_sample_rows": int(len(x_global)),
        "base_value": base_value,
        "top_global_features": global_importance.head(10).to_dict(orient="records"),
        "representative_cases": case_summary,
        "interpretation_caution": (
            "SHAP describes how the fitted model uses features; it does not establish "
            "causal effects or prove that changing a feature would change repayment."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Module 4 SHAP explainability analysis completed.")
    print("SHAP explains: base tuned Random Forest")
    print("Calibration shown separately: isotonic")
    print()
    print("Top global SHAP features")
    print(global_importance.head(10).to_string(index=False))
    print()
    print("Representative local cases")
    for case in case_summary:
        print(
            f"{case['case']}: raw={case['raw_risk']:.4f}, "
            f"calibrated={case['calibrated_risk']:.4f}, "
            f"observed={case['observed_outcome']}"
        )
        for item in case["top_contributors"][:3]:
            print(
                f"  - {item['feature']}: SHAP={item['shap_value']:.6f} "
                f"({item['direction']})"
            )
    print()
    print(f"Global table written to: {GLOBAL_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Local table written to: {LOCAL_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Summary JSON written to: {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Global bar figure: {GLOBAL_BAR.relative_to(PROJECT_ROOT)}")
    print(f"Global beeswarm figure: {GLOBAL_BEESWARM.relative_to(PROJECT_ROOT)}")
    print(
        "Local figures: "
        + ", ".join(
            str((FIGURE_DIR / f"shap_local_{name}.png").relative_to(PROJECT_ROOT))
            for name in ["low_risk", "typical_risk", "high_risk"]
        )
    )


if __name__ == "__main__":
    main()
