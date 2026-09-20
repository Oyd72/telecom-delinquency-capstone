"""Generate counterfactual-style local explanations for the selected Module 4 model.

This is a contrastive explanation exercise, not a causal or prescriptive analysis.

Question:
    What small, in-distribution feature changes would be sufficient for the fitted
    model to cross the frozen development-selected classification threshold?

Safeguards:
- the operating threshold is read from the development-frozen threshold record;
- the selected model is fitted using the same training/calibration chronology;
- candidate feature values are drawn from observed training values near empirical
  quantiles rather than arbitrary out-of-range values;
- outputs are explicitly labelled as model-behaviour explanations, not recommended
  customer actions and not evidence that changing a feature would cause repayment.

The script examines three holdout cases:
- the closest score just below the threshold;
- the closest score just above the threshold;
- a representative high-risk case near the 90th percentile.

It searches single-feature changes first. If no single-feature change crosses the
threshold, it searches two-feature combinations among the five strongest global SHAP
features.
"""

from __future__ import annotations

import json
from itertools import combinations, product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"
THRESHOLD_PATH = PROJECT_ROOT / "reports/tables/module4_classification_threshold.json"

TABLE_DIR = PROJECT_ROOT / "reports/tables"
FIGURE_DIR = PROJECT_ROOT / "reports/figures/module4"

COUNTERFACTUAL_TABLE = TABLE_DIR / "module4_counterfactual_explanations.csv"
SUMMARY_JSON = TABLE_DIR / "module4_counterfactual_summary.json"
FIGURE_PATH = FIGURE_DIR / "counterfactual_risk_changes.png"

TARGET = "delinquent_5d"
DATE = "pdate"

TRAIN_END = pd.Timestamp("2016-07-06")
CALIBRATION_START = pd.Timestamp("2016-07-07")
CALIBRATION_END = pd.Timestamp("2016-07-13")
HOLDOUT_START = pd.Timestamp("2016-07-14")
HOLDOUT_END = pd.Timestamp("2016-07-23")

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

TOP_GLOBAL_FEATURES = [
    "cnt_ma_rech90",
    "sumamnt_ma_rech90",
    "sumamnt_ma_rech30",
    "cnt_ma_rech30",
    "daily_decr30",
]

SINGLE_QUANTILES = np.linspace(0.01, 0.99, 41)
PAIR_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def fit_selected_model(train: pd.DataFrame, calibration: pd.DataFrame):
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
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_cal, calibration[TARGET].to_numpy())

    return imputer, model, calibrator, x_train


def calibrated_score(frame, imputer, model, calibrator):
    x = imputer.transform(frame[FEATURES])
    raw = model.predict_proba(x)[:, 1]
    return np.asarray(calibrator.predict(raw), dtype=float)


def observed_grid(series: pd.Series, quantiles) -> list[float]:
    values = series.dropna().to_numpy(dtype=float)
    unique = np.unique(values)
    if len(unique) == 0:
        return []

    q_values = np.quantile(values, quantiles)
    snapped = []
    for q in q_values:
        idx = int(np.abs(unique - q).argmin())
        snapped.append(float(unique[idx]))
    return sorted(set(snapped))


def feature_iqr(x_train: pd.DataFrame, feature: str) -> float:
    q1 = float(x_train[feature].quantile(0.25))
    q3 = float(x_train[feature].quantile(0.75))
    return max(q3 - q1, 1e-9)


def choose_cases(holdout: pd.DataFrame, scores: np.ndarray, threshold: float) -> list[dict]:
    score_series = pd.Series(scores, index=holdout.index)

    below = score_series.loc[score_series < threshold]
    above = score_series.loc[score_series >= threshold]

    if below.empty or above.empty:
        raise ValueError("Holdout does not contain cases on both sides of threshold.")

    below_idx = (threshold - below).idxmin()
    above_idx = (above - threshold).idxmin()

    q90 = float(score_series.quantile(0.90))
    high_idx = (score_series - q90).abs().idxmin()

    return [
        {
            "case": "near_boundary_predicted_on_time",
            "index": int(below_idx),
            "target_direction": "increase_above_threshold",
        },
        {
            "case": "near_boundary_predicted_delinquent",
            "index": int(above_idx),
            "target_direction": "decrease_below_threshold",
        },
        {
            "case": "representative_high_risk",
            "index": int(high_idx),
            "target_direction": "decrease_below_threshold",
        },
    ]


def crossed(score: float, threshold: float, direction: str) -> bool:
    if direction == "increase_above_threshold":
        return score >= threshold
    return score < threshold


def search_single_feature(
    base_row: pd.DataFrame,
    base_score: float,
    direction: str,
    threshold: float,
    train: pd.DataFrame,
    x_train: pd.DataFrame,
    imputer,
    model,
    calibrator,
):
    candidates = []

    for feature in FEATURES:
        grid = observed_grid(train[feature], SINGLE_QUANTILES)
        if not grid:
            continue

        rows = []
        vals = []
        original = float(base_row.iloc[0][feature])
        for value in grid:
            if np.isclose(value, original, equal_nan=True):
                continue
            row = base_row.copy()
            row.loc[row.index[0], feature] = value
            rows.append(row.iloc[0][FEATURES].to_dict())
            vals.append(value)

        if not rows:
            continue

        frame = pd.DataFrame(rows)
        scores = calibrated_score(frame, imputer, model, calibrator)
        iqr = feature_iqr(x_train, feature)

        for value, score in zip(vals, scores):
            if crossed(float(score), threshold, direction):
                candidates.append(
                    {
                        "search_type": "single_feature",
                        "changed_features": [feature],
                        "changes": [
                            {
                                "feature": feature,
                                "from": original,
                                "to": float(value),
                                "delta": float(value - original),
                                "normalized_abs_change_iqr": float(abs(value - original) / iqr),
                            }
                        ],
                        "counterfactual_score": float(score),
                        "distance": float(abs(value - original) / iqr),
                    }
                )

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda x: (
            x["distance"],
            abs(x["counterfactual_score"] - threshold),
        ),
    )


def search_two_features(
    base_row: pd.DataFrame,
    direction: str,
    threshold: float,
    train: pd.DataFrame,
    x_train: pd.DataFrame,
    imputer,
    model,
    calibrator,
):
    candidates = []

    grids = {
        feature: observed_grid(train[feature], PAIR_QUANTILES)
        for feature in TOP_GLOBAL_FEATURES
    }

    for feature_a, feature_b in combinations(TOP_GLOBAL_FEATURES, 2):
        original_a = float(base_row.iloc[0][feature_a])
        original_b = float(base_row.iloc[0][feature_b])
        iqr_a = feature_iqr(x_train, feature_a)
        iqr_b = feature_iqr(x_train, feature_b)

        rows = []
        changes = []

        for value_a, value_b in product(grids[feature_a], grids[feature_b]):
            if np.isclose(value_a, original_a) and np.isclose(value_b, original_b):
                continue

            row = base_row.copy()
            row.loc[row.index[0], feature_a] = value_a
            row.loc[row.index[0], feature_b] = value_b
            rows.append(row.iloc[0][FEATURES].to_dict())

            distance = (
                abs(value_a - original_a) / iqr_a
                + abs(value_b - original_b) / iqr_b
            )
            changes.append(
                (
                    value_a,
                    value_b,
                    float(distance),
                )
            )

        if not rows:
            continue

        frame = pd.DataFrame(rows)
        scores = calibrated_score(frame, imputer, model, calibrator)

        for (value_a, value_b, distance), score in zip(changes, scores):
            if crossed(float(score), threshold, direction):
                candidates.append(
                    {
                        "search_type": "two_feature",
                        "changed_features": [feature_a, feature_b],
                        "changes": [
                            {
                                "feature": feature_a,
                                "from": original_a,
                                "to": float(value_a),
                                "delta": float(value_a - original_a),
                                "normalized_abs_change_iqr": float(abs(value_a - original_a) / iqr_a),
                            },
                            {
                                "feature": feature_b,
                                "from": original_b,
                                "to": float(value_b),
                                "delta": float(value_b - original_b),
                                "normalized_abs_change_iqr": float(abs(value_b - original_b) / iqr_b),
                            },
                        ],
                        "counterfactual_score": float(score),
                        "distance": float(distance),
                    }
                )

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda x: (
            x["distance"],
            abs(x["counterfactual_score"] - threshold),
        ),
    )


def main() -> None:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            "Classification threshold record not found. Run "
            "generate_module4_classification_artifacts.py first."
        )

    threshold_payload = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    threshold = float(threshold_payload["selected_threshold"])

    df = pd.read_csv(DATA_PATH)
    df[DATE] = pd.to_datetime(df[DATE], errors="raise")

    train = df.loc[df[DATE] <= TRAIN_END].copy()
    calibration = df.loc[
        (df[DATE] >= CALIBRATION_START) & (df[DATE] <= CALIBRATION_END)
    ].copy()
    holdout = df.loc[
        (df[DATE] >= HOLDOUT_START) & (df[DATE] <= HOLDOUT_END)
    ].copy()

    imputer, model, calibrator, x_train = fit_selected_model(train, calibration)
    holdout_scores = calibrated_score(holdout, imputer, model, calibrator)

    cases = choose_cases(holdout, holdout_scores, threshold)
    rows = []
    summaries = []

    for case in cases:
        idx = case["index"]
        base_row = holdout.loc[[idx]].copy()
        base_score = float(
            holdout_scores[holdout.index.get_loc(idx)]
        )

        result = search_single_feature(
            base_row=base_row,
            base_score=base_score,
            direction=case["target_direction"],
            threshold=threshold,
            train=train,
            x_train=x_train,
            imputer=imputer,
            model=model,
            calibrator=calibrator,
        )

        if result is None:
            result = search_two_features(
                base_row=base_row,
                direction=case["target_direction"],
                threshold=threshold,
                train=train,
                x_train=x_train,
                imputer=imputer,
                model=model,
                calibrator=calibrator,
            )

        case_summary = {
            "case": case["case"],
            "holdout_index": idx,
            "observed_outcome": int(base_row.iloc[0][TARGET]),
            "original_score": base_score,
            "threshold": threshold,
            "original_class": (
                "predicted_delinquent" if base_score >= threshold else "predicted_on_time"
            ),
            "target_direction": case["target_direction"],
            "counterfactual_found": result is not None,
        }

        if result is not None:
            case_summary.update(
                {
                    "search_type": result["search_type"],
                    "counterfactual_score": result["counterfactual_score"],
                    "counterfactual_class": (
                        "predicted_delinquent"
                        if result["counterfactual_score"] >= threshold
                        else "predicted_on_time"
                    ),
                    "normalized_distance": result["distance"],
                    "changes": result["changes"],
                }
            )

            for change in result["changes"]:
                rows.append(
                    {
                        "case": case["case"],
                        "holdout_index": idx,
                        "observed_outcome": int(base_row.iloc[0][TARGET]),
                        "original_score": base_score,
                        "threshold": threshold,
                        "original_class": case_summary["original_class"],
                        "counterfactual_score": result["counterfactual_score"],
                        "counterfactual_class": case_summary["counterfactual_class"],
                        "search_type": result["search_type"],
                        "feature": change["feature"],
                        "original_value": change["from"],
                        "counterfactual_value": change["to"],
                        "delta": change["delta"],
                        "normalized_abs_change_iqr": change[
                            "normalized_abs_change_iqr"
                        ],
                    }
                )

        summaries.append(case_summary)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    result_table = pd.DataFrame(rows)
    result_table.to_csv(COUNTERFACTUAL_TABLE, index=False)

    payload = {
        "analysis_type": "counterfactual_style_contrastive_explanation",
        "classification_threshold": threshold,
        "threshold_selected_on_holdout": False,
        "candidate_value_source": (
            "Observed training values nearest to empirical quantiles; no arbitrary "
            "out-of-range values are introduced."
        ),
        "single_feature_search": {
            "features": FEATURES,
            "quantile_grid_size": len(SINGLE_QUANTILES),
        },
        "two_feature_fallback": {
            "features": TOP_GLOBAL_FEATURES,
            "quantiles": PAIR_QUANTILES,
        },
        "cases": summaries,
        "interpretation_caution": (
            "These are contrastive explanations of fitted model behaviour. They are not "
            "causal claims, behavioural recommendations, or evidence that a customer could "
            "or should change the listed feature to alter repayment outcomes. Feature "
            "combinations may also be statistically uncommon even when each candidate value "
            "was observed in training data."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Visual summary
    plot_rows = []
    for item in summaries:
        if not item.get("counterfactual_found"):
            continue
        plot_rows.append(
            {
                "case": item["case"],
                "score_type": "Original",
                "score": item["original_score"],
            }
        )
        plot_rows.append(
            {
                "case": item["case"],
                "score_type": "Counterfactual",
                "score": item["counterfactual_score"],
            }
        )

    plot_df = pd.DataFrame(plot_rows)
    if not plot_df.empty:
        cases_order = [c["case"] for c in summaries if c.get("counterfactual_found")]
        x = np.arange(len(cases_order))
        width = 0.36

        original = [
            float(plot_df.loc[
                (plot_df["case"] == case)
                & (plot_df["score_type"] == "Original"),
                "score",
            ].iloc[0])
            for case in cases_order
        ]
        counter = [
            float(plot_df.loc[
                (plot_df["case"] == case)
                & (plot_df["score_type"] == "Counterfactual"),
                "score",
            ].iloc[0])
            for case in cases_order
        ]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(x - width / 2, original, width, label="Original score")
        ax.bar(x + width / 2, counter, width, label="Counterfactual score")
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.5,
            label=f"Frozen threshold ({threshold:.3f})",
        )
        ax.set_xticks(
            x,
            labels=[case.replace("_", " ").title() for case in cases_order],
            rotation=15,
            ha="right",
        )
        ax.set_ylabel("Calibrated delinquency probability")
        ax.set_title("Counterfactual-style score changes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print("Module 4 counterfactual-style explanation completed.")
    print(f"Frozen threshold: {threshold:.6f}")
    print("Threshold selected on holdout: False")
    print()

    for item in summaries:
        print(
            f"{item['case']}: original={item['original_score']:.6f} "
            f"({item['original_class']})"
        )
        if not item.get("counterfactual_found"):
            print("  No threshold-crossing counterfactual found in the constrained search.")
            continue

        print(
            f"  counterfactual={item['counterfactual_score']:.6f} "
            f"({item['counterfactual_class']}), search={item['search_type']}"
        )
        for change in item["changes"]:
            print(
                f"  - {change['feature']}: {change['from']:.6g} -> "
                f"{change['to']:.6g} (delta={change['delta']:.6g}, "
                f"|delta|/IQR={change['normalized_abs_change_iqr']:.3f})"
            )

    print()
    print(f"Table written to: {COUNTERFACTUAL_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Summary written to: {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Figure written to: {FIGURE_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
