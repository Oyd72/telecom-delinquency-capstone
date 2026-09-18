"""Generate clearly labelled synthetic repayment scenarios for the post-23 July period.

This script does NOT repair or extend observed ground truth.

It preserves the real post-23 July predictor values and creates synthetic delinquency
outcomes under three transparent scenarios:

1. model_consistent:
   Uses the selected Random Forest + isotonic calibrated probability directly.

2. historical_prevalence_aligned:
   Applies a constant log-odds shift so the mean synthetic probability matches the
   observed delinquency prevalence in the labelled 1 June-23 July population.

3. holdout_stress_aligned:
   Applies a constant log-odds shift so the mean synthetic probability matches the
   observed delinquency prevalence in the 14-23 July final holdout.

For each scenario, a reproducible Bernoulli draw creates one synthetic binary outcome.
The synthetic labels are for scenario/sensitivity analysis only. They must not be mixed
with observed labels for official model training, validation, or performance claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERIM_PATH = PROJECT_ROOT / "data/interim/sample_data_intw_cleaned.csv"
PROCESSED_PATH = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"

SYNTHETIC_DIR = PROJECT_ROOT / "data/synthetic"
OUTPUT_PATH = SYNTHETIC_DIR / "module4_post23_synthetic_scenarios.csv"

TABLE_DIR = PROJECT_ROOT / "reports/tables"
SUMMARY_TABLE = TABLE_DIR / "module4_synthetic_scenario_summary.csv"
SUMMARY_JSON = TABLE_DIR / "module4_synthetic_scenario_summary.json"

FIGURE_DIR = PROJECT_ROOT / "reports/figures/module4"
RATE_FIGURE = FIGURE_DIR / "synthetic_scenario_rates.png"
PROBABILITY_FIGURE = FIGURE_DIR / "synthetic_scenario_probability_distributions.png"

TARGET = "delinquent_5d"
DATE = "pdate"

MODELLING_END = pd.Timestamp("2016-07-23")
TRAIN_END = pd.Timestamp("2016-07-06")
CALIBRATION_START = pd.Timestamp("2016-07-07")
CALIBRATION_END = pd.Timestamp("2016-07-13")
HOLDOUT_START = pd.Timestamp("2016-07-14")
HOLDOUT_END = pd.Timestamp("2016-07-23")

RANDOM_SEED = 42

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


def make_selected_model() -> Pipeline:
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                FEATURES,
            )
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("prep", prep), ("model", model)])


def clip_probability(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def logit(p: np.ndarray) -> np.ndarray:
    p = clip_probability(p)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def shift_probabilities_to_mean(
    probabilities: np.ndarray,
    target_mean: float,
) -> tuple[np.ndarray, float]:
    """Shift all log-odds by one intercept so the mean probability hits target_mean."""
    base_logit = logit(probabilities)

    low, high = -12.0, 12.0
    for _ in range(100):
        mid = (low + high) / 2
        shifted = sigmoid(base_logit + mid)
        if float(shifted.mean()) < target_mean:
            low = mid
        else:
            high = mid

    shift = (low + high) / 2
    adjusted = clip_probability(sigmoid(base_logit + shift))
    return adjusted, float(shift)


def scenario_frame(
    post: pd.DataFrame,
    scenario: str,
    probabilities: np.ndarray,
    target_prevalence: float | None,
    log_odds_shift: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = post[[DATE, *FEATURES]].copy()
    out["scenario"] = scenario
    out["synthetic_probability"] = probabilities
    out["synthetic_delinquent_5d"] = rng.binomial(1, probabilities).astype("int8")
    out["synthetic_label"] = True
    out["synthetic_method"] = "random_forest_isotonic_conditional_bernoulli"
    out["synthetic_seed"] = RANDOM_SEED
    out["scenario_target_prevalence"] = target_prevalence
    out["scenario_log_odds_shift"] = log_odds_shift
    out["source_outcome_used_as_ground_truth"] = False
    out["source_period"] = "post_23_july_2016"
    return out


def main() -> None:
    interim = pd.read_csv(INTERIM_PATH)
    processed = pd.read_csv(PROCESSED_PATH)

    interim[DATE] = pd.to_datetime(interim[DATE], dayfirst=True, errors="raise")
    processed[DATE] = pd.to_datetime(processed[DATE], errors="raise")

    labelled = processed.loc[processed[DATE] <= MODELLING_END].copy()
    train = processed.loc[processed[DATE] <= TRAIN_END].copy()
    calibration = processed.loc[
        (processed[DATE] >= CALIBRATION_START)
        & (processed[DATE] <= CALIBRATION_END)
    ].copy()
    holdout = processed.loc[
        (processed[DATE] >= HOLDOUT_START)
        & (processed[DATE] <= HOLDOUT_END)
    ].copy()
    post = interim.loc[interim[DATE] > MODELLING_END].copy()

    if len(post) != 58_825:
        raise AssertionError(f"Unexpected post-23 July rows: {len(post):,}")

    model = make_selected_model()
    model.fit(train[FEATURES], train[TARGET])

    p_cal_raw = clip_probability(model.predict_proba(calibration[FEATURES])[:, 1])
    p_post_raw = clip_probability(model.predict_proba(post[FEATURES])[:, 1])

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p_cal_raw, calibration[TARGET].to_numpy())
    p_post_calibrated = clip_probability(isotonic.predict(p_post_raw))

    labelled_prevalence = float(labelled[TARGET].mean())
    holdout_prevalence = float(holdout[TARGET].mean())

    p_history, history_shift = shift_probabilities_to_mean(
        p_post_calibrated,
        labelled_prevalence,
    )
    p_stress, stress_shift = shift_probabilities_to_mean(
        p_post_calibrated,
        holdout_prevalence,
    )

    scenarios = [
        (
            "model_consistent",
            p_post_calibrated,
            float(p_post_calibrated.mean()),
            0.0,
        ),
        (
            "historical_prevalence_aligned",
            p_history,
            labelled_prevalence,
            history_shift,
        ),
        (
            "holdout_stress_aligned",
            p_stress,
            holdout_prevalence,
            stress_shift,
        ),
    ]

    rng = np.random.default_rng(RANDOM_SEED)
    frames = []
    summary_rows = []

    for scenario, probabilities, target_prevalence, shift in scenarios:
        frame = scenario_frame(
            post=post,
            scenario=scenario,
            probabilities=probabilities,
            target_prevalence=target_prevalence,
            log_odds_shift=shift,
            rng=rng,
        )
        frames.append(frame)

        summary_rows.append(
            {
                "scenario": scenario,
                "rows": int(len(frame)),
                "target_mean_probability": float(target_prevalence),
                "actual_mean_probability": float(frame["synthetic_probability"].mean()),
                "realized_synthetic_delinquency_rate": float(
                    frame["synthetic_delinquent_5d"].mean()
                ),
                "log_odds_shift": float(shift),
                "p05_probability": float(frame["synthetic_probability"].quantile(0.05)),
                "median_probability": float(frame["synthetic_probability"].median()),
                "p95_probability": float(frame["synthetic_probability"].quantile(0.95)),
            }
        )

    synthetic = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    synthetic.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_TABLE, index=False)

    payload = {
        "analysis_role": "synthetic_scenario_only",
        "observed_ground_truth_repaired": False,
        "official_training_population_extended": False,
        "source_post23_rows": int(len(post)),
        "scenario_count": len(scenarios),
        "random_seed": RANDOM_SEED,
        "labelled_prevalence": labelled_prevalence,
        "final_holdout_prevalence": holdout_prevalence,
        "base_post23_mean_calibrated_probability": float(p_post_calibrated.mean()),
        "scenarios": summary.to_dict(orient="records"),
        "usage_restrictions": [
            "Do not merge synthetic labels into observed labelled data for official model training.",
            "Do not report performance on synthetic labels as observed model performance.",
            "Do not interpret a synthetic scenario as the true post-23 July repayment history.",
            "Use only for scenario, robustness, sensitivity, or demonstration analysis.",
        ],
        "method_note": (
            "All scenarios preserve actual post-23 July predictor values. Baseline probabilities "
            "come from the selected Random Forest with development-selected isotonic calibration. "
            "Alternative scenarios apply one constant log-odds intercept shift, preserving case "
            "ranking while changing the population-level expected delinquency rate."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Visual 1: expected vs one reproducible realised scenario rate.
    rates = summary.set_index("scenario")[
        ["actual_mean_probability", "realized_synthetic_delinquency_rate"]
    ] * 100
    ax = rates.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Synthetic post-23 July delinquency scenarios")
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(["Expected probability", "Realised synthetic outcome"])
    ax.figure.tight_layout()
    ax.figure.savefig(RATE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)

    # Visual 2: probability distributions across scenarios.
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 0.8, 41)
    for scenario, group in synthetic.groupby("scenario"):
        ax.hist(
            group["synthetic_probability"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=scenario,
        )
    ax.set_title("Synthetic scenario risk-probability distributions")
    ax.set_xlabel("Synthetic delinquency probability")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROBABILITY_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Module 4 synthetic post-23 July scenario generation completed.")
    print("Observed post-23 July outcomes used as ground truth: False")
    print("Official training population extended: False")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Synthetic scenario dataset written to: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Scenario summary table written to: {SUMMARY_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Scenario summary JSON written to: {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Scenario-rate figure written to: {RATE_FIGURE.relative_to(PROJECT_ROOT)}")
    print(
        "Scenario probability figure written to: "
        f"{PROBABILITY_FIGURE.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
