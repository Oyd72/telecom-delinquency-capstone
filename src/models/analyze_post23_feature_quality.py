"""Assess the usability of post-23 July predictor data without using its outcome labels.

This experiment keeps the 24 July-21 August 2016 records separate from the
supervised modelling population and asks whether their *feature values* remain useful
for exploratory robustness/sensitivity work.

Checks:
- missingness by feature;
- min/max and out-of-training-range rates;
- median and interquartile-range shifts;
- Population Stability Index (PSI) using modelling-period quantile bins;
- selected-model score distribution on the final holdout vs post-23 July records.

The post-23 July source label is never used as ground truth in this analysis.

For score behaviour, the selected tuned Random Forest is trained through 6 July,
isotonic calibration is fitted on 7-13 July, and both the 14-23 July holdout and
24 July-21 August post period are scored. This preserves the same chronology used in
the calibrated-holdout sensitivity experiment.
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

FEATURE_TABLE = PROJECT_ROOT / "reports/tables/module4_post23_feature_quality.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports/tables/module4_post23_feature_quality_summary.json"
SCORE_TABLE = PROJECT_ROOT / "reports/tables/module4_post23_score_distribution.csv"

FIGURE_DIR = PROJECT_ROOT / "reports/figures/module4"
PSI_FIGURE = FIGURE_DIR / "post23_feature_psi.png"
SCORE_FIGURE = FIGURE_DIR / "post23_score_distribution.png"

TARGET = "delinquent_5d"
DATE = "pdate"

MODELLING_END = pd.Timestamp("2016-07-23")
TRAIN_END = pd.Timestamp("2016-07-06")
CALIBRATION_START = pd.Timestamp("2016-07-07")
CALIBRATION_END = pd.Timestamp("2016-07-13")
HOLDOUT_START = pd.Timestamp("2016-07-14")
HOLDOUT_END = pd.Timestamp("2016-07-23")

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


def quantile_bins(reference: pd.Series, n_bins: int = 10) -> np.ndarray:
    values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return np.array([-np.inf, np.inf], dtype=float)

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))

    if len(edges) < 2:
        v = float(edges[0])
        return np.array([-np.inf, v, np.inf], dtype=float)

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def psi(reference: pd.Series, comparison: pd.Series, n_bins: int = 10) -> float:
    edges = quantile_bins(reference, n_bins=n_bins)
    ref = pd.to_numeric(reference, errors="coerce")
    cmp = pd.to_numeric(comparison, errors="coerce")

    ref_missing = float(ref.isna().mean())
    cmp_missing = float(cmp.isna().mean())

    ref_nonmissing = ref.dropna()
    cmp_nonmissing = cmp.dropna()

    ref_counts, _ = np.histogram(ref_nonmissing, bins=edges)
    cmp_counts, _ = np.histogram(cmp_nonmissing, bins=edges)

    ref_props = ref_counts / max(len(ref), 1)
    cmp_props = cmp_counts / max(len(cmp), 1)

    ref_props = np.append(ref_props, ref_missing)
    cmp_props = np.append(cmp_props, cmp_missing)

    eps = 1e-6
    ref_props = np.clip(ref_props, eps, None)
    cmp_props = np.clip(cmp_props, eps, None)

    return float(np.sum((cmp_props - ref_props) * np.log(cmp_props / ref_props)))


def robust_scale_shift(reference: pd.Series, comparison: pd.Series) -> float | None:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cmp = pd.to_numeric(comparison, errors="coerce").dropna()
    if ref.empty or cmp.empty:
        return None

    ref_median = float(ref.median())
    cmp_median = float(cmp.median())
    q1 = float(ref.quantile(0.25))
    q3 = float(ref.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return None
    return float((cmp_median - ref_median) / iqr)


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


def score_distribution_summary(name: str, scores: np.ndarray) -> dict:
    s = pd.Series(scores, dtype=float)
    return {
        "population": name,
        "rows": int(len(s)),
        "mean_score": float(s.mean()),
        "std_score": float(s.std(ddof=0)),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
    }


def main() -> None:
    interim = pd.read_csv(INTERIM_PATH)
    processed = pd.read_csv(PROCESSED_PATH)

    interim[DATE] = pd.to_datetime(interim[DATE], dayfirst=True, errors="raise")
    processed[DATE] = pd.to_datetime(processed[DATE], errors="raise")

    missing_features = sorted(set(FEATURES) - set(interim.columns))
    if missing_features:
        raise ValueError(f"Missing interim features: {missing_features}")

    labelled = interim.loc[interim[DATE] <= MODELLING_END].copy()
    post = interim.loc[interim[DATE] > MODELLING_END].copy()

    if len(labelled) != 150_767:
        raise AssertionError(f"Unexpected labelled rows: {len(labelled):,}")
    if len(post) != 58_825:
        raise AssertionError(f"Unexpected post-23 July rows: {len(post):,}")

    rows = []
    for feature in FEATURES:
        ref = pd.to_numeric(labelled[feature], errors="coerce")
        cmp = pd.to_numeric(post[feature], errors="coerce")

        ref_nonmissing = ref.dropna()
        cmp_nonmissing = cmp.dropna()

        ref_min = float(ref_nonmissing.min()) if not ref_nonmissing.empty else None
        ref_max = float(ref_nonmissing.max()) if not ref_nonmissing.empty else None

        if ref_min is None or ref_max is None:
            out_of_range_rate = None
        else:
            out_of_range_rate = float(((cmp < ref_min) | (cmp > ref_max)).mean())

        feature_psi = psi(ref, cmp)
        shift = robust_scale_shift(ref, cmp)

        rows.append(
            {
                "feature": feature,
                "labelled_missing_rate": float(ref.isna().mean()),
                "post23_missing_rate": float(cmp.isna().mean()),
                "missing_rate_change": float(cmp.isna().mean() - ref.isna().mean()),
                "labelled_min": ref_min,
                "labelled_max": ref_max,
                "post23_min": float(cmp_nonmissing.min()) if not cmp_nonmissing.empty else None,
                "post23_max": float(cmp_nonmissing.max()) if not cmp_nonmissing.empty else None,
                "post23_out_of_labelled_range_rate": out_of_range_rate,
                "labelled_median": float(ref_nonmissing.median()) if not ref_nonmissing.empty else None,
                "post23_median": float(cmp_nonmissing.median()) if not cmp_nonmissing.empty else None,
                "robust_median_shift_in_labelled_iqr": shift,
                "psi": feature_psi,
                "psi_band": (
                    "low"
                    if feature_psi < 0.10
                    else "moderate"
                    if feature_psi < 0.25
                    else "high"
                ),
            }
        )

    feature_quality = pd.DataFrame(rows).sort_values("psi", ascending=False)

    # Selected-model score behaviour with frozen chronology.
    train = processed.loc[processed[DATE] <= TRAIN_END].copy()
    calibration = processed.loc[
        (processed[DATE] >= CALIBRATION_START)
        & (processed[DATE] <= CALIBRATION_END)
    ].copy()
    holdout = processed.loc[
        (processed[DATE] >= HOLDOUT_START)
        & (processed[DATE] <= HOLDOUT_END)
    ].copy()

    model = make_selected_model()
    model.fit(train[FEATURES], train[TARGET])

    p_cal_raw = np.clip(model.predict_proba(calibration[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    p_holdout_raw = np.clip(model.predict_proba(holdout[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    p_post_raw = np.clip(model.predict_proba(post[FEATURES])[:, 1], 1e-6, 1 - 1e-6)

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p_cal_raw, calibration[TARGET].to_numpy())

    p_holdout_iso = np.clip(isotonic.predict(p_holdout_raw), 1e-6, 1 - 1e-6)
    p_post_iso = np.clip(isotonic.predict(p_post_raw), 1e-6, 1 - 1e-6)

    score_summary = pd.DataFrame(
        [
            score_distribution_summary("final_holdout", p_holdout_iso),
            score_distribution_summary("post_23_july", p_post_iso),
        ]
    )

    FEATURE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    feature_quality.to_csv(FEATURE_TABLE, index=False)
    score_summary.to_csv(SCORE_TABLE, index=False)

    psi_series = feature_quality.set_index("feature")["psi"].sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    psi_series.plot(kind="barh", ax=ax)
    ax.set_title("Post-23 July feature distribution shift")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(PSI_FIGURE, dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.linspace(0, 1, 31)
    ax.hist(
        p_holdout_iso,
        bins=bins,
        density=True,
        alpha=0.55,
        label="14-23 July holdout",
    )
    ax.hist(
        p_post_iso,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Post-23 July",
    )
    ax.set_title("Selected-model calibrated score distributions")
    ax.set_xlabel("Calibrated predicted delinquency risk")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SCORE_FIGURE, dpi=170, bbox_inches="tight")
    plt.close(fig)

    high_psi = feature_quality.loc[feature_quality["psi"] >= 0.25, "feature"].tolist()
    moderate_psi = feature_quality.loc[
        (feature_quality["psi"] >= 0.10) & (feature_quality["psi"] < 0.25),
        "feature",
    ].tolist()
    material_missing_shift = feature_quality.loc[
        feature_quality["missing_rate_change"].abs() >= 0.01,
        "feature",
    ].tolist()
    material_out_of_range = feature_quality.loc[
        feature_quality["post23_out_of_labelled_range_rate"].fillna(0) >= 0.01,
        "feature",
    ].tolist()

    summary = {
        "analysis_role": "post_cutoff_feature_quality_and_score_behaviour",
        "outcome_labels_used_for_evaluation": False,
        "labelled_rows": int(len(labelled)),
        "post23_rows": int(len(post)),
        "labelled_period_start": str(labelled[DATE].min().date()),
        "labelled_period_end": str(labelled[DATE].max().date()),
        "post23_period_start": str(post[DATE].min().date()),
        "post23_period_end": str(post[DATE].max().date()),
        "feature_count": len(FEATURES),
        "high_psi_features": high_psi,
        "moderate_psi_features": moderate_psi,
        "features_with_abs_missing_rate_change_ge_1pct": material_missing_shift,
        "features_with_post23_out_of_range_rate_ge_1pct": material_out_of_range,
        "score_distribution": score_summary.to_dict(orient="records"),
        "interpretation_rule": {
            "psi_low": "<0.10",
            "psi_moderate": "0.10-0.25",
            "psi_high": ">=0.25",
            "note": (
                "PSI thresholds are used as descriptive drift bands, not as universal "
                "accept/reject criteria."
            ),
        },
        "synthetic_outcome_next_step": (
            "Only consider synthetic repayment outcomes if the post-23 July feature "
            "population is sufficiently interpretable for scenario analysis. Synthetic "
            "labels must remain explicitly separated from observed outcomes and must not "
            "be used to inflate official model performance or training evidence."
        ),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Post-23 July feature-quality analysis completed.")
    print("Post-23 July outcome labels used for evaluation: False")
    print()
    print("Feature shift summary")
    print(
        feature_quality[
            [
                "feature",
                "labelled_missing_rate",
                "post23_missing_rate",
                "post23_out_of_labelled_range_rate",
                "robust_median_shift_in_labelled_iqr",
                "psi",
                "psi_band",
            ]
        ].to_string(index=False)
    )
    print()
    print("Selected-model calibrated score distribution")
    print(score_summary.to_string(index=False))
    print()
    print("High-PSI features:", high_psi)
    print("Moderate-PSI features:", moderate_psi)
    print("Missingness shifts >=1 percentage point:", material_missing_shift)
    print("Out-of-range rates >=1%:", material_out_of_range)
    print()
    print(f"Feature table written to: {FEATURE_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Score table written to: {SCORE_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Summary written to: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"PSI figure written to: {PSI_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Score figure written to: {SCORE_FIGURE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
