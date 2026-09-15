"""Run Stage 2 filter-method feature screening on the model-ready dataset.

This script keeps feature screening leakage-aware by fitting all screening statistics on
an earlier temporal development subset only. The later subset is reserved and is not
used to rank or remove features.

Outputs include:
- per-feature relevance and redundancy indicators;
- highly correlated predictor pairs;
- an aggregate JSON summary;
- a correlation heatmap for presentation/reporting.

The script does not automatically remove features. It produces evidence for the next
feature-selection stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer


TARGET = "delinquent_5d"
DATE_FIELD = "pdate"
RANDOM_STATE = 42
REDUNDANCY_THRESHOLD = 0.90
DOMINANCE_THRESHOLD = 0.995

DISCRETE_FEATURES = {
    "cnt_ma_rech30",
    "cnt_ma_rech90",
    "cnt_da_rech30",
    "cnt_da_rech90",
    "prior_tx_count",
    "is_repeat_customer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("reports/tables/filter_feature_screening.csv"),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("reports/tables/high_correlation_pairs.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/filter_feature_screening_summary.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("reports/figures/filter_correlation_heatmap.png"),
    )
    parser.add_argument(
        "--development-fraction",
        type=float,
        default=0.80,
        help="Approximate earlier fraction used for feature screening.",
    )
    return parser.parse_args()


def temporal_development_split(
    df: pd.DataFrame, fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if not 0.5 <= fraction < 1.0:
        raise ValueError("development-fraction must be between 0.5 and 1.0")

    ordered = df.sort_values(DATE_FIELD, kind="stable").copy()
    target_index = max(0, min(len(ordered) - 1, int(np.floor(len(ordered) * fraction)) - 1))
    cutoff = ordered.iloc[target_index][DATE_FIELD]

    development = ordered.loc[ordered[DATE_FIELD] <= cutoff].copy()
    holdout = ordered.loc[ordered[DATE_FIELD] > cutoff].copy()

    if development.empty or holdout.empty:
        raise ValueError("Temporal split produced an empty development or holdout set.")

    return development, holdout, cutoff


def high_correlation_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= REDUNDANCY_THRESHOLD:
                rows.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "spearman_correlation": float(value),
                        "absolute_correlation": float(abs(value)),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "spearman_correlation",
                "absolute_correlation",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        "absolute_correlation", ascending=False, kind="stable"
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

    features = [c for c in df.columns if c not in {DATE_FIELD, TARGET}]
    if not features:
        raise ValueError("No predictor columns found.")

    development, holdout, cutoff = temporal_development_split(
        df, args.development_fraction
    )

    X_dev = development[features]
    y_dev = development[TARGET].astype(int)

    # Fit imputation only on the development subset. The imputed matrix is used only
    # for mutual-information screening; the model-ready dataset itself is not changed.
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X_dev), columns=features, index=X_dev.index
    )

    discrete_mask = np.array([feature in DISCRETE_FEATURES for feature in features])
    mi = mutual_info_classif(
        X_imp,
        y_dev,
        discrete_features=discrete_mask,
        random_state=RANDOM_STATE,
    )

    pearson_target = X_dev.corrwith(y_dev, method="pearson")
    spearman_target = X_dev.corrwith(y_dev, method="spearman")

    rows: list[dict] = []
    for idx, feature in enumerate(features):
        series = X_dev[feature]
        nonmissing = series.dropna()
        value_counts = nonmissing.value_counts(normalize=True, dropna=False)
        dominant_share = float(value_counts.iloc[0]) if not value_counts.empty else np.nan
        nunique = int(nonmissing.nunique())
        variance = float(nonmissing.var()) if len(nonmissing) > 1 else 0.0

        rows.append(
            {
                "feature": feature,
                "missing_count_development": int(series.isna().sum()),
                "missing_rate_development": float(series.isna().mean()),
                "nunique_development": nunique,
                "variance_development": variance,
                "dominant_value_share": dominant_share,
                "constant_flag": bool(nunique <= 1),
                "near_constant_flag": bool(
                    pd.notna(dominant_share) and dominant_share >= DOMINANCE_THRESHOLD
                ),
                "pearson_with_target": (
                    float(pearson_target[feature])
                    if pd.notna(pearson_target[feature])
                    else None
                ),
                "abs_pearson_with_target": (
                    float(abs(pearson_target[feature]))
                    if pd.notna(pearson_target[feature])
                    else None
                ),
                "spearman_with_target": (
                    float(spearman_target[feature])
                    if pd.notna(spearman_target[feature])
                    else None
                ),
                "abs_spearman_with_target": (
                    float(abs(spearman_target[feature]))
                    if pd.notna(spearman_target[feature])
                    else None
                ),
                "mutual_information": float(mi[idx]),
                "treated_as_discrete_for_mi": bool(discrete_mask[idx]),
            }
        )

    screening = pd.DataFrame(rows).sort_values(
        "mutual_information", ascending=False, kind="stable"
    )

    spearman_matrix = X_dev.corr(method="spearman")
    pairs = high_correlation_pairs(spearman_matrix)

    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.pairs.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)

    screening.to_csv(args.table, index=False)
    pairs.to_csv(args.pairs, index=False)

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(spearman_matrix.to_numpy(), vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=7)
    ax.set_title("Spearman correlation among Stage 1-approved predictors\n(training-only development subset)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_rows": int(len(df)),
        "development_rows": int(len(development)),
        "holdout_rows": int(len(holdout)),
        "development_fraction_requested": float(args.development_fraction),
        "temporal_cutoff_inclusive": cutoff.date().isoformat(),
        "development_date_min": development[DATE_FIELD].min().date().isoformat(),
        "development_date_max": development[DATE_FIELD].max().date().isoformat(),
        "holdout_date_min": holdout[DATE_FIELD].min().date().isoformat(),
        "holdout_date_max": holdout[DATE_FIELD].max().date().isoformat(),
        "development_delinquency_rate": float(y_dev.mean()),
        "holdout_delinquency_rate": float(holdout[TARGET].mean()),
        "feature_count": int(len(features)),
        "constant_features": screening.loc[screening["constant_flag"], "feature"].tolist(),
        "near_constant_features": screening.loc[
            screening["near_constant_flag"], "feature"
        ].tolist(),
        "high_correlation_pair_count": int(len(pairs)),
        "redundancy_threshold_abs_spearman": REDUNDANCY_THRESHOLD,
        "top_10_mutual_information": screening.head(10)[
            ["feature", "mutual_information"]
        ].to_dict(orient="records"),
        "automatic_feature_removal": False,
        "note": (
            "Filter outputs are diagnostic evidence only. Feature removal decisions are "
            "deferred until later embedded/wrapper and stability checks."
        ),
    }

    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Feature screening table written to {args.table}")
    print(f"High-correlation pairs written to {args.pairs}")
    print(f"Correlation heatmap written to {args.figure}")
    print(f"Screening summary written to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
