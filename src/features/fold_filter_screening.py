"""Run Stage 2 filter screening across calendar-aware chronological folds.

The purpose is to assess whether feature relevance is stable across short temporal
segments instead of relying on one arbitrary 80/20 chronological block. Because the
modelling period contains only June and part of July 2016, the folds are aligned to
comparable early/late month positions where possible:

- June 1-13
- June 14-30
- July 1-13
- July 14-23

For each fold the script calculates Pearson and Spearman association with the target,
mutual information, missingness, and dominant-value share. Mutual-information
imputation is fitted independently inside each fold.

A sensitivity view is also produced with the observation-window-sensitive variables
`prior_tx_count` and `is_repeat_customer` excluded. No feature is removed
automatically.
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
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer


TARGET = "delinquent_5d"
DATE_FIELD = "pdate"
RANDOM_STATE = 42
HISTORY_FEATURES = {"prior_tx_count", "is_repeat_customer"}
DISCRETE_FEATURES = {
    "cnt_ma_rech30",
    "cnt_ma_rech90",
    "cnt_da_rech30",
    "cnt_da_rech90",
    "prior_tx_count",
    "is_repeat_customer",
}

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
        "--fold-table",
        type=Path,
        default=Path("reports/tables/filter_fold_metrics.csv"),
    )
    parser.add_argument(
        "--stability-table",
        type=Path,
        default=Path("reports/tables/filter_stability_summary.csv"),
    )
    parser.add_argument(
        "--sensitivity-table",
        type=Path,
        default=Path("reports/tables/filter_stability_without_history.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/filter_fold_screening_summary.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("reports/figures/filter_mi_stability.png"),
    )
    return parser.parse_args()


def _fold_metrics(
    fold: pd.DataFrame,
    features: list[str],
    fold_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    X = fold[features]
    y = fold[TARGET].astype(int)

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X), columns=features, index=X.index
    )
    discrete_mask = np.array([feature in DISCRETE_FEATURES for feature in features])
    mi = mutual_info_classif(
        X_imp,
        y,
        discrete_features=discrete_mask,
        random_state=RANDOM_STATE,
    )

    pearson = X.corrwith(y, method="pearson")
    spearman = X.corrwith(y, method="spearman")

    rows: list[dict] = []
    for idx, feature in enumerate(features):
        series = X[feature]
        nonmissing = series.dropna()
        shares = nonmissing.value_counts(normalize=True)
        dominant_share = float(shares.iloc[0]) if not shares.empty else np.nan

        rows.append(
            {
                "fold": fold_name,
                "fold_start": start.date().isoformat(),
                "fold_end": end.date().isoformat(),
                "fold_rows": int(len(fold)),
                "fold_delinquency_rate": float(y.mean()),
                "feature": feature,
                "missing_rate": float(series.isna().mean()),
                "dominant_value_share": dominant_share,
                "pearson_with_target": float(pearson[feature]) if pd.notna(pearson[feature]) else np.nan,
                "spearman_with_target": float(spearman[feature]) if pd.notna(spearman[feature]) else np.nan,
                "mutual_information": float(mi[idx]),
            }
        )

    result = pd.DataFrame(rows)
    result["mi_rank_within_fold"] = result["mutual_information"].rank(
        ascending=False, method="average"
    )
    result["abs_spearman_rank_within_fold"] = result["spearman_with_target"].abs().rank(
        ascending=False, method="average"
    )
    return result


def _aggregate_stability(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby("feature", as_index=False).agg(
        folds_observed=("fold", "nunique"),
        mean_mutual_information=("mutual_information", "mean"),
        std_mutual_information=("mutual_information", "std"),
        min_mutual_information=("mutual_information", "min"),
        max_mutual_information=("mutual_information", "max"),
        mean_mi_rank=("mi_rank_within_fold", "mean"),
        std_mi_rank=("mi_rank_within_fold", "std"),
        mean_abs_spearman=("spearman_with_target", lambda s: float(s.abs().mean())),
        std_abs_spearman=("spearman_with_target", lambda s: float(s.abs().std())),
        mean_abs_spearman_rank=("abs_spearman_rank_within_fold", "mean"),
        mean_missing_rate=("missing_rate", "mean"),
        max_missing_rate=("missing_rate", "max"),
    )

    # Rank-stability ratio is diagnostic only: lower values imply a more stable rank.
    grouped["mi_rank_cv"] = grouped["std_mi_rank"] / grouped["mean_mi_rank"].replace(0, np.nan)
    grouped = grouped.sort_values(
        ["mean_mi_rank", "std_mi_rank"], ascending=[True, True], kind="stable"
    )
    return grouped


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
    all_metrics: list[pd.DataFrame] = []
    fold_summary: list[dict] = []

    for fold_name, start_raw, end_raw in FOLDS:
        start = pd.Timestamp(start_raw)
        end = pd.Timestamp(end_raw)
        fold = df.loc[df[DATE_FIELD].between(start, end, inclusive="both")].copy()
        if fold.empty:
            raise ValueError(f"Calendar-aware fold {fold_name} is empty.")

        fold_summary.append(
            {
                "fold": fold_name,
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "rows": int(len(fold)),
                "delinquency_rate": float(fold[TARGET].mean()),
                "repeat_customer_share": (
                    float(fold["is_repeat_customer"].mean())
                    if "is_repeat_customer" in fold.columns
                    else None
                ),
            }
        )
        all_metrics.append(_fold_metrics(fold, features, fold_name, start, end))

    fold_metrics = pd.concat(all_metrics, ignore_index=True)
    stability = _aggregate_stability(fold_metrics)

    sensitivity_metrics = fold_metrics.loc[
        ~fold_metrics["feature"].isin(HISTORY_FEATURES)
    ].copy()
    sensitivity = _aggregate_stability(sensitivity_metrics)

    args.fold_table.parent.mkdir(parents=True, exist_ok=True)
    args.stability_table.parent.mkdir(parents=True, exist_ok=True)
    args.sensitivity_table.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)

    fold_metrics.to_csv(args.fold_table, index=False)
    stability.to_csv(args.stability_table, index=False)
    sensitivity.to_csv(args.sensitivity_table, index=False)

    top_features = stability.head(12)["feature"].tolist()
    plot = fold_metrics.loc[fold_metrics["feature"].isin(top_features)].copy()
    pivot = plot.pivot(index="feature", columns="fold", values="mutual_information")
    pivot = pivot.reindex(top_features)

    fig, ax = plt.subplots(figsize=(10, max(5, len(top_features) * 0.42)))
    image = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mutual-information stability across calendar-aware folds")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_rows": int(len(df)),
        "folds": fold_summary,
        "feature_count": int(len(features)),
        "history_features_treated_as_sensitivity": sorted(HISTORY_FEATURES),
        "top_10_by_mean_mi_rank": stability.head(10)[
            [
                "feature",
                "mean_mutual_information",
                "std_mutual_information",
                "mean_mi_rank",
                "std_mi_rank",
                "mean_abs_spearman",
            ]
        ].to_dict(orient="records"),
        "top_10_without_history_features": sensitivity.head(10)[
            [
                "feature",
                "mean_mutual_information",
                "std_mutual_information",
                "mean_mi_rank",
                "std_mi_rank",
                "mean_abs_spearman",
            ]
        ].to_dict(orient="records"),
        "automatic_feature_removal": False,
        "interpretation_note": (
            "Filter statistics are compared across calendar-aware chronological folds. "
            "The folds are diagnostics, not proof of recurring seasonality; only two partial months are available. "
            "Repeat-customer history variables are analysed separately because of left-censoring."
        ),
    }

    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Per-fold metrics written to {args.fold_table}")
    print(f"Stability summary written to {args.stability_table}")
    print(f"Sensitivity summary written to {args.sensitivity_table}")
    print(f"MI stability figure written to {args.figure}")
    print(f"Screening summary written to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
