"""Diagnose intra-month temporal patterns and observation-window effects.

This analysis is exploratory. It tests two plausible explanations for the drift observed
between earlier and later modelling periods without assuming either explanation is true:

1. calendar/pay-cycle effects: borrowing and repayment behaviour may vary by position in
   the month;
2. observation-window effects: repeat-customer indicators may rise mechanically because
   the dataset begins on 1 June 2016 and earlier customer history is unobserved.

The script therefore combines the model-ready dataset with the controlled interim dataset
(which still contains ``msisdn``) and produces descriptive diagnostics only. It does not
change the modelling population or automatically remove any feature.
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


DATE_FIELD = "pdate"
TARGET = "delinquent_5d"
MODELLING_CUTOFF = pd.Timestamp("2016-07-23")
KEY_DRIFT_FEATURES = ["daily_decr30", "daily_decr90", "rental30", "rental90"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True, type=Path)
    parser.add_argument("--interim", required=True, type=Path)
    parser.add_argument(
        "--daily-table",
        type=Path,
        default=Path("reports/tables/temporal_daily_patterns.csv"),
    )
    parser.add_argument(
        "--month-position-table",
        type=Path,
        default=Path("reports/tables/month_position_patterns.csv"),
    )
    parser.add_argument(
        "--paired-dom-table",
        type=Path,
        default=Path("reports/tables/paired_day_of_month_patterns.csv"),
    )
    parser.add_argument(
        "--window-table",
        type=Path,
        default=Path("reports/tables/observation_window_diagnostics.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/temporal_pattern_summary.json"),
    )
    parser.add_argument(
        "--figure-daily",
        type=Path,
        default=Path("reports/figures/daily_delinquency_repeat_share.png"),
    )
    parser.add_argument(
        "--figure-drift",
        type=Path,
        default=Path("reports/figures/daily_key_feature_medians.png"),
    )
    return parser.parse_args()


def _month_position(day: pd.Series) -> pd.Categorical:
    bins = [0, 7, 15, 23, 31]
    labels = ["days_01_07", "days_08_15", "days_16_23", "days_24_end"]
    return pd.cut(day, bins=bins, labels=labels, include_lowest=True, right=True)


def _daily_patterns(df: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "records": (TARGET, "size"),
        "delinquency_rate": (TARGET, "mean"),
        "repeat_customer_share": ("is_repeat_customer", "mean"),
        "prior_tx_median": ("prior_tx_count", "median"),
        "prior_tx_mean": ("prior_tx_count", "mean"),
    }
    for feature in KEY_DRIFT_FEATURES:
        aggregations[f"{feature}_median"] = (feature, "median")
        aggregations[f"{feature}_mean"] = (feature, "mean")

    daily = df.groupby(DATE_FIELD).agg(**aggregations).reset_index()
    daily["month"] = daily[DATE_FIELD].dt.to_period("M").astype(str)
    daily["day_of_month"] = daily[DATE_FIELD].dt.day
    daily["days_since_start"] = (daily[DATE_FIELD] - daily[DATE_FIELD].min()).dt.days
    return daily


def _month_position_patterns(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["month"] = temp[DATE_FIELD].dt.to_period("M").astype(str)
    temp["day_of_month"] = temp[DATE_FIELD].dt.day
    temp["month_position"] = _month_position(temp["day_of_month"])

    agg: dict[str, tuple[str, str]] = {
        "records": (TARGET, "size"),
        "delinquency_rate": (TARGET, "mean"),
        "repeat_customer_share": ("is_repeat_customer", "mean"),
        "prior_tx_median": ("prior_tx_count", "median"),
    }
    for feature in KEY_DRIFT_FEATURES:
        agg[f"{feature}_median"] = (feature, "median")

    return (
        temp.groupby(["month", "month_position"], observed=True)
        .agg(**agg)
        .reset_index()
    )


def _paired_day_of_month(daily: pd.DataFrame) -> pd.DataFrame:
    # Only days present in both June and July can provide a like-for-like calendar-position
    # comparison. With only two observed months this is descriptive, not proof of seasonality.
    metrics = [
        "delinquency_rate",
        "repeat_customer_share",
        *[f"{feature}_median" for feature in KEY_DRIFT_FEATURES],
    ]
    june = daily.loc[daily["month"] == "2016-06", ["day_of_month", *metrics]].copy()
    july = daily.loc[daily["month"] == "2016-07", ["day_of_month", *metrics]].copy()
    paired = june.merge(july, on="day_of_month", suffixes=("_jun", "_jul"), how="inner")
    for metric in metrics:
        paired[f"{metric}_difference_jul_minus_jun"] = (
            paired[f"{metric}_jul"] - paired[f"{metric}_jun"]
        )
    return paired.sort_values("day_of_month")


def _observation_window_diagnostics(interim: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"msisdn", "pdate"}
    missing = required - set(interim.columns)
    if missing:
        raise ValueError(f"Interim data missing required columns: {sorted(missing)}")

    work = interim.copy()
    work["pdate"] = pd.to_datetime(work["pdate"], dayfirst=True, errors="coerce")
    work = work.loc[work["pdate"] <= MODELLING_CUTOFF].copy()
    if work["pdate"].isna().any():
        raise ValueError("Unparseable pdate values found in interim data.")

    first_seen = work.groupby("msisdn")["pdate"].transform("min")
    work["first_observed_date"] = first_seen
    work["observed_history_days"] = (work["pdate"] - first_seen).dt.days
    work["newly_observed_customer"] = (work["observed_history_days"] == 0).astype(int)

    daily = (
        work.groupby("pdate")
        .agg(
            records=("msisdn", "size"),
            unique_customers=("msisdn", "nunique"),
            newly_observed_customer_share=("newly_observed_customer", "mean"),
            observed_history_days_median=("observed_history_days", "median"),
            observed_history_days_mean=("observed_history_days", "mean"),
        )
        .reset_index()
    )
    daily["days_since_start"] = (daily["pdate"] - daily["pdate"].min()).dt.days

    corr_new = daily["days_since_start"].corr(daily["newly_observed_customer_share"], method="spearman")
    corr_hist = daily["days_since_start"].corr(daily["observed_history_days_mean"], method="spearman")

    summary = {
        "interim_modelling_period_rows": int(len(work)),
        "unique_customers": int(work["msisdn"].nunique()),
        "dataset_start": work["pdate"].min().date().isoformat(),
        "dataset_end_used": work["pdate"].max().date().isoformat(),
        "spearman_days_since_start_vs_new_customer_share": float(corr_new),
        "spearman_days_since_start_vs_mean_observed_history_days": float(corr_hist),
        "left_censoring_note": (
            "Customer history before the dataset start is unobserved. A rising repeat-customer "
            "share or observed-history length over calendar time may therefore be partly mechanical."
        ),
    }
    return daily, summary


def main() -> int:
    args = parse_args()
    processed = pd.read_csv(args.processed)
    interim = pd.read_csv(args.interim)

    required_processed = {
        DATE_FIELD,
        TARGET,
        "is_repeat_customer",
        "prior_tx_count",
        *KEY_DRIFT_FEATURES,
    }
    missing = required_processed - set(processed.columns)
    if missing:
        raise ValueError(f"Processed data missing required columns: {sorted(missing)}")

    processed[DATE_FIELD] = pd.to_datetime(processed[DATE_FIELD], errors="coerce")
    if processed[DATE_FIELD].isna().any():
        raise ValueError("Unparseable pdate values found in processed data.")

    daily = _daily_patterns(processed)
    month_position = _month_position_patterns(processed)
    paired = _paired_day_of_month(daily)
    window_daily, window_summary = _observation_window_diagnostics(interim)

    # Calendar-position similarity across June and July. With only two months these are
    # descriptive signals and must not be interpreted as established salary-cycle effects.
    paired_correlations: dict[str, float | None] = {}
    for metric in [
        "delinquency_rate",
        "repeat_customer_share",
        *[f"{feature}_median" for feature in KEY_DRIFT_FEATURES],
    ]:
        left = paired[f"{metric}_jun"]
        right = paired[f"{metric}_jul"]
        correlation = left.corr(right, method="spearman") if len(paired) >= 3 else np.nan
        paired_correlations[metric] = None if pd.isna(correlation) else float(correlation)

    summary = {
        "processed_rows": int(len(processed)),
        "processed_date_min": processed[DATE_FIELD].min().date().isoformat(),
        "processed_date_max": processed[DATE_FIELD].max().date().isoformat(),
        "paired_day_of_month_count": int(len(paired)),
        "paired_june_july_spearman": paired_correlations,
        "observation_window": window_summary,
        "interpretation_note": (
            "The diagnostics test, but do not assume, calendar/pay-cycle effects and left-censoring. "
            "Only two partial calendar months are available, so recurring monthly seasonality cannot "
            "be established conclusively from this dataset alone."
        ),
        "automatic_split_decision": False,
    }

    for path in [
        args.daily_table,
        args.month_position_table,
        args.paired_dom_table,
        args.window_table,
        args.summary,
        args.figure_daily,
        args.figure_drift,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    daily.to_csv(args.daily_table, index=False)
    month_position.to_csv(args.month_position_table, index=False)
    paired.to_csv(args.paired_dom_table, index=False)
    window_daily.to_csv(args.window_table, index=False)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily[DATE_FIELD], daily["delinquency_rate"], label="Delinquency rate")
    ax.plot(daily[DATE_FIELD], daily["repeat_customer_share"], label="Repeat-customer share")
    ax.set_xlabel("Date")
    ax.set_ylabel("Share")
    ax.set_title("Daily delinquency and observed repeat-customer share")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.figure_daily, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for feature in KEY_DRIFT_FEATURES:
        ax.plot(daily[DATE_FIELD], daily[f"{feature}_median"], label=feature)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily median")
    ax.set_title("Daily medians of strongest temporal-drift features")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.figure_drift, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Daily temporal patterns written to {args.daily_table}")
    print(f"Month-position patterns written to {args.month_position_table}")
    print(f"Paired day-of-month comparison written to {args.paired_dom_table}")
    print(f"Observation-window diagnostics written to {args.window_table}")
    print(f"Summary written to {args.summary}")
    print(f"Figures written to {args.figure_daily} and {args.figure_drift}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
