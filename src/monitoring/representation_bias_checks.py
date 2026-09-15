"""Representation and bias diagnostics for the Module 3 telecom pipeline.

The source data do not contain usable demographic protected characteristics. This
suite therefore does not claim demographic fairness assessment. It measures only
attributes genuinely available in the project and keeps the main operational slice
(first-time versus returning borrowers) explicitly labelled as left-censored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame


DEFAULT_DATA = Path("data/processed/telecom_delinquency_model_ready.csv")
DEFAULT_INTERIM = Path("data/interim/sample_data_intw_cleaned.csv")
DEFAULT_TABLE = Path("reports/tables/representation_bias_by_group.csv")
DEFAULT_SUMMARY = Path("reports/tables/representation_bias_summary.json")


def _count_metric(y_true, y_pred):
    return int(len(y_true))


def _positive_rate(y_true, y_pred):
    return float(np.mean(y_true)) if len(y_true) else float("nan")


def evaluate_representation(data: pd.DataFrame, interim: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    required = {"delinquent_5d", "is_repeat_customer", "pdate"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Processed dataset is missing required columns: {missing}")

    work = data.copy()
    work["borrower_history_group"] = np.where(
        work["is_repeat_customer"].astype(int).eq(1), "returning", "first_time"
    )

    metric_frame = MetricFrame(
        metrics={"rows": _count_metric, "delinquency_rate": _positive_rate},
        y_true=work["delinquent_5d"].astype(int),
        y_pred=work["delinquent_5d"].astype(int),
        sensitive_features=work["borrower_history_group"],
    )

    by_group = metric_frame.by_group.reset_index()
    by_group["share"] = by_group["rows"] / len(work)

    group_rates = dict(zip(by_group["borrower_history_group"], by_group["delinquency_rate"]))
    returning_rate = group_rates.get("returning")
    first_time_rate = group_rates.get("first_time")
    rate_difference = (
        float(returning_rate - first_time_rate)
        if returning_rate is not None and first_time_rate is not None
        else None
    )

    work["month_position"] = pd.to_datetime(work["pdate"]).dt.strftime("%Y-%m")
    temporal = (
        work.groupby(["month_position", "borrower_history_group"], observed=True)
        .agg(rows=("delinquent_5d", "size"), delinquency_rate=("delinquent_5d", "mean"))
        .reset_index()
    )
    temporal["share_within_period"] = temporal["rows"] / temporal.groupby("month_position")["rows"].transform("sum")

    pcircle = {
        "available_for_check": False,
        "unique_nonmissing_values": None,
        "usable_for_group_comparison": False,
        "note": "pcircle is not present in the processed model-ready dataset.",
    }
    if interim is not None and "pcircle" in interim.columns:
        unique_values = interim["pcircle"].dropna().astype(str).unique().tolist()
        pcircle = {
            "available_for_check": True,
            "unique_nonmissing_values": len(unique_values),
            "usable_for_group_comparison": len(unique_values) > 1,
            "note": (
                "pcircle is constant and therefore cannot support group-comparison fairness analysis."
                if len(unique_values) <= 1
                else "pcircle has multiple values and would require substantive review before any fairness interpretation."
            ),
        }

    summary = {
        "input_rows": int(len(work)),
        "protected_characteristics_available": False,
        "protected_characteristics_note": (
            "The dataset does not provide usable demographic protected characteristics. "
            "The project therefore does not infer or manufacture them."
        ),
        "operational_slice": "first-time versus returning borrower",
        "operational_slice_is_protected_attribute": False,
        "left_censoring_caveat": (
            "Repeat-customer status is derived only from history visible after the dataset start, so apparent first-time/returning representation changes over time are partly mechanical."
        ),
        "group_metrics": by_group.to_dict(orient="records"),
        "returning_minus_first_time_delinquency_rate": rate_difference,
        "pcircle_check": pcircle,
        "fairlearn_usage": (
            "Fairlearn MetricFrame is used for transparent grouped measurement only. "
            "No Equalized Odds, demographic-parity, or other protected-group fairness claim is made because suitable protected attributes and final model predictions are not available in this dataset."
        ),
        "automatic_bias_conclusion": False,
        "interpretation_note": (
            "Observed group differences are diagnostic. They must not be interpreted as evidence of protected-group discrimination. "
            "The first-time/returning slice is operational and affected by left-censoring."
        ),
        "temporal_group_metrics": temporal.to_dict(orient="records"),
    }
    return by_group, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run representation and bias diagnostics.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--interim", default=str(DEFAULT_INTERIM))
    parser.add_argument("--table", default=str(DEFAULT_TABLE))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    interim_path = Path(args.interim)
    interim = pd.read_csv(interim_path) if interim_path.exists() else None

    table, summary = evaluate_representation(data, interim)

    table_path = Path(args.table)
    summary_path = Path(args.summary)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Representation table written to {table_path}")
    print(f"Representation summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
