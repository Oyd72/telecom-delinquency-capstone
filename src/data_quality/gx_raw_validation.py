"""Run the initial blocking Great Expectations suite against the raw telecom dataset.

This script intentionally validates only rules supported by the current data dictionary
and cleaning policy. Dataset-specific anomaly thresholds remain diagnostic and are not
encoded here as universal validity limits.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import great_expectations as gx


EXPECTED_COLUMNS = [
    "msisdn",
    "aon",
    "daily_decr30",
    "daily_decr90",
    "rental30",
    "rental90",
    "last_rech_date_ma",
    "last_rech_date_da",
    "last_rech_amt_ma",
    "cnt_ma_rech30",
    "fr_ma_rech30",
    "sumamnt_ma_rech30",
    "medianamnt_ma_rech30",
    "medianmarechprebal30",
    "cnt_ma_rech90",
    "fr_ma_rech90",
    "sumamnt_ma_rech90",
    "medianamnt_ma_rech90",
    "medianmarechprebal90",
    "cnt_da_rech30",
    "fr_da_rech30",
    "cnt_da_rech90",
    "fr_da_rech90",
    "cnt_loans30",
    "amnt_loans30",
    "maxamnt_loans30",
    "medianamnt_loans30",
    "cnt_loans90",
    "amnt_loans90",
    "maxamnt_loans90",
    "medianamnt_loans90",
    "payback30",
    "payback90",
    "pcircle",
    "pdate",
    "label",
]

COUNT_FIELDS = [
    "cnt_ma_rech30",
    "cnt_ma_rech90",
    "cnt_da_rech30",
    "cnt_da_rech90",
    "cnt_loans30",
    "cnt_loans90",
]

NONNEGATIVE_AMOUNT_FIELDS = [
    "last_rech_amt_ma",
    "sumamnt_ma_rech30",
    "sumamnt_ma_rech90",
    "medianamnt_ma_rech30",
    "medianamnt_ma_rech90",
    "amnt_loans30",
    "amnt_loans90",
]

DURATION_FIELDS_WITH_HARD_NONNEGATIVE_RULE = [
    "aon",
    "last_rech_date_ma",
    "last_rech_date_da",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to the raw telecom delinquency CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tables/gx_raw_validation.json"),
        help="Where to write the JSON validation result.",
    )
    return parser.parse_args()


def load_validation_copy(path: Path) -> pd.DataFrame:
    """Load a validation copy without modifying the raw source file."""
    df = pd.read_csv(path)
    # Convert only in memory so parse failures become null and can be caught by GX.
    df["pdate"] = pd.to_datetime(df["pdate"], dayfirst=True, errors="coerce")
    return df


def build_suite(context: gx.data_context.AbstractDataContext):
    suite = context.suites.add(
        gx.ExpectationSuite(name="telecom_raw_blocking_expectations")
    )

    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=EXPECTED_COLUMNS,
            severity="critical",
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=5000,
            severity="critical",
        )
    )

    for field in ["msisdn", "pdate", "label", *COUNT_FIELDS]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=field,
                severity="critical",
            )
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="label",
            value_set=[0, 1],
            severity="critical",
        )
    )

    for field in DURATION_FIELDS_WITH_HARD_NONNEGATIVE_RULE:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=field,
                min_value=0,
                severity="critical",
            )
        )

    for field in COUNT_FIELDS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=field,
                min_value=0,
                severity="critical",
            )
        )

    for field in NONNEGATIVE_AMOUNT_FIELDS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=field,
                min_value=0,
                severity="critical",
            )
        )

    return suite


def run_validation(df: pd.DataFrame):
    # Disable GX telemetry for this academic pipeline run.
    os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas("telecom_pandas")
    asset = data_source.add_dataframe_asset(name="telecom_raw_dataframe")
    batch_definition = asset.add_batch_definition_whole_dataframe(
        "telecom_raw_whole_dataframe"
    )

    suite = build_suite(context)

    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="telecom_raw_blocking_validation",
            data=batch_definition,
            suite=suite,
        )
    )

    return validation_definition.run(batch_parameters={"dataframe": df})


def main() -> int:
    args = parse_args()
    df = load_validation_copy(args.data)
    result = run_validation(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result.to_json_dict(), handle, indent=2, default=str)

    print(result.describe())
    print(f"Validation result written to {args.output}")

    # A failing raw-data suite is expected to reveal cleaning needs. Return non-zero so
    # orchestration/CI can distinguish a failed blocking validation from a clean run.
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
