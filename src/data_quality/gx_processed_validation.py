"""Validate the first model-ready telecom delinquency dataset.

This validation checks that the processed output has the approved structure and
retains the point-in-time and governance constraints defined for Module 3.
Cleaning-introduced missing values are allowed in approved predictors where the
pipeline deliberately converted invalid/contaminated source values to missing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import great_expectations as gx


MODELLING_CUTOFF = pd.Timestamp("2016-07-23")

APPROVED_PREDICTORS = [
    "aon",
    "daily_decr30",
    "daily_decr90",
    "rental30",
    "rental90",
    "last_rech_date_ma",
    "last_rech_date_da",
    "last_rech_amt_ma",
    "cnt_ma_rech30",
    "sumamnt_ma_rech30",
    "medianamnt_ma_rech30",
    "medianmarechprebal30",
    "cnt_ma_rech90",
    "sumamnt_ma_rech90",
    "medianamnt_ma_rech90",
    "medianmarechprebal90",
    "cnt_da_rech30",
    "cnt_da_rech90",
    "prior_tx_count",
    "is_repeat_customer",
]

EXPECTED_COLUMNS = ["pdate", *APPROVED_PREDICTORS, "delinquent_5d"]

NONNEGATIVE_DURATION_FIELDS = ["aon", "last_rech_date_ma", "last_rech_date_da"]
NONNEGATIVE_AMOUNT_FIELDS = [
    "last_rech_amt_ma",
    "sumamnt_ma_rech30",
    "sumamnt_ma_rech90",
    "medianamnt_ma_rech30",
    "medianamnt_ma_rech90",
]
NONNEGATIVE_COUNT_FIELDS = [
    "cnt_ma_rech30",
    "cnt_ma_rech90",
    "cnt_da_rech30",
    "cnt_da_rech90",
    "prior_tx_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tables/gx_processed_validation.json"),
    )
    return parser.parse_args()


def load_validation_copy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pdate"] = pd.to_datetime(df["pdate"], errors="coerce")
    return df


def build_suite(context: gx.data_context.AbstractDataContext):
    suite = context.suites.add(
        gx.ExpectationSuite(name="telecom_processed_blocking_expectations")
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
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="pdate",
            severity="critical",
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="pdate",
            max_value=MODELLING_CUTOFF,
            severity="critical",
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="delinquent_5d",
            severity="critical",
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="delinquent_5d",
            value_set=[0, 1],
            severity="critical",
        )
    )

    for field in NONNEGATIVE_DURATION_FIELDS:
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

    for field in NONNEGATIVE_COUNT_FIELDS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=field,
                min_value=0,
                severity="critical",
            )
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="prior_tx_count",
            severity="critical",
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="is_repeat_customer",
            severity="critical",
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="is_repeat_customer",
            value_set=[0, 1],
            severity="critical",
        )
    )

    return suite


def run_validation(df: pd.DataFrame):
    os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("telecom_processed_pandas")
    asset = data_source.add_dataframe_asset(name="telecom_processed_dataframe")
    batch_definition = asset.add_batch_definition_whole_dataframe(
        "telecom_processed_whole_dataframe"
    )
    suite = build_suite(context)
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="telecom_processed_blocking_validation",
            data=batch_definition,
            suite=suite,
        )
    )
    return validation_definition.run(batch_parameters={"dataframe": df})


def supplemental_checks(df: pd.DataFrame) -> dict:
    prior_numeric = pd.to_numeric(df["prior_tx_count"], errors="coerce")
    prior_fractional = int(((prior_numeric % 1) != 0).fillna(False).sum())
    repeat_expected = (prior_numeric > 0).astype("int8")
    repeat_actual = pd.to_numeric(df["is_repeat_customer"], errors="coerce")
    repeat_mismatch = int((repeat_expected != repeat_actual).sum())

    forbidden_columns = [column for column in ["msisdn", "label"] if column in df.columns]

    return {
        "prior_tx_count_fractional_count": prior_fractional,
        "repeat_customer_logic_mismatch_count": repeat_mismatch,
        "forbidden_columns_present": forbidden_columns,
        "success": (
            prior_fractional == 0
            and repeat_mismatch == 0
            and len(forbidden_columns) == 0
        ),
    }


def main() -> int:
    args = parse_args()
    df = load_validation_copy(args.data)
    result = run_validation(df)
    supplemental = supplemental_checks(df)

    payload = {
        "great_expectations": result.to_json_dict(),
        "supplemental_checks": supplemental,
        "overall_success": bool(result.success and supplemental["success"]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    print(result.describe())
    print(json.dumps(supplemental, indent=2))
    print(f"Overall processed-data validation success: {payload['overall_success']}")
    print(f"Validation result written to {args.output}")

    return 0 if payload["overall_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
