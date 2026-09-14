"""Validate the cleaned interim telecom dataset with Great Expectations.

The suite checks that cleaning resolved the hard semantic failures found in the raw data
while preserving the expected schema and structural fields. Integer semantics for count
fields are checked on non-missing values with a small pandas guard because cleaned invalid
count values are intentionally represented as missing, which causes CSV ingestion to use
floating dtypes even when every retained value is mathematically integral.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import great_expectations as gx
import pandas as pd


EXPECTED_COLUMNS = [
    "label",
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

CLEANED_DURATION_FIELDS = [
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
        help="Path to the cleaned interim telecom CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tables/gx_interim_validation.json"),
        help="Where to write the combined validation result.",
    )
    return parser.parse_args()


def load_validation_copy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pdate"] = pd.to_datetime(df["pdate"], dayfirst=True, errors="coerce")
    return df


def check_integer_semantics(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Check that every retained non-missing count value is mathematically integral."""
    results: dict[str, dict[str, object]] = {}
    for field in COUNT_FIELDS:
        values = pd.to_numeric(df[field], errors="coerce")
        nonmissing = values.dropna()
        fractional_mask = (nonmissing % 1).abs() > 1e-12
        fractional_count = int(fractional_mask.sum())
        results[field] = {
            "success": fractional_count == 0,
            "fractional_count": fractional_count,
            "retained_nonmissing_count": int(nonmissing.shape[0]),
        }
    return results


def build_suite(context: gx.data_context.AbstractDataContext):
    suite = context.suites.add(
        gx.ExpectationSuite(name="telecom_interim_blocking_expectations")
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

    # Structural fields must remain complete after cleaning.
    for field in ["msisdn", "pdate", "label"]:
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

    # Invalid negative values should have become missing; retained values must be valid.
    for field in CLEANED_DURATION_FIELDS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=field,
                min_value=0,
                severity="critical",
            )
        )

    # Cleaned count columns may contain missing values where invalid source values were
    # removed, but any retained value must be non-negative.
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
    os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas("telecom_interim_pandas")
    asset = data_source.add_dataframe_asset(name="telecom_interim_dataframe")
    batch_definition = asset.add_batch_definition_whole_dataframe(
        "telecom_interim_whole_dataframe"
    )

    suite = build_suite(context)
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="telecom_interim_blocking_validation",
            data=batch_definition,
            suite=suite,
        )
    )
    return validation_definition.run(batch_parameters={"dataframe": df})


def main() -> int:
    args = parse_args()
    df = load_validation_copy(args.data)

    gx_result = run_validation(df)
    integer_checks = check_integer_semantics(df)
    integer_success = all(item["success"] for item in integer_checks.values())

    combined = {
        "gx_success": bool(gx_result.success),
        "integer_semantics_success": integer_success,
        "overall_success": bool(gx_result.success) and integer_success,
        "integer_semantics": integer_checks,
        "great_expectations": gx_result.to_json_dict(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, default=str)

    print(gx_result.describe())
    print("Integer semantics checks:")
    print(json.dumps(integer_checks, indent=2))
    print(f"Overall cleaned-data validation success: {combined['overall_success']}")
    print(f"Validation result written to {args.output}")

    return 0 if combined["overall_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
