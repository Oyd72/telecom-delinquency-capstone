"""Build the first model-ready telecom delinquency dataset from interim-cleaned data.

The script:
- parses and checks the decision date;
- derives strictly prior customer-participation features using `msisdn`;
- restricts the labelled modelling population to records through 23 July 2016;
- converts the source success label into a delinquency target;
- keeps only the approved predictor set plus the temporal control field;
- removes `msisdn` and all excluded/restricted fields from the processed output;
- preserves cleaning-introduced missing values for later leakage-safe imputation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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

CONTROL_FIELDS = ["pdate"]
TARGET_FIELD = "delinquent_5d"

REQUIRED_INPUT_FIELDS = [
    "label",
    "msisdn",
    "pdate",
    *[field for field in APPROVED_PREDICTORS if field not in {"prior_tx_count", "is_repeat_customer"}],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to the validated interim-cleaned CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/telecom_delinquency_model_ready.csv"),
        help="Path for the model-ready CSV.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/model_dataset_summary.json"),
        help="Path for an aggregate model-dataset summary.",
    )
    return parser.parse_args()


def _derive_strictly_prior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add prior transaction counts based only on strictly earlier dates.

    Transactions occurring on the same date are treated as contemporaneous and do not
    count as prior transactions for one another.
    """
    counts_by_date = (
        df.groupby(["msisdn", "pdate"], dropna=False)
        .size()
        .rename("transactions_on_date")
        .reset_index()
        .sort_values(["msisdn", "pdate"], kind="stable")
    )

    counts_by_date["prior_tx_count"] = (
        counts_by_date.groupby("msisdn")["transactions_on_date"].cumsum()
        - counts_by_date["transactions_on_date"]
    )

    enriched = df.merge(
        counts_by_date[["msisdn", "pdate", "prior_tx_count"]],
        on=["msisdn", "pdate"],
        how="left",
        validate="many_to_one",
    )
    enriched["prior_tx_count"] = enriched["prior_tx_count"].astype("int64")
    enriched["is_repeat_customer"] = (enriched["prior_tx_count"] > 0).astype("int8")
    return enriched


def build_model_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing_columns = sorted(set(REQUIRED_INPUT_FIELDS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required input columns: {missing_columns}")

    working = df.copy(deep=True)
    working["pdate"] = pd.to_datetime(
        working["pdate"], dayfirst=True, errors="coerce"
    )
    if working["pdate"].isna().any():
        raise ValueError("Interim dataset contains unparseable pdate values.")

    if working["msisdn"].isna().any():
        raise ValueError("Interim dataset contains missing msisdn values required for chronology.")

    if not working["label"].isin([0, 1]).all():
        raise ValueError("Interim dataset contains invalid label values.")

    enriched = _derive_strictly_prior_features(working)

    later_mask = enriched["pdate"] > MODELLING_CUTOFF
    later_rows = enriched.loc[later_mask]
    modelling = enriched.loc[~later_mask].copy()

    modelling[TARGET_FIELD] = (1 - modelling["label"].astype("int8")).astype("int8")

    output_columns = [*CONTROL_FIELDS, *APPROVED_PREDICTORS, TARGET_FIELD]
    model_ready = modelling[output_columns].copy()

    if "msisdn" in model_ready.columns or "label" in model_ready.columns:
        raise AssertionError("Identifier/source target unexpectedly present in processed output.")

    missing_by_predictor = {
        field: int(model_ready[field].isna().sum()) for field in APPROVED_PREDICTORS
    }

    summary = {
        "input_rows": int(len(df)),
        "modelling_cutoff_inclusive": MODELLING_CUTOFF.date().isoformat(),
        "post_cutoff_rows_excluded": int(later_mask.sum()),
        "post_cutoff_label_values": sorted(
            [int(value) for value in later_rows["label"].dropna().unique().tolist()]
        ),
        "output_rows": int(len(model_ready)),
        "approved_predictor_count": int(len(APPROVED_PREDICTORS)),
        "approved_predictors": APPROVED_PREDICTORS,
        "control_fields": CONTROL_FIELDS,
        "target": TARGET_FIELD,
        "delinquent_count": int(model_ready[TARGET_FIELD].sum()),
        "delinquent_rate": float(model_ready[TARGET_FIELD].mean()),
        "repeat_customer_count": int(model_ready["is_repeat_customer"].sum()),
        "missing_values_by_predictor": missing_by_predictor,
        "identifier_present_in_output": False,
        "source_label_present_in_output": False,
        "imputation_applied": False,
    }

    return model_ready, summary


def main() -> int:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Interim dataset not found: {args.data}")

    df = pd.read_csv(args.data)
    model_ready, summary = build_model_dataset(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    model_ready.to_csv(args.output, index=False)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Model-ready data written to {args.output}")
    print(f"Model-dataset summary written to {args.summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
