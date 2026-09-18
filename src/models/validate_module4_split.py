from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/interim/sample_data_intw_cleaned.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports/tables/module4_split_summary.json"

HOLDOUT_START = pd.Timestamp("2016-07-14")
MODELLING_CUTOFF = pd.Timestamp("2016-07-23")


def build_split_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required = {"pdate", "label", "msisdn"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    work["pdate"] = pd.to_datetime(work["pdate"], dayfirst=True, errors="raise")
    work["delinquent_5d"] = 1 - work["label"].astype(int)

    modelling = work.loc[work["pdate"] <= MODELLING_CUTOFF].copy()
    post_period = work.loc[work["pdate"] > MODELLING_CUTOFF].copy()
    development = modelling.loc[modelling["pdate"] < HOLDOUT_START].copy()
    final_holdout = modelling.loc[modelling["pdate"] >= HOLDOUT_START].copy()

    # Freeze the agreed population and chronological split.
    assert len(work) == 209_592, f"Unexpected cleaned source row count: {len(work):,}"
    assert len(modelling) == 150_767, (
        f"Unexpected modelling population: {len(modelling):,}"
    )
    assert len(post_period) == 58_825, (
        f"Unexpected post-cutoff population: {len(post_period):,}"
    )
    assert development["pdate"].max() < final_holdout["pdate"].min()
    assert final_holdout["pdate"].min() == HOLDOUT_START
    assert final_holdout["pdate"].max() == MODELLING_CUTOFF
    assert modelling.index.intersection(post_period.index).empty
    assert development.index.intersection(final_holdout.index).empty
    assert len(development) + len(final_holdout) == len(modelling)
    assert post_period["label"].eq(1).all(), (
        "Unexpected post-cutoff target pattern."
    )

    split_summary = pd.DataFrame(
        [
            {
                "population": "development",
                "purpose": "training + time-aware CV",
                "start_date": str(development["pdate"].min().date()),
                "end_date": str(development["pdate"].max().date()),
                "rows": int(len(development)),
                "delinquent_rows": int(development["delinquent_5d"].sum()),
                "delinquency_rate": float(development["delinquent_5d"].mean()),
            },
            {
                "population": "final_holdout",
                "purpose": "untouched final evaluation",
                "start_date": str(final_holdout["pdate"].min().date()),
                "end_date": str(final_holdout["pdate"].max().date()),
                "rows": int(len(final_holdout)),
                "delinquent_rows": int(final_holdout["delinquent_5d"].sum()),
                "delinquency_rate": float(final_holdout["delinquent_5d"].mean()),
            },
            {
                "population": "post_23_july",
                "purpose": "exploratory robustness/distribution checks only",
                "start_date": str(post_period["pdate"].min().date()),
                "end_date": str(post_period["pdate"].max().date()),
                "rows": int(len(post_period)),
                "delinquent_rows": int(post_period["delinquent_5d"].sum()),
                "delinquency_rate": float(post_period["delinquent_5d"].mean()),
            },
        ]
    )

    development_customers = set(development["msisdn"].dropna())
    holdout_customer_seen = final_holdout["msisdn"].isin(development_customers)

    customer_check = pd.Series(
        {
            "holdout_rows": int(len(final_holdout)),
            "holdout_rows_seen_customer": int(holdout_customer_seen.sum()),
            "holdout_rows_unseen_customer": int((~holdout_customer_seen).sum()),
            "share_unseen_customer": float((~holdout_customer_seen).mean()),
        },
        name="value",
    )

    return split_summary, customer_check


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    split_summary, customer_check = build_split_summary(df)

    payload = {
        "source_rows": int(len(df)),
        "date_min": str(pd.to_datetime(df["pdate"], dayfirst=True).min().date()),
        "date_max": str(pd.to_datetime(df["pdate"], dayfirst=True).max().date()),
        "modelling_cutoff_inclusive": str(MODELLING_CUTOFF.date()),
        "holdout_start": str(HOLDOUT_START.date()),
        "splits": split_summary.to_dict(orient="records"),
        "customer_continuity": customer_check.to_dict(),
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Frozen split checks passed.")
    print()
    print(split_summary.to_string(index=False))
    print()
    print("Customer continuity")
    print(customer_check.to_string())
    print()
    print(f"Summary written to: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
