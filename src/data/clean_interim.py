"""Create an interim cleaned copy of the telecom delinquency dataset.

This script implements the currently agreed interim cleaning rules:
- negative duration values in `aon`, `last_rech_date_ma`, and `last_rech_date_da`
  are flagged and converted to missing;
- values in the separated high-confidence contamination regime (>= 500,000)
  are flagged and converted to missing for `aon`, `last_rech_date_ma`,
  `last_rech_date_da`, `fr_ma_rech30`, and `fr_da_rech30`;
- fractional values in the genuine count fields `cnt_da_rech30` and
  `cnt_loans90` are flagged and converted to missing;
- valid integer-like count values are retained;
- one copy of an exact duplicate row is removed;
- every changed value or removed row is written automatically to a companion
  audit log without recording `msisdn`.

The 500,000 boundary is dataset-specific. It reflects the large empirical gap
between the ordinary value range and a separate implausible numeric regime; it
is not presented as a universal business maximum.

The raw source file is read only and is never overwritten.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DURATION_FIELDS = ["aon", "last_rech_date_ma", "last_rech_date_da"]
FRACTIONAL_COUNT_FIELDS = ["cnt_da_rech30", "cnt_loans90"]
SEPARATED_CONTAMINATION_FIELDS = [
    "aon",
    "last_rech_date_ma",
    "last_rech_date_da",
    "fr_ma_rech30",
    "fr_da_rech30",
]
SEPARATED_CONTAMINATION_MIN = 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to the immutable raw CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/sample_data_intw_cleaned.csv"),
        help="Path for the interim cleaned CSV.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/interim/cleaning_audit_log.csv"),
        help="Path for the row-level cleaning audit log.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/cleaning_summary.json"),
        help="Path for the aggregate cleaning summary.",
    )
    return parser.parse_args()


def _is_fractional(series: pd.Series) -> pd.Series:
    """Return a boolean mask for finite numeric values that are not integers."""
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric)
    return finite & ~np.isclose(numeric, np.round(numeric), rtol=0.0, atol=1e-12)


def _audit_rows(
    df: pd.DataFrame,
    mask: pd.Series,
    field: str,
    rule_id: str,
    treatment: str,
    run_id: str,
    run_timestamp: str,
) -> list[dict]:
    rows: list[dict] = []
    for idx, original_value in df.loc[mask, field].items():
        rows.append(
            {
                "run_id": run_id,
                "run_timestamp_utc": run_timestamp,
                "source_row_number": int(idx) + 1,
                "field": field,
                "original_value": original_value,
                "rule_triggered": rule_id,
                "treatment_applied": treatment,
            }
        )
    return rows


def _audit_duplicate_rows(
    df: pd.DataFrame,
    mask: pd.Series,
    run_id: str,
    run_timestamp: str,
) -> list[dict]:
    """Create audit entries for removed duplicate rows without copying row contents."""
    rows: list[dict] = []
    for idx in df.index[mask]:
        rows.append(
            {
                "run_id": run_id,
                "run_timestamp_utc": run_timestamp,
                "source_row_number": int(idx) + 1,
                "field": "__row__",
                "original_value": "<exact duplicate row>",
                "rule_triggered": "exact_duplicate",
                "treatment_applied": "remove_duplicate_copy",
            }
        )
    return rows


def clean_dataframe(df: pd.DataFrame, run_id: str, run_timestamp: str):
    """Return an interim cleaned copy, audit table, and aggregate summary."""
    cleaned = df.copy(deep=True)
    audit_records: list[dict] = []
    changed_cells = 0
    summary: dict[str, object] = {
        "run_id": run_id,
        "run_timestamp_utc": run_timestamp,
        "input_rows": int(len(df)),
        "rules": {},
    }

    for field in DURATION_FIELDS:
        numeric = pd.to_numeric(cleaned[field], errors="coerce")
        mask = numeric < 0
        count = int(mask.sum())
        audit_records.extend(
            _audit_rows(
                cleaned,
                mask,
                field,
                rule_id="negative_duration",
                treatment="set_to_missing",
                run_id=run_id,
                run_timestamp=run_timestamp,
            )
        )
        cleaned.loc[mask, field] = np.nan
        changed_cells += count
        summary["rules"][f"{field}:negative_duration"] = count

    for field in SEPARATED_CONTAMINATION_FIELDS:
        numeric = pd.to_numeric(cleaned[field], errors="coerce")
        mask = numeric >= SEPARATED_CONTAMINATION_MIN
        count = int(mask.sum())
        audit_records.extend(
            _audit_rows(
                cleaned,
                mask,
                field,
                rule_id="separated_contamination_regime",
                treatment="set_to_missing",
                run_id=run_id,
                run_timestamp=run_timestamp,
            )
        )
        cleaned.loc[mask, field] = np.nan
        changed_cells += count
        summary["rules"][f"{field}:separated_contamination_regime"] = count

    for field in FRACTIONAL_COUNT_FIELDS:
        mask = _is_fractional(cleaned[field])
        count = int(mask.sum())
        audit_records.extend(
            _audit_rows(
                cleaned,
                mask,
                field,
                rule_id="fractional_count",
                treatment="set_to_missing",
                run_id=run_id,
                run_timestamp=run_timestamp,
            )
        )
        cleaned.loc[mask, field] = np.nan
        changed_cells += count

        # After removing fractional contamination, preserve valid count semantics
        # using pandas' nullable integer type so missing values remain explicit.
        cleaned[field] = pd.to_numeric(cleaned[field], errors="coerce").round().astype("Int64")
        summary["rules"][f"{field}:fractional_count"] = count

    # An exact duplicate is a record-level issue rather than an unusual repeat-customer
    # event. Keep the first occurrence and remove later exact copies only.
    duplicate_mask = cleaned.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    audit_records.extend(
        _audit_duplicate_rows(
            cleaned,
            duplicate_mask,
            run_id=run_id,
            run_timestamp=run_timestamp,
        )
    )
    cleaned = cleaned.loc[~duplicate_mask].copy()
    summary["rules"]["exact_duplicate:removed"] = duplicate_count

    audit = pd.DataFrame(
        audit_records,
        columns=[
            "run_id",
            "run_timestamp_utc",
            "source_row_number",
            "field",
            "original_value",
            "rule_triggered",
            "treatment_applied",
        ],
    )

    summary["changed_cells"] = int(changed_cells)
    summary["output_rows"] = int(len(cleaned))
    summary["rows_removed"] = duplicate_count

    return cleaned, audit, summary


def main() -> int:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Raw dataset not found: {args.data}")

    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("clean_%Y%m%dT%H%M%S_%fZ")

    raw = pd.read_csv(args.data)
    cleaned, audit, summary = clean_dataframe(raw, run_id, run_timestamp)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(args.output, index=False)
    audit.to_csv(args.audit, index=False)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Interim cleaned data written to {args.output}")
    print(f"Cleaning audit log written to {args.audit}")
    print(f"Aggregate summary written to {args.summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
