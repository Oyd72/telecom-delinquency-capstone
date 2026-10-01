"""Prepare monitoring populations from the canonical model-ready dataset.

The final project demonstrates monitoring with data already used legitimately in the
model-development workflow:
1. development reference: 1 June–13 July 2016;
2. final holdout: 14–23 July 2016.

No post-23-July records are used by the final-project monitoring solution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"
DEFAULT_METADATA = PROJECT_ROOT / "models/selected_model_metadata.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/monitoring"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports/monitoring/monitoring_population_summary.json"

DEVELOPMENT_END = pd.Timestamp("2016-07-13")
HOLDOUT_START = pd.Timestamp("2016-07-14")
HOLDOUT_END = pd.Timestamp("2016-07-23")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def prepare_monitoring_populations(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = {"pdate", "delinquent_5d", *features}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Model-ready data are missing required columns: {missing}")

    work = df.copy()
    work["pdate"] = pd.to_datetime(work["pdate"], errors="raise")

    development = work.loc[work["pdate"] <= DEVELOPMENT_END].copy()
    holdout = work.loc[
        (work["pdate"] >= HOLDOUT_START) & (work["pdate"] <= HOLDOUT_END)
    ].copy()

    output_columns = ["pdate", *features, "delinquent_5d"]
    development_out = development[output_columns].copy()
    holdout_out = holdout[output_columns].copy()

    if len(development_out) + len(holdout_out) != len(work):
        raise ValueError(
            "The canonical model-ready dataset contains rows outside the expected "
            "1 June–23 July 2016 monitoring demonstration window."
        )

    summary = {
        "source": "data/processed/telecom_delinquency_model_ready.csv",
        "source_rows": int(len(work)),
        "source_date_min": work["pdate"].min().date().isoformat(),
        "source_date_max": work["pdate"].max().date().isoformat(),
        "feature_count": int(len(features)),
        "features": features,
        "development_reference": {
            "period": "2016-06-01 through 2016-07-13",
            "rows": int(len(development_out)),
            "purpose": "Reference distribution for monitoring demonstration",
        },
        "final_holdout": {
            "period": "2016-07-14 through 2016-07-23",
            "rows": int(len(holdout_out)),
            "purpose": (
                "Chronologically later comparison population and trusted labelled "
                "evaluation period"
            ),
        },
        "post_23_july_data_used": False,
        "monitoring_design_note": (
            "The final project demonstrates monitoring on the existing development "
            "reference and final holdout. In operational use the same monitoring logic "
            "would be applied to future batches as they arrive."
        ),
    }
    return development_out, holdout_out, summary


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Model-ready dataset not found: {args.input}")
    if not args.metadata.exists():
        raise FileNotFoundError(f"Selected model metadata not found: {args.metadata}")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    features = list(metadata["features"])
    df = pd.read_csv(args.input)

    development, holdout, summary = prepare_monitoring_populations(df, features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    development_path = args.output_dir / "development_reference.csv"
    holdout_path = args.output_dir / "final_holdout_labelled.csv"

    development.to_csv(development_path, index=False)
    holdout.to_csv(holdout_path, index=False)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Development reference written to: {development_path}")
    print(f"Final holdout written to: {holdout_path}")
    print(f"Monitoring population summary written to: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
