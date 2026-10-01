"""Prepare transparent monitoring populations from the cleaned interim dataset.

The script creates three monitoring populations:
1. development reference: 1 June–13 July 2016;
2. final labelled evaluation: 14–23 July 2016;
3. post-23-July diagnostic: records after 23 July 2016.

The post-23-July source labels are deliberately NOT copied into the diagnostic output.
They show an unexplained all-successful regime and are not treated as ground truth.
That period is therefore restricted to label-free drift and population diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/interim/sample_data_intw_cleaned.csv"
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    required = {"pdate", "label", *features}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Cleaned interim data are missing required columns: {missing}")

    work = df.copy()
    work["pdate"] = pd.to_datetime(work["pdate"], dayfirst=True, errors="raise")

    development = work.loc[work["pdate"] <= DEVELOPMENT_END].copy()
    holdout = work.loc[
        (work["pdate"] >= HOLDOUT_START) & (work["pdate"] <= HOLDOUT_END)
    ].copy()
    diagnostic = work.loc[work["pdate"] > HOLDOUT_END].copy()

    # Convert reliable source labels to the project's delinquency target only for
    # development and final-holdout populations.
    development["delinquent_5d"] = (1 - development["label"].astype("int8")).astype("int8")
    holdout["delinquent_5d"] = (1 - holdout["label"].astype("int8")).astype("int8")

    labelled_columns = ["pdate", *features, "delinquent_5d"]
    diagnostic_columns = ["pdate", *features]

    development_out = development[labelled_columns].copy()
    holdout_out = holdout[labelled_columns].copy()
    diagnostic_out = diagnostic[diagnostic_columns].copy()

    diagnostic_source_label_values = sorted(
        int(value) for value in diagnostic["label"].dropna().unique().tolist()
    )

    summary = {
        "source_rows": int(len(work)),
        "source_date_min": work["pdate"].min().date().isoformat(),
        "source_date_max": work["pdate"].max().date().isoformat(),
        "feature_count": int(len(features)),
        "features": features,
        "development_reference": {
            "period": "2016-06-01 through 2016-07-13",
            "rows": int(len(development_out)),
            "labels_reliable_for_project_evaluation": True,
        },
        "final_labelled_evaluation": {
            "period": "2016-07-14 through 2016-07-23",
            "rows": int(len(holdout_out)),
            "labels_reliable_for_project_evaluation": True,
        },
        "post_23_july_diagnostic": {
            "period": "after 2016-07-23",
            "rows": int(len(diagnostic_out)),
            "source_label_values_observed": diagnostic_source_label_values,
            "labels_copied_to_monitoring_output": False,
            "labels_treated_as_ground_truth": False,
            "permitted_use": "label-free feature, population, missingness and prediction-score drift diagnostics only",
            "prohibited_use": "supervised performance, calibration or outcome-based fairness claims",
        },
    }

    return development_out, holdout_out, diagnostic_out, summary


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Cleaned interim dataset not found: {args.input}")
    if not args.metadata.exists():
        raise FileNotFoundError(f"Selected model metadata not found: {args.metadata}")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    features = list(metadata["features"])
    df = pd.read_csv(args.input)

    development, holdout, diagnostic, summary = prepare_monitoring_populations(df, features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    development_path = args.output_dir / "development_reference.csv"
    holdout_path = args.output_dir / "final_holdout_labelled.csv"
    diagnostic_path = args.output_dir / "post_23_july_diagnostic.csv"

    development.to_csv(development_path, index=False)
    holdout.to_csv(holdout_path, index=False)
    diagnostic.to_csv(diagnostic_path, index=False)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Development reference written to: {development_path}")
    print(f"Final labelled holdout written to: {holdout_path}")
    print(f"Post-23-July diagnostic population written to: {diagnostic_path}")
    print(f"Monitoring population summary written to: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
