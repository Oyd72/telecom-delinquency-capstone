"""Batch inference for the packaged selected Module 4 model.

Example:
    python src/inference/predict_selected_model.py --input input.csv --output predictions.csv

The input CSV must contain the 12 required feature columns. Extra columns are preserved
only when --keep-extra-columns is supplied. The script returns raw and calibrated
five-day delinquency probabilities; it does not make an approval/decline decision.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = PROJECT_ROOT / "models/selected_random_forest_isotonic.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--keep-extra-columns", action="store_true")
    return parser.parse_args()


def validate_input(df: pd.DataFrame, required_features: list[str]) -> None:
    missing = sorted(set(required_features) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"Duplicate input columns are not allowed: {duplicate_columns}")


def predict_frame(df: pd.DataFrame, package: dict, keep_extra_columns: bool = False) -> pd.DataFrame:
    features = package["features"]
    validate_input(df, features)

    x = package["imputer"].transform(df[features])
    raw = np.asarray(package["model"].predict_proba(x)[:, 1], dtype=float)
    calibrated = np.asarray(package["calibrator"].predict(raw), dtype=float)

    if keep_extra_columns:
        out = df.copy()
    else:
        out = pd.DataFrame(index=df.index)

    out["raw_delinquency_probability"] = raw
    out["calibrated_delinquency_probability"] = calibrated
    out["model_name"] = package["model_name"]
    out["artifact_version"] = package["artifact_version"]
    return out


def main() -> None:
    args = parse_args()

    if not args.artifact.exists():
        raise FileNotFoundError(
            f"Model artefact not found: {args.artifact}. "
            "Run src/models/package_selected_model.py first."
        )

    package = joblib.load(args.artifact)
    df = pd.read_csv(args.input)
    predictions = predict_frame(
        df,
        package,
        keep_extra_columns=args.keep_extra_columns,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    print("Inference completed.")
    print(f"Rows scored: {len(predictions):,}")
    print(f"Output: {args.output}")
    print(
        "Mean calibrated delinquency probability: "
        f"{predictions['calibrated_delinquency_probability'].mean():.6f}"
    )


if __name__ == "__main__":
    main()
