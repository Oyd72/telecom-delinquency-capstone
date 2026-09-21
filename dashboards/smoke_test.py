"""Smoke test for the Module 5 Streamlit dashboard dependencies and frozen model.

Run from the repository root:
    python dashboards/smoke_test.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboards.dashboard_utils import load_evidence, load_model_package  # noqa: E402
from src.inference.predict_selected_model import predict_frame  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    evidence = load_evidence()
    metadata = evidence["metadata"]
    package = load_model_package()

    model_path = PROJECT_ROOT / "models" / metadata["artifact_filename"]
    actual_hash = sha256(model_path)
    expected_hash = metadata["artifact_sha256"]
    if actual_hash != expected_hash:
        raise RuntimeError(
            "Frozen model artefact hash does not match selected_model_metadata.json: "
            f"expected {expected_hash}, got {actual_hash}"
        )

    features = metadata["features"]
    row = pd.DataFrame([{feature: np.nan for feature in features}], columns=features)
    result = predict_frame(row, package)

    probability = float(result.iloc[0]["calibrated_delinquency_probability"])
    if not 0.0 <= probability <= 1.0:
        raise RuntimeError(f"Prediction is outside [0, 1]: {probability}")

    robustness = evidence["robustness"]
    if robustness.empty:
        raise RuntimeError("Operational robustness evidence is empty.")

    threshold_tradeoff = evidence["threshold_tradeoff"]
    required_tradeoff_columns = {"threshold", "precision", "recall", "f1", "flagged_rate"}
    if threshold_tradeoff.empty or not required_tradeoff_columns.issubset(threshold_tradeoff.columns):
        raise RuntimeError("Module 5 threshold trade-off evidence is missing or incomplete.")

    from dashboards.dashboard_utils import local_median_sensitivity

    sensitivity = local_median_sensitivity(row, package)
    if not sensitivity.empty:
        raise RuntimeError("Blank-input sensitivity helper should return an empty frame.")

    print("Dashboard smoke test passed.")
    print("Blank-input sensitivity helper: passed")
    print(f"Model artefact: {model_path}")
    print(f"SHA-256 verified: {actual_hash}")
    print(f"Required features: {len(features)}")
    print(f"Test calibrated probability: {probability:.6f}")
    print("Module 4 evidence files loaded successfully.")
    print(f"Threshold trade-off rows: {len(threshold_tradeoff)}")


if __name__ == "__main__":
    main()
