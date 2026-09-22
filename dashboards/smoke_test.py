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
from fastapi.testclient import TestClient  # noqa: E402
from src.api.app import app as prediction_api  # noqa: E402


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

    client = TestClient(prediction_api)
    api_response = client.post("/predict", json={feature: None for feature in features})
    if api_response.status_code != 200:
        raise RuntimeError(
            f"Module 4 prediction API smoke test failed: "
            f"{api_response.status_code} {api_response.text}"
        )
    probability = float(api_response.json()["calibrated_delinquency_probability"])

    from dashboards.dashboard_utils import case_feature_values
    high_risk_values = case_feature_values(evidence["shap"], "high_risk")
    high_risk_payload = {
        feature: float(high_risk_values[feature]) for feature in features
    }
    high_risk_response = client.post("/predict", json=high_risk_payload)
    if high_risk_response.status_code != 200:
        raise RuntimeError(
            f"Representative high-risk API preset failed: "
            f"{high_risk_response.status_code} {high_risk_response.text}"
        )
    high_risk_probability = float(
        high_risk_response.json()["calibrated_delinquency_probability"]
    )
    expected_high_risk = 0.47058823529411764
    if abs(high_risk_probability - expected_high_risk) > 1e-9:
        raise RuntimeError(
            "Representative high-risk preset does not reproduce the committed Module 4 "
            f"risk: expected {expected_high_risk:.12f}, got {high_risk_probability:.12f}"
        )
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
    print("Module 4 prediction API: passed")
    print(f"Representative high-risk API preset: {high_risk_probability:.6f}")
    print("Blank-input sensitivity helper: passed")
    print(f"Model artefact: {model_path}")
    print(f"SHA-256 verified: {actual_hash}")
    print(f"Required features: {len(features)}")
    print(f"Test calibrated probability: {probability:.6f}")
    print("Module 4 evidence files loaded successfully.")
    print(f"Threshold trade-off rows: {len(threshold_tradeoff)}")


if __name__ == "__main__":
    main()
