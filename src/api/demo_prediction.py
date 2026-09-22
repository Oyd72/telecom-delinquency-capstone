"""Standalone demonstration of the Module 4 FastAPI prediction endpoint.

Run from the repository root:
    python src/api/demo_prediction.py

This demo uses FastAPI's in-process TestClient so it exercises the same /predict
API contract without requiring a separately hosted API server.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api.app import app


HIGH_RISK_EXAMPLE = {
    "cnt_ma_rech90": 2.0,
    "daily_decr30": 1544.65,
    "last_rech_date_ma": 3.0,
    "sumamnt_ma_rech90": 2309.0,
    "aon": 638.0,
    "last_rech_amt_ma": 770.0,
    "daily_decr90": 1555.5,
    "sumamnt_ma_rech30": 770.0,
    "medianamnt_ma_rech30": 770.0,
    "medianmarechprebal90": 16.0,
    "rental30": 12969.51,
    "cnt_ma_rech30": 1.0,
}


def main() -> None:
    client = TestClient(app)

    response = client.post("/predict", json=HIGH_RISK_EXAMPLE)
    response.raise_for_status()
    result = response.json()

    print("Module 4 API demo")
    print("-----------------")
    print("Endpoint: POST /predict")
    print(f"Model: {result['model_name']}")
    print(f"Artifact version: {result['artifact_version']}")
    print(
        "Calibrated five-day delinquency probability: "
        f"{result['calibrated_delinquency_probability']:.1%}"
    )
    print(
        "Raw model probability: "
        f"{result['raw_delinquency_probability']:.1%}"
    )


if __name__ == "__main__":
    main()
