"""FastAPI endpoint for the packaged Module 4 delinquency model.

Run locally with:
    uvicorn src.api.app:app --reload

The endpoint returns probabilities only. It does not make approval or decline decisions.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.inference.predict_selected_model import predict_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = PROJECT_ROOT / "models/selected_random_forest_isotonic.joblib"

app = FastAPI(
    title="Telecom delinquency prediction API",
    version="1.0.0",
    description=(
        "Returns raw and calibrated probabilities of five-day delinquency for one transaction. "
        "The service is intended for decision support and does not implement an approval/decline rule."
    ),
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cnt_ma_rech90: Annotated[float | None, Field(description="Count of MA recharges in 90-day window")]
    daily_decr30: Annotated[float | None, Field(description="30-day daily decrement measure")]
    last_rech_date_ma: Annotated[float | None, Field(description="MA recharge recency measure")]
    sumamnt_ma_rech90: Annotated[float | None, Field(description="Sum of MA recharge amount in 90-day window")]
    aon: Annotated[float | None, Field(description="Account/network tenure measure")]
    last_rech_amt_ma: Annotated[float | None, Field(description="Most recent MA recharge amount")]
    daily_decr90: Annotated[float | None, Field(description="90-day daily decrement measure")]
    sumamnt_ma_rech30: Annotated[float | None, Field(description="Sum of MA recharge amount in 30-day window")]
    medianamnt_ma_rech30: Annotated[float | None, Field(description="Median MA recharge amount in 30-day window")]
    medianmarechprebal90: Annotated[float | None, Field(description="Median MA recharge pre-balance in 90-day window")]
    rental30: Annotated[float | None, Field(description="30-day rental/activity measure")]
    cnt_ma_rech30: Annotated[float | None, Field(description="Count of MA recharges in 30-day window")]


class PredictionResponse(BaseModel):
    raw_delinquency_probability: float
    calibrated_delinquency_probability: float
    model_name: str
    artifact_version: str


@lru_cache(maxsize=1)
def get_model_package():
    if not DEFAULT_ARTIFACT.exists():
        raise FileNotFoundError(
            f"Model artefact not found at {DEFAULT_ARTIFACT}. "
            "Run src/models/package_selected_model.py first."
        )
    return joblib.load(DEFAULT_ARTIFACT)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        package = get_model_package()
        row = pd.DataFrame([request.model_dump()])
        result = predict_frame(row, package).iloc[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionResponse(
        raw_delinquency_probability=float(result["raw_delinquency_probability"]),
        calibrated_delinquency_probability=float(result["calibrated_delinquency_probability"]),
        model_name=str(result["model_name"]),
        artifact_version=str(result["artifact_version"]),
    )
