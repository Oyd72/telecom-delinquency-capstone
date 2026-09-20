from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference.predict_selected_model import predict_frame, validate_input


FEATURES = [
    "cnt_ma_rech90",
    "daily_decr30",
    "last_rech_date_ma",
    "sumamnt_ma_rech90",
    "aon",
    "last_rech_amt_ma",
    "daily_decr90",
    "sumamnt_ma_rech30",
    "medianamnt_ma_rech30",
    "medianmarechprebal90",
    "rental30",
    "cnt_ma_rech30",
]


class DummyImputer:
    def transform(self, x):
        return np.asarray(x, dtype=float)


class DummyModel:
    def predict_proba(self, x):
        p = np.full(len(x), 0.25)
        return np.column_stack([1 - p, p])


class DummyCalibrator:
    def predict(self, p):
        return np.asarray(p) * 0.8


def package():
    return {
        "features": FEATURES,
        "imputer": DummyImputer(),
        "model": DummyModel(),
        "calibrator": DummyCalibrator(),
        "model_name": "test_model",
        "artifact_version": "test",
    }


def test_missing_required_feature_rejected():
    df = pd.DataFrame({feature: [1.0] for feature in FEATURES[:-1]})
    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_input(df, FEATURES)


def test_prediction_contract_returns_probabilities_and_metadata():
    df = pd.DataFrame({feature: [1.0, 2.0] for feature in FEATURES})
    out = predict_frame(df, package())

    assert list(out.columns) == [
        "raw_delinquency_probability",
        "calibrated_delinquency_probability",
        "model_name",
        "artifact_version",
    ]
    assert np.allclose(out["raw_delinquency_probability"], 0.25)
    assert np.allclose(out["calibrated_delinquency_probability"], 0.20)
    assert (out["model_name"] == "test_model").all()
