from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import src.api.app as api_module


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
        p = np.full(len(x), 0.30)
        return np.column_stack([1 - p, p])


class DummyCalibrator:
    def predict(self, p):
        return np.asarray(p) * 0.75


def dummy_package():
    return {
        "features": FEATURES,
        "imputer": DummyImputer(),
        "model": DummyModel(),
        "calibrator": DummyCalibrator(),
        "model_name": "test_model",
        "artifact_version": "test",
    }


def valid_payload():
    return {feature: 1.0 for feature in FEATURES}


def test_health_endpoint():
    client = TestClient(api_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr(api_module, "get_model_package", dummy_package)
    client = TestClient(api_module.app)

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["raw_delinquency_probability"] == 0.30
    assert body["calibrated_delinquency_probability"] == 0.225
    assert body["model_name"] == "test_model"
    assert body["artifact_version"] == "test"


def test_predict_rejects_missing_feature(monkeypatch):
    monkeypatch.setattr(api_module, "get_model_package", dummy_package)
    client = TestClient(api_module.app)

    payload = valid_payload()
    payload.pop("cnt_ma_rech30")
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
