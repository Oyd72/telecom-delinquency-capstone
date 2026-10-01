# Model artefacts

This directory contains the packaged selected model and its traceability metadata.

## Selected Module 4 model

Run:

```powershell
python src\models\package_selected_model.py
```

This creates:

- `selected_random_forest_isotonic.joblib` — fitted imputer, Random Forest, isotonic calibrator, and feature contract;
- `selected_model_metadata.json` — privacy-safe metadata, model parameters, training/calibration dates, SHA-256 hash, and known limitations.

For this capstone repository, the selected `.joblib` artefact is committed together with its metadata so the FastAPI service, Streamlit applications and Docker Compose stack are reproducible from the repository. Other model artefacts remain excluded by default.

Batch inference is provided by:

`src/inference/predict_selected_model.py`

The inference script returns raw and calibrated delinquency probabilities only. It does not make lending or approval decisions.

## API inference

A single-record FastAPI endpoint is available at:

`src/api/app.py`

Run it locally with:

```powershell
uvicorn src.api.app:app --reload
```

The service exposes:

- `GET /health` — basic availability check;
- `POST /predict` — accepts the 12 model features and returns raw and calibrated five-day delinquency probabilities.

The endpoint does not make an approval or decline decision.
