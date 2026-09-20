# Model artefacts

This directory is reserved for packaged model artefacts.

## Selected Module 4 model

Run:

```powershell
python src\models\package_selected_model.py
```

This creates locally:

- `selected_random_forest_isotonic.joblib` — fitted imputer, Random Forest, isotonic calibrator, and feature contract;
- `selected_model_metadata.json` — privacy-safe metadata, model parameters, training/calibration dates, SHA-256 hash, and known limitations.

The binary `.joblib` file is intentionally excluded from GitHub. The metadata JSON is allowed in the repository so that the packaged model can be identified and checked without committing the binary artefact.

Batch inference is provided by:

`src/inference/predict_selected_model.py`

The inference script returns raw and calibrated delinquency probabilities only. It does not make lending or approval decisions.
