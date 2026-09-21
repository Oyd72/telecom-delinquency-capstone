# Module 5 Streamlit deployment

This note documents the deployment path for the Module 5 stakeholder dashboard.

It is intentionally additive. Existing Module 4 evidence paths and linked artefacts are not renamed, moved, or replaced.

## Application entry point

- Repository: `Oyd72/telecom-delinquency-capstone`
- Branch: `main`
- Streamlit entry point: `dashboards/streamlit_app.py`
- Dashboard-specific dependencies: `dashboards/requirements.txt`

## Local verification

From the repository root:

```powershell
python -m pip install -r dashboards/requirements.txt
python dashboards/smoke_test.py
streamlit run dashboards/streamlit_app.py
```

The smoke test verifies:

- the frozen Module 4 model artefact can be loaded;
- the SHA-256 matches `models/selected_model_metadata.json`;
- all 12 required model features are available;
- a calibrated prediction can be produced;
- the committed Module 4 dashboard evidence files can be loaded.

GitHub Actions repeats these checks and also starts Streamlit headlessly and checks its health endpoint.

## Live deployment

Public URL:

https://telecom-delinquency-capstone-3zecwziyb4rhssv8uavlm2.streamlit.app/

## Streamlit Community Cloud settings

Create a new app from the GitHub repository using:

- Repository: `Oyd72/telecom-delinquency-capstone`
- Branch: `main`
- Main file path: `dashboards/streamlit_app.py`

After deployment:

1. Open the public application URL.
2. Check all four dashboard tabs.
3. Run at least one prediction in the Prediction explorer.
4. Confirm the operating-threshold explanation appears.
5. Confirm the Explainability & what-if charts render.
6. Confirm the Fairness & limits view renders.
7. Add the final public URL to the repository README.
8. Re-open the public URL immediately before assignment submission.

## Module 4 compatibility

The Module 5 dashboard consumes the existing Module 4 artefacts at their current paths. Do not rename or move those files because they are already cited from the submitted Module 4 paper.

The deployed dashboard must use the frozen artefact:

`models/selected_random_forest_isotonic.joblib`

with the SHA-256 recorded in:

`models/selected_model_metadata.json`

The dashboard does not retrain or recalibrate the model.
