# Telecom delinquency capstone

Business analytics capstone on five-day repayment delinquency in telecom-enabled microcredit.

## Project scope

The project uses historical telecom microcredit data for academic work. The aim is to rank repayment risk at the time of a credit request. It does not automate lending decisions and it does not attempt to predict long-term default.

The modelling population ends on 23 July 2016. Records after that date are kept outside ordinary supervised modelling because every later outcome is successful repayment and the source material does not explain whether the change comes from sampling, labelling, extraction, or the business process itself.

Post-23-July artefacts are retained solely as historical Module 4 exploratory evidence and are not part of the final-project monitoring implementation.

## Current pipeline

The Prefect flow in `src/pipeline/prefect_etl.py` runs the Module 3 data path in this order:

`raw validation → cleaning → interim validation → model-ready transformation → processed validation → representation diagnostics`

Raw validation is diagnostic: it is meant to show the defects present in the untouched source. Interim and processed validation are blocking controls. The pipeline also writes a privacy-safe audit trail for data access, transformations, validation results, identifier removal, and run completion or failure.

## Repository structure

### Active areas

- `data/raw/` – local source data; customer-level files are not committed
- `data/interim/` – cleaned and intermediate data; `msisdn` may still be present where chronology requires it
- `data/processed/` – model-ready row-level data; `msisdn` and the source label are removed
- `src/data/` – cleaning logic
- `src/data_quality/` – Great Expectations validation scripts for raw, interim, and processed data
- `src/features/` – model-dataset construction, temporal diagnostics, and feature-selection work
- `src/models/` – model development, calibration, explainability, robustness, feature-sensitivity experiments, and selected-model packaging; see `src/models/README.md` for an index
- `src/inference/` – stable batch inference contract for the packaged selected model
- `src/api/` – FastAPI `/predict` endpoint for single-record scoring
- `src/pipeline/` – Prefect orchestration
- `src/privacy/` – privacy-safe pipeline audit logging
- `src/monitoring/` – model monitoring, drift metrics, Evidently integration, performance monitoring, and representation/operational diagnostics
- `tests/unit/` – tests for transformations, privacy controls, representation checks, packaged inference, and the FastAPI endpoint
- `tests/validation/` – pipeline and validation contract tests
- `reports/` – reproducible aggregate outputs for analysis, validation, privacy, and presentation; row-level generated data stay local
- `docs/` – data dictionary, methodology, decision records, governance material, and the Module 4 experiment record
- `Dockerfile` – container build for the ETL pipeline
- `Dockerfile.api`, `Dockerfile.stakeholder`, `Dockerfile.monitoring` – service-specific container builds
- `docker-compose.yml` – local multi-container stack for the data pipeline, API, stakeholder dashboard, and monitoring dashboard

### Additional project areas

- `notebooks/` – exploratory and explanatory notebooks used for interactive analysis; reusable project logic is implemented in `src/`
- `models/` – committed selected model binary, privacy-safe metadata, and artefact documentation
- `dashboards/` – Module 5 stakeholder dashboard, final-project monitoring dashboard, and deployment dependencies
- `config/` – shared configuration, including monitoring thresholds
- `.github/workflows/` – final-project CI/CD automation


## Data and privacy position

**Raw data acquisition.** The raw dataset is not committed to this repository. To reproduce the pipeline, obtain the original Kaggle telecom delinquency dataset and place the source CSV at `data/raw/sample_data_intw.csv`. The repository keeps the `data/raw/` directory visible through `.gitkeep`, while Git excludes the customer-level source file itself.

Raw, interim, processed, and synthetic row-level datasets stay out of GitHub. `.gitignore` excludes their contents while keeping the directory structure visible. Synthetic scenarios are reproducible from the committed generator script; only aggregate summaries and figures are suitable repository evidence.

`msisdn` is kept only for the short part of preprocessing where customer-level chronology is needed. It is removed before the model-ready dataset is written and is never an approved predictor. The processed behavioural data are still treated as restricted analytical data; removing the identifier is not taken to mean that the dataset is fully anonymous.

The main governance documents are:

- `docs/governance/data_governance_framework.md` – access, use, retention, lineage, auditability, and exceptions
- `docs/governance/data_anonymization_plan.md` – identifier minimisation and privacy audit logging
- `docs/governance/data_cleaning_policy.md` – standing cleaning rules
- `docs/governance/data_cleaning_narrative.md` – what the cleaning and validation work found in this dataset
- `docs/governance/representation_bias_assessment.md` – scope and limits of representation/bias checks
- `docs/governance/continuous_fairness_monitoring_plan.md` – continuous fairness-monitoring design, escalation logic, and protected-group data contingency
- `docs/governance/ethical_incident_response_plan.md` – detection, severity, containment, investigation, escalation, remediation, and resumption criteria for responsible-AI incidents
- `docs/governance/ethical_decommissioning_plan.md` – governed retirement, evidence preservation, access revocation, and closure controls
- `docs/governance/algorithmic_impact_assessment.md` – consolidated impact assessment covering purpose, stakeholders, benefits, risks, oversight, redress, and lifecycle controls
- `docs/governance/regulatory_compliance_assessment.md` – applicability-oriented GDPR, EU AI Act, CCPA/CPRA, and HIPAA assessment
- `docs/governance/public_trust_statement.md` – plain-language public explanation of model purpose, limits, fairness, privacy, monitoring, and challenge rights
- `docs/governance/feature_selection.md` – feature eligibility and selection method
- `docs/governance/model_development_narrative.md` – modelling decisions and results
- `docs/governance/model_decision_log.md` – compact decision record and current model position

`docs/data_dictionary.md` gives the current interpretation and modelling status of each field. `docs/data_dictionary_changelog.md` records material changes to those interpretations. The Module 4 modelling trail is consolidated in `docs/model-documentation/module4_experiment_record.md`, with a short navigation note in `docs/model-documentation/README.md`.

## Feature-selection history

`src/features/filter_screening.py` is kept as the first Stage 2 single-window diagnostic. The current method is `src/features/fold_filter_screening.py`, which compares filter evidence across calendar-aware folds. The older script remains so the methodological history is visible, but it is no longer the baseline for feature selection.

## Validation and testing

Great Expectations is used through Python scripts rather than through a separate `great_expectations/` project directory:

- `src/data_quality/gx_raw_validation.py`
- `src/data_quality/gx_interim_validation.py`
- `src/data_quality/gx_processed_validation.py`

Pytest checks reusable transformation and control logic. The Prefect flow and the Docker image have both been run successfully end to end on the project data.

## Dependencies

The requirements files serve different runtime scopes:

- `requirements.txt` – wider analytical and development environment
- `requirements-pipeline.txt` – containerised ETL and Module 3 control path
- `requirements-api.txt` – FastAPI inference service
- `requirements-monitoring.txt` – monitoring calculations, Evidently AI, Fairlearn and monitoring dashboard
- `dashboards/requirements.txt` – stakeholder dashboard

Keeping runtime dependencies separate avoids putting the full analytical environment into every container.

## Report evidence

Row-level generated data are not committed. This includes synthetic scenario rows. `reports/README.md` explains which aggregate outputs are suitable for repository or assignment evidence and which should remain local. Figures used by the Module 4 experiment record live under `reports/figures/module4/`. The final Module 4 evidence also includes a formal fairness report, MLflow evidence, conventional classification artefacts, constrained counterfactual explanations, a packaged model metadata record, and a tested FastAPI `/predict` endpoint.

## Public prediction API

The packaged FastAPI inference service is deployed publicly on Render:

- API base URL: `https://telecom-delinquency-capstone-api.onrender.com`
- Health check: `https://telecom-delinquency-capstone-api.onrender.com/health`
- Interactive API documentation: `https://telecom-delinquency-capstone-api.onrender.com/docs`
- Prediction endpoint: `POST https://telecom-delinquency-capstone-api.onrender.com/predict`

The public endpoint was verified against the representative Module 4/5 typical-risk case. It returned the expected frozen-model values:

- raw delinquency probability: `0.26910022741067435`
- calibrated delinquency probability: `0.10013440860215053`
- model: `telecom_delinquency_random_forest_isotonic`
- artefact version: `1.0.0`

The service returns model probabilities only. It does not make approval or decline decisions.

## Module 5 stakeholder dashboard

The Module 5 dashboard is implemented in `dashboards/streamlit_app.py`. It reuses the frozen Module 4 evidence and inference contract rather than changing the submitted Module 4 artefacts or their paths.

Live stakeholder dashboard:

https://telecom-delinquency-capstone-3zecwziyb4rhssv8uavlm2.streamlit.app/

Dashboard sections:

- Overview – business purpose, final-holdout performance and responsible-use boundary
- Prediction explorer – calibrated five-day delinquency probability and the frozen 17.51% operating threshold
- Explainability & what-if – simplified SHAP drivers and constrained counterfactual examples
- Fairness & limits – demographic fairness feasibility, operational robustness and model limitations

For local use:

```powershell
pip install -r dashboards/requirements.txt
streamlit run dashboards/streamlit_app.py
```

Before deployment, the dashboard-specific smoke test verifies that the frozen model loads, its SHA-256 matches the committed Module 4 metadata, a calibrated prediction can be produced, and the committed evidence files can be read:

```powershell
python dashboards/smoke_test.py
```

Live scoring requires the frozen Module 4 artefact at `models/selected_random_forest_isotonic.joblib`. The dashboard does not retrain or recalibrate the model.

## Final-project monitoring dashboard

The governance-facing monitoring dashboard is implemented in `dashboards/monitoring_dashboard.py`. It is intentionally separate from the stakeholder/XAI dashboard.

The monitoring demonstration compares the development reference period (1 June–13 July 2016) with the chronologically later final holdout (14–23 July 2016). It presents:

- custom Python PSI, KS and missingness monitoring;
- Evidently AI drift results as separate independent evidence;
- final-holdout performance and calibration metrics;
- model version, operating threshold and SHA-256 traceability;
- fairness-feasibility and operational-robustness limitations.

Monitoring evidence is versioned under `reports/monitoring/`.

Live monitoring dashboard:

https://telecom-delinquency-monitoring.streamlit.app/

## Docker Compose

The local multi-container stack is defined in `docker-compose.yml`:

- Prefect ETL/data pipeline: batch service using the root `Dockerfile`
- FastAPI prediction service: port 8000
- stakeholder dashboard: port 8501
- monitoring dashboard: port 8502

Run locally with:

```powershell
docker compose build
docker compose up -d
docker compose ps
```

The full Compose setup has been locally verified. The Prefect ETL pipeline builds and completes end to end inside its container, while the API and both dashboard services run successfully. The stack therefore satisfies the final-project requirement for a multi-container pipeline + API + monitoring setup.

## CI/CD

GitHub Actions provides automated validation and deployment through `.github/workflows/final-project-ci-cd.yml`.

The workflow runs automated tests, deployment-entry-point compilation, dashboard smoke checks, Docker Compose validation, and a monitoring-dashboard health check. After successful validation on the deployment branches, it triggers the Render API deployment through a protected repository secret. The Streamlit applications remain linked to their repository branches and redeploy through Streamlit Community Cloud.

## Delivery approach

The work is organised into four Scrum-style two-week sprints: data understanding and setup; data preparation and EDA; modelling and evaluation; fairness, explainability, governance, and reporting. GitHub issue status shows where the work stands now. Sprint labels show the iteration to which each item belongs.