# Telecom delinquency capstone

Business analytics capstone on five-day repayment delinquency in telecom-enabled microcredit.

## Project scope

The project uses historical telecom microcredit data for academic work. The aim is to rank repayment risk at the time of a credit request. It does not automate lending decisions and it does not attempt to predict long-term default.

The modelling population ends on 23 July 2016. Records after that date are kept outside ordinary supervised modelling because every later outcome is successful repayment and the source material does not explain whether the change comes from sampling, labelling, extraction, or the business process itself.

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
- `src/models/` – model comparison and calibration experiments
- `src/pipeline/` – Prefect orchestration
- `src/privacy/` – privacy-safe pipeline audit logging
- `src/monitoring/` – representation and operational slice diagnostics
- `tests/unit/` – tests for transformations, privacy controls, and representation checks
- `tests/validation/` – pipeline and validation contract tests
- `reports/` – reproducible aggregate outputs for analysis, validation, privacy, and presentation
- `docs/` – data dictionary, methodology, decision records, and governance material
- `Dockerfile` – container build for the ETL pipeline

### Reserved areas

- `notebooks/` – exploratory notebooks if needed; reusable logic belongs in `src/`
- `models/` – persisted model artefacts and metadata once a final artefact is produced
- `dashboards/` – Power BI deliverables
- `config/` – shared configuration if project parameters are externalised
- `.github/workflows/` – CI automation if introduced later

Empty reserved directories remain visible through `.gitkeep` files.

## Data and privacy position

Raw, interim, and processed customer-level datasets stay out of GitHub. `.gitignore` excludes the contents of all three data folders.

`msisdn` is kept only for the short part of preprocessing where customer-level chronology is needed. It is removed before the model-ready dataset is written and is never an approved predictor. The processed behavioural data are still treated as restricted analytical data; removing the identifier is not taken to mean that the dataset is fully anonymous.

The main governance documents are:

- `docs/governance/data_governance_framework.md` – access, use, retention, lineage, auditability, and exceptions
- `docs/governance/data_anonymization_plan.md` – identifier minimisation and privacy audit logging
- `docs/governance/data_cleaning_policy.md` – standing cleaning rules
- `docs/governance/data_cleaning_narrative.md` – what the cleaning and validation work found in this dataset
- `docs/governance/representation_bias_assessment.md` – scope and limits of representation/bias checks
- `docs/governance/feature_selection.md` – feature eligibility and selection method
- `docs/governance/model_development_narrative.md` – modelling decisions and results
- `docs/governance/model_decision_log.md` – compact decision record and current model position

`docs/data_dictionary.md` gives the current interpretation and modelling status of each field. `docs/data_dictionary_changelog.md` records material changes to those interpretations.

## Feature-selection history

`src/features/filter_screening.py` is kept as the first Stage 2 single-window diagnostic. The current method is `src/features/fold_filter_screening.py`, which compares filter evidence across calendar-aware folds. The older script remains so the methodological history is visible, but it is no longer the baseline for feature selection.

## Validation and testing

Great Expectations is used through Python scripts rather than through a separate `great_expectations/` project directory:

- `src/data_quality/gx_raw_validation.py`
- `src/data_quality/gx_interim_validation.py`
- `src/data_quality/gx_processed_validation.py`

Pytest checks reusable transformation and control logic. The Prefect flow and the Docker image have both been run successfully end to end on the project data.

## Dependencies

The two requirements files serve different purposes:

- `requirements.txt` – the wider analytical and development environment
- `requirements-pipeline.txt` – the smaller set needed for the containerised ETL and Module 3 control path

Keeping the pipeline dependencies separate avoids putting the full analytical environment into the Docker image.

## Report evidence

Row-level generated data are not committed. `reports/README.md` explains which aggregate outputs are suitable for the repository or assignment evidence and which should remain local.

## Delivery approach

The work is organised into four Scrum-style two-week sprints: data understanding and setup; data preparation and EDA; modelling and evaluation; fairness, explainability, governance, and reporting. GitHub issue status shows where the work stands now. Sprint labels show the iteration to which each item belongs.