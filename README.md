# Telecom delinquency capstone

Business analytics capstone project for predicting five-day repayment delinquency in telecom-enabled microcredit.

## Project scope

The project uses historical telecom microcredit data for academic and demonstration purposes. It focuses on risk ranking at the time of a credit request and does not automate final customer decisions or predict long-term default.

The current modelling population contains records through 23 July 2016. Later records are excluded from ordinary supervised modelling because that block contains only successful repayment outcomes and its data-generation regime cannot be established from the source documentation.

## Current pipeline

The implemented Prefect flow in `src/pipeline/prefect_etl.py` orchestrates the reproducible Module 3 path:

`raw validation → cleaning → interim validation → model-ready transformation → processed validation → representation diagnostics`

Raw validation is diagnostic because it is intended to expose known source-quality defects before cleaning. Interim and processed validation are blocking controls. Privacy-safe audit logging records pipeline access, transformation, validation, identifier minimisation, and completion/failure events.

## Repository structure

### Active project areas

- `data/raw/` – local source data; customer-level contents are excluded from Git
- `data/interim/` – local cleaned/intermediate data; may still contain `msisdn` where chronology requires it
- `data/processed/` – local model-ready row-level analytical data; `msisdn` and the source label are removed
- `src/data/` – cleaning logic
- `src/data_quality/` – Great Expectations-based raw, interim, and processed validation scripts
- `src/features/` – model-dataset construction, temporal diagnostics, and feature-selection experiments
- `src/models/` – model comparison and calibration experiments
- `src/pipeline/` – Prefect orchestration
- `src/privacy/` – privacy-safe pipeline audit logging
- `src/monitoring/` – representation and operational bias/slice diagnostics
- `tests/unit/` – focused tests for transformations, privacy controls, and representation diagnostics
- `tests/validation/` – lightweight pipeline/validation contract tests
- `reports/` – reproducible aggregate analytical, validation, privacy, and presentation outputs
- `docs/` – data dictionary, methodology, decision records, and governance documentation
- `Dockerfile` – reproducible container build for the ETL pipeline

### Reserved / later-delivery areas

- `notebooks/` – exploratory notebooks if needed; reusable logic belongs in `src/`
- `models/` – persisted model artefacts and metadata when a final artefact is produced
- `dashboards/` – Power BI deliverables
- `config/` – shared configuration if project parameters are externalised later
- `.github/workflows/` – CI automation if introduced

Empty reserved directories are retained with `.gitkeep` files so the intended project structure remains visible.

## Data and privacy position

Customer-level source, interim, and processed datasets are not committed to GitHub. `.gitignore` excludes the contents of all three data directories.

`msisdn` is retained only temporarily where customer-level chronology is required. It is removed before the model-ready dataset is written and is not an approved predictor. The processed behavioural dataset remains restricted analytical data; identifier removal is not treated as proof of full anonymisation.

The main governance controls are documented in:

- `docs/governance/data_governance_framework.md` – umbrella rules for access, use, retention, lineage, auditability, and exceptions
- `docs/governance/data_anonymization_plan.md` – identifier minimisation and privacy audit logging
- `docs/governance/data_cleaning_policy.md` – standing cleaning rules
- `docs/governance/data_cleaning_narrative.md` – what the cleaning and validation work found and changed in this dataset
- `docs/governance/representation_bias_assessment.md` – scope and limitations of representation/bias checks
- `docs/governance/feature_selection.md` – feature eligibility and selection methodology
- `docs/governance/model_development_narrative.md` – analytical modelling story
- `docs/governance/model_decision_log.md` – compact decision trail and current modelling position

`docs/data_dictionary.md` records current field interpretation and modelling status, while `docs/data_dictionary_changelog.md` records material changes to those interpretations.

## Feature-selection history

`src/features/filter_screening.py` is retained as the initial Stage 2 single-development-window diagnostic. It has been superseded for the current methodology by `src/features/fold_filter_screening.py`, which evaluates filter evidence across calendar-aware folds. The earlier script remains for methodological lineage and should not be mistaken for the current selection baseline.

## Validation and testing

Great Expectations is used through Python validation scripts rather than a standalone `great_expectations/` project directory:

- `src/data_quality/gx_raw_validation.py`
- `src/data_quality/gx_interim_validation.py`
- `src/data_quality/gx_processed_validation.py`

Pytest complements dataset validation by checking reusable transformation and control logic. The Prefect flow and Docker image have both been executed successfully end to end on the project data.

## Dependencies

Two dependency files are intentional:

- `requirements.txt` – the wider analytical/development environment, including modelling, explainability, notebooks, validation, orchestration, testing, and Fairlearn
- `requirements-pipeline.txt` – the smaller dependency set required by the containerised ETL and Module 3 control path

Keeping the pipeline dependency set separate reduces the Docker image to what is needed for reproducible execution.

## Report evidence

Generated row-level data are not committed. `reports/README.md` explains which privacy-safe aggregate outputs are suitable for repository or assignment evidence and which outputs should remain local.

## Delivery approach

The work is organised into four Scrum-style two-week sprints: data understanding and setup; data preparation and EDA; modelling and evaluation; fairness, explainability, governance, and reporting. GitHub issue status records current completion state, while sprint labels record the iteration to which the work logically belongs.