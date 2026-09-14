# Telecom delinquency capstone

Business analytics capstone project for predicting five-day repayment delinquency in telecom-enabled microcredit.

## Project scope

The project uses historical telecom microcredit data for academic and demonstration purposes. It focuses on risk ranking at the time of a credit request and does not automate final customer decisions or predict long-term default.

## Repository structure

- `data/raw/` – source data (not committed when customer-level)
- `data/interim/` – cleaned, flagged, pseudonymised, or otherwise intermediate data products
- `data/processed/` – final modelling-ready data produced by the pipeline
- `notebooks/` – exploratory analysis and modelling notebooks; reusable logic is moved into `src/`
- `src/` – reusable pipeline, data-quality, transformation, privacy, monitoring, and modelling code
  - `src/data/` – ingestion and cleaning logic
  - `src/data_quality/` – profiling, anomaly analysis, and data-quality checks
  - `src/features/` – feature engineering and transformation logic
  - `src/pipeline/` – pipeline orchestration, including Prefect flow code
  - `src/privacy/` – identifier minimisation and pseudonymisation logic
  - `src/monitoring/` – bias/slice checks and audit logging
- `models/` – model artefacts and metadata
- `dashboards/` – Power BI outputs
- `tests/` – automated unit, pipeline, and validation tests
  - `tests/unit/` – unit tests for reusable functions
  - `tests/validation/` – tests for validation and pipeline behaviour
- `config/` – project configuration, paths, thresholds, and pipeline parameters
- `reports/` – reproducible analysis outputs
  - `reports/figures/` – generated plots and presentation-ready visuals
  - `reports/tables/` – generated analytical and validation tables
- `docs/` – documentation, data dictionary, governance, and methodology material
  - `docs/governance/` – governance, privacy, bias, and control documentation
  - `docs/governance/data_cleaning_policy.md` – standing cleaning rules and treatment principles used by the pipeline
  - `docs/governance/data_cleaning_narrative.md` – dataset-specific account of what the validation and cleaning steps found, changed, and left unresolved
  - `docs/data_dictionary.md` – current operational data dictionary
  - `docs/data_dictionary_changelog.md` – human-readable audit trail of material dictionary changes
- `great_expectations/` – Great Expectations project/configuration and validation artefacts
- `.github/workflows/` – workflow automation
- `Dockerfile` – reproducible container build for the pipeline

## Data cleaning documentation

The cleaning policy and the cleaning narrative serve different purposes. The policy defines the rules the pipeline is expected to follow, including how hard-invalid values, contamination, missing values, and audit logging are treated. The narrative describes what happened when those rules were applied to this dataset: the issues found in the raw data, the transformations made in the first cleaning pass, the validation outcome, and the questions that remain open.

Together with the data dictionary and its change log, these documents provide both the operational rules and the reasoning trail behind the evolving treatment of the data.

## Planned delivery

The work is organised into four sprints using a Scrum-style two-week sprint approach: data understanding and setup; data preparation and EDA; modelling and evaluation; fairness, explainability and reporting.
