# Telecom delinquency capstone

Business analytics capstone project for predicting five-day repayment delinquency in telecom-enabled microcredit.

## Project scope

The project uses historical telecom microcredit data for academic and demonstration purposes. It focuses on risk ranking at the time of a credit request and does not automate final customer decisions or predict long-term default.

## Repository structure

- `data/raw/` – source data (not committed when customer-level)
- `data/interim/` – intermediate data products
- `data/processed/` – modelling-ready data
- `notebooks/` – exploratory analysis and modelling notebooks
- `src/` – reusable source code
- `models/` – model artefacts and metadata
- `dashboards/` – Power BI outputs
- `tests/` – tests and validation checks
- `config/` – project configuration
- `reports/` – analysis outputs and figures
- `docs/` – documentation and governance material
- `.github/workflows/` – workflow automation

## Planned delivery

The work is organised into four sprints: data understanding and setup; data preparation and EDA; modelling and evaluation; fairness, explainability and reporting.
