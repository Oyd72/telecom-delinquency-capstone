# Data governance framework

## Purpose and scope

This framework defines how data used in the telecom delinquency capstone should be accessed, used, transformed, retained, and evidenced throughout the analytical lifecycle. It is a project-level governance framework for the academic pipeline rather than a statement of legal obligations for a production lender.

The framework applies to:

- the original telecom microcredit source dataset;
- interim cleaned data;
- the processed model-ready dataset;
- validation, privacy, bias, and modelling reports;
- audit logs and lineage records;
- model artefacts and documentation produced by the project.

It builds on the data cleaning policy, anonymization and privacy logging plan, representation/bias assessment, data dictionary, and model decision records already maintained in the repository.

## Governance principles

The project follows these operating principles:

- **Purpose limitation:** data are used only for the stated academic objective of analysing and predicting five-day repayment delinquency in telecom-enabled microcredit.
- **Data minimisation:** fields are retained only where they are needed for cleaning, chronology, modelling, validation, or governance evidence.
- **Chronology and leakage control:** only information defensibly available at the intended scoring point may be used as a predictor.
- **Traceability:** material transformations, validation outcomes, privacy controls, and modelling decisions must be reproducible from code, reports, and governance records.
- **Separation of raw and derived data:** the source dataset remains unchanged; cleaning and modelling occur on separate outputs.
- **No unsupported inference:** the project does not invent demographic protected characteristics or infer them merely to enable fairness analysis.
- **Human accountability:** model outputs are analytical risk signals, not autonomous final lending decisions.

## Roles and responsibilities

The project is currently maintained by one analyst, so several roles may be held by the same person. The roles are nevertheless kept conceptually separate so the framework can scale to collaborative use.

| Role | Main responsibilities |
| --- | --- |
| Project owner / accountable analyst | Defines analytical purpose, approves material methodological decisions, and ensures assignment requirements are met. |
| Data custodian | Controls access to raw and interim customer-level data, protects local copies, and enforces retention/deletion rules. |
| Pipeline / model developer | Implements transformations, validation, feature engineering, model code, and tests. |
| Governance reviewer | Reviews privacy, data quality, representation/bias, lineage, and documented limitations before outputs are relied on. |
| Report consumer | Uses only approved reports, model outputs, and documentation for the stated academic purpose. |

In a production setting these responsibilities should not automatically be concentrated in one individual.

## Data classification

For this project, artefacts are classified according to the level of customer-level information they contain.

### Restricted customer-level data

Includes:

- `data/raw/sample_data_intw.csv`;
- interim data containing `msisdn`;
- any other temporary file from which individual customer activity can be linked directly.

These data require the strongest access restriction because they contain the identifier-like `msisdn` field and row-level behavioural data.

### Restricted analytical data

Includes:

- `data/processed/telecom_delinquency_model_ready.csv`;
- row-level modelling datasets from which `msisdn` has been removed.

Removal of `msisdn` reduces identification risk but does not make the dataset inherently anonymous. Row-level behavioural data remain restricted analytical material.

### Controlled project evidence

Includes:

- cleaning audit logs;
- privacy audit logs;
- Great Expectations results;
- representation/bias summaries;
- model evaluation tables and generated figures.

These artefacts should not contain raw customer identifiers. They may be shared within the project or included in assignment evidence where they do not expose row-level personal data.

### Repository-safe artefacts

Includes:

- source code;
- tests;
- configuration without secrets;
- governance documents;
- aggregate reports and figures that do not disclose individual records.

These are suitable for the GitHub repository.

## Access control policy

Access follows least-privilege and need-to-know principles.

### Raw and interim data

- Raw and interim customer-level datasets are stored locally and are not committed to GitHub.
- Access is limited to the project owner or another specifically authorised collaborator who needs the data for preprocessing, validation, or controlled analysis.
- Raw data must not be sent by email, placed in public cloud folders, or copied into presentation material.
- `msisdn` may be used only during the limited preprocessing steps that require customer-level chronology.
- Credentials, API keys, or secrets must not be stored in notebooks, scripts, audit logs, or repository files.

The repository `.gitignore` excludes the contents of `data/raw/`, `data/interim/`, and `data/processed/` from version control while keeping only placeholder files.

### Processed data

- The model-ready dataset remains restricted to analytical use even after `msisdn` is removed.
- It may be used for approved modelling, validation, representation diagnostics, and reproducibility checks.
- It must not be republished as an open dataset through this repository.

### Reports and documentation

- Aggregate metrics, figures, methodology documents, and privacy-safe audit evidence may be stored in the repository.
- Any report intended for broader sharing must be checked for row-level values, identifiers, or unnecessarily detailed customer information.

## Permitted and prohibited use

### Permitted use

The data may be used to:

- validate and clean the historical source extract;
- derive leakage-conscious analytical features;
- develop and compare delinquency models;
- assess calibration, temporal stability, representation, and data quality;
- produce academic reports, dashboards, and reproducibility evidence;
- demonstrate governance controls required by the course.

### Prohibited or out-of-scope use

The project data and outputs must not be used to:

- make autonomous final lending or adverse customer decisions;
- attempt to re-identify individuals after identifier removal;
- infer demographic or protected characteristics that are absent from the dataset;
- use `msisdn` as a predictive model feature;
- treat the post-23-July all-success block as ordinary labelled training data without resolving its data-generation anomaly;
- use unresolved or potentially leakage-prone fields as production predictors without documented review;
- claim demographic fairness where suitable protected attributes are not available;
- represent the academic model as production-ready without further validation, governance, monitoring, and legal review.

## Data quality and validation controls

Data are promoted through the pipeline only when the control appropriate to that stage has been satisfied.

- Raw-data Great Expectations validation is diagnostic. Its purpose is to expose known source defects before cleaning.
- Cleaning is governed by `docs/governance/data_cleaning_policy.md`; unusual values are not changed solely because they are statistically extreme.
- Interim validation is a blocking gate. The cleaned data cannot progress if the agreed semantic and structural checks fail.
- Processed-data validation is also a blocking gate and confirms, among other controls, that forbidden identifier fields are absent.
- Pytest verifies reusable transformation and privacy-control logic on controlled examples.
- The Prefect flow preserves the execution order and fails if a blocking downstream control fails.

## Privacy and identifier minimisation

The detailed privacy design is maintained in `docs/governance/data_anonymization_plan.md`.

The principal rules are:

- `msisdn` is retained only while customer-level grouping and chronology require it;
- it is not written to the model-ready dataset;
- no persistent hashed or tokenised substitute is retained where the analytical purpose does not require customer linkage;
- the source `label` is removed after `delinquent_5d` is derived;
- privacy audit events record processing metadata but not raw identifier values;
- the project does not treat identifier removal alone as proof that row-level behavioural data are anonymous.

## Data lineage and change control

Lineage is maintained through the fixed pipeline sequence:

`raw source → raw validation → cleaned interim data → interim validation → model-ready transformation → processed validation → representation diagnostics → modelling and reporting`

Supporting evidence includes:

- the immutable source file;
- `data/interim/cleaning_audit_log.csv`;
- `reports/tables/cleaning_summary.json`;
- Great Expectations validation outputs;
- `reports/privacy/privacy_audit_log.jsonl`;
- data dictionary and changelog;
- feature-selection and modelling narratives;
- the model decision log;
- Git commit history.

Material changes to cleaning rules, feature eligibility, model specification, or governance controls should be documented before being treated as the new project baseline.

## Retention and deletion policy

Retention is based on the academic purpose and reproducibility needs rather than indefinite storage.

### Raw customer-level data

Retain only while needed to complete the course project, verify submitted work, and cover any applicable grading or appeal period. After that purpose has ended, the local raw dataset should be deleted unless there is a separate legitimate reason and permission to retain it.

### Interim customer-level data

Interim datasets are reproducible from the raw source and pipeline code. They should be deleted when no longer needed for active development and, at the latest, when the raw dataset is deleted.

### Processed model-ready data

The processed dataset should be retained only for active modelling, reproducibility, and assessment evidence. Because it is reproducible and still contains row-level behavioural data, it should also be deleted when the academic project and any review period are complete unless a separate approved purpose exists.

### Audit logs and aggregate validation evidence

Privacy-safe audit logs, aggregate validation outputs, governance documents, code, and non-identifying figures may be retained with the academic project record because they provide traceability without intentionally retaining direct customer identifiers.

### Repository material

Source code, tests, methodology, governance documents, and aggregate outputs may remain in GitHub after the course, provided they contain no restricted row-level data, secrets, or direct identifiers.

Deletion of local data should include unnecessary duplicate copies and exported working files, not only the primary project folders.

## Audit logging and accountability

The pipeline-level privacy audit log records:

- pipeline start and completion/failure;
- raw dataset access for validation;
- validation outcomes;
- cleaning and transformation events;
- identifier minimisation;
- representation-diagnostic execution.

The log is intentionally privacy-safe and does not store raw `msisdn` values or identifiable feature values. It complements, rather than replaces, the cell-level cleaning audit and Git history.

## Representation and bias governance

The dataset does not contain usable demographic protected attributes. The project therefore limits automated bias assessment to defensible operational slices and clearly separates those from demographic fairness claims.

The current representation suite compares first-time and returning borrowers as an operational diagnostic and records the left-censoring limitation. `pcircle` is constant and cannot support meaningful group comparison.

Observed differences trigger interpretation and documentation, not an automatic declaration that the dataset or model is fair or unfair.

## Model and output governance

The model is intended to produce a delinquency probability or risk ranking for analytical use. Governance controls include:

- exclusion of direct identifiers and unresolved high-risk features;
- chronology-aware feature construction and forward-chaining evaluation;
- documented calibration limitations;
- explainability through coefficients, permutation importance, and SHAP where appropriate;
- retention of a simpler preferred specification with a broader challenger model;
- explicit acknowledgement that short observation history limits production conclusions.

Any future production deployment would require fresh validation on representative current data, formal access controls, operational monitoring, incident handling, documented model ownership, and applicable legal/compliance review.

## Incident and exception handling

A governance exception includes any event such as:

- restricted data being committed to GitHub or otherwise exposed;
- a direct identifier appearing in processed model output or privacy audit logs;
- a blocking validation failure being bypassed;
- use of an excluded or leakage-prone predictor without review;
- unexplained material changes in target prevalence or feature distributions;
- corruption or loss of lineage evidence.

If an exception occurs, processing should stop where feasible, the affected output should not be relied on, the event should be documented, and the control or dataset should be corrected before the pipeline resumes.

## Review and maintenance

This framework should be reviewed when:

- the dataset or source changes;
- new collaborators obtain access;
- feature definitions or model purpose change materially;
- new personal or protected attributes are introduced;
- retention needs change;
- the project moves from academic demonstration toward operational use.

The governance framework does not replace the more detailed documents in `docs/governance/`. It provides the umbrella rules that connect access control, permitted use, privacy, validation, retention, auditability, representation checks, and model governance into one lifecycle view.
