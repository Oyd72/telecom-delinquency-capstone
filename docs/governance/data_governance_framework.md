# Data governance framework

## Purpose and scope

This framework sets the working rules for how data are handled in the telecom delinquency capstone: who should have access, what the data may be used for, how long different artefacts should be kept, and what evidence is needed to show that the pipeline was run properly.

It applies to the original source extract, cleaned interim data, the model-ready dataset, validation and modelling outputs, audit logs, lineage records, model artefacts, and the project documentation.

This is an academic project framework, not a substitute for the policies, legal analysis, or operational controls a production lender would need.

## Core principles

The project follows a small set of practical rules:

- **Use the data only for the stated purpose.** The purpose here is analysis and prediction of five-day repayment delinquency in telecom-enabled microcredit.
- **Keep only what is needed.** A field should have a clear reason to exist in the pipeline.
- **Respect chronology.** A predictor must be defensibly available at the intended scoring point.
- **Keep the processing traceable.** Important transformations, validation results, privacy controls, and modelling decisions should be reproducible from code and records.
- **Leave the source intact.** Raw data are not overwritten; cleaning and modelling use derived copies.
- **Do not invent missing group attributes.** The project does not infer protected characteristics simply to make a fairness analysis possible.
- **Keep human accountability.** Model output is a risk signal, not an autonomous lending decision.

## Roles

One person currently performs most project tasks, but it is still useful to separate the responsibilities conceptually.

| Role | Responsibility |
| --- | --- |
| Project owner / accountable analyst | Defines the analytical purpose, approves material methodological choices, and keeps the work aligned with the assignment. |
| Data custodian | Controls access to raw and interim customer-level data and applies retention/deletion rules. |
| Pipeline / model developer | Maintains transformations, validation, feature engineering, model code, and tests. |
| Governance reviewer | Reviews privacy, data quality, representation, lineage, and documented limitations. |
| Report consumer | Uses approved outputs and documentation only for the stated project purpose. |

In a production setting, these responsibilities should not automatically sit with one person.

## Data classification

### Restricted customer-level data

This includes the raw source file, interim data containing `msisdn`, and any temporary file that still allows direct customer-level linkage.

These files receive the strictest treatment because they combine row-level behavioural data with an identifier-like field.

### Restricted analytical data

This includes the processed model-ready dataset and other row-level modelling tables after `msisdn` has been removed.

Removing the identifier lowers risk but does not make the behavioural data automatically anonymous. These files remain restricted analytical material.

### Controlled project evidence

Cleaning audits, privacy audits, Great Expectations results, representation summaries, model-evaluation tables, and generated figures fall into this category.

They should not contain raw customer identifiers. They can be used as assignment evidence when they do not expose row-level personal data.

### Repository-safe material

Source code, tests, non-secret configuration, governance documents, and aggregate reports or figures can be stored in GitHub when they contain no restricted row-level data.

## Access rules

Raw and interim customer-level data stay local and are not committed to GitHub. Access is limited to the project owner or an explicitly authorised collaborator who genuinely needs the data for preprocessing, validation, or analysis.

Raw data should not be copied into email, public cloud folders, presentations, or other unnecessary locations. `msisdn` is used only for the short part of preprocessing that requires customer-level chronology. Secrets and credentials do not belong in scripts, notebooks, logs, or repository files.

The model-ready dataset remains restricted even after `msisdn` has been removed. It can be used for modelling, validation, representation checks, and reproducibility, but it should not be republished through the repository as an open dataset.

Aggregate reports and documentation may be shared more broadly after checking that they do not expose row-level values or identifiers.

## Permitted and out-of-scope uses

The data may be used to clean and validate the historical extract, derive leakage-conscious features, compare delinquency models, assess calibration and temporal stability, run representation checks, and produce the academic deliverables and reproducibility evidence.

The project does not use the data or model to make autonomous lending decisions, attempt re-identification, infer missing protected characteristics, use `msisdn` as a predictor, or present the academic model as production-ready. The post-23-July all-success block also remains outside ordinary supervised training unless its data-generation anomaly can be explained.

## Data quality controls

Different stages have different validation roles.

Raw Great Expectations checks are diagnostic. Their job is to show what is wrong with the untouched source.

Cleaning follows `docs/governance/data_cleaning_policy.md`, which deliberately avoids changing values simply because they are statistically unusual.

Interim validation is blocking. Processed validation is also blocking and includes checks that forbidden identifier fields are absent. Pytest checks reusable transformation and control logic. Prefect preserves the order of these steps and stops the flow if a blocking control fails.

## Privacy and identifier minimisation

The detailed privacy design is in `docs/governance/data_anonymization_plan.md`.

In summary, `msisdn` is retained only while grouping and chronology require it, then removed. No persistent hash or token is kept when the model does not need customer linkage. The source `label` is removed after `delinquent_5d` is derived. Privacy-audit events record processing metadata rather than identifier values. Identifier removal alone is not treated as proof that the remaining row-level behavioural data are anonymous.

## Lineage and change control

The main lineage is:

`raw source → raw validation → cleaned interim data → interim validation → model-ready transformation → processed validation → representation diagnostics → modelling and reporting`

Evidence for that path comes from the source file, cleaning audit, cleaning summary, Great Expectations outputs, privacy audit, data dictionary and change log, modelling documentation, model decision log, and Git history.

A material change to cleaning rules, feature eligibility, model specification, or governance controls should be documented before it becomes the new project baseline.

## Retention and deletion

Retention is tied to the academic purpose, not to indefinite storage.

### Raw customer-level data

Keep the raw source only while it is needed to complete the project, verify the submission, and cover any relevant grading or appeal period. After that, delete the local copy unless there is a separate authorised reason to retain it.

### Interim customer-level data

Interim data can be regenerated from the source and pipeline code. Delete them when they are no longer needed for active work and no later than the raw data.

### Processed model-ready data

Keep the processed row-level dataset only for active modelling, reproducibility, and assessment evidence. It is still restricted analytical data and should also be deleted once the academic purpose and review period end, unless another approved purpose exists.

### Audit logs and aggregate evidence

Privacy-safe logs, aggregate validation outputs, governance documents, code, and non-identifying figures may remain with the academic project record because they provide useful traceability without intentionally retaining direct customer identifiers.

### Repository material

Code, tests, methodology, governance documentation, and aggregate outputs may remain in GitHub after the course as long as they contain no restricted row-level data, secrets, or direct identifiers.

Deletion should cover unnecessary duplicate copies and exported working files as well as the main project folders.

## Audit logging

The pipeline-level privacy audit records pipeline start and completion or failure, raw-data access for validation, validation outcomes, transformations, identifier removal, and execution of representation diagnostics.

It is intentionally separate from the cleaning audit. The cleaning audit records changes to values. The privacy audit records the processing sequence and privacy-relevant controls.

## Representation and bias governance

The source data do not contain usable demographic protected attributes. The project therefore limits bias assessment to operational slices that are actually present in the data.

The current check compares first-time and returning borrowers and keeps the left-censoring limitation explicit. `pcircle` is constant and cannot support a regional comparison. Differences are reported for interpretation; they are not converted into an automatic declaration that the data are fair or unfair.

## Model and output governance

The model is intended to produce a delinquency probability or risk ranking for analytical use.

The main controls are exclusion of direct identifiers and unresolved fields, chronology-aware feature construction, forward-chaining evaluation, documented calibration limitations, explainability through several methods, and a preference for the simpler 12-feature specification where performance is effectively retained.

The short observation history remains a significant limitation. Any real production use would require fresh validation on representative current data, formal access controls, monitoring, incident handling, named model ownership, and relevant legal/compliance review.

## Incident and exception handling

Examples of governance exceptions include restricted data being committed to GitHub, a direct identifier appearing in processed output or privacy logs, bypass of a blocking validation failure, use of an excluded leakage-prone field without review, unexplained changes in target prevalence or feature distributions, or loss of lineage evidence.

If that happens, processing should stop where practical. The affected output should not be relied on until the event is documented and the underlying control or data problem has been corrected.

## Review

Review this framework when the source data change, new collaborators receive access, the model purpose or feature definitions change materially, new personal or protected attributes appear, retention needs change, or the work moves beyond an academic demonstration.

The framework is the umbrella document. The more detailed cleaning, privacy, representation, and modelling files remain the working evidence underneath it.