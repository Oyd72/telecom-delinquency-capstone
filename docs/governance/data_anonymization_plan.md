# Data anonymization and privacy logging plan

## Purpose

This document describes how the telecom delinquency pipeline handles identifier-like data and what is written to the privacy audit log. It builds on the earlier privacy planning and on the controls already implemented in the cleaning and model-dataset steps.

The approach is based on minimisation. The pipeline keeps only what is needed for the analytical task and removes direct identifier-like data before the model-ready dataset is written.

## Treatment of `msisdn`

`msisdn` is not an approved modelling feature. It is kept temporarily because repeated records for the same customer have to be linked when chronology-sensitive features are created.

The rule is simple:

1. keep `msisdn` only while customer grouping and chronological derivation need it;
2. do not copy it into transformation audit records;
3. remove it before writing the model-ready dataset;
4. check that it is absent from the processed output;
5. do not create a reversible identifier mapping in the analytical repository.

`build_model_dataset.py` removes `msisdn` after the strictly prior customer-history features have been derived. This is treated as a privacy control, not just as a modelling choice.

## Other fields

The source `label` is removed after `delinquent_5d` has been created. Keeping both would add no analytical value and could create accidental target leakage.

The source data do not contain usable demographic protected characteristics. The project does not try to manufacture or infer them for privacy or fairness work.

Fields with unresolved meaning are controlled through feature eligibility rather than altered for privacy reasons. The main privacy measure is to avoid retaining data that the model does not need.

## Why no persistent pseudonym is kept

The processed dataset does not contain a hashed or tokenised replacement for `msisdn`. A persistent pseudonym would still allow customer-level linkage, and the current model does not need that capability.

Where linkage is required during preprocessing, the original identifier is used only for that step and then removed. If future monitoring genuinely requires longitudinal linkage after scoring, that should be designed separately, with a keyed pseudonymisation scheme and the key held outside the analytical repository.

## Cleaning audit and privacy audit

The cleaning audit and the privacy audit serve different purposes.

The cleaning audit records which values were changed and which rule was applied. It does not add `msisdn` to those records.

The privacy audit records the processing sequence itself: when a dataset was accessed, transformed, validated, or stripped of identifiers. It does not reproduce row-level personal data.

Each privacy-audit event can include:

- UTC timestamp;
- pipeline run ID;
- pipeline stage;
- event type;
- status;
- logical input/output paths where relevant;
- validation result where relevant;
- confirmation of identifier removal where relevant.

The privacy audit must not contain raw `msisdn` values, raw records, identifiable feature values, secrets, credentials, or other unnecessary personal data. The logger also rejects keys such as `msisdn`, `raw_identifier`, and `original_value` if a caller tries to add them directly.

## Events recorded

The verified flow records:

- pipeline start;
- raw-data access for validation;
- raw-validation outcome and whether it is diagnostic or blocking;
- cleaning completion;
- interim validation;
- model-ready transformation and identifier removal;
- confirmation that `msisdn` and the source `label` are not retained in processed output;
- processed validation;
- pipeline completion or failure.

## Output and retention

The pipeline-level privacy audit is written as append-only JSON Lines to:

`reports/privacy/privacy_audit_log.jsonl`

Each line is independently parseable and every run has its own run ID.

For this academic project, the log is kept with the evidence needed to show reproducibility and privacy controls. A production system would need a formal retention period set by the organisation rather than by this project.

## What counts as successful privacy treatment

The control is working as intended when:

- customer chronology can be derived before identifier removal;
- `msisdn` is absent from the model-ready dataset;
- the original source `label` is absent from the model-ready dataset;
- cleaning and privacy audit logs do not contain `msisdn` values;
- the audit trail is detailed enough to reconstruct the processing sequence without reproducing identifiable records;
- automated tests confirm identifier removal and safe audit logging.

## Verification

The privacy controls were verified on 15 September 2026.

At that point the unit-test suite ran seven tests and all passed, including the privacy-specific checks for JSONL logging and rejection of forbidden personal-data keys. The Prefect ETL also completed end to end with privacy logging enabled.

The resulting audit trail contained the expected sequence from pipeline start through raw access, validation, cleaning, identifier minimisation, processed validation, and completion. In the identifier-removal event, `msisdn` appears only as the name of the field being controlled. No identifier value is recorded, and the event confirms `identifier_retained_in_output: false`.

For the present Module 3 scope, these controls cover the privacy-treatment and pipeline-audit-logging requirement.