# Data anonymization and privacy logging plan

## Purpose

This document defines the privacy treatment for the telecom delinquency pipeline and the events recorded in the privacy audit log. It builds on the Module 1–2 privacy approach and the controls implemented during cleaning and model-dataset preparation.

The aim is data minimisation rather than synthetic replacement of every value. The pipeline retains only the information needed for the analytical purpose and removes direct identifier-like data before the model-ready dataset is created.

## Identifier treatment

### `msisdn`

`msisdn` is treated as a direct identifier-like field and is not an approved modelling feature.

It is required temporarily during controlled preprocessing because repeated records for the same customer must be linked to derive chronology-sensitive features such as strictly prior transaction counts. For that limited purpose, `msisdn` may exist in the raw and interim processing stages.

The privacy rule is therefore:

1. retain `msisdn` only while customer-level grouping and chronological derivation require it;
2. do not copy `msisdn` into transformation audit records;
3. remove `msisdn` before writing the model-ready dataset;
4. validate that `msisdn` is absent from the processed output;
5. do not expose a reversible identifier mapping as part of the analytical repository.

The current `build_model_dataset.py` implementation removes `msisdn` after the strictly prior customer-history features are derived. This behaviour is treated as a privacy control rather than an incidental modelling choice.

## Other fields

The source `label` is also removed from the model-ready dataset after the analytical target `delinquent_5d` is derived. This prevents the original outcome representation from being retained redundantly.

No demographic protected characteristics are available in the source dataset. The pipeline must not infer or manufacture such attributes for anonymisation or fairness analysis.

Fields whose meanings remain unresolved are governed through feature eligibility rather than altered merely for privacy reasons. Data minimisation is achieved by excluding fields that are not required for the approved analytical purpose.

## Pseudonymisation position

The project does not create a persistent hashed or tokenised substitute for `msisdn` in the model-ready dataset. A pseudonym would still permit customer-level linkage and would therefore retain information that the current modelling specification does not need.

Where temporary customer linkage is necessary during preprocessing, the original identifier is confined to that stage and then removed. This is stronger minimisation for the stated modelling purpose than carrying a pseudonymous identifier forward unnecessarily.

If a future monitoring requirement genuinely needs longitudinal customer linkage after model scoring, a separate keyed pseudonymisation design should be introduced with the key stored outside the analytical repository. That is outside the present Module 3 scope.

## Privacy-safe transformation audit

The existing cleaning audit log already follows an important privacy rule: it records source row number, field, original value, triggered rule and treatment, but it does not copy `msisdn` into the log.

The Module 3 privacy audit log extends this from cell-level cleaning evidence to pipeline-level processing evidence. It records processing events without reproducing raw personal data.

Each event contains the information needed for traceability, including:

- UTC timestamp;
- pipeline run identifier;
- pipeline stage;
- event type;
- status;
- logical input/output asset paths where relevant;
- validation outcome where relevant;
- identifier-removal confirmation where relevant.

The privacy audit log must not contain raw `msisdn` values, raw records, identifiable feature values, secrets, credentials, or other unnecessary personal data.

The logger also rejects forbidden personal-data keys such as `msisdn`, `raw_identifier`, and `original_value` if a caller attempts to add them directly to an audit event.

## Events captured

The verified end-to-end flow records:

- pipeline run start;
- raw dataset access for validation;
- raw validation result, including whether findings are diagnostic or blocking;
- cleaning transformation completion;
- interim validation result;
- model-ready transformation and explicit identifier minimisation;
- confirmation that `msisdn` and the source `label` are not retained in processed output;
- processed-data validation result;
- pipeline completion or failure.

The cell-level cleaning audit and pipeline-level privacy audit serve different purposes and remain separate. The cleaning audit explains what data values were altered. The privacy audit explains when and why datasets were accessed or transformed and confirms that identifier minimisation occurred.

## Output and retention

The pipeline-level privacy audit is written as append-only JSON Lines to:

`reports/privacy/privacy_audit_log.jsonl`

Each event is independently parseable and each pipeline run is distinguished by a run identifier.

For the academic project, the log is retained with the project evidence needed to demonstrate reproducibility and privacy controls. In a production implementation, retention should be set by the organisation's formal records-retention policy rather than inferred from this project.

## Validation criteria

The privacy treatment is considered successful when all of the following are true:

- customer-level chronology can be derived correctly before identifier removal;
- `msisdn` is absent from the model-ready dataset;
- the original source `label` is absent from the model-ready dataset;
- cleaning and pipeline audit logs do not contain `msisdn` values;
- privacy audit events provide enough information to reconstruct the processing sequence without reproducing identifiable records;
- automated tests verify identifier-removal and privacy-safe logging behaviour.

## Verification result

The privacy controls were verified on 15 September 2026.

The unit-test suite ran seven tests and all seven passed, including tests confirming that structured JSONL events are written without identifier values and that forbidden personal-data keys are rejected.

The Prefect ETL was then run end to end with privacy audit logging enabled. The flow completed successfully, with interim and processed validation passing.

The resulting JSONL audit trail contained the expected sequence of events: pipeline start, raw data access, diagnostic raw validation findings, cleaning completion, interim validation, identifier minimisation, processed validation, and pipeline completion. The identifier-minimisation event records only the field name `msisdn` as control metadata and confirms `identifier_retained_in_output: false`; it does not contain any raw identifier value.

The implemented controls therefore satisfy the Module 3 privacy-treatment and pipeline-audit-logging requirement for the current project scope.
