# Data anonymization and privacy logging plan

## Purpose

This document defines the privacy treatment for the telecom delinquency pipeline and the events that must be recorded in the privacy audit log. It builds on the Module 1–2 privacy approach and the controls already implemented during cleaning and model-dataset preparation.

The aim is data minimisation rather than synthetic replacement of every value. The pipeline should retain only the information needed for the analytical purpose and remove direct identifier-like data before the model-ready dataset is created.

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

The current `build_model_dataset.py` implementation already removes `msisdn` after the strictly prior customer-history features are derived. This plan formalises that behaviour as a privacy control rather than treating it as an incidental modelling choice.

## Other fields

The source `label` is also removed from the model-ready dataset after the analytical target `delinquent_5d` is derived. This prevents the original outcome representation from being retained redundantly.

No demographic protected characteristics are available in the source dataset. The pipeline must not infer or manufacture such attributes for anonymisation or fairness analysis.

Fields whose meanings remain unresolved are governed through feature eligibility rather than altered merely for privacy reasons. Data minimisation is achieved by excluding fields that are not required for the approved analytical purpose.

## Pseudonymisation position

The project does not create a persistent hashed or tokenised substitute for `msisdn` in the model-ready dataset. A pseudonym would still permit customer-level linkage and would therefore retain information that the current modelling specification does not need.

Where temporary customer linkage is necessary during preprocessing, the original identifier is confined to that stage and then removed. This is stronger minimisation for the stated modelling purpose than carrying a pseudonymous identifier forward unnecessarily.

If a future monitoring requirement genuinely needs longitudinal customer linkage after model scoring, a separate keyed pseudonymisation design should be introduced with the key stored outside the analytical repository. That is outside the present Module 3 scope.

## Privacy-safe transformation audit

The existing cleaning audit log already follows an important privacy rule: it records source row number, field, original value, triggered rule, treatment and run metadata, but it does not copy `msisdn` into the log.

The Module 3 privacy audit log extends this from cell-level cleaning evidence to pipeline-level processing evidence. It should record processing events without reproducing raw personal data.

Each event should contain:

- UTC timestamp;
- pipeline run identifier;
- pipeline stage;
- action or event type;
- input asset path or logical name;
- output asset path or logical name where relevant;
- row count where relevant;
- fields added, removed or transformed where relevant;
- validation outcome where relevant;
- status (`started`, `completed`, `failed`);
- a short privacy-safe message.

The privacy audit log must not contain:

- `msisdn` values;
- raw records;
- model feature values for identifiable individuals;
- secrets, credentials or local user paths that are not needed for traceability.

## Events to capture

At minimum, the end-to-end flow should record:

- raw dataset access for validation;
- raw validation completion and whether quality exceptions were found;
- cleaning start and completion;
- cleaning output row count and number of changed cells / removed rows;
- interim validation result;
- model-ready transformation start and completion;
- explicit identifier-removal event confirming removal of `msisdn` and source `label` from the processed output;
- processed-data validation result;
- pipeline completion or failure.

The cell-level cleaning audit and pipeline-level privacy audit serve different purposes and should remain separate. The cleaning audit explains what data values were altered. The privacy audit explains when and why datasets were accessed or transformed and confirms that identifier minimisation occurred.

## Output and retention

The pipeline-level privacy audit should be written as an append-only JSON Lines file under `reports/audit/` so each processing event is independently parseable and later runs can be distinguished by run identifier.

For the academic project, the log is retained with the project evidence needed to demonstrate reproducibility and privacy controls. In a production implementation, retention should be set by the organisation's formal records-retention policy rather than inferred from this project.

## Validation criteria

The privacy treatment is considered successful when all of the following are true:

- customer-level chronology can be derived correctly before identifier removal;
- `msisdn` is absent from the model-ready dataset;
- the original source `label` is absent from the model-ready dataset;
- cleaning and pipeline audit logs do not contain `msisdn`;
- privacy audit events provide enough information to reconstruct the processing sequence without reproducing identifiable records;
- automated tests verify the identifier-removal and privacy-safe logging behaviour.

## Current status

Identifier removal from the processed dataset and privacy-safe cleaning audit behaviour are already implemented. The next implementation step is to add the pipeline-level JSONL privacy audit logger and connect it to the Prefect flow, followed by focused Pytest coverage for those controls.
