# Data cleaning narrative

This file tells the story of what happened to the telecom delinquency data as the project moved from the raw extract to the model-ready table. The cleaning policy sets the standing rules; this narrative records what those rules found in this particular dataset and what was done about it.

## Starting point

The source file contains 209,593 records and 36 columns. It is kept unchanged in `data/raw/`. All cleaning is done on a separate copy in `data/interim/`, which makes it possible to compare every treatment with the original data.

Field meanings come first from the description supplied with the Kaggle dataset. Those meanings are then checked against the CSV itself. Where the documentation and the observed values do not line up cleanly, the uncertainty is left visible rather than filled in by assumption.

## What the raw validation found

The first Great Expectations run was deliberately executed on the untouched source. A fully green result was not the objective. The point was to show which defects were already present before cleaning began.

The raw data passed the expected-schema check, minimum row count, completeness checks for `msisdn`, `pdate`, and `label`, the binary target rule, and the non-negative checks applied to clearly defined monetary fields.

Five known problems remained:

- 1,539 negative values in `aon`;
- 1,315 negative values in `last_rech_date_ma`;
- 14 negative values in `last_rech_date_da`;
- fractional values in `cnt_da_rech30`;
- fractional values in `cnt_loans90`.

The raw validation therefore returned an overall failure. In this stage that is expected: the validator is showing the defects that the cleaning step is supposed to address.

## What was cleaned, and what was not

The project treats a semantic contradiction differently from an unusual observation.

Negative network tenure and negative elapsed days are invalid under the documented field meanings. Fractional values are also invalid in genuine count fields. Those cases can be treated with confidence.

A rare but possible value is different. Large positive counts are not removed simply because they sit in the tail. The same caution is used for negative balance-like values and negative daily-decrement values because the source material does not prove that adjustments, reversals, or debit states are impossible.

In other words, statistical extremeness is not enough on its own to justify changing a value.

## First cleaning pass

The first executable cleaning pass applied only the high-confidence rules already supported by the dictionary and policy. It set the following values to missing in the interim copy and wrote the original values to the audit log:

- 1,539 negative `aon` values;
- 1,315 negative `last_rech_date_ma` values;
- 14 negative `last_rech_date_da` values;
- 1,047 fractional `cnt_da_rech30` values;
- 1,047 fractional `cnt_loans90` values.

That made 4,962 changed cells. No rows were removed, so the interim table still had 209,593 records.

The run produced:

- `data/interim/sample_data_intw_cleaned.csv`
- `data/interim/cleaning_audit_log.csv`
- `reports/tables/cleaning_summary.json`

The raw file was not touched.

## Extended cleaning pass

The script was then extended to cover the other high-confidence treatments already documented in the policy.

The separated upper contamination regimes in `aon`, `last_rech_date_ma`, `last_rech_date_da`, `fr_ma_rech30`, and `fr_da_rech30` were set to missing in the interim copy. These cut-offs come from clear discontinuities in this dataset, including large empty gaps and implausible jumps. They are not presented as general telecom business limits.

The one exact duplicate row found during profiling was also removed. Repeated customers or repeated dates were not treated as duplicates unless every field in the row matched.

After the extended pass:

- 11,239 cells had been changed;
- 1 exact duplicate row had been removed;
- 209,592 rows remained.

Every change is represented in the cleaning audit, and the aggregate counts are kept in the cleaning summary.

## Validation after cleaning

The interim Great Expectations suite is different from the raw suite because missing values introduced deliberately by cleaning are now expected.

The interim validation confirmed the expected structure, valid target values, required structural fields, and non-negative retained values in the cleaned duration fields. Count columns were checked for mathematical integer semantics rather than relying on pandas dtype, since a count column with missing values may reload as floating point.

No fractional values remained in the cleaned count fields, and the interim validation passed overall. For example, `cnt_loans90` retained 208,545 non-missing valid values, which is consistent with the 1,047 fractional values removed from analytical use and the one duplicate row removed from the dataset.

## Defining the modelling period

The cleaned data are not treated as one uniform labelled period. After 23 July 2016 there are 58,825 records and every one of them is labelled as successful repayment.

That break is too sharp to dismiss as ordinary class imbalance. The source documentation does not explain whether the change comes from sampling, labelling, extraction, or a business-process change. Because the cause is unknown, the later block is not mixed into ordinary supervised training.

The model-ready population therefore uses records through 23 July 2016. After duplicate removal, this gives 150,767 modelling rows, including 26,162 delinquent cases. The later records are kept for lineage and possible drift diagnostics, but not as a conventional labelled validation set.

## Prefect orchestration

The raw-to-processed path is orchestrated in `src/pipeline/prefect_etl.py`:

`raw validation → cleaning → interim validation → model-ready transformation → processed validation → representation diagnostics`

The flow was tested locally on 15 September 2026 and completed successfully.

Raw validation remains diagnostic. The untouched source is expected to fail checks that identify defects later corrected by cleaning. Interim and processed validation are different: both are blocking gates. If either fails, the pipeline stops.

That distinction lets the project preserve evidence of source-data problems without allowing bad cleaned or processed data to pass quietly downstream.

## Tests

The project now has both unit tests and lightweight pipeline-contract tests. After the repository housekeeping pass, `python -m pytest tests -v` collected 12 tests and all 12 passed.

The tests cover, among other things:

- integer-like versus fractional count values;
- application of the agreed cleaning rules;
- duplicate removal and cleaning audit behaviour;
- the modelling cutoff and target construction;
- removal of `msisdn` and the source `label` from processed data;
- strictly earlier-date logic for customer-history features;
- rejection of invalid source labels;
- privacy-safe audit logging;
- representation-diagnostic input requirements;
- existence of the Prefect stage scripts and preservation of the raw/interim/processed path contract.

Pytest and Great Expectations serve different purposes. Great Expectations checks datasets at stage boundaries. Pytest checks that the code implementing transformations and controls behaves as intended on controlled examples.

## Docker verification

The ETL path is containerised with the root `Dockerfile`, `requirements-pipeline.txt`, and `.dockerignore`. The image uses `python:3.13-slim` and starts the Prefect flow automatically.

The image built successfully on 15 September 2026 as `telecom-delinquency-pipeline:latest`. It was then run with the local `data` and `reports` folders mounted into the container. The complete flow ran successfully inside Docker: raw validation reported the expected diagnostic findings, cleaning completed, interim validation passed, the model-ready data were rebuilt, processed validation passed, and the flow ended in `Completed` state.

This shows that the ETL is not dependent on the developer's local virtual environment.

## Privacy audit verification

The Prefect flow also writes a pipeline-level privacy audit to `reports/privacy/privacy_audit_log.jsonl`.

The verified run recorded pipeline start, raw-data access, raw-validation findings, cleaning, interim validation, identifier minimisation, processed validation, and completion. `msisdn` appears only as the name of the controlled field; no identifier value is written to the log. The processed output also excludes the original source `label` once `delinquent_5d` has been derived.

This audit is separate from the cell-level cleaning audit. The cleaning audit answers “what value changed and why?” The privacy audit answers “what processing step happened, on which asset, and did the privacy control succeed?”

## Remaining limitations

The basic pipeline is now reproducible and verified, but some analytical questions remain deliberately unresolved.

The `fr_*` variables are still excluded because their exact construction is unclear. `medianamnt_loans30` and `medianamnt_loans90` remain unresolved. `payback30` and `payback90` stay outside the approved predictor set until point-in-time leakage can be ruled out. The `maxamnt_loans30/90` fields are treated as consistency checks rather than independent predictors. Loan count and amount fields also remain outside the first approved set until their treatment of the current transaction can be verified.

The representation/bias checks and the broader data-governance framework have now been implemented as separate controls. Presentation and final reporting remain later delivery work.

## How the governance files fit together

`data_cleaning_policy.md` contains the standing treatment rules. `data_anonymization_plan.md` covers identifier minimisation and privacy logging. This file records what happened when those rules were applied to the telecom data.

The data dictionary gives the current interpretation and modelling status of each field. The dictionary change log explains how those interpretations evolved. Git history remains the exact record of code and document changes.