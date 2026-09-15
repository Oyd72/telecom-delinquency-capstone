# Data cleaning narrative

This document explains how the raw telecom delinquency data have been assessed, cleaned, and validated so far. It complements the formal cleaning policy by describing what actually happened in this dataset and why each treatment was applied.

## Starting point

The project works from the original Kaggle telecom delinquency dataset with 209,593 records and 36 columns. The raw file is treated as immutable. All cleaning is performed on a separate copy written to `data/interim/`, so the source data remain available for comparison and reproducibility.

Field interpretation is based on the supplied Kaggle data description first, then tested against the observed CSV structure and distributions. Where the documentation and the data do not align cleanly, the ambiguity is documented rather than resolved by assumption.

## What the raw-data validation found

The first Great Expectations validation was deliberately run against the untouched raw dataset. Its purpose was not to produce a green result, but to identify hard structural and semantic violations that the cleaning stage would need to address.

The raw dataset passed the structural checks for the expected 36-column schema, the minimum row count, completeness of `msisdn`, `pdate`, and `label`, the binary target rule `label ∈ {0,1}`, and the non-negative rules for the clearly defined monetary fields tested.

Five blocking issues remained:

- `aon` contained 1,539 negative values, which conflict with its documented meaning as age on the cellular network in days;
- `last_rech_date_ma` contained 1,315 negative elapsed-day values;
- `last_rech_date_da` contained 14 negative elapsed-day values;
- `cnt_da_rech30` was loaded as a floating-point column because it contains fractional values, which violates count semantics;
- `cnt_loans90` showed the same issue.

The raw Great Expectations suite therefore failed overall, as expected. That failure was evidence that the validation logic was detecting the known data-quality issues rather than an implementation failure.

## Why only some unusual values were cleaned

The project distinguishes between values that conflict with documented field semantics and values that are merely unusual.

Negative values in `aon` and the two recharge-date fields were treated as invalid because elapsed days and network tenure cannot logically be negative under the documented meaning.

Fractional values in genuine count fields were also treated as invalid because a count must be integer-valued. In contrast, large positive counts were not removed simply because they were rare. A value such as 203 recharges in 30 days may be unusual, but it is not impossible on semantic grounds.

The same caution applies to several balance-like and expenditure variables. Negative balances or daily-decrement values were not automatically deleted because the available documentation does not establish that such values are impossible. They may reflect adjustments, reversals, or an undocumented encoding.

This distinction is deliberate: statistical extremeness alone is not treated as proof of invalidity.

## Initial cleaning pass

The first executable cleaning stage implemented only the high-confidence rules already supported by the data dictionary and cleaning policy.

The pipeline converted the following invalid values to missing in the interim analytical copy while preserving the original values in the automated audit log:

- 1,539 negative `aon` values;
- 1,315 negative `last_rech_date_ma` values;
- 14 negative `last_rech_date_da` values;
- 1,047 fractional `cnt_da_rech30` values;
- 1,047 fractional `cnt_loans90` values.

This produced 4,962 changed cells in total. No rows were removed, and the interim dataset retained all 209,593 records.

The cleaning run produced three outputs:

- `data/interim/sample_data_intw_cleaned.csv` – the interim cleaned dataset;
- `data/interim/cleaning_audit_log.csv` – the automated record of altered cells and the rules applied;
- `reports/tables/cleaning_summary.json` – an aggregate summary of the cleaning run.

The raw file was not overwritten.

## Extended cleaning pass

After the initial rules had been validated, the cleaning script was extended to implement the remaining high-confidence treatments already documented in the policy.

The separated contamination regimes identified in the upper tails of `aon`, `last_rech_date_ma`, `last_rech_date_da`, `fr_ma_rech30`, and `fr_da_rech30` were converted to missing in the interim copy. These treatments are based on clear discontinuities in this dataset, including large empty gaps and implausible jumps, and are not presented as universal business thresholds.

The script also removed the one exact duplicate row identified during profiling. Repeated customers and repeated transaction dates were not treated as duplicates unless the complete record was identical.

The extended cleaning run produced:

- 11,239 changed cells;
- 1 exact duplicate row removed;
- 209,592 output rows.

The audit log records every altered value and the duplicate-removal event, while the aggregate summary records the counts by rule. The immutable raw dataset remains unchanged.

## Validation after cleaning

A separate Great Expectations validation was run against the interim cleaned dataset. The cleaned-data validation is intentionally different from the raw validation because values converted to missing are expected at this stage.

The interim validation confirmed that the dataset still has the expected structure, valid target values, and required structural fields. It also confirmed that retained values in the cleaned duration fields are non-negative.

The count fields were checked separately for mathematical integer semantics so that missing values introduced by cleaning would not create false failures simply because pandas reloads the column as floating point. No retained fractional values remained after the extended cleaning pass, and the overall cleaned-data validation passed.

For example, `cnt_loans90` retained 208,545 non-missing valid values in the 209,592-row interim dataset, exactly reflecting the 1,047 fractional values removed from analytical use together with the single duplicate-row removal.

## Transition to the modelling population

After cleaning, the dataset is not used for supervised modelling as one undifferentiated time period. Records dated after **23 July 2016** form a distinct block of 58,825 observations in which the target is **100% successful repayment**.

This discontinuity is not treated as ordinary class imbalance. The available documentation does not establish why the later period contains no delinquent outcomes, so the project does not assume that its sampling or label-generation process is comparable with the earlier period. Possible explanations could include a change in data extraction, labelling maturity, business process, or sampling, but none of these is established by the source documentation.

For that reason, the later observations are excluded from the ordinary labelled modelling population. Including them in supervised training would artificially increase the proportion of successful cases and could distort learned relationships between predictors and delinquency. They are retained for lineage and may later support separate diagnostic analysis, such as checking distributional shift or covariate drift, but they are not used as a conventional labelled validation set.

The first model-ready dataset therefore uses records through 23 July 2016 only. After removal of the single exact duplicate, this produces 150,767 modelling rows, including 26,162 delinquent cases.

## What remains unresolved

The current interim dataset has passed the agreed cleaning-stage validation, but several modelling decisions remain intentionally cautious.

The `fr_*` fields remain excluded from the approved model feature set because their exact construction is unresolved. `medianamnt_loans30` and `medianamnt_loans90` are likewise retained only for provenance and analysis because their encoding cannot yet be reconciled confidently with the documented meaning.

`payback30` and `payback90` remain excluded pending point-in-time leakage review, and the `maxamnt_loans30/90` fields are treated as consistency checks rather than independent predictors. Loan count and amount fields also remain outside the first approved predictor set until it can be established whether their historical-window construction excludes the current transaction.

Further steps therefore focus on the approved model-ready feature set, statistical feature-selection experiments, privacy treatment, and end-to-end pipeline orchestration rather than additional arbitrary cleaning.

## Relationship to the other governance artefacts

This narrative explains the observed sequence of work on this specific dataset.

`docs/governance/data_cleaning_policy.md` defines the standing rules and treatment principles that the pipeline is expected to follow. It answers questions such as what counts as a hard-invalid value, when values may be converted to missing, and how audit logging should work.

This narrative answers a different question: what did those rules reveal when applied to the telecom delinquency data, what changed during cleaning, how the result was validated, and how the cleaned population was transitioned into a defensible modelling population.

`docs/data_dictionary.md` provides the current operational meaning and modelling status of each field, while `docs/data_dictionary_changelog.md` explains how those interpretations changed over time. Git history remains the authoritative technical record of the exact code and document changes.
