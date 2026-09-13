# Data cleaning and treatment policy

This policy defines how the Module 3 pipeline will treat ambiguous, invalid, contaminated, and unusual values in the Delinquency Telecom Dataset. It is designed to keep the raw data intact and make every transformation reproducible.

## Evidence hierarchy

1. The supplied Kaggle field description is the primary semantic source.
2. The actual CSV structure and observed distributions are used to test whether the documented semantics are consistent with the data.
3. Cross-field and within-customer consistency checks provide additional evidence.
4. Statistical anomaly methods are diagnostic and do not, by themselves, prove that a value is invalid.
5. Where documentation and observed encoding conflict, the ambiguity is recorded rather than silently resolved.

## General treatment rules

- `data/raw/` is immutable. Source values are never overwritten.
- Cleaning takes place in `data/interim/` and preserves an audit trail of every flagged or changed value.
- A questionable cell is normally flagged and converted to missing in the cleaned analytical copy instead of causing the entire row to be deleted.
- Rows are removed only where there is a defensible record-level reason (such as an exact duplicate or an unusable target/date record). Any record or value removed or altered during pipeline execution must be recorded automatically in the pipeline audit log, including the applicable rule and treatment. Git history separately records changes to the code, configuration, and governance decisions that define those rules.
- Model-ready data in `data/processed/` contain only approved predictors and derived fields.
- Arbitrary winsorisation or clipping is not used. Monetary heavy tails are retained unless semantic, cross-field, or longitudinal evidence supports a stronger conclusion.
- Dataset-specific contamination boundaries are not presented as universal business limits.

## Structural fields

### `label`

- Meaning: repayment within five days, `1 = success`, `0 = failure`.
- Rule: non-null and restricted to `{0,1}`.
- Treatment: a record with an invalid or missing target is quarantined from supervised modelling and logged.

### `pdate`

- Meaning: transaction / loan date.
- Rule: non-null and parseable as a date.
- Treatment: invalid dates are quarantined because chronology and point-in-time controls cannot be applied safely.

### `msisdn`

- Meaning: mobile/user identifier.
- Rule: non-null for customer-history and split logic.
- Treatment: retained only in controlled intermediate processing for grouping, chronology, repeat-customer analysis, and leakage-safe splits. It is removed or pseudonymised before model-ready output and is never used as a predictor.

### `pcircle`

- Meaning: telecom circle.
- Observed structure: all records contain `UPW`.
- Treatment: retained in raw/intermediate data for provenance, excluded from the model-ready feature set. `UPW` is not treated as a universal validity rule.

## High-confidence semantic violations or contamination

The current dataset contains several fields where semantic rules and/or an abrupt separate numeric regime provide stronger evidence than statistical outlier status alone.

### `aon`

- Meaning: age on cellular network in days.
- Hard-invalid values: negative values.
- Contamination evidence: ordinary values are integer-like and extend to approximately 2,440 days, followed by a large empty gap and a separate decimal-valued regime above approximately 500,000 days. Repeat-customer histories also show impossible jumps into and out of this regime.
- Treatment: preserve the raw value and add an audit flag. Negative values and values belonging to the separated high-confidence contamination regime are converted to missing in the cleaned analytical copy. The empirical gap is documented as dataset-specific evidence, not as a universal maximum tenure rule.

### `last_rech_date_ma`, `last_rech_date_da`

- Meaning: days associated with the most recent recharge.
- Hard-invalid values: negative elapsed-day values.
- Contamination evidence: normal integer-like values extend to approximately 113–115 days, followed by a separate decimal-valued regime above approximately 500,000.
- Treatment: preserve raw values and audit flags; semantic-invalid and high-confidence contamination values become missing in the cleaned analytical copy.

### Genuine count fields

The following are treated as counts: `cnt_ma_rech30`, `cnt_ma_rech90`, `cnt_da_rech30`, `cnt_da_rech90`, `cnt_loans30`, `cnt_loans90`.

- Hard rules: non-null, non-negative, integer-valued.
- Large positive integers are not rejected merely because they are rare.
- Fractional values violate count semantics and are treated as contaminated values.
- In the current dataset, the fractional/extreme regimes in `cnt_da_rech30` and `cnt_loans90` are therefore cleaned to missing while retaining raw values and flags.
- Related 30-day and 90-day counts are checked for consistency; inconsistencies are flagged rather than silently corrected.

### `fr_ma_rech30`, `fr_da_rech30`

- Source meaning: recharge-frequency-related measures; exact construction is not documented.
- Contamination evidence: both show an abrupt separate regime above approximately 500,000.
- Treatment: preserve and flag the separated contamination regime in intermediate data, but exclude the fields from modelling unless the encoding is clarified. The remaining values are not equated with recharge counts.

### `fr_ma_rech90`, `fr_da_rech90`

- Exact construction remains unresolved.
- Treatment: preserve in intermediate data for analysis, but exclude from the approved model feature set unless the encoding is clarified. No cleaning rule is inferred solely from the word “frequency.”

## Loan-amount fields

### `amnt_loans30`, `amnt_loans90`

- Meaning: total loan amount over the relevant window.
- Treatment: retain non-negative values, monitor distribution and consistency with loan counts. No arbitrary upper clipping is applied.

### `maxamnt_loans30`, `maxamnt_loans90`

The source description says principal options are 5 and 10, with repayment amounts of 6 and 12. The observed core encoding of the maximum fields is `0/6/12`.

- `maxamnt_loans30` can be reconstructed from `cnt_loans30` and `amnt_loans30` for approximately 99.5% of records; the mismatches are the same non-core contamination regime in the supplied maximum field.
- `maxamnt_loans90` is likewise effectively reconstructible from `cnt_loans90` and `amnt_loans90` in the clean portion of the data.
- Treatment: both maximum fields are treated as derived/redundant consistency checks rather than independent model predictors. The pipeline derives an expected maximum from the count/total relationship after the 6/12 encoding is confirmed and compares it with the supplied value. Mismatches are logged rather than silently corrected.

### `medianamnt_loans30`, `medianamnt_loans90`

- Source label: median loan amount.
- Observed values: `0`, `0.5`, `1`, `1.5`, `2`, `3`, which are not consistent with a straightforward median of 6/12-valued loan amounts.
- Treatment: retain in raw/intermediate data but exclude from modelling unless the encoding can be verified. Do not reconstruct or reinterpret the values from the field name alone.

## Monetary and balance fields

### Recharge amounts and totals

Fields such as `last_rech_amt_ma`, `sumamnt_ma_rech30`, `sumamnt_ma_rech90`, `medianamnt_ma_rech30`, and `medianamnt_ma_rech90` may be heavily skewed.

- Non-negative amount semantics are used where documented.
- Large positive values are monitored using robust quantiles, distribution plots, and cross-field relationships.
- Large values are not removed solely because they are statistically extreme.

### Balance-like fields

`rental30`, `rental90`, `medianmarechprebal30`, and `medianmarechprebal90` are balance-like measures.

- Negative values are not automatically invalid because the source documentation does not establish that negative balances are impossible.
- Preserve values and monitor distributional discontinuities and 30/90-day consistency.

### `daily_decr30`, `daily_decr90`

- Source meaning: average daily amount spent from the main account.
- Negative values are suspicious but not automatically invalid because refunds, adjustments, or undocumented encoding cannot be ruled out.
- Treatment: preserve and flag for sensitivity analysis; do not silently delete or clip.

## Repayment-history fields

### `payback30`, `payback90`

- Meaning: average payback time in days over the relevant historical window.
- Data-quality rule: non-negative duration.
- Modelling restriction: excluded from the approved feature set until it is verified that the calculation uses only completed prior loans and information available at the time of the current credit decision.
- Reason: otherwise the fields may introduce point-in-time or target leakage.

## Derived customer-history fields

The pipeline may derive:

- `prior_tx_count`: number of transactions for the same `msisdn` on strictly earlier dates;
- `is_repeat_customer`: whether at least one strictly earlier transaction exists.

These are derived before `msisdn` is removed/pseudonymised. They are calculated using strict chronology so that later transactions cannot influence earlier records.

The higher observed repayment rate among returning customers is treated as an association, not a causal effect. The full credit approval process is unknown, and returning borrowers may have been selected under different approval or eligibility rules from first-time borrowers.

## Duplicate records

The source data contain one exact duplicate row. Exact duplicates are logged and one copy is removed in the interim cleaned dataset. Repeated customers or repeated dates are not treated as duplicates unless the entire record is identical.

## Audit fields

Cleaning should generate explicit indicators rather than silently modifying data. At minimum, the interim dataset or companion audit table should record:

- record identifier / row lineage;
- field name;
- original value;
- rule triggered;
- treatment applied;
- pipeline run timestamp or run identifier.

Field-level flags may also be retained where useful, for example `aon_invalid_flag`, `cnt_loans90_invalid_flag`, or a general `data_quality_flag_count`.

## Validation relationship

Great Expectations will implement:

- blocking structural and semantic rules where defensible;
- diagnostic expectations for distributions, categories, and cross-field consistency;
- observed contamination rates as monitoring evidence, not as universal business thresholds.

The cleaning policy is the bridge between the data dictionary and executable validation. Any material change to this policy or to the dictionary must be recorded in `docs/data_dictionary_changelog.md` and Git history.
