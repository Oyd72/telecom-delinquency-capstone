# Data cleaning and treatment policy

This policy sets out how the Module 3 pipeline handles ambiguous, invalid, contaminated, and unusual values in the Delinquency Telecom Dataset. The raw source stays unchanged; every treatment is applied to derived data and is reproducible from code and logs.

## Evidence used

The supplied Kaggle field description is the main reference for field meaning. I check it against the CSV structure, observed distributions, cross-field relationships, and within-customer history. Statistical anomaly methods are useful for finding suspicious values, but they are not enough on their own to prove that a value is wrong. Where the documentation and the data disagree, the uncertainty is recorded rather than resolved by assumption.

## General rules

- `data/raw/` is immutable.
- Cleaning happens in `data/interim/`.
- Questionable cells are usually flagged and set to missing in the cleaned copy rather than causing the whole row to be removed.
- Rows are removed only for a defensible record-level reason, such as an exact duplicate or an unusable target/date record.
- Runtime changes are written to the cleaning audit log. Git history separately records changes to rules and code.
- `data/processed/` contains only approved predictors, controls, and derived fields.
- Heavy tails are not clipped or winsorised simply because they are extreme.
- Boundaries observed in this dataset are not presented as universal business limits.

## Structural fields

### `label`

`label` records whether repayment occurred within five days (`1 = success`, `0 = failure`). It must be present and restricted to `{0,1}`. Invalid or missing targets are kept out of supervised modelling and logged.

### `pdate`

`pdate` must be present and parse as a valid date. Records with unusable dates cannot safely enter chronology-based processing and are therefore quarantined.

### `msisdn`

`msisdn` is treated as an identifier-like customer field. It is kept only while grouping and chronology require it. It is never a predictor and is removed before the model-ready dataset is written.

### `pcircle`

The source field represents telecom circle. In this dataset every record contains `UPW`. The field is retained in raw/intermediate data for provenance but excluded from modelling. `UPW` is not treated as a universal validity rule.

## High-confidence invalid or contaminated values

### `aon`

`aon` is age on cellular network in days. Negative values are invalid. The dataset also contains a separate upper regime, above a large empty gap, with decimal values around 500,000 days and above. Repeat-customer records show implausible jumps into and out of this regime. These values are treated as high-confidence contamination in this dataset and are set to missing in the cleaned copy, with the original value retained in the audit log.

### `last_rech_date_ma`, `last_rech_date_da`

These fields are elapsed-day measures associated with the latest recharge. Negative values are invalid. Their upper tails also contain a separate implausible regime above roughly 500,000. Negative and high-confidence contaminated values are therefore set to missing in the cleaned copy and logged.

### Count fields

The following are treated as genuine counts: `cnt_ma_rech30`, `cnt_ma_rech90`, `cnt_da_rech30`, `cnt_da_rech90`, `cnt_loans30`, and `cnt_loans90`.

Counts must be non-null, non-negative, and integer-valued. Large positive integers are not rejected simply because they are unusual. Fractional values violate count semantics and are treated as contamination. Related 30-day and 90-day counts are also checked for consistency, but inconsistencies are flagged rather than silently rewritten.

### `fr_ma_rech30`, `fr_da_rech30`

The source describes these as frequency-related measures but does not define the calculation. Both fields contain a clearly separated upper regime above about 500,000. That regime is flagged and set to missing in the interim data. The remaining values are retained for analysis but the fields stay out of the approved predictor set until their meaning is clearer.

### `fr_ma_rech90`, `fr_da_rech90`

Their construction is also unresolved. They are retained in intermediate data but excluded from the approved model feature set. No rule is inferred simply from the word “frequency.”

## Loan-amount fields

### `amnt_loans30`, `amnt_loans90`

These totals are retained when non-negative. Their distributions and consistency with loan counts are monitored; no arbitrary upper clipping is applied.

### `maxamnt_loans30`, `maxamnt_loans90`

The source notes principal options of 5 and 10 with repayment amounts of 6 and 12, while the observed maximum fields mainly use `0/6/12`. The supplied maximum can be reconstructed from count and total amount for almost all clean records. These fields are therefore treated as derived consistency checks rather than independent predictors. Mismatches are logged rather than silently corrected.

### `medianamnt_loans30`, `medianamnt_loans90`

The observed values do not behave like straightforward medians of 6/12-valued loans. They remain in raw/intermediate data for provenance but are excluded from modelling until the encoding can be verified.

## Monetary and balance fields

Recharge amounts and totals can be highly skewed. Where the field meaning supports non-negative values, that rule is applied, but large positive values are not deleted merely because they are rare.

`rental30`, `rental90`, `medianmarechprebal30`, and `medianmarechprebal90` are balance-like fields. Negative values are retained because the documentation does not show that a negative balance is impossible.

`daily_decr30` and `daily_decr90` represent average daily spend. Negative values are suspicious, but refunds, adjustments, or undocumented encoding cannot be ruled out. They are therefore retained and considered in sensitivity analysis rather than removed automatically.

## Repayment-history fields

`payback30` and `payback90` are non-negative duration measures, but they are excluded from the approved feature set until it is clear that they use only completed prior loans and information available at the current scoring point. Otherwise they could introduce point-in-time or target leakage.

## Derived customer-history fields

The pipeline derives:

- `prior_tx_count` — transactions for the same `msisdn` on strictly earlier dates;
- `is_repeat_customer` — whether at least one strictly earlier transaction exists.

They are calculated before `msisdn` is removed. Same-day transactions do not count as prior transactions for one another. Because pre-June history is unavailable, these variables are treated cautiously and are not interpreted as causal measures of creditworthiness.

## Duplicate records

The source contains one exact duplicate row. One copy is removed and the event is logged. Repeated customers or dates are not treated as duplicates unless the full record is identical.

## Cleaning audit

The cleaning audit should make changes visible rather than hiding them. At minimum it records row lineage, field name, original value, rule triggered, treatment applied, and the pipeline run identifier or timestamp. Field-level quality flags may also be retained where useful.

## Relationship with validation

Great Expectations implements blocking checks where the semantics are clear and diagnostic checks where uncertainty remains. Observed contamination rates are monitoring evidence, not business limits.

This policy sits between the data dictionary and the executable validation code. Material changes should be reflected in `docs/data_dictionary_changelog.md` and Git history.