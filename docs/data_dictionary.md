# Data dictionary

This dictionary covers the public **Delinquency Telecom Dataset** used in the capstone. The field meanings come from the descriptions supplied with the dataset and are checked against the CSV itself. The current file has **209,593 transaction-level records and 36 columns**. For labelled modelling, the project uses records through **23 July 2016**. The 58,825 later records are all labelled as successful repayment and are kept outside the ordinary modelling population.

## How field meanings are assessed

The supplied Kaggle description is the starting point for field meaning. I then check whether the observed values and structure are consistent with that description. Cross-field and within-customer checks provide further evidence where they are useful. Statistical anomaly methods are treated as diagnostics, not as proof that a value is wrong. Where the documentation and the observed encoding do not line up, the uncertainty is left visible rather than resolved by guesswork.

Blocking rules are therefore limited to cases where the field meaning supports them. An unusual value is not automatically an invalid value.

## Field assessment

| Field | Source-supported meaning | Type | Pipeline role | Validation / treatment | Notes / limitations |
|---|---|---|---|---|---|
| `msisdn` | Mobile number / user identifier | Identifier-like | Grouping and chronology only | Require non-null. Retain in controlled intermediate data for repeat-customer analysis and leakage-safe splitting; remove or pseudonymise before model-ready output. | Public data do not establish whether values can be linked back to identifiable persons. Treat proportionately as identifier-like, not as a model feature. |
| `aon` | Age on cellular network in days | Numeric duration | Candidate after cleaning | Non-negative. Negative values are invalid. Preserve raw value and audit flag; values in the separated high-confidence contamination regime are converted to missing in the cleaned analytical copy. | Very large values are incompatible with the field meaning and repeat-customer histories show implausible jumps. The empirical boundary is dataset-specific and is not presented as a universal tenure limit. |
| `daily_decr30` | Daily amount spent from main account, averaged over last 30 days | Numeric monetary/activity | Candidate with caution | Require non-null. Negative values are investigated rather than automatically rejected. Monitor robust tail behaviour and 30/90-day consistency. | A negative “amount spent” is suspicious, but undocumented adjustments/refunds cannot be ruled out. |
| `daily_decr90` | Daily amount spent from main account, averaged over last 90 days | Numeric monetary/activity | Candidate with caution | Same treatment as `daily_decr30`. | Same semantic uncertainty around negative values. |
| `rental30` | Average main-account balance over last 30 days | Numeric balance | Candidate with caution | Require non-null. Do not reject negative values solely from the field name; monitor distribution and consistency. | Negative balances may be meaningful if debit/overdrawn states are possible. |
| `rental90` | Average main-account balance over last 90 days | Numeric balance | Candidate with caution | Same treatment as `rental30`. | Same balance-sign caveat. |
| `last_rech_date_ma` | Number of days associated with the last main-account recharge | Numeric duration | Candidate after cleaning | Non-negative. Negative values are invalid. Preserve raw value and audit flag; values in the separated high-confidence contamination regime are converted to missing in the cleaned analytical copy. | Source wording says “number of days till last recharge”; exact operational construction is not fully documented. |
| `last_rech_date_da` | Number of days associated with the last data-account recharge | Numeric duration | Candidate after cleaning | Same treatment as `last_rech_date_ma`. | Extreme discontinuity and negative values require cleaning. |
| `last_rech_amt_ma` | Amount of last main-account recharge | Numeric monetary | Candidate | Require non-null and non-negative. Monitor heavy tail diagnostically. | Large monetary values are not automatically invalid. |
| `cnt_ma_rech30` | Number of main-account recharges in last 30 days | Count | Candidate | Must be non-null, non-negative and integer-valued. Extreme positive values are monitored but not rejected solely for being large. | Genuine count field. |
| `fr_ma_rech30` | Recharge-frequency-related measure over last 30 days | Numeric, exact construction unresolved | Exclude from approved model feature set unless clarified | Preserve in intermediate data. Flag the separated contamination regime. Do not assume this is equivalent to recharge count. | Source uses “frequency” but does not explain construction. It is empirically distinct from `cnt_ma_rech30`. |
| `sumamnt_ma_rech30` | Total main-account recharge amount over last 30 days | Numeric monetary | Candidate | Require non-null and non-negative; monitor robust tail behaviour. | Heavy tails may be legitimate. |
| `medianamnt_ma_rech30` | Median main-account recharge amount over last 30 days | Numeric monetary | Candidate with caution | Require non-null and non-negative; monitor distribution and consistency. | Meaning is clear, but exact source computation is not independently verified. |
| `medianmarechprebal30` | Median main-account balance just before recharge over last 30 days | Numeric balance | Candidate with caution | Require non-null. Do not reject negatives automatically; monitor extreme discontinuities. | Balance sign may be meaningful. |
| `cnt_ma_rech90` | Number of main-account recharges in last 90 days | Count | Candidate | Must be non-null, non-negative and integer-valued. Extreme positive values are monitored but not automatically rejected. | Check consistency with the 30-day count. |
| `fr_ma_rech90` | Recharge-frequency-related measure over last 90 days | Numeric, exact construction unresolved | Exclude from approved model feature set unless clarified | Preserve in intermediate data for analysis; no cleaning rule is inferred solely from the word “frequency.” | Exact distinction from recharge count is not documented. |
| `sumamnt_ma_rech90` | Total main-account recharge amount over last 90 days | Numeric monetary | Candidate | Require non-null and non-negative; monitor robust tail behaviour. | Check consistency with 30-day total. |
| `medianamnt_ma_rech90` | Median main-account recharge amount over last 90 days | Numeric monetary | Candidate with caution | Require non-null and non-negative; monitor distribution. | Exact computation not independently verified. |
| `medianmarechprebal90` | Median main-account balance just before recharge over last 90 days | Numeric balance | Candidate with caution | Same treatment as `medianmarechprebal30`. | Balance sign may be meaningful. |
| `cnt_da_rech30` | Number of data-account recharges in last 30 days | Count | Candidate after cleaning | Must be non-null, non-negative and integer-valued. Non-integer values are treated as contaminated; raw values and audit flags are preserved. Extreme positive integers are monitored rather than rejected solely for being large. | Genuine count field. |
| `fr_da_rech30` | Data-account recharge-frequency-related measure over last 30 days | Numeric, exact construction unresolved | Exclude from approved model feature set unless clarified | Preserve in intermediate data and flag the separated contamination regime. | Exact construction is undocumented. |
| `cnt_da_rech90` | Number of data-account recharges in last 90 days | Count | Candidate after cleaning | Same count rule as `cnt_da_rech30`: non-null, non-negative, integer-valued; monitor extreme positives and contamination patterns. | Apply consistent 30/90-day count logic. |
| `fr_da_rech90` | Data-account recharge-frequency-related measure over last 90 days | Numeric, exact construction unresolved | Exclude from approved model feature set unless clarified | Preserve in intermediate data for analysis; no cleaning rule is inferred solely from the word “frequency.” | Exact construction is undocumented. |
| `cnt_loans30` | Number of loans taken by the user in last 30 days | Count | Candidate | Must be non-null, non-negative and integer-valued. Extreme positives are monitored. | Confirm point-in-time construction and that only prior loans are included. |
| `amnt_loans30` | Total amount of loans taken by the user in last 30 days | Numeric monetary | Candidate | Require non-null and non-negative; monitor consistency with loan counts and derived maximum. | Confirm whether current transaction is excluded. |
| `maxamnt_loans30` | Maximum loan amount in last 30 days | Derived / discrete, source encoding ambiguous | Derived consistency check; generally redundant | Do not enforce `{5,10}`. Recompute the expected maximum from `cnt_loans30` and `amnt_loans30` after confirming the observed 6/12 encoding; compare the supplied value against the derived result. Mismatches are logged rather than silently corrected. | Source notes principal options of 5 and 10 with repayment amounts of 6 and 12, while observed core values are 0/6/12. The supplied field is reconstructible for about 99.5% of records. |
| `medianamnt_loans30` | Median loan amount in last 30 days | Numeric, encoding unresolved | Exclude from modelling unless clarified | Retain in raw/intermediate data; do not infer or reconstruct its meaning from the field name alone. | Observed encoding is not consistent with a straightforward median of 6/12-valued loans. |
| `cnt_loans90` | Number of loans taken by the user in last 90 days | Count | Candidate after cleaning | Must be non-null, non-negative and integer-valued. Non-integer values are treated as contaminated; raw values and audit flags are preserved. Extreme positive integers are monitored. | Check consistency with `cnt_loans30`. |
| `amnt_loans90` | Total amount of loans taken by the user in last 90 days | Numeric monetary | Candidate | Require non-null and non-negative; monitor cross-field consistency. | Confirm point-in-time construction. |
| `maxamnt_loans90` | Maximum loan amount in last 90 days | Derived / discrete | Derived consistency check; generally redundant | Recompute expected value from `cnt_loans90` and `amnt_loans90` once the 6/12 encoding is confirmed. Compare supplied value against the derived result. Mismatches are logged rather than silently corrected. | Observed values are 0/6/12 and the field is reconstructible for almost all clean rows; mismatches concentrate in contaminated count records. |
| `medianamnt_loans90` | Median loan amount in last 90 days | Numeric, encoding unresolved | Exclude from modelling unless clarified | Retain in raw/intermediate data; do not reconstruct or assume meaning. | Observed values such as 0, 0.5, 1, 1.5, 2 and 3 are not consistent with a straightforward median of 6/12 loan amounts. |
| `payback30` | Average payback time in days over last 30 days | Numeric duration | Exclude from approved feature set pending leakage review | Require non-null and non-negative. Use only if confirmed to contain information from completed prior loans available at scoring time. | High leakage risk if current-loan or future outcome information is included. |
| `payback90` | Average payback time in days over last 90 days | Numeric duration | Exclude from approved feature set pending leakage review | Same treatment as `payback30`. | Same point-in-time leakage concern. |
| `pcircle` | Telecom circle | Categorical | Exclude from modelling | Column must be present and categories logged. Do not hard-code `UPW` as universally valid. Drop before model-ready feature set. | Current dataset contains only `UPW`, so there is no predictive variation and no basis for regional fairness comparison. |
| `pdate` | Date | Date | Temporal filtering / lineage | Must parse as a valid date and be non-null. Use for chronology, modelling-period restriction and split logic; exclude as default predictor. | Essential for point-in-time controls and derivation of prior participation. |
| `label` | Whether the loan was repaid within 5 days: 1 = success, 0 = failure | Binary | Target | Blocking rule: non-null and values restricted to `{0,1}`. Invalid/missing targets are quarantined from supervised modelling and logged. | Does not distinguish late repayment from permanent non-payment. |

## Derived fields used in the pipeline

| Derived field | Meaning | Use | Limitation |
|---|---|---|---|
| `prior_tx_count` | Number of transactions for the same `msisdn` on strictly earlier dates | Candidate behavioural feature and analysis variable | Must be calculated chronologically to avoid future leakage. |
| `is_repeat_customer` | Whether the customer has at least one strictly earlier transaction | Candidate behavioural feature and evaluation slice | Returning borrowers may be a selected population because the underlying credit approval procedure is unknown. |

The difference between first-time and returning borrowers is observational. The two groups may have been approved under different rules, so the association with five-day repayment should not be read as causal.

## Modelling position

### Candidate families

Subject to cleaning and point-in-time checks, the candidate families are network tenure, recharge behaviour, selected borrowing-history fields, and derived prior-participation measures built only from earlier transactions.

### Fields kept out of the approved predictor set

- `msisdn` — identifier-like and never a predictor;
- `pcircle` — constant in this dataset;
- `pdate` — used for chronology, not as a default predictor;
- all `fr_*` fields — exact construction unresolved;
- `maxamnt_loans30` and `maxamnt_loans90` — treated as derived consistency checks;
- `medianamnt_loans30` and `medianamnt_loans90` — unresolved encoding;
- `payback30` and `payback90` — held back until point-in-time availability is verified;
- any other field whose meaning or encoding remains unresolved.

## Data-quality and governance rules

1. An unusual value is not automatically an invalid value.
2. Blocking rules are used only where the field meaning supports them.
3. Statistical outliers remain diagnostic unless semantic, cross-field, or longitudinal evidence justifies stronger treatment.
4. Count fields must be non-negative and integer-valued; rare large counts are monitored rather than rejected simply because they are large.
5. Related 30-day and 90-day measures are checked for consistency where the nested windows make that comparison meaningful.
6. `data/raw/` is left unchanged. Cleaning is carried out in `data/interim/`.
7. Changes made during pipeline execution are logged with the rule and treatment applied. Git history separately records changes to code and governance decisions.
8. `msisdn` is kept only as long as grouping and chronology require it, then removed before model-ready output.
9. Records after 23 July 2016 remain outside the ordinary labelled modelling population because all later outcomes are successful.
10. Returning-customer effects are reviewed separately because the underlying approval process is unknown.
11. Great Expectations checks are derived from this dictionary and the cleaning policy, not from arbitrary thresholds.

## Source and status

Source: Sivakrishna3311, *Delinquency Telecom Dataset*, Kaggle, together with the field-description image supplied with the dataset.

Status: **working Module 3 data dictionary**. It is aligned with `docs/governance/data_cleaning_policy.md` and replaces the preliminary Module 2 version. Material changes should be supported by source evidence or reproducible analysis and recorded in `docs/data_dictionary_changelog.md` and Git history.