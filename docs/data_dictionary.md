# Data dictionary

This dictionary is based on the public **Delinquency Telecom Dataset** used for the capstone project and on the source field descriptions supplied with the dataset. The current CSV contains **209,593 transaction-level observations and 36 columns**. The labelled modelling population is restricted to records dated through **23 July 2016**; all 58,825 later observations are labelled successful and are kept outside the ordinary labelled modelling population.

## Evidence hierarchy

Field treatment follows this order:

1. the supplied Kaggle data description is the primary semantic source;
2. the actual CSV structure and observed distributions are used to test whether those semantics are consistent with the data;
3. cross-field and within-customer consistency checks provide additional evidence;
4. statistical anomaly methods are diagnostic and do not, by themselves, prove that a value is invalid;
5. where documentation and observed encoding conflict, the ambiguity is documented rather than silently resolved;
6. external material may support generic telecom terminology, but is not used to invent undocumented business rules.

Structural validity is not the same as dataset-specific normality. Blocking validation rules are used only where the field semantics support them. Distributional anomalies are otherwise retained as diagnostic flags until stronger evidence exists.

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

## Derived fields proposed for the pipeline

| Derived field | Meaning | Use | Limitation |
|---|---|---|---|
| `prior_tx_count` | Number of transactions for the same `msisdn` on strictly earlier dates | Candidate behavioural feature and analysis variable | Must be calculated chronologically to avoid future leakage. |
| `is_repeat_customer` | Whether the customer has at least one strictly earlier transaction | Candidate behavioural feature and evaluation slice | Returning borrowers may be a selected population because the underlying credit approval procedure is unknown. |

The observed association between repeat participation and five-day repayment must therefore not be interpreted causally. First-time and returning borrowers may have been approved under different eligibility or underwriting rules, meaning they may not come from the same underlying applicant population.

## Modelling and validation position

### Candidate predictors

Subject to cleaning, point-in-time checks and leakage review, candidate families include:

- network tenure: `aon` after treatment of invalid/contaminated values;
- recharge behaviour: count fields, totals, last recharge amount and selected median measures whose meaning is supported;
- borrowing history: loan counts and totals;
- derived prior-participation features calculated only from strictly earlier transactions.

### Excluded or restricted fields

- `msisdn` — identifier-like field, never a predictive feature;
- `pcircle` — constant in the current dataset;
- `pdate` — temporal control rather than a default predictor;
- all `fr_*` fields — exact construction unresolved, therefore excluded from the approved model feature set unless clarified;
- `maxamnt_loans30` and `maxamnt_loans90` — treated as derived/redundant consistency checks rather than independent predictors;
- `medianamnt_loans30` and `medianamnt_loans90` — unresolved encoding;
- `payback30` and `payback90` — excluded pending point-in-time/leakage verification;
- any field whose meaning or encoding remains unresolved;
- any value identified as a hard semantic violation or high-confidence contamination, with the original raw value preserved for auditability.

## Data-quality and governance principles

1. Structural validity is not the same as dataset-specific normality.
2. Hard blocking rules are used only where source semantics support them.
3. Statistical outliers are not automatically deleted; unusual values remain diagnostic unless stronger semantic, cross-field, or longitudinal evidence supports invalidation.
4. Genuine count fields must be non-negative and integer-valued; extreme positive counts are monitored rather than rejected solely for being large.
5. 30-day and 90-day totals/counts are checked for internal consistency where the nested-window logic supports it.
6. `data/raw/` is immutable. Cleaning occurs in `data/interim/`; raw values are preserved when cleaning flags or replaces questionable observations.
7. Any record or value removed or altered during pipeline execution must be recorded automatically in the pipeline audit log, including the applicable rule and treatment. Git history separately records changes to code, configuration, and governance decisions.
8. `msisdn` is retained only where necessary for controlled grouping, chronology and split checks, then removed or pseudonymised before model-ready output.
9. Records after 23 July 2016 remain outside the ordinary labelled modelling population because all later records are labelled successful.
10. Returning-customer effects are evaluated separately because the underlying approval process is unknown and selection effects are plausible.
11. Final Great Expectations rules will be derived from this dictionary and the cleaning policy rather than from arbitrary thresholds.

## Source and status

Source: Sivakrishna3311, *Delinquency Telecom Dataset*, Kaggle, plus the accompanying field-description image supplied with the dataset.

Status: **Domain-informed working data dictionary for Module 3**. It is aligned with `docs/governance/data_cleaning_policy.md` and supersedes the preliminary Module 2 version. Future material changes must be supported by additional source evidence or reproducible analysis and recorded in `docs/data_dictionary_changelog.md` and Git history.
