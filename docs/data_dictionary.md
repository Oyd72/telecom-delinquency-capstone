# Preliminary data dictionary

This dictionary is based on the public **Delinquency Telecom Dataset** used for the capstone project. The source dataset contains 209,593 transaction-level observations and 37 columns. The modelling subset is restricted to records dated through 23 July 2016.

The original dataset documentation is limited. Several descriptions below are therefore **preliminary interpretations from the column names and prior exploratory work**, not confirmed business definitions. Variables whose meaning, construction, or point-in-time availability cannot be verified should not be used as model predictors.

## Field assessment

| Field | Preliminary interpretation | Type | Proposed use | Notes / limitations |
|---|---|---|---|---|
| `Unnamed: 0` | Row/index field | Integer | Exclude | Administrative index; no predictive meaning. |
| `msisdn` | Mobile subscriber identifier / phone-related customer ID | Identifier | Exclude as predictor | Customer identifier. Retain only for grouping, deduplication and train/test split checks. Exclude from the model feature set and from scoring inputs. |
| `aon` | Age on network / customer tenure | Numeric | Candidate | Likely useful tenure measure. Check units and implausible values before use. |
| `daily_decr30` | 30-day decrease/activity-derived measure | Numeric | Candidate, verify | Exact construction is not clearly documented. Use only if meaning and scoring-time availability are confirmed. |
| `daily_decr90` | 90-day decrease/activity-derived measure | Numeric | Candidate, verify | Exact construction is not clearly documented. |
| `rental30` | 30-day rental/account-activity measure | Numeric | Hold / verify | Meaning is insufficiently documented. Exclude unless clarified. |
| `rental90` | 90-day rental/account-activity measure | Numeric | Hold / verify | Meaning is insufficiently documented. Exclude unless clarified. |
| `last_rech_date_ma` | Time/date-related measure for latest main-account recharge | Numeric | Candidate, verify | Confirm encoding and that it represents information known at scoring time. |
| `last_rech_date_da` | Time/date-related measure for latest data/account recharge | Numeric | Candidate, verify | Confirm encoding and business meaning before use. |
| `last_rech_amt_ma` | Amount of latest main-account recharge | Numeric | Candidate | Plausible point-in-time recharge feature; validate scale and outliers. |
| `cnt_ma_rech30` | Count of main-account recharges in 30-day window | Numeric | Candidate | Recharge-frequency indicator. |
| `fr_ma_rech30` | Frequency-related main-account recharge measure over 30 days | Numeric | Hold / verify | Encoding/meaning needs confirmation before modelling. |
| `sumamnt_ma_rech30` | Total main-account recharge amount over 30 days | Numeric | Candidate | Aggregate recharge behaviour. |
| `medianamnt_ma_rech30` | Median main-account recharge amount over 30 days | Numeric | Candidate, verify | Check calculation and distribution. |
| `medianmarechprebal30` | Median pre-recharge balance over 30 days | Numeric | Candidate, verify | Exact business definition should be confirmed. |
| `cnt_ma_rech90` | Count of main-account recharges in 90-day window | Numeric | Candidate | Recharge-frequency indicator. |
| `fr_ma_rech90` | Frequency-related main-account recharge measure over 90 days | Numeric | Hold / verify | Encoding/meaning needs confirmation before modelling. |
| `sumamnt_ma_rech90` | Total main-account recharge amount over 90 days | Numeric | Candidate | Aggregate recharge behaviour. |
| `medianamnt_ma_rech90` | Median main-account recharge amount over 90 days | Numeric | Candidate, verify | Check calculation and distribution. |
| `medianmarechprebal90` | Median pre-recharge balance over 90 days | Numeric | Candidate, verify | Exact business definition should be confirmed. |
| `cnt_da_rech30` | Count of data/account recharges in 30-day window | Numeric | Candidate, verify | Confirm meaning of `da` and scoring-time availability. |
| `fr_da_rech30` | Frequency-related data/account recharge measure over 30 days | Numeric | Hold / verify | Meaning/encoding needs confirmation. |
| `cnt_da_rech90` | Count of data/account recharges in 90-day window | Numeric | Candidate, verify | Confirm meaning of `da`. |
| `fr_da_rech90` | Frequency-related data/account recharge measure over 90 days | Numeric | Hold / verify | Meaning/encoding needs confirmation. |
| `cnt_loans30` | Count of loans/credit events in 30-day window | Numeric | Candidate | Borrowing-history indicator. Confirm that only prior loans are included. |
| `amnt_loans30` | Aggregate loan/credit amount over 30 days | Numeric | Candidate | Confirm units and point-in-time construction. |
| `maxamnt_loans30` | Maximum loan/credit amount over 30 days | Numeric | Candidate | Confirm units and whether current transaction is included. |
| `medianamnt_loans30` | Median loan/credit amount over 30 days | Numeric | Hold / verify | Construction and point-in-time validity should be verified. |
| `cnt_loans90` | Count of loans/credit events in 90-day window | Numeric | Candidate | Borrowing-history indicator. |
| `amnt_loans90` | Aggregate loan/credit amount over 90 days | Numeric | Candidate | Confirm units and point-in-time construction. |
| `maxamnt_loans90` | Maximum loan/credit amount over 90 days | Numeric | Candidate | Confirm whether current transaction is included. |
| `medianamnt_loans90` | Median loan/credit amount over 90 days | Numeric | Hold / verify | Construction and point-in-time validity should be verified. |
| `payback30` | Historical repayment/payback measure over 30 days | Numeric | Hold / verify | Potentially useful but high leakage risk if construction includes the current loan or future outcomes. Must be verified before use. |
| `payback90` | Historical repayment/payback measure over 90 days | Numeric | Hold / verify | Potential leakage risk; confirm calculation window and availability at scoring time. |
| `pcircle` | Telecom circle / operating region | Categorical | Candidate for analysis; cautious modelling use | May capture geographic or operational segmentation. Review for proxy-discrimination concerns. |
| `pdate` | Transaction / loan issue date | Date | Use for splitting, not primary predictor | Essential for temporal filtering and train/validation/test splits. Avoid using raw date as a shortcut predictor without justification. |
| `label` | Repayment within five days: 1 = repaid, 0 = not repaid within five days | Binary target | Target | Does not distinguish late repayment from permanent default. |

## Initial modelling position

### Likely candidate predictors

Subject to data-quality and point-in-time checks, the strongest initial candidates are:

- network tenure: `aon`;
- recharge activity: `last_rech_amt_ma`, `cnt_ma_rech30`, `sumamnt_ma_rech30`, `cnt_ma_rech90`, `sumamnt_ma_rech90`;
- borrowing history: `cnt_loans30`, `amnt_loans30`, `maxamnt_loans30`, `cnt_loans90`, `amnt_loans90`, `maxamnt_loans90`;
- selected additional activity variables where their definitions can be verified.

### Excluded or restricted fields

- `Unnamed: 0` — administrative index;
- `msisdn` — identifier, not a predictive feature;
- `pdate` — used for temporal filtering and evaluation rather than as a default model feature;
- fields with unresolved meaning or encoding, including rental, recharge-frequency, median-loan and payback-related variables, unless verified;
- any variable found to contain future information or information unavailable at the time of scoring.

## Data-quality and governance checks before modelling

1. Confirm the exact meaning, units and construction of candidate fields.
2. Check whether each predictor would have been available immediately after loan issuance, before the five-day outcome was known.
3. Test for implausible values and extreme outliers, especially tenure and amount variables.
4. Confirm that historical loan/payback features do not include the current transaction or future information.
5. Keep records after 23 July 2016 outside the labelled modelling population because all later observations are labelled successful.
6. Use `msisdn` only to identify repeat customers and to test performance on previously unseen customers.
7. Review geographic or operational segmentation variables such as `pcircle` for proxy-discrimination risk before modelling.
8. Record any final exclusions and transformations in the modelling documentation.

## Source and status

Source: Sivakrishna3311, *Delinquency Telecom Dataset*, Kaggle. Public metadata also describes the dataset as a telecom–microfinance microcredit sample with a five-day repayment label.

Status: **Preliminary data dictionary**. It should be updated as field definitions and exploratory checks are completed during Sprint 1 and Sprint 2.
