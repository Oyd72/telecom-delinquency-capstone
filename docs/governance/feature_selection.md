# Approved feature set for the first model-ready dataset

This document defines the predictors, control fields, target, and exclusions used to create the first model-ready dataset for Module 3. It is deliberately conservative: a field is included only where its meaning is sufficiently supported and there is no unresolved identifier, redundancy, or obvious point-in-time concern.

## Modelling population

The ordinary labelled modelling population is restricted to records dated through **23 July 2016**. All later records remain outside this population because the 58,825 later observations in the source data are labelled successful, which would make them unsuitable for ordinary supervised model development without further explanation.

The model-ready dataset is built from the validated interim-cleaned dataset, not directly from the raw source.

## Target

The source field `label` is defined as `1 = repaid within five days` and `0 = failure to repay within five days`.

For modelling clarity, the processed dataset converts this into:

- `delinquent_5d = 1` when `label = 0`;
- `delinquent_5d = 0` when `label = 1`.

The original `label` field is not carried into the processed modelling table because it is an exact inverse of the target and would create target leakage if accidentally used as a predictor.

## Approved predictors

The first model-ready dataset contains the following predictors:

### Network tenure

- `aon`

### Account activity and balances

- `daily_decr30`
- `daily_decr90`
- `rental30`
- `rental90`

### Main-account recharge behaviour

- `last_rech_date_ma`
- `last_rech_amt_ma`
- `cnt_ma_rech30`
- `sumamnt_ma_rech30`
- `medianamnt_ma_rech30`
- `medianmarechprebal30`
- `cnt_ma_rech90`
- `sumamnt_ma_rech90`
- `medianamnt_ma_rech90`
- `medianmarechprebal90`

### Data-account recharge behaviour

- `last_rech_date_da`
- `cnt_da_rech30`
- `cnt_da_rech90`

### Derived prior-participation features

- `prior_tx_count` — number of transactions for the same `msisdn` on **strictly earlier dates**;
- `is_repeat_customer` — `1` when at least one strictly earlier transaction exists, otherwise `0`.

The derived customer-history fields are calculated while `msisdn` is still available in controlled processing. Transactions on the same date do not count as prior transactions for one another. `msisdn` is then removed from the processed output.

## Control field retained outside the predictor set

- `pdate` is retained in the processed dataset for temporal splitting, reproducibility, and lineage. It is **not an approved predictor**.

## Fields excluded from the first model-ready predictor set

- `msisdn` — identifier-like; required only for controlled grouping and chronology, then removed;
- `pcircle` — constant in the current dataset;
- all `fr_*` fields — exact construction unresolved;
- `maxamnt_loans30`, `maxamnt_loans90` — redundant/derived consistency checks;
- `medianamnt_loans30`, `medianamnt_loans90` — encoding unresolved;
- `payback30`, `payback90` — excluded pending confirmation that only information available before the current outcome is used;
- `cnt_loans30`, `amnt_loans30`, `cnt_loans90`, `amnt_loans90` — **not included in the first model-ready dataset until point-in-time construction is verified**. The source describes them as historical-window variables, but does not establish whether the current credit transaction is included. They remain analytically useful and may be reconsidered later if leakage can be ruled out;
- source `label` — replaced by `delinquent_5d` and removed from the predictor table;
- `pdate` — retained only as a temporal control, not as a predictor.

## Missing values

Values converted to missing during cleaning remain missing in the model-ready dataset. They are **not imputed before train/test splitting**. Any imputation used for modelling will be fitted inside the training pipeline/folds so that information from validation or test data cannot influence training-time preprocessing.

## Interpretation of repeat-customer features

`prior_tx_count` and `is_repeat_customer` describe observed prior participation, not inherent creditworthiness. Returning borrowers may be a selected population because the full credit-approval procedure is unknown and repeat approvals may follow different eligibility or underwriting rules. Any association with repayment therefore remains observational rather than causal.

## Status

This is the approved feature set for the **first Module 3 model-ready dataset**. It can be revised only where additional source evidence or reproducible analysis supports a change. Material revisions should be reflected in the data dictionary/change log and Git history.
