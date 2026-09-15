# Approved feature set for the first model-ready dataset

This document defines the predictors, control fields, target, and exclusions used to create the first model-ready dataset for Module 3. It is deliberately conservative: a field is included only where its meaning is sufficiently supported. The current feature set should not be interpreted as the result of a single conventional statistical feature-selection algorithm. It represents **Stage 1: governance and point-in-time eligibility screening**. Formal statistical feature selection will follow as a separate analytical stage using established feature-selection methods.

## Feature-selection methodology

The project uses a staged approach so that predictive usefulness is assessed only after a feature has passed basic semantic, temporal, data-quality, and governance checks.

### Stage 1: governance and point-in-time eligibility screening

A field is eligible for the first model-ready dataset only where all of the following can be defended:

- its source-supported meaning is sufficiently clear;
- it can plausibly be available at the time of the current credit decision;
- any known data-quality problems have a documented and reproducible treatment;
- it does not function primarily as an identifier, constant field, redundant derived measure, or unresolved encoding;
- its use does not create an obvious target- or point-in-time leakage risk.

This stage is a precondition for statistical feature selection rather than a substitute for it. A statistically predictive variable will not be retained if its meaning, timing, or governance position cannot be defended.

### Stage 2: statistical relevance and redundancy screening

The eligible predictors will later be assessed with established **filter methods**, including appropriate measures such as correlation structure, low-variance checks, mutual information, and univariate predictive screening where suitable for the variable type and target.

The purpose of this stage is to identify weak, redundant, or highly overlapping predictors before more computationally intensive selection methods are used.

### Stage 3: embedded and wrapper methods

The project will then compare established model-based selection approaches. These may include:

- L1-regularised logistic regression as an embedded feature-selection method;
- recursive feature elimination (RFE), likely using logistic regression or another suitable baseline estimator.

These methods will be applied within the training data only so that validation and test information do not influence feature selection.

### Stage 4: nonlinear and model-agnostic confirmation

For nonlinear models, feature contribution will be compared using tree-based importance and model-agnostic methods (such as permutation importance and SHAP).

No single importance method will be treated as authoritative. The objective is to compare whether important variables remain stable across model families and evaluation methods.

### Selection principle

The final feature set will therefore be based on converging evidence rather than on one algorithm. The intended sequence is:

**semantic and governance eligibility → point-in-time eligibility → data-quality eligibility → filter methods → embedded/wrapper methods → nonlinear/model-agnostic confirmation → stability across validation splits**

A feature may be statistically strong and still be rejected if its timing or meaning cannot be defended. On the contrary, a semantically valid feature may remain available for modelling even if it is later removed because it adds little predictive value.

## Modelling population

The ordinary labelled modelling population is restricted to records dated through **23 July 2016**. All later records remain outside this population because the 58,825 later observations in the source data are labelled successful.

The reason is methodological rather than simply numerical. After 23 July 2016, the target distribution changes discontinuously to **100% successful repayment**. The available documentation does not explain whether this reflects a change in sampling, labelling, business process, extract construction, or another data-generation mechanism. Those later records are therefore not assumed to be comparable with the earlier mixed-outcome population.

Including them in ordinary supervised training would artificially increase the share of successful cases and could distort both class balance and estimated predictor relationships. They are retained outside the modelling population for lineage and may later be used as a separate diagnostic population, for example to assess distributional shift or covariate drift. They are not treated as a conventional labelled validation set.

The model-ready dataset is built from the validated interim-cleaned dataset, not directly from the raw source.

## Target

The source field `label` is defined as `1 = repaid within five days` and `0 = failure to repay within five days`.

For modelling clarity, the processed dataset converts this into:

- `delinquent_5d = 1` when `label = 0`;
- `delinquent_5d = 0` when `label = 1`.

The original `label` field is not carried into the processed modelling table because it is an exact inverse of the target and would create target leakage if accidentally used as a predictor.

## Approved predictors

The first model-ready dataset contains the following predictors after Stage 1 eligibility screening:

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

This is the approved feature set for the **first Module 3 model-ready dataset** after Stage 1 governance and point-in-time eligibility screening. It is not yet the final statistically selected feature set. Later stages will compare established filter, embedded, wrapper, and model-agnostic methods, with material revisions reflected in the data dictionary/change log and Git history.
