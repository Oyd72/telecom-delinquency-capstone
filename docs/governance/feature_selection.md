# Approved feature set for the first model-ready dataset

This document defines the predictors, control fields, target, and exclusions used to create the first model-ready dataset for Module 3. It is deliberately conservative: a field is included only where its meaning is sufficiently supported. The current feature set should not be interpreted as the result of a single conventional statistical feature-selection algorithm. It represents **Stage 1: governance and point-in-time eligibility screening**. Formal statistical feature selection follows as a separate analytical stage using established feature-selection methods.

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

### Temporal diagnostics before Stage 2 interpretation

Before interpreting filter-method results, the project tested whether a simple earlier-development / later-holdout split could be treated as approximately comparable over time. The initial 80/20 chronological split placed records through 13 July 2016 in development and 14–23 July 2016 in the later subset.

The comparison showed material temporal differences. Delinquency increased from about 16.55% in the earlier period to 20.78% in the later period. Several account-behaviour variables also shifted strongly, especially `daily_decr30`, `daily_decr90`, `rental30`, and `rental90`.

Further diagnostics then compared matching days of the month across June and July. The day-of-month pattern was highly stable for several account-behaviour variables: Spearman correlations were approximately 0.94 for `daily_decr30/90` and about 0.98 for `rental30/90`. Delinquency itself showed only a more moderate June–July day-of-month correlation of about 0.34.

This pattern means that at least part of the apparent temporal drift may reflect **calendar-position or pay-cycle-related composition effects**, rather than a simple change in the underlying population. Salary timing or other recurring income cycles are plausible explanations, but the dataset contains no salary-payment dates, so no causal explanation is asserted.

The derived repeat-customer variables also show a separate observation-window problem. Customer history before 1 June 2016 is unobserved. The share of apparently new customers falls almost monotonically as the dataset progresses (Spearman correlation about -0.99 with days since the observation start), while mean observed history length rises almost perfectly over time (about 0.998). This is strong evidence of **left-censoring / observation-window bias**. `prior_tx_count` and `is_repeat_customer` therefore remain useful for exploratory analysis but will be subjected to explicit sensitivity analysis rather than treated automatically as stable production predictors.

### Stage 2: statistical relevance, redundancy, and temporal stability screening

The eligible predictors are assessed using established **filter methods**, including correlation structure, low-variance checks, mutual information, and suitable univariate predictive screening.

Stage 2 is no longer interpreted from one arbitrary 80% chronological block alone. Filter evidence will be examined across **multiple chronological development folds**, with calendar-position diagnostics considered alongside the statistical rankings. The aim is to identify predictors whose relevance is reasonably stable across time rather than strong only in one particular segment of June or July.

The analysis will also compare results **with and without `prior_tx_count` and `is_repeat_customer`** because of the documented left-censoring risk.

No feature is removed automatically on the basis of one correlation, PSI value, mutual-information score, or single fold. Stage 2 produces evidence for later model-based selection.

### Stage 3: embedded and wrapper methods

The project will compare established model-based selection approaches. These may include:

- L1-regularised logistic regression as an embedded feature-selection method;
- recursive feature elimination (RFE), likely using logistic regression or another suitable baseline estimator.

These methods will be applied within the training data only so that validation and test information do not influence feature selection. The same temporal and repeat-customer sensitivity considerations used in Stage 2 will be carried forward.

### Stage 4: nonlinear and model-agnostic confirmation

For nonlinear models, feature contribution will be compared using tree-based importance and model-agnostic methods such as permutation importance and SHAP.

No single importance method will be treated as authoritative. The objective is to compare whether important variables remain stable across model families, temporal folds, and evaluation methods.

### Selection principle

The final feature set will therefore be based on converging evidence rather than on one algorithm. The intended sequence is:

**semantic and governance eligibility → point-in-time eligibility → data-quality eligibility → temporal/calendar diagnostics → filter methods across chronological folds → embedded/wrapper methods → nonlinear/model-agnostic confirmation → stability and sensitivity checks**

A feature may be statistically strong and still be rejected if its timing or meaning cannot be defended. Conversely, a semantically valid feature may remain available for modelling even if it is later removed because it adds little predictive value.

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

Because customer history before 1 June 2016 is not observed, these two derived fields are **provisional analytical predictors**. Their usefulness will be compared with models that exclude them.

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

The observation-window analysis adds a second limitation: customers with activity before 1 June may be misclassified as first-time customers early in the dataset simply because earlier history is unavailable. This is why repeat-customer features require sensitivity analysis rather than unconditional inclusion.

## Status

This is the approved feature set for the **first Module 3 model-ready dataset** after Stage 1 governance and point-in-time eligibility screening. It is not yet the final statistically selected feature set. Later stages compare filter, embedded, wrapper, and model-agnostic methods while explicitly accounting for temporal/calendar effects and observation-window bias. Material revisions are reflected in the data dictionary/change log, narrative documentation, and Git history.
