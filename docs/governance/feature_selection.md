# Approved feature set for the first model-ready dataset

This document explains how the first Module 3 predictor set was assembled. The starting point is deliberately conservative: a field enters the first model-ready table only if its meaning, timing, and treatment can be defended. This is **Stage 1: governance and point-in-time eligibility screening**, not the final statistical feature-selection result.

## How features are selected

The project uses several stages. Statistical importance is considered only after a field has passed the basic semantic, timing, data-quality, and governance checks.

### Stage 1: eligibility screening

A field can enter the first candidate set when:

- its meaning is sufficiently clear from the source material;
- it could plausibly be available at the time of the credit decision;
- known data-quality problems have a documented treatment;
- it is not primarily an identifier, constant, redundant derived measure, or unresolved encoding;
- there is no obvious target or point-in-time leakage problem.

A predictive variable is not automatically acceptable if its meaning or timing cannot be defended.

### Temporal checks before statistical screening

The first chronological 80/20 split used records through 13 July 2016 for development and 14–23 July for the later subset. The two periods were not as similar as initially hoped. Delinquency rose from about 16.55% to 20.78%, and several account-behaviour variables shifted sharply.

The next check compared matching days of the month across June and July. `daily_decr30`, `daily_decr90`, `rental30`, and `rental90` showed strong day-of-month similarity across the two months, while delinquency itself was only moderately correlated. This makes a recurring calendar or pay-cycle effect plausible, although the dataset does not contain salary dates and cannot prove that explanation.

A separate problem affects the derived customer-history fields. History before 1 June 2016 is missing. As the observation window progresses, the apparent share of new customers falls almost monotonically while observed customer-history length rises. `prior_tx_count` and `is_repeat_customer` are therefore affected by left-censoring and are treated as sensitivity variables rather than automatically stable predictors.

### Stage 2: filter methods across time

The eligible fields are screened using correlation, low-variance checks, mutual information, and other univariate evidence. These statistics are compared across calendar-aware chronological folds rather than taken from one arbitrary development block.

The analysis is also repeated without `prior_tx_count` and `is_repeat_customer`. No feature is removed from a single score or single fold. Stage 2 provides evidence for later model-based selection.

### Stage 3: embedded and wrapper methods

L1-regularised logistic regression is used as an embedded method and recursive feature elimination (RFE) as a wrapper method. Preprocessing and selection are fitted inside the relevant training data only. Temporal sensitivity and the history-variable caveat remain in place.

### Stage 4: nonlinear confirmation

Tree-based models are then used to check whether the same signals persist under nonlinear modelling. Permutation importance and SHAP provide complementary views of feature contribution. No single importance measure is treated as decisive.

### Overall selection rule

The final feature position is based on agreement across several kinds of evidence:

**semantic eligibility → point-in-time eligibility → data-quality treatment → temporal checks → filter methods → embedded/wrapper methods → nonlinear importance → stability and sensitivity analysis**

A field may be statistically strong and still be excluded for timing or governance reasons. A valid field may also be dropped later if it adds little predictive value.

## Modelling population

Ordinary supervised modelling is limited to records dated through **23 July 2016**. The 58,825 later records are all labelled as successful repayment. The source material does not explain whether that break comes from sampling, labelling, extraction, or a business-process change, so the later block is not assumed to be comparable with the earlier mixed-outcome population.

The model-ready table is built from the validated interim data, not directly from the raw CSV.

## Target

The source uses `label = 1` for repayment within five days and `label = 0` for failure to repay within five days.

For modelling, this becomes:

- `delinquent_5d = 1` when `label = 0`;
- `delinquent_5d = 0` when `label = 1`.

The original `label` is then removed. Keeping both would add no information and could create accidental target leakage.

## Stage 1 approved predictors

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

### Derived customer-history fields

- `prior_tx_count` — number of transactions for the same `msisdn` on strictly earlier dates;
- `is_repeat_customer` — `1` when at least one strictly earlier transaction exists, otherwise `0`.

These two fields are created while `msisdn` is still available. Same-day transactions do not count as prior transactions for each other. `msisdn` is removed afterwards.

Because customer history before 1 June is not visible, the two history variables are provisional analytical predictors. Later modelling compares results with and without them.

## Temporal control

`pdate` stays in the processed dataset for splitting, lineage, and reproducibility. It is not an approved predictor.

## Fields excluded at Stage 1

- `msisdn` — identifier-like and used only for controlled grouping/chronology;
- `pcircle` — constant in the current dataset;
- all `fr_*` fields — exact construction unresolved;
- `maxamnt_loans30`, `maxamnt_loans90` — derived/redundant consistency checks;
- `medianamnt_loans30`, `medianamnt_loans90` — encoding unresolved;
- `payback30`, `payback90` — held back until point-in-time availability can be confirmed;
- `cnt_loans30`, `amnt_loans30`, `cnt_loans90`, `amnt_loans90` — not used in the first model-ready table until it is clear whether the current transaction is included in their historical windows;
- source `label` — replaced by `delinquent_5d`;
- `pdate` — retained as a control, not a predictor.

## Missing values

Values set to missing during cleaning remain missing in the model-ready table. They are not imputed before train/test splitting. Any imputation used by a model is fitted inside the training pipeline or fold so that validation and test data do not influence preprocessing.

## Repeat-customer interpretation

`prior_tx_count` and `is_repeat_customer` describe observed prior participation, not inherent creditworthiness. The approval process is unknown, so returning borrowers may represent a selected group. The missing pre-June history also means that some early records may be wrongly classified as first-time simply because earlier activity is outside the extract.

For both reasons, the relationship between repeat participation and repayment is treated as observational and tested through sensitivity analysis.

## Current status

This document records the **Stage 1** predictor set and the methodology used to move beyond it. It is not the final model specification. Later stages compare statistical and model-based evidence across time, and material changes are recorded in the modelling narrative, model decision log, data dictionary, and Git history.