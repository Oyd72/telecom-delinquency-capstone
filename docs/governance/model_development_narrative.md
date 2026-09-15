# Model development narrative

This document records the analytical decisions made after the cleaned modelling population was created. It complements the cleaning narrative by explaining how feature selection and validation strategy evolve as new evidence is produced.

## Starting modelling population

The model-ready population contains 150,767 records dated from 1 June through 23 July 2016, with 26,162 delinquent cases. Records after 23 July remain outside ordinary supervised modelling because the later block contains only successful repayments and may follow a different data-generation or labelling process.

The first processed dataset contains the Stage 1-approved candidate predictors plus `pdate` as a temporal control. `msisdn` and the original source `label` are removed before modelling.

## Stage 1: governance and point-in-time eligibility

Feature selection begins with a non-statistical eligibility screen. Variables are admitted only where their meaning, timing, data-quality treatment, and governance position can be defended. Identifier-like fields, unresolved encodings, redundant derived fields, and variables with unresolved point-in-time leakage risk are excluded before statistical selection begins.

This stage produced the first candidate predictor set documented in `docs/governance/feature_selection.md`.

## Initial temporal split and representativeness check

An initial 80/20 chronological split was tested rather than assumed to be representative. Records through 13 July 2016 formed the earlier development subset; 14–23 July formed the later subset.

The comparison showed material differences:

- delinquency increased from about 16.55% to 20.78%;
- repeat-customer share increased from about 6.84% to 12.94%;
- `daily_decr30` and `daily_decr90` showed very large PSI values of about 2.9;
- `rental30` and `rental90` also showed substantial temporal distribution shifts.

This meant that the later subset could not simply be described as distributionally representative of the earlier subset.

## Calendar-position and possible pay-cycle effects

The apparent temporal drift was investigated further instead of being accepted at face value. Matching day-of-month patterns were compared across June and July.

Several account-behaviour variables were strikingly stable by day of month across the two months:

- `daily_decr30` and `daily_decr90`: Spearman correlation about 0.94;
- `rental30`: about 0.98;
- `rental90`: about 0.98.

Delinquency itself showed only a moderate paired day-of-month correlation of about 0.34.

This suggests that part of the measured temporal drift may arise from calendar-position effects. A salary or recurring-income cycle is a plausible explanation, but the dataset does not contain salary-payment dates or another variable that could establish this causally. The project therefore records this as a hypothesis, not as a fact.

The validation strategy is adjusted accordingly: a simple 80/20 chronological split is not treated as the sole basis for feature selection or final validation. Feature stability will instead be checked across multiple chronological folds while calendar-position effects are considered explicitly.

## Observation-window bias in repeat-customer features

The derived fields `prior_tx_count` and `is_repeat_customer` were designed to use only strictly earlier transactions. This protects against future leakage, but a separate limitation emerged: customer history before 1 June 2016 is unavailable.

The diagnostics show a nearly monotonic decline in the share of apparently new customers as the dataset progresses (Spearman correlation about -0.99 with days since the dataset start), while mean observed history length rises almost perfectly over time (about 0.998).

This is strong evidence of left-censoring / observation-window bias. Someone appearing early in June may be a long-standing borrower whose earlier activity is simply outside the extract, while a customer appearing in July has had much more opportunity to accumulate visible history.

For that reason, `prior_tx_count` and `is_repeat_customer` remain useful exploratory variables but are no longer treated as automatically stable production predictors. Later feature-selection and modelling stages will compare results with and without these variables.

## Revised Stage 2 strategy

Stage 2 uses established filter methods such as correlation analysis, low-variance checks, mutual information, and univariate relevance measures. However, rankings will not be interpreted from one arbitrary chronological block alone.

The revised approach is to:

- calculate filter evidence across multiple chronological development folds;
- examine whether feature relevance is reasonably stable across those folds;
- retain calendar-position diagnostics as context when account-behaviour variables appear to shift;
- run sensitivity analysis with and without `prior_tx_count` and `is_repeat_customer`;
- avoid automatic feature removal from any one filter statistic.

The purpose is to separate genuine predictive relevance from artefacts caused by observation-window position or short-term calendar composition.

## Stage 2 fold-stability results

The revised filter screening used four calendar-aware chronological folds rather than a single 80/20 block:

- 1–13 June: 33,647 records, delinquency 15.85%;
- 14–30 June: 49,507 records, delinquency 16.02%;
- 1–13 July: 38,965 records, delinquency 17.83%;
- 14–23 July: 28,648 records, delinquency 20.78%.

The repeat-customer share increased from about 2.13% in early June to 12.94% in late July. In light of the left-censoring diagnostics, this increase is treated as partly mechanical and is not interpreted as proof of a true change in borrower composition.

Across the four folds, the strongest and most consistently relevant feature families were account spending/decrement and main-account recharge behaviour. The highest average mutual-information rankings included `sumamnt_ma_rech90`, `daily_decr90`, `daily_decr30`, `sumamnt_ma_rech30`, `cnt_ma_rech90`, and `cnt_ma_rech30`.

`daily_decr30` and `daily_decr90` had the highest mean mutual information of about 0.151, with mean absolute Spearman relationships to delinquency of about 0.437. Their mutual-information values varied more across folds than several recharge variables, which is consistent with the previously identified calendar-position effects. They therefore remain strong candidates, but their temporal stability needs to be considered in later model-based stages rather than being accepted on filter strength alone.

Recharge totals and counts were somewhat more stable across folds. `cnt_ma_rech90`, for example, had a mean mutual information of about 0.099 and a relatively low standard deviation of the mutual-information rank (about 1.26). `sumamnt_ma_rech90` also ranked strongly, although its rank varied somewhat more.

Several mid-ranked predictors, including `last_rech_date_ma`, `medianmarechprebal30`, `medianmarechprebal90`, and `last_rech_amt_ma`, showed lower average mutual information but comparatively stable values across folds. They are not removed at this stage because filter evidence alone is not sufficient to judge their incremental value once correlated predictors are modelled jointly.

The sensitivity analysis that excluded `prior_tx_count` and `is_repeat_customer` produced the same top ten ranked non-history features. This confirms that the principal Stage 2 ranking is not being driven by the left-censored customer-history variables.

No feature is removed automatically after Stage 2. The filter results are treated as evidence for Stage 3, where embedded and wrapper methods can test which variables retain value when predictors are considered jointly.

## Stage 3: embedded and wrapper selection

Stage 3 compared two established model-based approaches across the same four calendar-aware folds: L1-regularised logistic regression as an embedded method and recursive feature elimination (RFE) with logistic regression as a wrapper method. Preprocessing was fitted separately within each fold so that imputation and scaling did not borrow information across periods.

The L1 result was only partly selective. In the first three folds, cross-validation chose a weak regularisation setting (`C = 10`) and retained all 20 candidate predictors. In the late-July fold, stronger regularisation (`C ≈ 0.0139`) retained 15 predictors. This means that simple L1 selection frequency should not be interpreted as strong evidence on its own: in most folds the fitted penalty was too weak to generate much sparsity.

RFE was more discriminating because it was explicitly asked to retain 10 predictors per fold. Only three features were selected by RFE in all four folds: `daily_decr30`, `cnt_ma_rech90`, and `sumamnt_ma_rech30`. Of these, `daily_decr30` and `cnt_ma_rech90` were also selected by L1 in every fold and therefore showed the strongest agreement between the embedded and wrapper approaches.

`daily_decr30` had the strongest joint stability: it was selected by both methods in all folds, had a mean absolute L1 coefficient of about 5.84, consistent coefficient sign, and an average RFE rank of 1.0. `cnt_ma_rech90` showed the same all-fold selection pattern, with a mean absolute L1 coefficient of about 1.59 and consistent sign.

Several other variables remained credible but less stable across methods. `daily_decr90`, `sumamnt_ma_rech90`, `last_rech_amt_ma`, `aon`, and `medianamnt_ma_rech30` were retained by L1 in every fold and by RFE in three of four folds. `daily_decr90` showed some coefficient-sign instability, while `sumamnt_ma_rech90` had only 50% sign consistency, suggesting that multicollinearity or changing relationships with correlated recharge variables may be affecting coefficient interpretation.

`cnt_ma_rech30` and `rental90` were retained by L1 in all folds but by RFE in only half. They therefore remain candidates rather than confirmed selections.

The left-censored history variable `is_repeat_customer` illustrates why sensitivity analysis remains necessary. L1 retained it in every fold, but RFE selected it in only one of four folds and its coefficient sign was not fully stable. This result is not treated as evidence that repeat-customer status is a reliable production predictor, particularly because the observation-window analysis already showed that its apparent prevalence changes mechanically over time.

No final feature set is declared after Stage 3. The principal evidence so far favours `daily_decr30` and `cnt_ma_rech90` as the most stable candidates across filter, embedded, and wrapper methods, while several additional recharge and account-behaviour variables remain plausible. The next stage should test nonlinear and model-agnostic importance before any irreversible feature removal is made.

## Narrative status

This file is the running narrative for model-development decisions. It should be updated whenever a material modelling choice changes because of new evidence. Exact code changes remain traceable through Git history, while generated analytical outputs remain under `reports/`.
