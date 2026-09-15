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

## Stage 4: nonlinear and model-agnostic confirmation

Stage 4 used forward-chaining XGBoost models so that each evaluation period was predicted only from earlier data. Feature importance was measured on the next chronological period with both permutation importance and SHAP. The process was repeated with all 20 features and with the two left-censored customer-history features removed.

Predictive performance remained materially useful across all three forward evaluations. With all features, ROC-AUC was about 0.896 for late June, 0.823 for early July, and 0.832 for late July. Average precision was about 0.704, 0.502, and 0.580 respectively. Performance therefore declined after the earliest evaluation period but remained substantially above random ranking in the later periods.

Removing `prior_tx_count` and `is_repeat_customer` did not reduce performance in a meaningful way. ROC-AUC changed from approximately 0.896 to 0.896 in late June, 0.823 to 0.825 in early July, and 0.832 to 0.832 in late July. Average precision was likewise effectively unchanged or slightly better without the history variables. This supports excluding those fields from the primary production-oriented candidate set: their left-censoring risk is real, while their incremental predictive contribution appears negligible.

Across permutation importance and SHAP, `cnt_ma_rech90` was the most consistently strong nonlinear predictor. It ranked first or near first under both approaches and remained the top feature after the history variables were removed. `last_rech_date_ma` also showed strong and comparatively stable importance, ranking second by mean permutation importance and remaining near the top by SHAP.

`daily_decr30` remained a major predictor, particularly under SHAP, where its average absolute contribution was the largest of the candidate variables. Its permutation rank was less stable, which is consistent with the earlier finding that this feature has strong calendar-position structure and is correlated with related account-behaviour variables. This divergence is interpreted as evidence that the variable is important but shares predictive information with other features rather than as a reason to remove it.

`aon` and `sumamnt_ma_rech90` remained credible across the nonlinear methods and also had support from earlier stages. `rental30`, `last_rech_amt_ma`, and `medianmarechprebal90` showed secondary but persistent nonlinear importance. `daily_decr90` had relatively weak permutation importance but stronger SHAP importance, again suggesting shared or interacting information with `daily_decr30` rather than a clean independent effect.

The forward-chaining sensitivity path without history variables produced a very similar top-importance structure. This is important because it shows that the principal model signal is not dependent on the potentially biased repeat-customer fields.

## Consolidated candidate feature position after Stages 2–4

The project does not treat any single feature-selection method as authoritative. The current recommendation is based on convergence across filter methods, linear embedded/wrapper methods, nonlinear importance, temporal stability, and governance constraints.

A **core candidate set** is supported most consistently by the combined evidence:

- `cnt_ma_rech90` — strong and stable across filter, RFE, permutation importance, and SHAP;
- `daily_decr30` — strong across all stages, but requiring explicit monitoring for calendar-position sensitivity;
- `last_rech_date_ma` — moderate filter evidence but strong nonlinear importance and stable practical interpretation;
- `sumamnt_ma_rech90` — strong filter evidence and persistent nonlinear importance;
- `aon` — consistent model-based contribution across linear and nonlinear methods;
- `last_rech_amt_ma` — stable secondary contribution across methods.

Several **secondary candidates** remain reasonable and should be tested in model-comparison runs rather than removed immediately: `daily_decr90`, `sumamnt_ma_rech30`, `medianamnt_ma_rech30`, `medianmarechprebal90`, `rental30`, and `cnt_ma_rech30`. Their evidence is less uniform, often because of correlation with stronger variables or calendar-position effects.

`prior_tx_count` and `is_repeat_customer` should remain outside the primary production-oriented feature set unless a future dataset provides reliable pre-observation customer history. Their exclusion is supported both by governance reasoning and by the Stage 4 sensitivity result showing essentially unchanged predictive performance without them.

## Feature-set model comparison

Three non-history feature variants were compared under identical forward-chaining XGBoost settings: the full 18-feature non-history set, a 12-feature core-plus-secondary set, and a compact six-feature core. The comparison considered discrimination, calibration, and top-20% delinquent capture rather than ROC-AUC alone.

The **full 18-feature set** had the strongest average results overall: mean ROC-AUC about 0.851, mean average precision about 0.596, mean Brier score about 0.117, mean expected calibration error about 0.075, and mean top-20% capture about 60.7%.

The **12-feature core-plus-secondary set** performed almost identically on discrimination and business ranking: mean ROC-AUC about 0.850, mean average precision about 0.594, and mean top-20% capture about 60.1%. Its calibration was only slightly weaker, with mean Brier score about 0.118 and mean expected calibration error about 0.078. In practical terms, reducing the feature set from 18 to 12 removed one third of the predictors at a very small cost in discrimination and capture.

The **six-feature compact core** retained useful ranking performance but showed a clearer loss in calibration and some discrimination. Mean ROC-AUC fell to about 0.844, mean average precision to about 0.587, and mean expected calibration error increased to about 0.124. Its mean top-20% capture remained close to the 12-feature set at about 60.0%, but calibration deteriorated sharply in the early-July evaluation. The compact model is therefore considered too aggressive a reduction for the primary specification at this stage.

A separate issue emerged across all three variants in late July. Mean predicted risk substantially exceeded the observed delinquency rate, producing calibration gaps of about 0.16–0.17. Because this appears across the full and reduced feature sets, it is interpreted primarily as a temporal-calibration problem rather than a feature-count problem. The model therefore still requires explicit calibration assessment and likely post-hoc calibration fitted without using future data.

### Preferred specification after feature-set comparison

The current preferred modelling specification is the **12-feature core-plus-secondary set**:

- `cnt_ma_rech90`
- `daily_decr30`
- `last_rech_date_ma`
- `sumamnt_ma_rech90`
- `aon`
- `last_rech_amt_ma`
- `daily_decr90`
- `sumamnt_ma_rech30`
- `medianamnt_ma_rech30`
- `medianmarechprebal90`
- `rental30`
- `cnt_ma_rech30`

This choice is an interpretation based on parsimony and converging evidence rather than a statistically proven optimum. The 12-feature set sacrifices very little discrimination or top-risk capture relative to the full 18-feature model while reducing complexity and avoiding six weaker predictors. The 18-feature model remains a useful benchmark/challenger rather than being discarded.

## Temporal calibration experiment

The preferred 12-feature specification was then tested with a leakage-safe post-hoc calibration design. For each evaluation period, the base XGBoost model was trained on earlier data, the calibrator was fitted on the immediately preceding seven-day window, and performance was measured only on the subsequent evaluation period. This preserved chronology and prevented the evaluation fold from influencing either model fitting or calibration fitting.

Three probability outputs were compared: the uncalibrated model, Platt scaling, and isotonic calibration.

On average, both calibration methods improved probability calibration relative to the uncalibrated model. Mean expected calibration error fell from about 0.114 for the uncalibrated model to about 0.067 with Platt scaling and about 0.068 with isotonic calibration. Mean absolute calibration-in-the-large error similarly fell from about 0.107 to about 0.059 with Platt and about 0.062 with isotonic calibration. Brier score also improved from about 0.128 uncalibrated to about 0.118 with either calibration method.

Platt scaling preserved ROC-AUC, average precision, and top-20% capture exactly because it applies a monotonic logistic transformation to the model score. Isotonic calibration produced very similar calibration performance but slightly reduced average ROC-AUC and average precision, reflecting the fact that its stepwise mapping can introduce tied scores and modestly alter ranking metrics.

The aggregate averages, however, conceal an important period-specific result. In late July, both calibration methods corrected the large overprediction problem very effectively. The uncalibrated mean predicted risk was about 36.8% against an observed delinquency rate of 20.8%, with ECE about 0.160. Platt scaling reduced mean predicted risk to about 18.2% and ECE to about 0.029; isotonic produced a very similar result.

In early July, the direction was different. The uncalibrated model already underpredicted risk, with mean predicted risk about 8.8% against an observed delinquency rate of 17.8%. Fitting the calibrator on the final seven days of June pushed predicted risk lower still: about 7.0% under Platt and 6.3% under isotonic. Both methods therefore worsened calibration for that period.

This means that post-hoc calibration is **temporally regime-sensitive** in this dataset. A calibrator estimated from the immediately preceding week can be highly beneficial when the recent window resembles the next period, but can move probabilities in the wrong direction when the delinquency regime changes. The result is consistent with the earlier evidence of changing target prevalence and calendar-related structure.

### Calibration decision

No universal calibrator is frozen at this stage. Platt scaling is the leading candidate because it has the best aggregate calibration metrics while leaving ranking performance unchanged, but the early-July deterioration prevents treating it as a generally reliable solution without further testing.

The next calibration question is therefore not simply “Platt or isotonic?” but whether calibration should adapt to recent prevalence and temporal regime, and how stable that adaptation is under rolling evaluation. The project should test rolling calibration windows and recent-period prevalence before a final calibrated production specification is declared.

## Narrative status

This file is the running narrative for model-development decisions. It should be updated whenever a material modelling choice changes because of new evidence. Exact code changes remain traceable through Git history, while generated analytical outputs remain under `reports/`.
