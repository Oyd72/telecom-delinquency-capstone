# Model development narrative

This file records how the modelling approach developed after the cleaned modelling population was created. It is meant to explain the sequence of decisions: what we tested, what we found, and why the next step changed.

## Starting modelling population

The model-ready population has 150,767 records dated from 1 June through 23 July 2016, including 26,162 delinquent cases. Records after 23 July are not used in ordinary supervised modelling because every later outcome is successful repayment and the source does not explain why that regime changes.

The first processed table contains the Stage 1-approved predictors plus `pdate` as a temporal control. `msisdn` and the source `label` are removed before modelling.

## Stage 1: eligibility before statistics

Feature selection starts with a non-statistical screen. A field has to make sense semantically, be available at the intended scoring point, have a defensible data-quality treatment, and avoid obvious leakage or identifier problems. Unresolved encodings and redundant derived fields are held back even if they might look predictive.

The resulting Stage 1 set is documented in `docs/governance/feature_selection.md`.

## Initial chronological split

The first test used an 80/20 chronological split: records through 13 July for development and 14–23 July as the later subset.

The two periods differed more than expected:

- delinquency rose from about 16.55% to 20.78%;
- repeat-customer share rose from about 6.84% to 12.94%;
- `daily_decr30` and `daily_decr90` had PSI values around 2.9;
- `rental30` and `rental90` also shifted substantially.

That made it inappropriate to describe the later block as simply representative of the earlier one.

## Calendar-position effects

Before treating the differences as straightforward temporal drift, matching days of the month were compared across June and July.

The day-of-month patterns were very similar for several variables:

- `daily_decr30` and `daily_decr90`: Spearman correlation about 0.94;
- `rental30`: about 0.98;
- `rental90`: about 0.98.

Delinquency itself was much less stable by day of month, with a correlation of about 0.34.

A recurring income or pay-cycle effect is therefore plausible, but it cannot be established from this dataset because salary dates are not available. The practical consequence is more important than the explanation: one chronological split is not enough to judge feature stability.

## Observation-window bias in customer history

`prior_tx_count` and `is_repeat_customer` use only strictly earlier records, so they do not look forward in time. Even so, they have a different problem: customer history before 1 June 2016 is invisible.

As the dataset progresses, the apparent new-customer share falls almost monotonically (Spearman about -0.99 with days since the start) while observed history length rises almost perfectly (about 0.998). A customer seen early in June can therefore look “new” simply because earlier activity is outside the extract.

For that reason, the two history variables are kept for exploration and sensitivity testing rather than treated as automatically reliable production predictors.

## Stage 2: filter screening across folds

Stage 2 uses filter methods such as correlation, low-variance checks, mutual information, and univariate relevance. The analysis was revised so that those statistics are compared across four calendar-aware folds:

- 1–13 June: 33,647 records, delinquency 15.85%;
- 14–30 June: 49,507 records, delinquency 16.02%;
- 1–13 July: 38,965 records, delinquency 17.83%;
- 14–23 July: 28,648 records, delinquency 20.78%.

Repeat-customer share rose from about 2.13% in early June to 12.94% in late July, which is interpreted cautiously because of left-censoring.

The most persistent filter signals came from account decrement/spending and main-account recharge behaviour. `daily_decr30` and `daily_decr90` had mean mutual information around 0.151 and mean absolute Spearman relationships to delinquency around 0.437. `sumamnt_ma_rech90`, `sumamnt_ma_rech30`, `cnt_ma_rech90`, and `cnt_ma_rech30` also ranked strongly.

The decrement variables varied more across folds than some recharge variables, which fits the earlier calendar-position finding. `cnt_ma_rech90`, for example, had a mean mutual information around 0.099 with comparatively stable ranking.

Removing the two history variables did not change the top ten non-history features. No feature was dropped automatically after Stage 2.

## Stage 3: L1 and RFE

Stage 3 compared L1-regularised logistic regression with recursive feature elimination (RFE), using the same four chronological folds and fitting preprocessing inside each fold.

L1 was not very selective in the first three folds: cross-validation chose `C = 10`, and all 20 candidate features remained. In late July, stronger regularisation (`C ≈ 0.0139`) retained 15. This is why L1 selection frequency alone is not treated as strong evidence.

RFE was more selective because it was asked to keep 10 predictors per fold. Three fields were selected by RFE in every fold: `daily_decr30`, `cnt_ma_rech90`, and `sumamnt_ma_rech30`. Of these, `daily_decr30` and `cnt_ma_rech90` were also selected by L1 in every fold.

`daily_decr30` had the strongest joint stability, with consistent coefficient sign and an average RFE rank of 1.0. `cnt_ma_rech90` showed the same all-fold selection pattern. Several other fields remained plausible but were less stable, including `daily_decr90`, `sumamnt_ma_rech90`, `last_rech_amt_ma`, `aon`, and `medianamnt_ma_rech30`.

`is_repeat_customer` is a useful example of why sensitivity testing matters. L1 retained it in every fold, but RFE selected it in only one and its coefficient sign was not fully stable. Given the known left-censoring issue, that is not enough to justify relying on it.

No final feature set was declared after Stage 3.

## Stage 4: nonlinear confirmation

Stage 4 used forward-chaining XGBoost. Each evaluation period was predicted only from earlier data. Feature contribution was assessed with both permutation importance and SHAP, once with all 20 features and again with the two history variables removed.

With all features, ROC-AUC was about 0.896 in late June, 0.823 in early July, and 0.832 in late July. Average precision was about 0.704, 0.502, and 0.580.

Removing `prior_tx_count` and `is_repeat_customer` made almost no difference. ROC-AUC was roughly 0.896, 0.825, and 0.832 without them, with average precision also effectively unchanged. That gave a practical reason to exclude the history variables from the primary specification: their governance problem is real, while their incremental value is negligible.

`cnt_ma_rech90` was the most consistently strong nonlinear predictor across permutation importance and SHAP. `last_rech_date_ma` was also strong. `daily_decr30` remained important, especially under SHAP, although its permutation rank moved more, which is consistent with shared information and calendar structure.

`aon`, `sumamnt_ma_rech90`, `rental30`, `last_rech_amt_ma`, and `medianmarechprebal90` showed continuing value. `daily_decr90` had weaker permutation importance but stronger SHAP importance, again suggesting overlap with `daily_decr30` rather than no signal.

## Candidate feature position after Stages 2–4

No single selection method is treated as authoritative. The strongest combined support was for:

- `cnt_ma_rech90`;
- `daily_decr30`;
- `last_rech_date_ma`;
- `sumamnt_ma_rech90`;
- `aon`;
- `last_rech_amt_ma`.

A second group remained worth testing rather than dropping immediately: `daily_decr90`, `sumamnt_ma_rech30`, `medianamnt_ma_rech30`, `medianmarechprebal90`, `rental30`, and `cnt_ma_rech30`.

The history variables stayed outside the primary production-oriented set because more reliable pre-observation history would be needed to use them confidently.

## Comparing feature-set variants

Three non-history XGBoost variants were compared under the same forward-chaining setup:

- full non-history set: 18 features;
- core-plus-secondary set: 12 features;
- compact core: 6 features.

The 18-feature model had mean ROC-AUC about 0.851, mean average precision about 0.596, mean Brier score about 0.117, mean ECE about 0.075, and mean top-20% capture about 60.7%.

The 12-feature model was almost the same on discrimination and ranking: mean ROC-AUC about 0.850, average precision about 0.594, and top-20% capture about 60.1%. Calibration was only slightly weaker, with mean Brier around 0.118 and ECE around 0.078.

The six-feature model remained useful for ranking but gave up more calibration quality. Mean ROC-AUC fell to about 0.844, average precision to about 0.587, and mean ECE rose to about 0.124.

The 12-feature specification was therefore preferred as the working model: it removes one third of the predictors from the full set for very little loss in useful performance. The 18-feature version remains a challenger.

The preferred 12 features are:

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

This is a reasoned choice based on parsimony and converging evidence, not a claim that a mathematical optimum has been proven.

## Calibration

All three feature-set variants overpredicted risk badly in late July, which pointed to a temporal calibration problem rather than a feature-count problem.

The 12-feature model was therefore tested with leakage-safe post-hoc calibration. For each evaluation period, the base model was trained on earlier data, the calibrator was fitted on the immediately preceding seven-day window, and performance was measured on the next period.

Three outputs were compared: uncalibrated probabilities, Platt scaling, and isotonic calibration.

On average, both calibration methods improved the probability estimates. Mean ECE fell from about 0.114 uncalibrated to about 0.067 with Platt and 0.068 with isotonic. Mean absolute calibration-in-the-large error fell from about 0.107 to about 0.059 with Platt and 0.062 with isotonic. Mean Brier score improved from about 0.128 to about 0.118.

Platt preserved ROC-AUC, average precision, and top-20% capture because it is monotonic. Isotonic slightly reduced ranking metrics because its stepwise mapping can create ties.

The period-level results mattered more than the averages. In late July, the uncalibrated model predicted about 36.8% risk against an observed delinquency rate of 20.8%, with ECE around 0.160. Platt reduced mean predicted risk to about 18.2% and ECE to about 0.029; isotonic was similar.

Early July moved in the opposite direction. The uncalibrated model already underpredicted risk: about 8.8% predicted against 17.8% observed. Platt and isotonic pushed the probabilities lower still. So a calibrator fitted on the most recent week can help a great deal in one regime and hurt in another.

Platt remains the leading calibration candidate, but no universal calibrator was frozen from this result.

## Rolling calibration windows

Platt scaling was then tested with 3-, 5-, 7-, and 10-day recent windows while holding the underlying XGBoost model fixed within each evaluation fold.

Across 12 fold/window comparisons, Platt improved ECE in 11 and reduced the absolute calibration gap in 10. Brier score improved in two of the three evaluation folds for every window length.

There was no single best window. Three days worked best in both July folds, while ten days worked better in late June. The relationship between recent prevalence mismatch and calibration benefit was only weak to moderate (Spearman around -0.30 for Brier change and -0.36 for ECE). The stronger descriptive relationship was between the signed prevalence gap and the direction of the probability shift (about -0.71).

With only three evaluation folds, that is not enough to justify an automatic prevalence-triggered rule. The current position is to keep Platt as the preferred method, treat the recent calibration window as something to monitor, and require a longer history before fixing an adaptive production policy.

## Pipeline and test status

The modelling population can now be regenerated through the verified Prefect ETL in `src/pipeline/prefect_etl.py`. Raw validation is diagnostic; interim and processed validation are blocking. The flow has run successfully both locally and inside Docker.

The repository test suite now contains 12 passing tests across `tests/unit/` and `tests/validation/`. These checks do not change the modelling conclusions, but they make the transformations and pipeline contract easier to reproduce and harder to break accidentally.

## Status of this narrative

This is the running account of material modelling decisions. New evidence that changes the model position should be reflected here and in `model_decision_log.md`. Exact code changes remain in Git history, while generated evidence stays under `reports/`.