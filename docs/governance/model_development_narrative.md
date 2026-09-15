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

## Narrative status

This file is the running narrative for model-development decisions. It should be updated whenever a material modelling choice changes because of new evidence. Exact code changes remain traceable through Git history, while generated analytical outputs remain under `reports/`.
