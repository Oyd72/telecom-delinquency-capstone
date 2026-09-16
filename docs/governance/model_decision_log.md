# Model decision log

This is the short-form record of the main modelling decisions. The longer reasoning is in `model_development_narrative.md`; this file is meant to make the trigger, decision, and follow-up easy to scan.

| ID | Trigger / evidence | Decision | Why | Status / follow-up |
| --- | --- | --- | --- | --- |
| MD-01 | Every record after 23 July 2016 is labelled successful. | Limit ordinary supervised modelling to records through 23 July 2016. | The later block may come from a different sampling, labelling, or business-process regime and would distort class balance if mixed into training. | Active. Later block kept for separate diagnostics. |
| MD-02 | Point-in-time construction of several loan-history fields cannot be verified. | Keep unresolved loan-history variables out of the first approved predictor set. | The current transaction may be included in historical aggregates, which would create leakage. | Active until source construction is verified. |
| MD-03 | `msisdn` is needed for customer chronology but is identifier-like. | Use it only during controlled derivation and remove it from model-ready output. | Chronology is needed; identifier use as a predictor is not. | Implemented and validated. |
| MD-04 | The first 80/20 chronological split showed large PSI and target-rate differences. | Do not assume the later period is representative of the earlier one. | The gap may reflect drift, calendar composition, or both. | Replaced by calendar-aware folds and forward chaining. |
| MD-05 | June and July show similar day-of-month patterns for `daily_decr*` and `rental*`. | Treat calendar position / possible pay-cycle effects as a modelling consideration. | Part of the apparent drift may be tied to position within the month. | Active limitation; no causal claim. |
| MD-06 | Apparent new-customer share falls as observed customer-history length rises. | Treat `prior_tx_count` and `is_repeat_customer` as left-censored sensitivity variables rather than default production predictors. | History before 1 June is missing, so both variables partly reflect observation-window position. | Excluded from primary model variants; retained for sensitivity analysis. |
| MD-07 | Stage 2 filter results vary by period but repeatedly highlight recharge and decrement variables. | Compare filter evidence across several calendar-aware folds. | Feature relevance should not depend on one arbitrary time block. | Implemented. |
| MD-08 | L1 retained all 20 features in three folds because regularisation was weak; RFE was more selective. | Do not treat L1 selection frequency alone as proof that a feature is necessary. | Weak regularisation produces little sparsity. | Reflected in interpretation. |
| MD-09 | Stages 2–4 repeatedly support `cnt_ma_rech90` and `daily_decr30`, with several secondary variables also recurring. | Compare full, reduced, and compact feature sets before dropping variables permanently. | Selection evidence needs to be checked against actual predictive and calibration performance. | Completed. |
| MD-10 | Removing the history variables barely changes forward-chaining ROC-AUC or average precision. | Leave `prior_tx_count` and `is_repeat_customer` out of the primary production-oriented specification. | They carry left-censoring risk and add little predictive value. | Current primary position. |
| MD-11 | The 12-feature model performs almost the same as the 18-feature model on discrimination and top-20% capture; the six-feature version calibrates worse. | Prefer the 12-feature core-plus-secondary model and keep the 18-feature version as a challenger. | It gives a simpler specification for very little performance loss. | Current preferred specification. |
| MD-12 | All feature-set variants overpredict risk in late July. | Test explicit temporal calibration before finalising the model. | The problem appears temporal rather than caused by feature count alone. | Completed with past-only calibration windows. |
| MD-13 | Platt and isotonic calibration improve late-July calibration but worsen early July; Platt is strongest on aggregate and preserves ranking. | Do not freeze one universal calibrator. Keep Platt as the leading candidate and test rolling windows. | Calibration depends on the recent regime. | Completed by rolling-window diagnostics. |
| MD-14 | Across 3/5/7/10-day windows, Platt improves ECE in 11 of 12 comparisons and absolute calibration gap in 10 of 12, but the best window changes by fold. Only three evaluation folds are available. | Do not create an automatic prevalence-triggered calibration rule. Keep Platt as the preferred method but treat window length as a monitored parameter. | The evidence supports regime sensitivity but not a validated adaptive rule. | Current position. More history is needed before fixing a production calibration policy. |

## Current preferred feature set

The current 12-feature core-plus-secondary specification is:

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

This is still a working specification rather than a frozen production model. Any material change should be added here with the evidence that prompted it and reflected in the longer narrative.