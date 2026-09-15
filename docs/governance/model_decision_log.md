# Model decision log

This log provides a compact record of material modelling decisions, the evidence that triggered them, and their current status. It complements `model_development_narrative.md`: the narrative explains the analytical story in prose, while this log is intended as a quick audit trail.

| ID | Trigger / evidence | Decision | Why | Status / follow-up |
| --- | --- | --- | --- | --- |
| MD-01 | Records after 23 July 2016 are all labelled successful. | Restrict ordinary supervised modelling to records through 23 July 2016. | The later block may follow a different sampling, labelling, or business-process regime and would distort class balance if treated as ordinary training data. | Active. Later block retained only for separate diagnostics. |
| MD-02 | Point-in-time construction of several loan-history fields cannot be verified. | Exclude unresolved loan-history variables from the first approved predictor set. | Avoid target or point-in-time leakage where the current transaction may be included in historical aggregates. | Active until source construction can be verified. |
| MD-03 | `msisdn` is needed to derive observed prior participation but is identifier-like. | Use `msisdn` only during controlled feature derivation and remove it from model-ready output. | Preserve chronology while preventing identifier use as a predictor. | Implemented and validated. |
| MD-04 | Initial chronological 80/20 split showed large PSI and target-rate differences. | Do not assume the later split is representative of the earlier period. | The difference may reflect temporal drift, calendar composition, or both. | Replaced by calendar-aware fold diagnostics and forward chaining. |
| MD-05 | June/July day-of-month patterns for `daily_decr*` and `rental*` are highly correlated. | Treat calendar / possible pay-cycle effects as a modelling consideration. | Apparent drift may partly reflect position within the month rather than a permanent population shift. | Active limitation; do not claim causality. |
| MD-06 | New-customer share falls almost monotonically with time while observed customer-history length rises. | Treat `prior_tx_count` and `is_repeat_customer` as left-censored sensitivity variables rather than default production predictors. | Customer history before 1 June is unavailable, so these fields partly encode observation-window position. | Excluded from primary model variants; retained for sensitivity analysis. |
| MD-07 | Stage 2 filter results vary by period but identify recurring recharge and decrement signals. | Use multiple calendar-aware folds instead of a single feature-ranking sample. | Feature relevance should be stable across time segments rather than accepted from one arbitrary block. | Implemented. |
| MD-08 | L1 retained all 20 features in three folds because regularisation was weak, whereas RFE was more selective. | Do not use L1 selection frequency alone as evidence of feature necessity. | Lack of sparsity under weak regularisation is not equivalent to strong feature support. | Implemented in interpretation. |
| MD-09 | Stages 2-4 converge on `cnt_ma_rech90` and `daily_decr30`; several additional variables show persistent secondary value. | Compare full, reduced, and compact model variants before removing features irreversibly. | Converging feature-selection evidence should be tested against actual predictive and calibration performance. | Completed. |
| MD-10 | Removing history variables produces virtually unchanged forward-chaining ROC-AUC and average precision. | Exclude `prior_tx_count` and `is_repeat_customer` from the primary production-oriented specification. | Their governance / left-censoring risk is material while incremental predictive value is negligible. | Current primary position. |
| MD-11 | 12-feature model is almost indistinguishable from the 18-feature model on discrimination and top-20% capture; six-feature model degrades calibration. | Prefer the 12-feature core-plus-secondary specification; keep 18-feature model as challenger. | Better parsimony with negligible loss in ranking performance and only minor calibration cost. | Current preferred specification. |
| MD-12 | All model variants substantially overpredict risk in late July. | Investigate explicit temporal calibration before finalising the model specification. | The calibration problem persists across feature counts and appears temporal rather than caused by model complexity alone. | Completed with past-only calibration windows. |
| MD-13 | Platt and isotonic calibration materially improve late-July calibration, but both worsen calibration in early July; Platt has the best aggregate calibration while preserving ranking exactly. | Do not freeze a universal post-hoc calibrator. Keep Platt scaling as the leading calibration candidate and test rolling recent-window stability. | Calibration quality is regime-sensitive. A method that corrects late-July overprediction can worsen earlier underprediction, so aggregate averages are not enough for a production decision. | Completed by rolling-window diagnostic. |
| MD-14 | Rolling 3/5/7/10-day tests show Platt improves ECE in 11 of 12 comparisons and absolute calibration gap in 10 of 12, but the best window differs by fold (3 days in both July folds, 10 days in late June). Recent-window prevalence mismatch has only weak-to-moderate descriptive association with calibration benefit, and only three evaluation folds are available. | Do not create an automatic prevalence-triggered calibration rule. Retain Platt as the preferred calibration method, but treat calibration-window length as a monitored operating parameter rather than a fixed universal constant. | Shorter recent windows appear more responsive in July, while late June favours a longer window; the evidence is too limited to justify an adaptive rule. The negative correlation between signed prevalence gap and probability shift supports regime sensitivity but is descriptive, not a validated decision rule. | Current position. Use Platt as candidate calibration with explicit temporal monitoring; final production calibration policy requires more historical periods. |

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

This remains a working specification rather than a final frozen model. Material changes should be added to this log with the evidence that triggered them and reflected in the narrative document.
