# Module 4 fairness assessment

## Scope

A conventional demographic fairness analysis is not possible with this dataset. The model-ready data do not contain usable protected characteristics such as sex, race or ethnicity, disability, religion, or nationality. The project therefore does not calculate demographic parity, Equalized Odds, Equal Opportunity, or protected-group false-positive/false-negative gaps.

That is a limitation of the evidence, not a finding that the model is fair.

The analysis takes a deliberately conservative position: protected groups are not inferred from unrelated behavioural fields, and operational variables are not relabelled as demographic attributes simply to make fairness metrics calculable.

## Fairness-feasibility result

| Question | Result |
|---|---|
| Direct protected-group fields available | No |
| Demographic group metrics supportable | No |
| Protected groups inferred from proxies | No |
| Demographic fairness claim supported | No |
| Operational robustness analysis available | Yes |
| Proxy effects ruled out | No |

The model therefore carries a standing fairness limitation. Recharge behaviour, account tenure, and other behavioural inputs could still correlate with characteristics that are not observed in the dataset. SHAP and feature-name review cannot establish whether such indirect proxy relationships exist.

## What can be tested responsibly

Two kinds of checks remain useful even though they are not demographic fairness tests.

The first looks at **representation**. Earlier pipeline analysis showed that first-time borrowers account for about **92%** of the model-ready records, while returning borrowers make up about **8%**. Their observed delinquency rates also differ substantially: approximately **18.44%** for first-time borrowers and **4.83%** for returning borrowers. This grouping is operational rather than protected, and it is affected by left-censoring because customer history before June 2016 is unavailable. The result is therefore treated as a representation warning, not evidence of discrimination.

The second looks at **model robustness across operational segments** on the final holdout. These segments are based on account tenure and 90-day recharge frequency.

### Account-tenure segments

| Tenure band | Rows | Delinquency rate | ROC-AUC | Brier | Top-20% capture |
|---|---:|---:|---:|---:|---:|
| 1–256 | 7,096 | 26.58% | 0.8093 | 0.1491 | 47.56% |
| 256–534 | 7,057 | 20.29% | 0.8098 | 0.1264 | 51.12% |
| 534–972 | 7,073 | 18.89% | 0.8344 | 0.1147 | 56.51% |
| 972–2404 | 7,068 | 16.89% | 0.8507 | 0.0999 | 62.23% |

The model remains discriminative in every tenure quartile, but performance improves with longer account history. This matters operationally because newer accounts are both riskier in the observed data and harder to rank.

### Recharge-frequency segments

| 90-day recharge count band | Rows | Delinquency rate | ROC-AUC | Brier | Top-20% capture |
|---|---:|---:|---:|---:|---:|
| 0–2 | 7,385 | 49.98% | 0.6893 | 0.2275 | 30.59% |
| 2–5 | 7,165 | 19.25% | 0.6793 | 0.1460 | 36.69% |
| 5–10 | 7,587 | 8.33% | 0.6746 | 0.0739 | 39.24% |
| 10–132 | 6,511 | 3.86% | 0.6850 | 0.0362 | 42.23% |

Within narrow recharge bands the ROC-AUC falls to about 0.67–0.69. This is not unexpected because recharge behaviour is one of the model's main sources of separation. Once the population is stratified by recharge activity, much of that variation has already been removed.

These results are evidence about robustness, not fairness.

## Sensitivity to influential features

A controlled perturbation test was run on the strongest SHAP features. Each selected input was moved by ±10% of its training interquartile range, while staying inside the training range.

Most observations showed little or no movement in calibrated probability. The largest p95 change was approximately **5.63 percentage points** for an increase in `sumamnt_ma_rech90`. The largest share of observations moving by at least ten percentage points was **1.15%**.

This supports local stability under modest input changes, but it does not show that the features are normatively appropriate or free of proxy effects.

## Bias mitigation position

No demographic bias mitigation algorithm has been applied. Doing so would require a defensible protected-group attribute and a measurable disparity objective. Neither is available here.

The controls used instead are methodological and governance-oriented:

- direct identifiers are excluded from modelling;
- only point-in-time-valid predictors are used;
- protected groups are not inferred;
- operational segment performance is monitored separately;
- the model is limited to decision support rather than autonomous credit decisions;
- demographic fairness must be reassessed if appropriate protected-group or vulnerability data become lawfully available.

The project also keeps the short historical window and shifting customer composition visible as model-risk issues rather than describing them as fairness findings.

## Residual risks

Three fairness-related uncertainties remain.

**Proxy risk.** Behavioural variables may correlate with unobserved protected characteristics. The current dataset cannot test this.

**Representation risk.** Returning customers are under-represented and customer-history status is affected by left-censoring. Performance evidence for this subgroup is therefore less secure than its raw record count might suggest.

**Operational-use risk.** Even a well-ranked risk score can create unfair outcomes if interventions are intrusive, thresholds are poorly chosen, or staff treat the score as determinative. The model should therefore support proportionate follow-up and human review rather than automatic adverse action.

## Conclusion

The available evidence does not support a demographic fairness claim, and the project does not make one.

What the evidence does support is narrower: the selected model remains useful across observed tenure segments, behaves less strongly within narrow recharge-frequency bands, and is generally stable under modest feature perturbations. Those findings help assess robustness, but they cannot substitute for protected-group fairness testing.

A real deployment would require a separate fairness assessment using lawful, relevant group information where appropriate, together with outcome monitoring and review of any material disparities.

## Supporting evidence

- `src/models/assess_module4_fairness_robustness.py`
- `docs/governance/representation_bias_assessment.md`
- `reports/tables/module4_fairness_robustness_summary.json`
- `reports/tables/module4_operational_robustness_segments.csv`
- `reports/tables/module4_feature_sensitivity.csv`
- `reports/figures/module4/operational_segment_roc_auc.png`
- `reports/figures/module4/feature_sensitivity_p95.png`
