# Continuous fairness monitoring plan

## Purpose

This plan defines how fairness-related monitoring would operate if the telecom delinquency model moved from academic demonstration into controlled operational use. It preserves the evidence limits already established in the project: the current dataset does not contain reliable protected demographic attributes, so the project cannot calculate or claim demographic fairness metrics such as Equalized Odds, demographic parity, disparate impact, or protected-group error-rate gaps.

The absence of those attributes is a limitation of the evidence, not a finding that the model is fair.

## Current monitoring position

The current implementation monitors only attributes that genuinely exist in the data.

Fairlearn `MetricFrame` is used for grouped measurement of operational slices, especially first-time versus returning borrowers. That grouping is useful for representation diagnostics but is not a protected characteristic and is affected by left-censoring because borrower history before June 2016 is unavailable.

The monitoring dashboard also reports operational robustness across account-tenure and recharge-frequency segments. These checks help identify instability or concentration of model error, but they are not demographic fairness tests.

## Monitoring cadence

In operational use, fairness monitoring would follow the same batch cadence as model monitoring.

For each new scored batch:

1. representation changes would be checked immediately using available operational attributes;
2. model-output distribution and proxy-risk indicators would be reviewed alongside ordinary drift evidence;
3. once trustworthy outcomes mature, segment-level error rates, recall, precision, calibration and false-positive / false-negative rates would be recomputed;
4. material changes would be routed into governance review rather than automatically labelled as discrimination.

The academic prototype demonstrates this process on historical data. It does not claim continuous production observation.

## Current fairness indicators

Until protected-group data become legitimately available, the following indicators remain in scope:

- representation by operational segment;
- changes in segment prevalence over time;
- segment-level ROC-AUC, recall, precision, Brier score and top-20% capture where labels are trustworthy;
- material changes in false-positive and false-negative rates across operational segments;
- concentration of adverse model outputs in narrowly defined behavioural segments;
- proxy-risk review for influential features where a plausible indirect relationship is identified;
- stability of feature importance and local explanations where useful for investigation.

These indicators may identify areas requiring review. They must not be described as evidence that a protected group is treated fairly or unfairly.

## What happens if protected-group data become available

If legitimate protected-group or vulnerability attributes become available, they would be incorporated only after a separate legal, privacy and governance assessment confirming that collection and processing are appropriate for fairness monitoring.

The monitoring framework would then be extended to:

- calculate group-specific selection, error and calibration metrics;
- use Fairlearn `MetricFrame` to compare protected groups consistently;
- calculate Equal Opportunity or Equalized Odds components where the decision context and outcome definition make those metrics appropriate;
- compare false-positive and false-negative rates across groups;
- examine demographic-parity or disparate-impact indicators only where they are substantively relevant to the operational decision;
- track intersectional groups where sample sizes are sufficient and privacy risk remains acceptable;
- apply minimum sample-size rules before publishing group metrics;
- suppress or qualify unstable estimates rather than presenting them as reliable findings.

No single fairness metric would be treated as a universal pass/fail test. Metric choice would depend on the intervention, business process, legal context and the harms associated with different errors.

## Escalation and review

A fairness alert should trigger investigation rather than automatic rejection or automated remediation.

Escalation would be appropriate where, for example:

- a protected or operational group shows a material deterioration in error rates relative to its own baseline;
- a disparity persists across several monitoring periods;
- a change in feature distribution disproportionately affects one group;
- a newly identified proxy relationship creates plausible discrimination risk;
- operational users report a pattern of adverse or disproportionate outcomes;
- monitoring evidence becomes unreliable because group labels, outcome labels or sample sizes are inadequate.

The review should consider data quality, population drift, threshold effects, sample size, operational practice and possible proxy mechanisms before deciding whether the model, threshold, workflow or data should change.

## Governance response

Material fairness findings would be documented and routed to the project governance owner and the proposed Ethics and Model Risk Review Committee.

Available responses include:

- additional validation;
- threshold review;
- restriction of a problematic feature;
- retraining or challenger-model comparison;
- modification of downstream human-review procedures;
- temporary suspension of a use case;
- collection of better evidence;
- decommissioning if the risk cannot be reduced to an acceptable level.

Any mitigation would be tested for both predictive and fairness effects before promotion.

## Audit trail

Each monitoring period should retain:

- model and threshold version;
- monitored population and dates;
- group definitions;
- sample sizes;
- metrics and confidence limitations;
- triggered alerts;
- analyst interpretation;
- governance decision;
- corrective action and follow-up status.

The project already records model version, SHA-256 fingerprint, monitoring periods and drift/performance evidence. Operational fairness monitoring would extend that audit trail rather than create a separate undocumented process.

## Relationship to the current project

Current evidence is intentionally narrower:

- demographic protected attributes are unavailable;
- protected groups are not inferred;
- Fairlearn is used only for transparent grouped measurement of observed operational slices;
- operational robustness is not presented as demographic fairness;
- the monitoring dashboard keeps this limitation visible.

This plan therefore describes how continuous fairness monitoring would operate without overstating what the current historical dataset can demonstrate.
