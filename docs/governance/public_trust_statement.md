# Public Trust Statement

## What this system does

This academic prototype estimates the probability that a telecom-enabled microcredit transaction will remain unpaid after five days.

Its purpose is to support risk prioritisation and follow-up. It is not designed to decide automatically whether a person receives credit.

## What information it uses

The model uses 12 behavioural features derived from recharge, account-tenure, decrement, rental, and related telecom-credit activity.

A customer identifier is used only during controlled preprocessing where chronology requires it. It is removed before the model-ready dataset is created and is not a predictor.

## What the output means

The model returns a probability of five-day delinquency.

A higher probability means that, based on patterns in the historical development data, the transaction looks more similar to cases that became delinquent.

The score is not proof that a person will fail to repay. It should not be treated as a statement about character, intent, or creditworthiness beyond the narrow model purpose.

## How people remain responsible

The project does not automate credit approval or refusal.

If used operationally, the score should support human review. Staff should be able to consider other relevant information, disagree with the model, and escalate cases where the output appears inappropriate.

## How the model can be explained

The project uses SHAP-based explanations to show which input features contributed most to model predictions.

These explanations describe how the model behaves. They do not prove that a feature caused a repayment outcome.

## Fairness limitations

The historical dataset does not contain reliable protected demographic attributes.

For that reason, the project cannot demonstrate demographic fairness and does not infer protected characteristics from behavioural data simply to generate fairness statistics.

Operational segments are monitored for robustness, but those checks are not presented as substitutes for protected-group fairness analysis.

If legitimate protected-group data became available in a real deployment, their use for fairness monitoring would require separate legal, privacy, and governance review.

## Accuracy and limitations

On the final historical holdout, the calibrated model achieved ROC-AUC of approximately 0.829 and captured about 54% of delinquent cases in the highest-risk 20% of scores.

Those results do not guarantee future performance.

The available historical period is short, the population changes over time, and several important features show temporal instability. The model therefore requires ongoing monitoring and fresh validation before reliance on a materially different population.

## Monitoring and accountability

The project monitors:

- input and prediction-score drift;
- missingness;
- performance and calibration where trustworthy outcomes exist;
- operational robustness;
- model version and fingerprint.

Monitoring alerts trigger investigation rather than automatic conclusions.

Material ethical, fairness, privacy, security, or model-validity concerns are covered by an incident-response process. The model can be suspended or decommissioned if risks cannot be reduced appropriately.

## Retraining

New trustworthy labelled data can be used to create a retrained candidate.

A candidate must pass defined quality gates and be recorded in MLflow. Passing those gates does not automatically replace the deployed model. Human review and approval are required before promotion.

## Privacy position

Raw and row-level analytical data are treated as restricted project data and are not published in GitHub.

Removing direct identifiers lowers risk but does not automatically make behavioural data anonymous.

## What this project does not claim

The project does not claim that:

- the model is production-ready;
- the score should determine credit approval or refusal;
- demographic fairness has been proven;
- SHAP explanations are causal;
- historical accuracy guarantees future accuracy;
- compliance with GDPR, the EU AI Act, CCPA, HIPAA, or lending law has been legally certified.

## Contact and challenge in a real deployment

A real operational version should provide a clear route for people to:

- ask how model-assisted processing affected them where applicable;
- correct inaccurate source information;
- request human review of consequential decisions;
- submit complaints or appeals.

Those customer-facing procedures are outside the scope of this academic prototype but are part of the recommended deployment controls.
