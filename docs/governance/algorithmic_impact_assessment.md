# Algorithmic Impact Assessment

## Assessment scope

This assessment covers the academic telecom five-day delinquency model and the surrounding pipeline, API, dashboards, monitoring, and retraining controls.

The model estimates the probability that a telecom-enabled microcredit transaction will remain unpaid after five days. The approved project purpose is **risk prioritisation and follow-up support**. It is not approved as the sole basis for automatic credit approval, refusal, pricing, or another decision with legal or similarly significant effects.

This assessment describes the current academic prototype. Any real deployment would require a fresh assessment for the actual organisation, jurisdiction, population, data sources, intervention, and decision process.

## System overview

| Item | Current project position |
|---|---|
| Model | Random Forest with isotonic calibration |
| Artefact version | 1.0.0 |
| Inputs | 12 behavioural / telecom-credit features |
| Output | Raw and calibrated five-day delinquency probability |
| Decision role | Advisory risk-prioritisation signal |
| Autonomous adverse decision | Not approved |
| Human review | Required for consequential use |
| Monitoring | Custom Python + Evidently AI |
| Explainability | SHAP-based global/local analysis and what-if examples |
| Retraining | Candidate-based, MLflow tracked, no automatic promotion |

## Stakeholders and potentially affected people

Direct stakeholders include:

- credit-risk and repayment teams using the score;
- model, data, compliance, privacy, security, and business owners;
- customers whose transaction data are scored;
- operational staff asked to review or act on higher-risk cases.

Potentially affected customers may experience different levels of follow-up attention if the score is used operationally. The model therefore has the capacity to influence treatment even where it does not make the final decision itself.

## Expected benefits

The intended benefits are limited and practical:

- prioritise manual follow-up toward transactions more likely to become five-day delinquent;
- concentrate limited operational capacity on higher-risk cases;
- provide a reproducible and explainable risk signal;
- improve visibility into model performance and population change.

These benefits are conditional on proportionate downstream use.

## Principal risks

### Incorrect or unstable predictions

The historical window is short and the final holdout shows temporal degradation relative to development. A materially changed population could make the model unreliable.

Controls:

- frozen holdout evaluation;
- calibration;
- drift and performance monitoring;
- separate monitoring dashboard;
- retraining candidate gates;
- human approval before promotion.

Residual risk: moderate. More real-world history would be required for production confidence.

### Unfair or disproportionate treatment

The dataset lacks reliable protected demographic attributes. Demographic fairness cannot be demonstrated, and indirect proxy effects cannot be ruled out.

Controls:

- protected groups are not inferred;
- Fairlearn is used only for supported grouped measurement;
- operational robustness is monitored;
- continuous fairness monitoring plan defines how legitimate protected-group data would be incorporated if lawfully available;
- consequential use remains human-controlled.

Residual risk: material uncertainty remains because evidence is incomplete.

### Automation bias and over-reliance

Users may treat a probability score as a definitive judgment even where the model is only advisory.

Controls:

- intended-use statements;
- dashboard explanations and limitations;
- no approval/decline decision in the inference layer;
- incident escalation for inappropriate autonomous use;
- public trust statement.

Residual risk depends heavily on operational training and process design.

### Privacy and data-protection risk

The source includes customer-level behavioural information and an identifier-like field used temporarily for chronology.

Controls:

- raw/interim data stay out of GitHub;
- `msisdn` is removed before model-ready output;
- row-level analytical data remain restricted;
- privacy-safe audit logging;
- data-governance and anonymisation plans.

Residual risk remains because de-identification does not automatically make behavioural records anonymous.

### Data-quality and lineage risk

Unexpected source defects, label anomalies, or pipeline changes could invalidate outputs.

Controls:

- Great Expectations checks;
- Prefect orchestration;
- blocking interim and processed validation;
- tests;
- lineage documentation;
- CI/CD.

### Explainability risk

SHAP explanations may be mistaken for causal explanations.

Controls:

- documentation explicitly limits SHAP to model-behaviour interpretation;
- local explanations are paired with model probabilities;
- explanations do not claim that changing a feature will cause repayment behaviour to change.

## Human oversight

The model should support, not replace, accountable human review.

A reviewer should be able to:

- see the calibrated score and model version;
- understand the principal model drivers;
- consider information outside the model;
- disagree with the score;
- escalate suspicious or harmful patterns;
- avoid treating a threshold crossing as an automatic adverse decision.

Human oversight is meaningful only if the organisation allows actual discretion and records how the model is used.

## Contestability and redress

A real deployment should provide a route for affected people to question consequential decisions where the model materially contributed.

Operational design should support:

- identification of the model/version used;
- reconstruction of relevant input and output evidence subject to privacy limits;
- human reconsideration;
- correction of inaccurate source data;
- complaint escalation.

The academic prototype does not operate a real customer-redress process.

## Monitoring and lifecycle controls

Ongoing controls include:

- PSI, KS, missingness and prediction-score drift;
- ROC-AUC, precision, recall, Brier score, calibration and top-20% capture once trustworthy outcomes mature;
- operational robustness and fairness-feasibility monitoring;
- model/version fingerprinting;
- ethical incident response;
- governed retraining;
- decommissioning criteria.

## Impact assessment conclusion

The model can be justified as an academic and decision-support prototype when its scope remains narrow, its limitations remain visible, and model output does not become an automatic adverse credit decision.

The strongest unresolved issues are temporal stability, incomplete demographic fairness evidence, and the risk of inappropriate operational reliance.

A production deployment should therefore be conditional on:

- fresh representative validation;
- documented legal and regulatory assessment;
- defined human-review and redress procedures;
- continuous monitoring;
- lawful fairness-data strategy where appropriate;
- incident-response and retirement controls;
- formal accountable-owner approval.
