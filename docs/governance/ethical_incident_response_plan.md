# Ethical incident-response plan

## Purpose

This plan defines how suspected ethical or responsible-AI incidents involving the telecom delinquency model should be identified, contained, investigated, documented, escalated, and resolved.

The model is an academic prototype intended for risk-prioritisation support rather than autonomous credit decisions. Even in that limited role, harmful outcomes can arise from data problems, model drift, unfair operational use, security or privacy failures, misleading explanations, or inappropriate reliance on the score.

The incident-response process therefore treats technical, operational, privacy, fairness, and human-oversight failures as potentially connected.

## What counts as an ethical incident

An ethical incident is any event, pattern, or credible concern suggesting that use of the model may create material harm, undermine lawful or responsible use, or invalidate the assumptions under which the model was approved.

Examples include:

- materially different error rates across relevant groups;
- evidence that model outputs are being treated as automatic approval or decline decisions;
- a significant degradation in predictive performance, calibration, or top-risk capture;
- a drift alert suggesting that the model is operating outside its validated population;
- evidence that a feature functions as an unacceptable proxy;
- data-quality or labelling failures that affect model outputs;
- unauthorised use of identifiers or restricted data;
- privacy or security incidents involving model inputs, outputs, logs, or artefacts;
- misleading or materially incomplete explanations shown to users or decision-makers;
- user complaints indicating a recurring pattern of disproportionate or harmful treatment;
- deployment of an unapproved model, threshold, or retraining candidate;
- failure of a control intended to preserve human review.

## Detection channels

Incidents may be identified through:

- the monitoring dashboard;
- automated drift, performance, calibration, and fairness checks;
- CI/CD failures;
- data-quality validation;
- privacy-audit logs;
- model artefact hash/version mismatch;
- operational staff reports;
- customer complaints or appeals;
- security alerts;
- governance or compliance review;
- audit findings.

No single source is treated as sufficient in every case. Monitoring alerts are signals for investigation, not automatic proof of harm.

## Severity levels

### Level 1 — observation

A weak or isolated signal with no current evidence of material impact.

Typical examples:

- a single monitoring threshold enters watch status;
- a small operational-segment difference appears once;
- a non-critical dashboard or explanation issue is detected.

Response:

- log the observation;
- assign an owner;
- continue monitoring;
- close only after rationale is recorded.

### Level 2 — material concern

Evidence suggests a control weakness or meaningful risk that could affect users, model validity, fairness, privacy, or operational decisions.

Typical examples:

- repeated or widening performance deterioration;
- a persistent segment-level error gap;
- unexplained model-input drift;
- use outside the documented purpose;
- a retraining candidate promoted without required approval;
- a data-quality problem affecting a meaningful portion of records.

Response:

- restrict the affected use where appropriate;
- preserve evidence;
- start formal investigation;
- notify the governance owner;
- assess whether threshold, model, data, or workflow changes are required.

### Level 3 — critical incident

There is evidence of serious or widespread harm, unlawful processing, major security/privacy compromise, or loss of meaningful human oversight.

Typical examples:

- automated adverse credit decisions contrary to the approved use;
- confirmed discrimination affecting a protected group;
- use of unlawfully obtained or highly sensitive data;
- material privacy/security breach involving model-related data;
- model behaviour that cannot be brought back within validated limits;
- systematic use of a materially incorrect model version.

Response:

- suspend the affected model or use case immediately where feasible;
- preserve logs and artefacts;
- escalate to senior governance, legal/privacy, security, and business owners;
- assess regulatory or contractual notification obligations;
- document containment and recovery actions;
- do not resume use until approval conditions are met.

## Immediate containment

The first objective is to limit further harm while preserving evidence.

Containment actions may include:

- disabling or pausing the affected endpoint or workflow;
- reverting to the last approved model and threshold;
- switching the model to advisory-only mode;
- requiring manual review for all affected cases;
- blocking a problematic feature or data feed;
- stopping retraining or deployment automation;
- suspending a specific customer segment or use case;
- preserving relevant logs, model hashes, datasets, configuration, and deployment records.

Containment should be proportionate. An alert should not automatically trigger complete shutdown if a narrower control can safely contain the risk.

## Investigation

The investigation should reconstruct what changed and what impact resulted.

At minimum, review:

- model version and SHA-256 fingerprint;
- operating threshold;
- deployment date and commit;
- affected population and time period;
- input-data quality and lineage;
- feature-distribution changes;
- prediction-score distribution;
- performance and calibration where trustworthy labels exist;
- fairness/segment evidence;
- explanation behaviour;
- downstream human decision process;
- customer or staff reports;
- relevant CI/CD and deployment logs;
- whether the incident arose from model design, data, implementation, misuse, or several factors together.

The investigation must distinguish observed facts from hypotheses.

## Roles and escalation

For this academic project, the proposed governance structure is:

- **Model owner / analytics lead** — coordinates technical investigation and evidence gathering.
- **Data owner** — investigates data quality, provenance, labelling, and lineage.
- **Privacy / compliance owner** — assesses privacy, lawful-use, transparency, and regulatory issues.
- **Security owner** — handles confidentiality, integrity, access, and infrastructure incidents.
- **Business owner** — assesses operational impact and downstream decision practices.
- **Ethics and Model Risk Review Committee** — reviews material incidents, approves remediation, and decides whether continued use is acceptable.

Where one person performs several roles in a small organisation, the responsibilities should still be documented separately.

## Corrective actions

Depending on the cause, remediation may include:

- data correction or source replacement;
- model retraining;
- threshold revision;
- feature removal;
- additional calibration;
- narrowing the approved population or use case;
- changes to human-review procedures;
- revised explanations or user notices;
- stronger access controls;
- additional monitoring;
- suspension or decommissioning.

Any model or threshold change must be revalidated before use.

## Resumption criteria

A suspended use may resume only when:

- the root cause is sufficiently understood;
- containment remains effective;
- corrective actions have been tested;
- performance, fairness, privacy, and operational controls are acceptable for the intended use;
- the approved model and threshold are traceable;
- governance approval is recorded.

If these conditions cannot be met, the model or affected use case should remain suspended and may proceed to decommissioning.

## Documentation and audit trail

Each material incident should record:

- incident ID;
- detection date and channel;
- severity;
- affected model/version/threshold;
- affected population and period;
- factual description;
- suspected and confirmed causes;
- containment actions;
- impact assessment;
- governance decisions;
- corrective actions;
- validation evidence;
- closure decision;
- residual risks;
- follow-up monitoring.

Records should avoid unnecessary personal data while preserving enough evidence for accountability.

## Communication

Internal communications should clearly distinguish:

- what is known;
- what is still under investigation;
- what controls have been applied;
- whether users or customers may have been affected;
- what further action is expected.

External or regulatory communication should be coordinated through the relevant legal, privacy, security, and business owners.

## Relationship to current project controls

The project already provides several inputs to this process:

- model version and SHA-256 traceability;
- CI/CD history;
- monitoring dashboard evidence;
- custom and Evidently drift checks;
- performance and calibration metrics;
- fairness limitations and operational segment analysis;
- privacy-safe pipeline audit logging;
- governed retraining with human approval before promotion.

This incident-response plan connects those controls into a single escalation and decision process.
