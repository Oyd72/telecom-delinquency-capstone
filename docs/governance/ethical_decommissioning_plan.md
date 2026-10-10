# Ethical decommissioning plan

## Purpose

This plan defines how the telecom delinquency model should be withdrawn when it is no longer suitable, lawful, reliable, or ethically acceptable for its approved use.

Decommissioning is treated as a governed lifecycle activity, not as a simple technical deletion. The objective is to stop inappropriate use, preserve enough evidence for accountability, protect affected data, and ensure that downstream users do not continue relying on an obsolete model.

## Decommissioning triggers

A decommissioning review should be opened where one or more of the following occurs:

- performance or calibration deterioration remains material after reasonable remediation;
- sustained drift shows that the deployment population is outside the validated scope;
- fairness-related risks cannot be reduced to an acceptable level;
- a prohibited or unacceptable use becomes embedded in operations;
- data provenance, quality, or lawful-use assumptions can no longer be defended;
- a privacy or security risk cannot be adequately mitigated;
- the business purpose materially changes;
- a successor model has been approved and the current model no longer serves a justified purpose;
- the underlying service or product is discontinued;
- legal or regulatory requirements make continued use inappropriate;
- the model cannot be maintained because critical data, expertise, or infrastructure are no longer available.

No single monitoring alert automatically requires retirement. Decommissioning follows evidence review unless immediate suspension is necessary to contain serious harm.

## Decision authority

The proposed Ethics and Model Risk Review Committee should approve permanent retirement of the model or a material use case.

Inputs should include:

- model-owner recommendation;
- monitoring evidence;
- incident history;
- privacy/compliance assessment;
- security considerations;
- business impact;
- successor-model status, if any.

For a critical incident, temporary suspension may occur before the formal decommissioning decision.

## Decommissioning process

### 1. Freeze change and identify scope

Record:

- model name and artefact version;
- SHA-256 fingerprint;
- operating threshold;
- deployed endpoints and dashboards;
- dependent systems and workflows;
- approved use cases;
- known users or teams;
- active retraining or deployment jobs.

Pause non-essential changes while the retirement scope is confirmed.

### 2. Stop new operational use

Depending on the environment:

- disable the public or internal scoring endpoint;
- remove the model from active routing;
- disable scheduled scoring jobs;
- stop automated retraining and deployment;
- remove the model from user-facing dashboards or mark it clearly as retired;
- ensure downstream systems no longer treat stale outputs as current.

Where a successor model exists, migration should occur only after that model has separately passed approval and deployment controls.

### 3. Preserve accountability evidence

Before deleting technical assets, retain the minimum evidence needed to reconstruct the model lifecycle:

- model card;
- approved metadata;
- SHA-256 fingerprint;
- source-code commit;
- configuration and threshold;
- validation and monitoring summaries;
- fairness limitations;
- incident and governance decisions;
- MLflow experiment and registry evidence;
- deployment and CI/CD records;
- decommissioning decision and effective date.

Preserved evidence should not include unnecessary row-level personal data.

### 4. Handle data and artefacts

Separate retirement of the model from retention of data.

The following should be reviewed against the project retention rules and any applicable legal requirements:

- raw and interim datasets;
- processed analytical data;
- candidate model binaries;
- the deployed binary;
- API or monitoring logs;
- privacy-audit logs;
- aggregate reports.

Unneeded row-level data and obsolete candidate artefacts should be securely deleted. Evidence required for audit or academic reproducibility may be retained in privacy-safe form.

### 5. Revoke access and automation

Remove or disable:

- service credentials no longer needed;
- deployment hooks;
- scheduled jobs;
- environment variables;
- model-serving permissions;
- storage access associated solely with the retired model.

Repository history may remain where needed for reproducibility, but active deployment paths should no longer point to the retired version.

### 6. Communicate retirement

Notify relevant users and owners of:

- retirement date;
- reason for retirement at an appropriate level of detail;
- systems or workflows affected;
- whether a replacement exists;
- what to do with previously generated scores;
- where residual issues should be reported.

External communications should avoid overstating technical conclusions and should be coordinated with legal, privacy, security, and business owners where relevant.

### 7. Verify closure

A closure review should confirm that:

- the model can no longer be invoked through active production paths;
- scheduled scoring and deployment are disabled;
- dashboards no longer present the model as active;
- retained evidence is complete;
- unneeded data and artefacts have been deleted or scheduled for deletion;
- successor-model references are correct;
- outstanding incidents or complaints have an owner.

## Historical scores

Retiring a model does not automatically make every historical score invalid. Their interpretation depends on the reason for decommissioning.

If retirement occurs because of serious model error, bias, unlawful processing, or data corruption, affected historical decisions should be identified and reviewed where feasible.

If retirement occurs because of normal replacement or end of service, existing records may remain valid historical evidence but should not be reused as current predictions.

## Rollback versus decommissioning

Rollback is a short-term recovery control, usually to a previously approved version.

Decommissioning is a permanent lifecycle decision.

An incident may therefore follow this path:

`incident → containment → rollback → investigation → remediation or decommissioning`

The two controls should not be treated as interchangeable.

## Relationship to the current project

The academic prototype already supports several parts of a defensible retirement process:

- frozen artefact versioning;
- SHA-256 model fingerprinting;
- Git commit history;
- MLflow evidence;
- CI/CD deployment records;
- monitoring evidence;
- documented fairness and privacy limitations;
- human approval before retraining-candidate promotion.

This plan adds the retirement controls needed to complete the model lifecycle.
