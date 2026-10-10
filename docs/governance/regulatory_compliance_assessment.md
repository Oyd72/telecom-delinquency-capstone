# Regulatory compliance assessment

## Purpose and limitation

This document maps the telecom delinquency prototype to regulatory themes named in the final-project brief and to the EU AI Act because the model concerns individual credit risk.

It is an academic compliance assessment, not legal advice and not a conclusion that every listed law applies to a real deployment. Applicability depends on the organisation, location of individuals, business model, data sources, and actual use of the model.

## GDPR

### Potential relevance

If the model processes personal data of individuals within GDPR scope, the processing would need to satisfy the GDPR principles of lawfulness, fairness, transparency, purpose limitation, data minimisation, accuracy, storage limitation, integrity/confidentiality, and accountability.

The project already implements several controls aligned with those principles:

- explicit analytical purpose;
- exclusion of unnecessary identifiers from model-ready data;
- restricted handling of row-level data;
- documented lineage and retention;
- privacy-safe audit logging;
- no protected-group inference merely to create fairness statistics;
- model transparency and limitations.

The legal basis for a real lender's processing is not established by this academic project and would require organisation-specific assessment.

### Automated decisions

The current inference layer returns a risk probability only and does not make a credit approval or refusal decision.

That design reduces, but does not eliminate, automated-decision risk. If a real organisation used the model output as the sole basis for a decision producing legal or similarly significant effects, additional GDPR requirements and safeguards would need to be assessed, including Article 22 where applicable.

### Data-subject transparency

A real deployment would need clear information about:

- the purpose of the processing;
- categories of data used;
- recipients;
- retention;
- individual rights;
- the role of automated or model-assisted processing where required.

The project's stakeholder dashboard and public trust statement demonstrate transparency design but are not a substitute for a statutory privacy notice.

### Source

Official GDPR text: https://eur-lex.europa.eu/eli/reg/2016/679/

## EU AI Act

### Creditworthiness classification risk

Annex III of the EU AI Act lists AI systems intended to evaluate the creditworthiness of natural persons or establish their credit score, except systems used to detect financial fraud, as a high-risk use case.

If this prototype were deployed so that its delinquency score materially evaluates a natural person's creditworthiness, that classification would need to be assessed seriously. Calling the score "decision support" does not by itself remove the question; the real intended purpose and influence on decisions matter.

For this academic project:

- the model is not represented as production-ready;
- the inference layer does not approve or decline credit;
- human oversight is required;
- no conclusion is made here that a hypothetical deployment falls outside high-risk classification.

### High-risk control themes

If the system were classified as high-risk, relevant control themes would include lifecycle risk management, data governance, technical documentation, logging, transparency to deployers, human oversight, accuracy/robustness/cybersecurity, conformity assessment, registration, and post-market monitoring.

The project demonstrates many of these concepts academically through:

- data-governance documentation;
- model card and experiment record;
- model/version traceability;
- monitoring and audit trails;
- SHAP-based explainability;
- CI/CD and tests;
- human-controlled retraining;
- incident-response and decommissioning plans.

This does not amount to formal EU AI Act conformity.

### Source

Official consolidated EU AI Act: https://eur-lex.europa.eu/eli/reg/2024/1689

## CCPA / CPRA

### Potential relevance

The California Consumer Privacy Act applies to qualifying for-profit businesses doing business in California that meet statutory thresholds. The California Attorney General describes thresholds including revenue, volume of California residents/households whose personal information is bought/sold/shared, or reliance on revenue from selling personal information.

This academic project does not establish that a particular business meets those thresholds or that California consumers are involved.

If applicable, relevant themes would include:

- notice at or before collection;
- access/know rights;
- deletion and correction rights subject to exceptions;
- opt-out rights for sale or sharing;
- limits concerning sensitive personal information where applicable;
- reasonable security and vendor controls.

Project controls such as data minimisation, restricted row-level storage, documentation, and transparency are directionally supportive but do not constitute CCPA compliance.

### Source

California Attorney General CCPA information: https://oag.ca.gov/privacy/ccpa

## HIPAA

### Applicability position

HIPAA is not a general US privacy law. HHS explains that the HIPAA Rules apply to covered entities and, for certain requirements, their business associates.

This telecom microcredit dataset is not described as health information, and the academic project is not operating as a HIPAA covered entity or business associate.

Accordingly, HIPAA does not appear directly applicable to the project as described.

If the system were later used by or for a HIPAA covered entity and involved protected health information, applicability would need to be reassessed, including whether a business-associate relationship existed.

### Source

HHS covered entities and business associates: https://www.hhs.gov/hipaa/for-professionals/covered-entities/index.html

## Cross-regulatory control map

| Control theme | Project evidence |
|---|---|
| Purpose limitation / intended use | README, data governance framework, model card |
| Data minimisation | anonymisation plan, removal of `msisdn`, feature review |
| Data quality | Great Expectations, Prefect pipeline, tests |
| Transparency | stakeholder dashboard, model card, public trust statement |
| Human oversight | advisory-only inference, no automatic credit decision |
| Traceability | model version, SHA-256, Git history, MLflow |
| Monitoring | custom Python, Evidently AI, performance/calibration monitoring |
| Fairness | Fairlearn-supported operational checks + continuous fairness plan |
| Incident handling | ethical incident-response plan |
| Lifecycle closure | ethical decommissioning plan |
| Change control | CI/CD, governed retraining candidate process |

## Compliance gaps before real deployment

A real deployment would still require organisation-specific work, including:

- documented legal basis and controller/processor roles;
- formal privacy notices and rights-handling procedures;
- data-retention schedule tied to actual legal/business needs;
- transfer/vendor assessment where relevant;
- security risk assessment;
- jurisdiction-specific lending/consumer-credit analysis;
- EU AI Act role and high-risk classification assessment;
- formal human-oversight and appeal/redress process;
- production monitoring and incident ownership;
- evidence that any protected-group processing for fairness monitoring is lawful and proportionate.

## Conclusion

The project contains a substantial governance foundation, but it should be described as **compliance-aware rather than legally certified**.

GDPR and the EU AI Act are the most directly relevant frameworks to evaluate for an EU-facing credit-risk deployment. CCPA would depend on California nexus and statutory business thresholds. HIPAA is not directly applicable on the facts currently available.
