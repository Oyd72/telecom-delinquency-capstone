# Technical RAID log

This log captures the main risks, assumptions, issues and dependencies that could affect the reliability or use of the telecom delinquency analytics project.

| Type | Item | Response |
|---|---|---|
| Risk | Class imbalance may reduce detection of delinquent cases | Use suitable evaluation metrics and test class-handling methods. |
| Risk | Data leakage could produce unrealistically strong model performance | Use only variables available at scoring time and keep training and evaluation data properly separated. |
| Risk | Temporal drift may reduce performance on more recent customers | Treat results as historical and require validation on recent data before operational use. |
| Risk | Proxy discrimination may affect some customer groups | Review feature use and compare error rates across relevant groups. |
| Risk | Users may rely too heavily on model outputs | Keep the model as decision support and require human review for consequential decisions. |
| Assumption | Historical patterns are sufficiently informative for demonstration | Treat results as illustrative rather than evidence of current operational performance. |
| Issue | Some variables have unclear meaning or documentation | Exclude them unless their meaning can be verified. |
| Dependency | Reliable access to the dataset and technical environment | Maintain versioned data and environment documentation. |

## Review approach

The RAID log should be reviewed at the end of each sprint and updated when new data-quality, modelling, fairness or implementation concerns are identified.
