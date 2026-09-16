# Technical RAID log

This log keeps track of the main risks, assumptions, issues, and dependencies that could affect how reliable or useful the telecom delinquency project is.

| Type | Item | Response |
|---|---|---|
| Risk | Class imbalance may make delinquent cases harder to detect | Use metrics that reflect minority-class performance and test class-handling methods where needed. |
| Risk | Data leakage could make model performance look unrealistically strong | Use only information available at scoring time and keep training and evaluation periods separate. |
| Risk | Temporal drift may reduce performance on later customers | Treat the results as historical and require validation on newer data before operational use. |
| Risk | Some features may act as proxies for customer groups | Review feature use and compare performance across relevant groups where the data support it. |
| Risk | Users may place too much weight on model output | Keep the model as decision support rather than an autonomous decision-maker. |
| Assumption | Historical patterns are informative enough for an academic demonstration | Treat results as illustrative, not as proof of current production performance. |
| Issue | Several variables have unclear definitions or encodings | Exclude them unless their meaning can be verified. |
| Dependency | Continued access to the dataset and working technical environment | Keep the environment and data-handling process documented and reproducible. |

## Review approach

Review this log at the end of each sprint and update it when new data-quality, modelling, fairness, or implementation concerns appear.