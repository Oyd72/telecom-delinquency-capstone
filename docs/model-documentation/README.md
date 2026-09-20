# Model documentation

The main Module 4 documents are:

- `module4_experiment_record.md` — chronological record of modelling decisions, experiments, results, interpretation, and figure references.
- `model_card.md` — compact technical and governance summary of the packaged selected model, including intended use, performance, limitations, inference contract, and monitoring expectations.

The record is intentionally comprehensive. It preserves the development path so that later conclusions can be traced back to the experiment that produced them.

The AI-use log is maintained separately in a private repository and is therefore not stored here.

For broader project governance and data-management documentation, use `../governance/` and the project-level data dictionary.


Supporting final evidence lives under `reports/`, notably:

- `reports/module4_fairness_report.md`
- `reports/module4_mlflow_evidence.md`
- `reports/tables/module4_classification_metrics.csv`
- `reports/tables/module4_counterfactual_explanations.csv`

The packaged-model metadata is maintained in `models/selected_model_metadata.json`, while the binary artefact itself remains local.
