# Report evidence

This directory contains reproducible analytical outputs created by the pipeline and modelling scripts. The local project may contain more generated files than the GitHub repository.

## Repository-safe outputs

Aggregate, privacy-safe evidence can be committed when it helps explain or reproduce the project. This includes:

- validation and data-quality summaries;
- cleaning and model-dataset summaries;
- temporal and feature-selection tables;
- model-comparison, tuning, calibration, robustness, and explainability summaries;
- aggregate representation or bias diagnostics;
- figures that do not expose individual-level records.

Module 4 figures used by the running experiment record are grouped under `figures/module4/`.

## Files that stay local

Do not commit:

- raw, interim, processed, or synthetic row-level datasets;
- extracts containing `msisdn` or another direct identifier;
- ad hoc row-level debugging exports;
- MLflow's local SQLite database (`mlflow.db`);
- logs or screenshots containing secrets, credentials, or unnecessary personal data.

The post-23 July synthetic scenarios are reproducible from `src/models/generate_post23_synthetic_scenarios.py`. Their row-level CSV stays local; the aggregate scenario summary and charts are sufficient repository evidence.

The privacy audit log is generated under `reports/privacy/`. It is designed not to contain raw identifiers, but it should still be checked before inclusion in a submission or commit.

## Source of truth

Generated reports show what happened in a run. The code that produced them lives under `src/`, while methodology, limitations, and decisions are documented under `docs/`.

For Module 4, the main interpretive record is:

`docs/model-documentation/module4_experiment_record.md`

This file links the experiments to the relevant tables and figures and should be treated as the first navigation point for modelling evidence.

## Module 4 fairness assessment

The formal Module 4 fairness artefact is `reports/module4_fairness_report.md`. It records why demographic fairness metrics cannot be validly calculated from the available data, separates fairness from operational robustness, and links to the supporting segment and sensitivity outputs.


## Module 4 validation evidence

The final Module 4 evidence is deliberately split between concise interpretive reports and reproducible tables/figures.

- `module4_fairness_report.md` — formal fairness-feasibility assessment, representation limits, operational robustness and residual risks.
- `module4_mlflow_evidence.md` — repository-safe summary of the local MLflow experiment history, including the tuning screenshot.
- `tables/module4_classification_metrics.csv` and `tables/module4_classification_threshold.json` — thresholded holdout accuracy, precision, recall, F1 and majority-class baseline.
- `tables/module4_counterfactual_explanations.csv` and `tables/module4_counterfactual_summary.json` — constrained contrastive explanations around the frozen operating threshold.
- `figures/module4/final_holdout_confusion_matrix.png` and `figures/module4/final_holdout_roc_curve.png` — conventional classification artefacts.
- `figures/module4/mlflow_model_tuning_runs.png` — direct MLflow Model training evidence showing tracked runs, metrics and parameters.

The complete modelling narrative remains in `docs/model-documentation/module4_experiment_record.md`.
