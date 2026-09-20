# Module 4 submission evidence map

This page is a navigation aid for the final Module 4 submission. It points to the repository evidence behind the concise written report.

| Assignment evidence | Repository location |
|---|---|
| Chronological split and validation design | `docs/model-documentation/module4_experiment_record.md` |
| Candidate models and tuning | `reports/tables/module4_candidate_model_summary.csv`, `reports/tables/module4_tuning_summary.csv` |
| MLflow tracking | `reports/module4_mlflow_evidence.md`, `reports/figures/module4/mlflow_model_tuning_runs.png` |
| Final holdout metrics | `reports/tables/module4_final_holdout_metrics.csv` |
| Precision, recall, F1, accuracy and threshold | `reports/tables/module4_classification_metrics.csv`, `reports/tables/module4_classification_threshold.json` |
| Confusion matrix and ROC curve | `reports/figures/module4/final_holdout_confusion_matrix.png`, `reports/figures/module4/final_holdout_roc_curve.png` |
| SHAP analysis notebook, global and local SHAP | `notebooks/module4_shap_analysis.ipynb`, `reports/tables/module4_shap_summary.json`, `reports/figures/module4/shap_global_importance.png`, local SHAP figures under `reports/figures/module4/` |
| Counterfactual-style explanations | `reports/tables/module4_counterfactual_summary.json`, `reports/figures/module4/counterfactual_risk_changes.png` |
| Fairness feasibility and robustness | `reports/module4_fairness_report.md` |
| Model card | `docs/model-documentation/model_card.md` |
| Packaged-model metadata and integrity hash | `models/selected_model_metadata.json` |
| Batch inference | `src/inference/predict_selected_model.py` |
| FastAPI `/predict` endpoint | `src/api/app.py`, `tests/unit/test_api.py` |
| Full experiment history | `docs/model-documentation/module4_experiment_record.md` |

Row-level project data, the binary model artefact and the MLflow SQLite database remain local by design.
