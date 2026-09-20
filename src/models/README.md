# Modelling scripts

This directory contains reproducible modelling and diagnostic scripts. They are kept flat rather than moved into nested subfolders because several scripts derive the project root from their current path. The index below is the navigation layer.

## Module 4 core modelling

### Dataset and feature checks

- `validate_module4_split.py` — freezes and validates the chronological development / holdout split.
- `review_module4_features.py` — checks the selected predictors for leakage, identifiers, duplicates, missingness, and obvious fairness-name flags.

### Candidate models and tuning

- `benchmark_module4_models.py` — compares logistic regression, Random Forest, and XGBoost.
- `tune_module4_models.py` — evaluates a small, controlled set of Random Forest and XGBoost configurations.
- `evaluate_module4_holdout.py` — performs the independent final-holdout evaluation.

### Calibration

- `assess_module4_calibration.py` — compares uncalibrated, Platt, and isotonic probabilities using development data only.
- `confirm_module4_calibration_holdout.py` — applies the frozen calibration method to the already-opened holdout as sensitivity evidence.

### Explainability and robustness

- `explain_module4_random_forest.py` — global and local SHAP analysis for the selected Random Forest.
- `assess_module4_fairness_robustness.py` — fairness-feasibility statement, operational subgroup robustness, and controlled feature perturbations.
- `analyze_post23_feature_quality.py` — examines post-23 July feature drift and score behaviour without using the later labels as ground truth.
- `generate_post23_synthetic_scenarios.py` — creates clearly labelled scenario-only synthetic outcomes from the real later-period predictors.

### Feature-set sensitivity and simplification

These scripts document the later feature-dependence investigation. They are retained because each answers a different question rather than representing duplicate implementations.

- `compare_temporal_light_variants.py` — compares 12-, 10-, and 7-feature specifications.
- `compare_dev_12_vs_7.py` — checks the 12- vs 7-feature comparison on development-only chronological folds.
- `incremental_addback_analysis.py` — adds the five excluded predictors back individually and in thematic groups.
- `check_decr30_decr90_redundancy.py` — tests whether the 30- and 90-day decrement features are redundant.
- `compare_model_8_vs_12.py` — direct comparison of the compact 8-feature challenger and the selected 12-feature model.

### Visual utility

- `plot_module4_experiments.py` — regenerates a small set of benchmark, tuning, and holdout summary figures from saved tables.

## Earlier modelling diagnostics retained for traceability

The following scripts pre-date the final Module 4 modelling sequence and remain because they document the feature-selection and temporal-analysis path used to reach the current specification:

- `compare_feature_variants.py`
- `temporal_calibration.py`
- `rolling_calibration_stability.py`

They should not be confused with the current final-model workflow.


### Packaging, assignment artefacts and experiment evidence

- `package_selected_model.py` — recreates and packages the selected Random Forest, fitted imputer and isotonic calibrator, and writes repository-safe metadata with a SHA-256 fingerprint.
- `generate_module4_classification_artifacts.py` — freezes a development-selected operating threshold and produces precision, recall, F1, accuracy, confusion-matrix and ROC evidence on the final holdout.
- `generate_module4_counterfactuals.py` — generates constrained counterfactual-style explanations using observed-range candidate values and the frozen threshold.
- `export_module4_mlflow_evidence.py` — exports repository-safe evidence from the local MLflow SQLite tracking database.
