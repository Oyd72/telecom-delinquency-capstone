# Module 4 MLflow evidence

The Module 4 modelling work was tracked locally with MLflow using a SQLite backend.
The database itself is intentionally excluded from GitHub, while this report and the
aggregate run export provide a repository-safe record of the experiments.

## Experiment coverage

| Experiment | Runs |
|---|---:|
| module4_candidate_model_benchmark | 9 |
| module4_model_tuning | 24 |
| module4_final_holdout_evaluation | 6 |
| module4_development_calibration | 18 |
| module4_calibrated_holdout_sensitivity | 4 |

**Total tracked runs:** 61

![MLflow experiment run counts](figures/module4/mlflow_experiment_run_counts.png)

## What was tracked

The runs record model and calibration choices, feature counts, chronology controls,
and the evaluation metrics produced by each stage. The exact fields vary by experiment
because benchmark, tuning, calibration and holdout evaluation answer different questions.

The full repository-safe export is available in:

- `reports/tables/module4_mlflow_runs.csv`
- `reports/tables/module4_mlflow_experiments.json`

For an assignment screenshot, the local MLflow UI can be opened with:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open `http://127.0.0.1:5000` and select a Module 4 experiment.
A screenshot should show the experiment name and the visible run/metric table.

## Interpretation

MLflow is used here for traceability rather than as a source of model truth. The formal
results remain the versioned tables, figures, model card and experiment record in GitHub.
