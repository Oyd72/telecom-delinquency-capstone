# Automated retraining

The final project includes a governed retraining mechanism implemented in
`src/models/retrain_with_new_data.py`.

The mechanism is intentionally **candidate-based**. New trustworthy labelled data can
trigger a retraining run, but a passing run does not automatically overwrite the
deployed model.

## Process

1. Validate the incoming labelled batch and feature contract.
2. Split the new batch chronologically into an update portion and a later validation portion.
3. Combine the update portion with the trusted historical development population.
4. Retrain the existing Random Forest architecture.
5. Refit isotonic calibration on the most recent seven-day labelled calibration slice.
6. Evaluate the candidate on the untouched later validation portion.
7. Log parameters, metrics, status and the candidate artefact to MLflow.
8. Apply policy quality gates.
9. Require human approval before any promotion or replacement of the deployed model.

The frozen version 1.0.0 model remains unchanged unless a separate governed promotion
decision is made.

In this academic project, the chronologically later final holdout is reused as the labelled batch for the retraining demonstration. This allows the retraining workflow, quality gates, MLflow tracking, and candidate-evaluation process to be demonstrated using available labelled data. Because the holdout is not genuinely new post-deployment data, the resulting retrained model is treated only as a candidate and not as a new production model.

Run locally after monitoring data have been prepared:

```powershell
python src/monitoring/prepare_monitoring_data.py
python src/models/retrain_with_new_data.py
```

Policy thresholds are defined in `config/retraining_policy.yaml`.
