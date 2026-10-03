# Monitoring evidence

This directory contains final-project model-monitoring evidence.

## Monitoring demonstration

Monitoring is demonstrated using populations already established in the model-development workflow:

- **Development reference:** 1 June–13 July 2016.
- **Final holdout:** 14–23 July 2016.

The final holdout is chronologically later than the development reference. It supports both:

- feature and prediction-score drift monitoring; and
- outcome-based performance and calibration monitoring.

The dashboard presents the project's custom Python monitoring results and Evidently AI results as separate evidence. Agreement and disagreement remain visible rather than being collapsed into a single combined drift verdict.

## Operational interpretation

In a real deployment, the same monitoring components would compare future scored batches with an approved reference population. Label-free drift checks could run immediately. Performance, calibration and outcome-based error analysis would run only after trustworthy outcomes had matured.

## Outputs

- `latest_monitoring_status.json` — machine-readable status and triggered metrics.
- `historical_monitoring_metrics.csv` — compact audit trail.
- `monitoring_population_summary.json` — reference/comparison population metadata.
- `feature_drift_final_holdout.csv` — custom project PSI, KS and missingness evidence.
- `evidently_final_holdout_drift.html` and `.json` — independent Evidently AI drift evidence.
- `dashboards/monitoring_dashboard.py` — separate Streamlit governance-facing monitoring dashboard.

## Fairness boundary

Protected demographic attributes are not available in the dataset. Operational segment monitoring must not be presented as demographic fairness evidence. Fairlearn and Evidently AI are used only where the available evidence supports their interpretation, with this limitation shown explicitly.
