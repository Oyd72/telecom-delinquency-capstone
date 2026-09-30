# Monitoring evidence

This directory is reserved for final-project model-monitoring evidence.

## Monitoring periods

- **Development reference:** 1 June–13 July 2016.
- **Final labelled evaluation:** 14–23 July 2016.
- **Post-23-July diagnostic period:** feature and prediction distributions may be used for
  label-free drift analysis only.

The post-23-July outcomes are all labelled as successful repayment. Because this abrupt
label regime change is unexplained, those labels are **not treated as ground truth**.
Accordingly, the project does not calculate accuracy, precision, recall, calibration,
outcome-based fairness, or other supervised performance measures from that period.

## Planned outputs

- `latest_monitoring_status.json` — machine-readable current status and triggered metrics.
- `historical_monitoring_metrics.csv` — compact audit trail across monitoring runs.
- Evidently AI monitoring artefacts will be added after the core calculations are verified.
- The separate Streamlit monitoring dashboard will consume these outputs but will remain
  distinct from the existing stakeholder/XAI dashboard.

## Fairness boundary

Protected demographic attributes are not available in the dataset. Operational segment
monitoring must not be presented as demographic fairness evidence. Fairlearn and Evidently
AI will be used only for analyses supported by the available data, with this limitation
displayed explicitly.
