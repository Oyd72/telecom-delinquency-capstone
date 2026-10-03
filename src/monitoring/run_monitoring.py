"""Run end-to-end monitoring calculations for the final-project demonstration.

The development period (1 June–13 July 2016) is the monitoring reference. The final
holdout (14–23 July 2016) is the chronologically later comparison population and also has
trusted labels, allowing both drift monitoring and performance/calibration monitoring.

No post-23-July records are used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.inference.predict_selected_model import predict_frame
from src.monitoring.drift_metrics import numeric_feature_drift, prediction_drift
from src.monitoring.performance_monitoring import evaluate_labelled_period


MODEL_PATH = PROJECT_ROOT / "models/selected_random_forest_isotonic.joblib"
METADATA_PATH = PROJECT_ROOT / "models/selected_model_metadata.json"
REFERENCE_PATH = PROJECT_ROOT / "data/monitoring/development_reference.csv"
HOLDOUT_PATH = PROJECT_ROOT / "data/monitoring/final_holdout_labelled.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports/monitoring"
STATUS_PATH = OUTPUT_DIR / "latest_monitoring_status.json"
FEATURE_DRIFT_PATH = OUTPUT_DIR / "feature_drift_final_holdout.csv"
HISTORY_PATH = OUTPUT_DIR / "historical_monitoring_metrics.csv"

OPERATING_THRESHOLD = 0.175141


def score_population(df: pd.DataFrame, package: dict, features: list[str]) -> pd.Series:
    predictions = predict_frame(df[features], package)
    return predictions["calibrated_delinquency_probability"]


def main() -> int:
    required = [MODEL_PATH, METADATA_PATH, REFERENCE_PATH, HOLDOUT_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required monitoring inputs are missing:\n- " + "\n- ".join(missing)
        )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    features = list(metadata["features"])
    package = joblib.load(MODEL_PATH)

    reference = pd.read_csv(REFERENCE_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)

    reference_scores = score_population(reference, package, features)
    holdout_scores = score_population(holdout, package, features)

    feature_drift = numeric_feature_drift(reference, holdout, features)
    score_drift = prediction_drift(reference_scores, holdout_scores)

    holdout_performance = evaluate_labelled_period(
        holdout["delinquent_5d"],
        holdout_scores,
        OPERATING_THRESHOLD,
        period_name="final_holdout_2016_07_14_to_2016_07_23",
        labels_reliable=True,
    )

    severities = set(feature_drift["severity"].astype(str))
    severities.add(str(score_drift["severity"]))
    if "escalate" in severities:
        overall_status = "escalate"
    elif "watch" in severities:
        overall_status = "watch"
    else:
        overall_status = "ok"

    status = {
        "model_name": metadata["model_name"],
        "artifact_version": metadata["artifact_version"],
        "artifact_sha256": metadata["artifact_sha256"],
        "operating_threshold": OPERATING_THRESHOLD,
        "reference_period": "2016-06-01 through 2016-07-13",
        "comparison_period": "2016-07-14 through 2016-07-23",
        "monitoring_mode": "historical_demonstration",
        "reference_rows": int(len(reference)),
        "comparison_rows": int(len(holdout)),
        "overall_drift_status": overall_status,
        "features_escalate": feature_drift.loc[
            feature_drift["severity"] == "escalate", "feature"
        ].tolist(),
        "features_watch": feature_drift.loc[
            feature_drift["severity"] == "watch", "feature"
        ].tolist(),
        "prediction_drift": score_drift,
        "final_holdout_performance": holdout_performance,
        "future_monitoring_note": (
            "In operational use the same monitoring logic would compare future scored "
            "batches with the approved reference population. Performance metrics would "
            "be calculated only after trustworthy outcome labels mature."
        ),
        "fairness_note": (
            "Protected demographic attributes are unavailable. Operational drift and "
            "robustness monitoring must not be presented as demographic fairness evidence."
        ),
        "post_23_july_data_used": False,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_drift.to_csv(FEATURE_DRIFT_PATH, index=False)
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")

    history_row = pd.DataFrame(
        [{
            "run_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "monitoring_period": status["comparison_period"],
            "monitoring_mode": status["monitoring_mode"],
            "overall_drift_status": overall_status,
            "labels_reliable": True,
            "notes": "Historical monitoring demonstration using the final holdout.",
        }]
    )
    history_row.to_csv(HISTORY_PATH, index=False)

    print("Monitoring run completed.")
    print()
    print("Final holdout performance")
    for key in [
        "roc_auc", "average_precision", "precision", "recall",
        "f1", "brier_score", "ece_10bin", "top20_capture",
    ]:
        print(f"  {key}: {holdout_performance[key]:.6f}")

    print()
    print("Development-reference versus final-holdout monitoring")
    print(f"  reference rows: {len(reference):,}")
    print(f"  comparison rows: {len(holdout):,}")
    print(f"  overall drift status: {overall_status}")
    print(f"  prediction PSI: {score_drift['psi']:.6f}")
    print(f"  prediction KS: {score_drift['ks_statistic']:.6f}")
    print(f"  reference mean risk: {score_drift['reference_mean']:.6f}")
    print(f"  holdout mean risk: {score_drift['current_mean']:.6f}")
    print()
    print("Feature drift summary")
    print(
        feature_drift[
            ["feature", "psi", "ks_statistic", "missingness_change_pp", "severity"]
        ].sort_values(["severity", "psi"], ascending=[True, False]).to_string(index=False)
    )
    print()
    print(f"Status written to: {STATUS_PATH}")
    print(f"Feature drift written to: {FEATURE_DRIFT_PATH}")
    print(f"History written to: {HISTORY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
