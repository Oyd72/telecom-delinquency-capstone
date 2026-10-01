"""Reconcile project PSI and Evidently AI drift evidence.

This script compares both implementations on the same development-reference and
final-holdout populations.

Classification:
- confirmed: both implementations meet/exceed the 0.10 PSI threshold;
- project_only: only the project implementation meets/exceeds the threshold;
- evidently_only: only Evidently meets/exceeds the threshold;
- no_drift: neither implementation meets/exceeds the threshold.

Differences remain visible because PSI can vary with binning and implementation details.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DRIFT_PATH = PROJECT_ROOT / "reports/monitoring/feature_drift_final_holdout.csv"
PROJECT_STATUS_PATH = PROJECT_ROOT / "reports/monitoring/latest_monitoring_status.json"
EVIDENTLY_JSON_PATH = PROJECT_ROOT / "reports/monitoring/evidently_final_holdout_drift.json"

OUTPUT_PATH = PROJECT_ROOT / "reports/monitoring/reconciled_drift_evidence.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports/monitoring/reconciled_drift_summary.json"

DRIFT_THRESHOLD = 0.10
PREDICTION_COLUMN = "calibrated_delinquency_probability"


def extract_evidently_psi(payload: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in payload.get("metrics", []):
        config = metric.get("config") or {}
        if config.get("type") != "evidently:metric_v2:ValueDrift":
            continue
        column = config.get("column")
        value = metric.get("value")
        if column is not None and isinstance(value, (int, float)):
            values[str(column)] = float(value)
    return values


def classify(project_psi: float, evidently_psi: float) -> str:
    project_drift = project_psi >= DRIFT_THRESHOLD
    evidently_drift = evidently_psi >= DRIFT_THRESHOLD
    if project_drift and evidently_drift:
        return "confirmed"
    if project_drift:
        return "project_only"
    if evidently_drift:
        return "evidently_only"
    return "no_drift"


def main() -> int:
    required = [PROJECT_DRIFT_PATH, PROJECT_STATUS_PATH, EVIDENTLY_JSON_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required drift evidence is missing:\n- " + "\n- ".join(missing)
        )

    project = pd.read_csv(PROJECT_DRIFT_PATH)
    project_status = json.loads(PROJECT_STATUS_PATH.read_text(encoding="utf-8"))
    evidently = json.loads(EVIDENTLY_JSON_PATH.read_text(encoding="utf-8"))
    evidently_psi = extract_evidently_psi(evidently)

    rows = []
    for _, row in project.iterrows():
        feature = str(row["feature"])
        if feature not in evidently_psi:
            raise ValueError(f"Evidently result missing feature: {feature}")
        ppsi = float(row["psi"])
        epsi = float(evidently_psi[feature])
        rows.append({
            "column": feature,
            "column_type": "model_feature",
            "project_psi": ppsi,
            "evidently_psi": epsi,
            "project_drift_at_0_10": ppsi >= DRIFT_THRESHOLD,
            "evidently_drift_at_0_10": epsi >= DRIFT_THRESHOLD,
            "evidence_classification": classify(ppsi, epsi),
        })

    if PREDICTION_COLUMN not in evidently_psi:
        raise ValueError("Evidently result missing calibrated prediction-score drift.")

    project_prediction_psi = float(project_status["prediction_drift"]["psi"])
    evidently_prediction_psi = float(evidently_psi[PREDICTION_COLUMN])
    rows.append({
        "column": PREDICTION_COLUMN,
        "column_type": "model_output",
        "project_psi": project_prediction_psi,
        "evidently_psi": evidently_prediction_psi,
        "project_drift_at_0_10": project_prediction_psi >= DRIFT_THRESHOLD,
        "evidently_drift_at_0_10": evidently_prediction_psi >= DRIFT_THRESHOLD,
        "evidence_classification": classify(
            project_prediction_psi, evidently_prediction_psi
        ),
    })

    reconciled = pd.DataFrame(rows)
    counts = reconciled["evidence_classification"].value_counts().to_dict()

    summary = {
        "comparison": (
            "Project PSI implementation versus Evidently AI PSI on the same "
            "development-reference and final-holdout populations."
        ),
        "drift_threshold": DRIFT_THRESHOLD,
        "post_23_july_data_used": False,
        "interpretation": {
            "confirmed": "Both implementations meet or exceed the PSI threshold.",
            "project_only": (
                "Only the project PSI implementation meets or exceeds the threshold; "
                "the signal is treated as method-sensitive."
            ),
            "evidently_only": (
                "Only Evidently AI meets or exceeds the threshold; "
                "the signal is treated as method-sensitive."
            ),
            "no_drift": "Neither implementation meets or exceeds the PSI threshold.",
        },
        "counts": {
            "confirmed": int(counts.get("confirmed", 0)),
            "project_only": int(counts.get("project_only", 0)),
            "evidently_only": int(counts.get("evidently_only", 0)),
            "no_drift": int(counts.get("no_drift", 0)),
            "total_columns": int(len(reconciled)),
        },
        "confirmed_columns": reconciled.loc[
            reconciled["evidence_classification"] == "confirmed", "column"
        ].tolist(),
        "method_sensitive_columns": reconciled.loc[
            reconciled["evidence_classification"].isin(["project_only", "evidently_only"]),
            "column",
        ].tolist(),
        "note": (
            "Differences between PSI values are expected because implementations may use "
            "different binning and edge-handling rules. Agreement is treated as convergent "
            "evidence; disagreement is retained visibly rather than resolved by selecting "
            "the larger value."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reconciled.to_csv(OUTPUT_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Drift evidence reconciliation completed.")
    print()
    print(reconciled.sort_values(
        ["evidence_classification", "project_psi"],
        ascending=[True, False],
    ).to_string(index=False))
    print()
    print("Classification counts")
    for key, value in summary["counts"].items():
        print(f"  {key}: {value}")
    print()
    print("Confirmed drift columns")
    for column in summary["confirmed_columns"]:
        print(f"  - {column}")
    print()
    print("Method-sensitive columns")
    for column in summary["method_sensitive_columns"]:
        print(f"  - {column}")
    print()
    print(f"Reconciled table written to: {OUTPUT_PATH}")
    print(f"Summary written to: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
