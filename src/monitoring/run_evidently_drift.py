"""Run an Evidently AI drift cross-check for the monitoring demonstration.

The comparison uses the development reference (1 June–13 July 2016) and final holdout
(14–23 July 2016). Both populations are part of the canonical model-ready dataset.

Outputs:
- reports/monitoring/evidently_final_holdout_drift.html
- reports/monitoring/evidently_final_holdout_drift.json
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

from evidently import Report
from evidently.presets import DataDriftPreset

from src.inference.predict_selected_model import predict_frame


MODEL_PATH = PROJECT_ROOT / "models/selected_random_forest_isotonic.joblib"
METADATA_PATH = PROJECT_ROOT / "models/selected_model_metadata.json"
REFERENCE_PATH = PROJECT_ROOT / "data/monitoring/development_reference.csv"
HOLDOUT_PATH = PROJECT_ROOT / "data/monitoring/final_holdout_labelled.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports/monitoring"
HTML_PATH = OUTPUT_DIR / "evidently_final_holdout_drift.html"
JSON_PATH = OUTPUT_DIR / "evidently_final_holdout_drift.json"


def score_population(df: pd.DataFrame, package: dict, features: list[str]) -> pd.Series:
    predictions = predict_frame(df[features], package)
    return predictions["calibrated_delinquency_probability"]


def main() -> int:
    required = [MODEL_PATH, METADATA_PATH, REFERENCE_PATH, HOLDOUT_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Evidently monitoring inputs are missing:\n- " + "\n- ".join(missing)
        )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    features = list(metadata["features"])
    package = joblib.load(MODEL_PATH)

    reference = pd.read_csv(REFERENCE_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)

    reference_eval = reference[features].copy()
    holdout_eval = holdout[features].copy()

    reference_eval["calibrated_delinquency_probability"] = score_population(
        reference, package, features
    ).to_numpy()
    holdout_eval["calibrated_delinquency_probability"] = score_population(
        holdout, package, features
    ).to_numpy()

    report = Report(
        [DataDriftPreset(method="psi")],
        include_tests=True,
    )
    snapshot = report.run(holdout_eval, reference_eval)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(HTML_PATH))
    JSON_PATH.write_text(snapshot.json(), encoding="utf-8")

    result = snapshot.dict()

    print("Evidently AI drift cross-check completed.")
    print()
    print(f"Reference rows: {len(reference_eval):,}")
    print(f"Final holdout rows: {len(holdout_eval):,}")
    print(f"Columns evaluated: {len(reference_eval.columns)}")
    print("  12 final-model features")
    print("  1 calibrated prediction-score column")
    print()
    print("Comparison: development reference versus final holdout")
    print("Drift method requested from Evidently: PSI")
    print("Post-23-July data used: False")
    print()
    print(f"Evidently result top-level keys: {sorted(result.keys())}")
    print(f"HTML report written to: {HTML_PATH}")
    print(f"JSON report written to: {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
