"""Run an Evidently AI drift cross-check on the final-project monitoring populations.

The comparison uses exactly the same development reference and post-23-July diagnostic
populations as the in-repository monitoring calculations. The diagnostic dataset contains
no outcome label. Evidently is therefore used here only for label-free feature and
prediction-score drift analysis.

Outputs:
- reports/monitoring/evidently_post_23_july_drift.html
- reports/monitoring/evidently_post_23_july_drift.json
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
DIAGNOSTIC_PATH = PROJECT_ROOT / "data/monitoring/post_23_july_diagnostic.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports/monitoring"
HTML_PATH = OUTPUT_DIR / "evidently_post_23_july_drift.html"
JSON_PATH = OUTPUT_DIR / "evidently_post_23_july_drift.json"


def score_population(df: pd.DataFrame, package: dict, features: list[str]) -> pd.Series:
    predictions = predict_frame(df[features], package)
    return predictions["calibrated_delinquency_probability"]


def main() -> int:
    required = [MODEL_PATH, METADATA_PATH, REFERENCE_PATH, DIAGNOSTIC_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Evidently monitoring inputs are missing:\n- " + "\n- ".join(missing)
        )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    features = list(metadata["features"])
    package = joblib.load(MODEL_PATH)

    reference = pd.read_csv(REFERENCE_PATH)
    diagnostic = pd.read_csv(DIAGNOSTIC_PATH)

    if "delinquent_5d" in diagnostic.columns or "label" in diagnostic.columns:
        raise AssertionError(
            "The post-23-July diagnostic dataset must not contain an outcome label."
        )

    reference_eval = reference[features].copy()
    diagnostic_eval = diagnostic[features].copy()

    reference_eval["calibrated_delinquency_probability"] = score_population(
        reference, package, features
    ).to_numpy()
    diagnostic_eval["calibrated_delinquency_probability"] = score_population(
        diagnostic, package, features
    ).to_numpy()

    # PSI is selected explicitly so the Evidently cross-check is directly comparable
    # with the project's own PSI calculations. Evidently still applies its own report
    # structure and test logic.
    report = Report(
        [DataDriftPreset(method="psi")],
        include_tests=True,
    )
    snapshot = report.run(diagnostic_eval, reference_eval)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(HTML_PATH))
    JSON_PATH.write_text(snapshot.json(), encoding="utf-8")

    result = snapshot.dict()

    print("Evidently AI drift cross-check completed.")
    print()
    print(f"Reference rows: {len(reference_eval):,}")
    print(f"Diagnostic rows: {len(diagnostic_eval):,}")
    print(f"Columns evaluated: {len(reference_eval.columns)}")
    print("  12 final-model features")
    print("  1 calibrated prediction-score column")
    print()
    print("Post-23-July labels treated as ground truth: False")
    print("Analysis type: label-free feature and prediction-score drift only")
    print("Drift method requested from Evidently: PSI")
    print()
    print(f"Evidently result top-level keys: {sorted(result.keys())}")
    print(f"HTML report written to: {HTML_PATH}")
    print(f"JSON report written to: {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
