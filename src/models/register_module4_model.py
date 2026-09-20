"""Register the selected Module 4 model in the local MLflow Model Registry.

Prerequisites:
- models/selected_random_forest_isotonic.joblib exists locally;
- mlflow.db exists locally;
- the packaged artefact has already passed the inference-contract tests.

The script:
1. logs the serialized selected model as an MLflow pyfunc model artefact;
2. creates a registered-model version;
3. assigns the alias "champion";
4. attempts to transition the version to the legacy "Production" stage for
   assignment compatibility, while also storing an explicit assignment-stage tag.

The local SQLite MLflow backend remains the registry store. The binary model can also
be committed separately as an assignment deliverable.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACT_PATH = PROJECT_ROOT / "models/selected_random_forest_isotonic.joblib"
METADATA_PATH = PROJECT_ROOT / "models/selected_model_metadata.json"
OUTPUT_PATH = PROJECT_ROOT / "reports/tables/module4_model_registry.json"

REGISTERED_MODEL_NAME = "telecom_delinquency_random_forest_isotonic"
EXPERIMENT_NAME = "module4_model_registry"


class TelecomDelinquencyPyfunc(PythonModel):
    """MLflow wrapper around the packaged imputer, Random Forest and calibrator."""

    def load_context(self, context):
        self.package = joblib.load(context.artifacts["package"])

    def predict(self, context, model_input, params=None):
        features = self.package["features"]
        missing = sorted(set(features) - set(model_input.columns))
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        frame = model_input.loc[:, features].copy()
        x = self.package["imputer"].transform(frame)
        raw = self.package["model"].predict_proba(x)[:, 1]
        calibrated = self.package["calibrator"].predict(raw)

        return pd.DataFrame(
            {
                "raw_delinquency_probability": raw,
                "calibrated_delinquency_probability": calibrated,
            },
            index=model_input.index,
        )


def main() -> None:
    if not MLFLOW_DB.exists():
        raise FileNotFoundError(f"MLflow database not found: {MLFLOW_DB}")
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Packaged model not found: {ARTIFACT_PATH}. "
            "Run src/models/package_selected_model.py first."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    package = joblib.load(ARTIFACT_PATH)
    features = package["features"]

    tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Aggregate-safe representative input: fitted training medians, not a customer row.
    median_input = pd.DataFrame(
        [dict(zip(features, package["imputer"].statistics_))],
        columns=features,
    )

    x = package["imputer"].transform(median_input)
    raw = package["model"].predict_proba(x)[:, 1]
    calibrated = package["calibrator"].predict(raw)
    example_output = pd.DataFrame(
        {
            "raw_delinquency_probability": raw,
            "calibrated_delinquency_probability": calibrated,
        }
    )
    signature = infer_signature(median_input, example_output)

    with mlflow.start_run(run_name="selected_model_registry_v1") as run:
        mlflow.log_params(
            {
                "artifact_version": metadata["artifact_version"],
                "model_family": "random_forest",
                "calibration_method": metadata["calibration_method"],
                "feature_count": metadata["feature_count"],
                "training_end": metadata["training_end"],
                "calibration_start": metadata["calibration_start"],
                "calibration_end": metadata["calibration_end"],
                "final_holdout_used_for_fitting": metadata[
                    "final_holdout_used_for_fitting"
                ],
            }
        )
        mlflow.set_tags(
            {
                "assignment_module": "4",
                "model_role": "formal_selected_model",
                "artifact_sha256": metadata["artifact_sha256"],
                "intended_use": "risk prioritisation support",
                "autonomous_credit_decision": "false",
            }
        )

        model_info = mlflow.pyfunc.log_model(
            name="selected_model",
            python_model=TelecomDelinquencyPyfunc(),
            artifacts={"package": str(ARTIFACT_PATH)},
            input_example=median_input,
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)

    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )
    matching = [
        version
        for version in versions
        if version.run_id == run_id
    ]
    if not matching:
        raise RuntimeError("Registered model version was not found for the new run.")

    model_version = max(matching, key=lambda item: int(item.version))
    version = str(model_version.version)

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "champion",
        version,
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "assignment_stage",
        "Production",
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "artifact_version",
        metadata["artifact_version"],
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "artifact_sha256",
        metadata["artifact_sha256"],
    )

    stage_status = "not_attempted"
    stage_error = None
    try:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        stage_status = "Production"
    except Exception as exc:  # modern MLflow deprecates model stages
        stage_status = "alias_and_tag_only"
        stage_error = str(exc)

    payload = {
        "tracking_backend": "sqlite",
        "registered_model_name": REGISTERED_MODEL_NAME,
        "registered_version": version,
        "alias": "champion",
        "assignment_stage_tag": "Production",
        "legacy_stage_status": stage_status,
        "legacy_stage_error": stage_error,
        "run_id": run_id,
        "model_uri": model_info.model_uri,
        "artifact_version": metadata["artifact_version"],
        "artifact_sha256": metadata["artifact_sha256"],
        "model_artifact_logged": True,
        "final_holdout_used_for_fitting": False,
        "note": (
            "MLflow model stages are deprecated in newer MLflow releases. "
            "The champion alias and explicit Production assignment-stage tag are the "
            "durable registry indicators; the script also attempts the legacy Production "
            "stage for assignment compatibility."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Module 4 MLflow Model Registry registration completed.")
    print(f"Registered model: {REGISTERED_MODEL_NAME}")
    print(f"Version: {version}")
    print("Alias: champion")
    print("Assignment-stage tag: Production")
    print(f"Legacy stage status: {stage_status}")
    print(f"Model artefact logged: True")
    print(f"Run ID: {run_id}")
    print(f"Model URI: {model_info.model_uri}")
    print(f"Summary written to: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    if stage_error:
        print()
        print("Legacy stage transition was not available in this MLflow version.")
        print("The champion alias and Production assignment-stage tag remain registered.")


if __name__ == "__main__":
    main()
