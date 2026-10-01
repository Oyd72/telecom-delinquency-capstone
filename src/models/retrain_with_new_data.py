"""Governed automated retraining for the telecom delinquency capstone.

The script demonstrates the final-project retraining requirement without silently
overwriting the frozen Module 4 model. A later labelled batch can be supplied as
"new data". The batch is split chronologically into an update portion and a validation
portion. A candidate model is retrained using the original trusted development data
plus the update portion, calibrated on a recent slice, evaluated on the untouched
validation portion, and logged to MLflow.

Passing quality gates produces a candidate artefact and MLflow run only. Promotion is
explicitly human-controlled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DATA = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"
DEFAULT_NEW_DATA = PROJECT_ROOT / "data/monitoring/final_holdout_labelled.csv"
POLICY_PATH = PROJECT_ROOT / "config/retraining_policy.yaml"
OUTPUT_DIR = PROJECT_ROOT / "models/candidates"
REPORT_PATH = PROJECT_ROOT / "reports/monitoring/latest_retraining_candidate.json"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

TARGET = "delinquent_5d"
DATE = "pdate"
ORIGINAL_DEVELOPMENT_END = pd.Timestamp("2016-07-13")
FEATURES = [
    "cnt_ma_rech90",
    "daily_decr30",
    "last_rech_date_ma",
    "sumamnt_ma_rech90",
    "aon",
    "last_rech_amt_ma",
    "daily_decr90",
    "sumamnt_ma_rech30",
    "medianamnt_ma_rech30",
    "medianmarechprebal90",
    "rental30",
    "cnt_ma_rech30",
]


def top_fraction_capture(y_true: np.ndarray, proba: np.ndarray, fraction: float = 0.20) -> float:
    n = max(1, int(np.ceil(len(proba) * fraction)))
    order = np.argsort(-proba)
    positives = np.asarray(y_true).sum()
    if positives == 0:
        return float("nan")
    return float(np.asarray(y_true)[order[:n]].sum() / positives)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy() -> dict:
    return yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))["retraining"]


def validate_columns(frame: pd.DataFrame, name: str) -> None:
    missing = sorted(set(FEATURES + [TARGET, DATE]) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a governed retraining candidate.")
    parser.add_argument("--base-data", default=str(DEFAULT_BASE_DATA))
    parser.add_argument("--new-data", default=str(DEFAULT_NEW_DATA))
    args = parser.parse_args()

    policy = load_policy()
    base = pd.read_csv(args.base_data)
    new = pd.read_csv(args.new_data)
    validate_columns(base, "Base data")
    validate_columns(new, "New data")

    base[DATE] = pd.to_datetime(base[DATE], errors="raise")
    new[DATE] = pd.to_datetime(new[DATE], errors="raise")
    new = new.sort_values(DATE).reset_index(drop=True)

    min_rows = int(policy["trigger"]["minimum_new_labelled_rows"])
    if len(new) < min_rows:
        raise ValueError(f"New labelled batch has {len(new):,} rows; at least {min_rows:,} are required.")

    validation_fraction = float(policy["evaluation"]["new_batch_validation_fraction"])
    split_index = int(np.floor(len(new) * (1 - validation_fraction)))
    if split_index <= 0 or split_index >= len(new):
        raise ValueError("New-data validation split is empty.")

    update_batch = new.iloc[:split_index].copy()
    validation = new.iloc[split_index:].copy()

    development = base.loc[base[DATE] <= ORIGINAL_DEVELOPMENT_END].copy()
    combined = pd.concat([development, update_batch], ignore_index=True).sort_values(DATE)

    # Keep a recent labelled slice exclusively for calibration.
    unique_dates = sorted(combined[DATE].dt.normalize().unique())
    if len(unique_dates) < 8:
        raise ValueError("Not enough dated observations for a 7-day calibration slice.")
    calibration_start = pd.Timestamp(unique_dates[-7])
    train = combined.loc[combined[DATE] < calibration_start].copy()
    calibration = combined.loc[combined[DATE] >= calibration_start].copy()

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[FEATURES])
    x_cal = imputer.transform(calibration[FEATURES])
    x_val = imputer.transform(validation[FEATURES])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=int(policy["evaluation"]["random_state"]),
    )
    model.fit(x_train, train[TARGET])

    raw_cal = model.predict_proba(x_cal)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_cal, calibration[TARGET].to_numpy())

    raw_val = model.predict_proba(x_val)[:, 1]
    calibrated_val = calibrator.predict(raw_val)

    metrics = {
        "roc_auc": float(roc_auc_score(validation[TARGET], calibrated_val)),
        "average_precision": float(average_precision_score(validation[TARGET], calibrated_val)),
        "brier_score": float(brier_score_loss(validation[TARGET], calibrated_val)),
        "top20_capture": top_fraction_capture(validation[TARGET].to_numpy(), calibrated_val),
    }

    gates = {
        "roc_auc": metrics["roc_auc"] >= float(policy["evaluation"]["minimum_roc_auc"]),
        "top20_capture": metrics["top20_capture"] >= float(policy["evaluation"]["minimum_top20_capture"]),
        "brier_score": metrics["brier_score"] <= float(policy["evaluation"]["maximum_brier_score"]),
    }
    passed = all(gates.values())

    package = {
        "artifact_version": "candidate-retrain",
        "model_name": "telecom_delinquency_random_forest_isotonic",
        "features": FEATURES,
        "imputer": imputer,
        "model": model,
        "calibrator": calibrator,
        "training_end": str(train[DATE].max().date()),
        "calibration_start": str(calibration[DATE].min().date()),
        "calibration_end": str(calibration[DATE].max().date()),
        "candidate_only": True,
        "automatic_promotion": False,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = OUTPUT_DIR / "retrained_candidate.joblib"
    joblib.dump(package, candidate_path)
    candidate_sha = sha256_file(candidate_path)

    tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("final_project_retraining")

    with mlflow.start_run(run_name="governed_retraining_candidate") as run:
        mlflow.log_params({
            "base_rows": len(development),
            "new_rows": len(new),
            "update_rows": len(update_batch),
            "validation_rows": len(validation),
            "feature_count": len(FEATURES),
            "automatic_promotion": False,
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "candidate_status": "passed_gates" if passed else "failed_gates",
            "human_approval_required": "true",
            "artifact_sha256": candidate_sha,
            "new_data_role": "chronologically later labelled batch",
        })
        mlflow.log_artifact(str(candidate_path), artifact_path="candidate_model")
        run_id = run.info.run_id

    summary = {
        "candidate_created": True,
        "candidate_only": True,
        "automatic_promotion": False,
        "human_approval_required": True,
        "new_data_rows": int(len(new)),
        "update_rows": int(len(update_batch)),
        "validation_rows": int(len(validation)),
        "validation_period": {
            "start": str(validation[DATE].min().date()),
            "end": str(validation[DATE].max().date()),
        },
        "metrics": metrics,
        "quality_gates": gates,
        "all_quality_gates_passed": passed,
        "candidate_artifact": str(candidate_path.relative_to(PROJECT_ROOT)),
        "candidate_sha256": candidate_sha,
        "mlflow_run_id": run_id,
        "promotion_decision": "requires human review; deployed model remains unchanged",
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Governed retraining run completed.")
    print(f"New labelled rows: {len(new):,}")
    print(f"Update rows: {len(update_batch):,}")
    print(f"Validation rows: {len(validation):,}")
    print()
    print("Candidate validation metrics")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    print()
    print("Quality gates")
    for key, value in gates.items():
        print(f"  {key}: {'PASS' if value else 'FAIL'}")
    print(f"All gates passed: {passed}")
    print("Automatic promotion: False")
    print(f"MLflow run ID: {run_id}")
    print(f"Summary written to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
