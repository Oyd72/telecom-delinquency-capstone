"""Package the selected Module 4 Random Forest and isotonic calibrator.

This script recreates the frozen selected model using the chronology already documented
in Module 4:

- Random Forest training: through 6 July 2016
- Isotonic calibration: 7-13 July 2016
- Final holdout: not used for fitting

The resulting local artefact contains:
- fitted median imputer;
- fitted Random Forest;
- fitted isotonic calibrator;
- ordered feature contract;
- training/calibration dates;
- model metadata.

The binary artefact stays local under models/. A privacy-safe metadata JSON is written
alongside it and is allowed to be committed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"

MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_PATH = MODEL_DIR / "selected_random_forest_isotonic.joblib"
METADATA_PATH = MODEL_DIR / "selected_model_metadata.json"

TARGET = "delinquent_5d"
DATE = "pdate"

TRAIN_END = pd.Timestamp("2016-07-06")
CALIBRATION_START = pd.Timestamp("2016-07-07")
CALIBRATION_END = pd.Timestamp("2016-07-13")

RANDOM_STATE = 42

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df[DATE] = pd.to_datetime(df[DATE], errors="raise")

    train = df.loc[df[DATE] <= TRAIN_END].copy()
    calibration = df.loc[
        (df[DATE] >= CALIBRATION_START)
        & (df[DATE] <= CALIBRATION_END)
    ].copy()

    if train.empty or calibration.empty:
        raise ValueError("Training or calibration partition is empty.")

    missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[FEATURES])
    x_cal = imputer.transform(calibration[FEATURES])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, train[TARGET])

    raw_cal = model.predict_proba(x_cal)[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_cal, calibration[TARGET].to_numpy())

    package = {
        "artifact_version": "1.0.0",
        "model_name": "telecom_delinquency_random_forest_isotonic",
        "features": FEATURES,
        "target_definition": "delinquent_5d = 1 when repayment was not completed within 5 days",
        "imputer": imputer,
        "model": model,
        "calibrator": calibrator,
        "training_end": str(TRAIN_END.date()),
        "calibration_start": str(CALIBRATION_START.date()),
        "calibration_end": str(CALIBRATION_END.date()),
        "random_state": RANDOM_STATE,
        "model_parameters": {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, ARTIFACT_PATH)

    metadata = {
        "artifact_version": package["artifact_version"],
        "model_name": package["model_name"],
        "artifact_filename": ARTIFACT_PATH.name,
        "artifact_sha256": sha256_file(ARTIFACT_PATH),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "target_definition": package["target_definition"],
        "training_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "training_end": package["training_end"],
        "calibration_start": package["calibration_start"],
        "calibration_end": package["calibration_end"],
        "final_holdout_used_for_fitting": False,
        "random_state": RANDOM_STATE,
        "model_parameters": package["model_parameters"],
        "calibration_method": "isotonic",
        "selected_model_status": (
            "Formal Module 4 selected model. The 8-feature model remains a documented challenger."
        ),
        "known_limitations": [
            "Available historical period is short.",
            "Final holdout showed temporal performance degradation beyond the pre-defined stability threshold.",
            "Demographic group-fairness metrics are not supported by the available data.",
            "Model is intended for risk prioritisation support, not autonomous credit decisions.",
        ],
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Selected Module 4 model package created.")
    print(f"Training rows: {len(train):,}")
    print(f"Calibration rows: {len(calibration):,}")
    print("Final holdout used for fitting: False")
    print(f"Binary artefact: {ARTIFACT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Metadata: {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"SHA-256: {metadata['artifact_sha256']}")


if __name__ == "__main__":
    main()
