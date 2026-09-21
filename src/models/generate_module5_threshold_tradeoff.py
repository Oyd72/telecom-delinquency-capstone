"""Generate a privacy-safe threshold trade-off table for the Module 5 dashboard.

This script reuses the frozen Module 4 modelling specification and final holdout.
It writes only aggregate threshold metrics. No row-level predictions are committed.

Run locally from the repository root:
    python src/models/generate_module5_threshold_tradeoff.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.generate_module4_classification_artifacts import (
    DATA_PATH,
    DATE,
    TARGET,
    FEATURES,
    HOLDOUT_START,
    HOLDOUT_END,
    fit_predict_calibrated,
)

OUTPUT_PATH = PROJECT_ROOT / "reports" / "tables" / "module5_threshold_tradeoff.csv"

TRAIN_END = pd.Timestamp("2016-07-06")
CAL_START = pd.Timestamp("2016-07-07")
CAL_END = pd.Timestamp("2016-07-13")

THRESHOLDS = np.arange(0.05, 0.405, 0.01)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df[DATE] = pd.to_datetime(df[DATE], errors="raise")

    train = df.loc[df[DATE] <= TRAIN_END].copy()
    calibration = df.loc[
        (df[DATE] >= CAL_START) & (df[DATE] <= CAL_END)
    ].copy()
    holdout = df.loc[
        (df[DATE] >= HOLDOUT_START) & (df[DATE] <= HOLDOUT_END)
    ].copy()

    probabilities = fit_predict_calibrated(train, calibration, holdout)
    y_true = holdout[TARGET].to_numpy(dtype=int)

    rows = []
    for threshold in THRESHOLDS:
        pred = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "flagged_rate": float(pred.mean()),
            }
        )

    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("Module 5 threshold trade-off table created.")
    print(f"Rows: {len(result)}")
    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print(result.head().to_string(index=False))
    print()
    print(result.tail().to_string(index=False))


if __name__ == "__main__":
    main()
