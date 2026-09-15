import pandas as pd
import pytest

from src.features.build_model_dataset import (
    APPROVED_PREDICTORS,
    TARGET_FIELD,
    build_model_dataset,
)


def _base_rows() -> pd.DataFrame:
    rows = [
        {
            "msisdn": 1001,
            "pdate": "01-06-2016",
            "label": 1,
        },
        {
            "msisdn": 1001,
            "pdate": "01-06-2016",
            "label": 0,
        },
        {
            "msisdn": 1001,
            "pdate": "05-06-2016",
            "label": 1,
        },
        {
            "msisdn": 2002,
            "pdate": "24-07-2016",
            "label": 1,
        },
    ]

    source_predictors = [
        feature
        for feature in APPROVED_PREDICTORS
        if feature not in {"prior_tx_count", "is_repeat_customer"}
    ]
    for row_number, row in enumerate(rows, start=1):
        for feature in source_predictors:
            row[feature] = float(row_number)
    return pd.DataFrame(rows)


def test_build_model_dataset_enforces_cutoff_target_and_identifier_removal():
    model_ready, summary = build_model_dataset(_base_rows())

    assert len(model_ready) == 3
    assert summary["post_cutoff_rows_excluded"] == 1
    assert "msisdn" not in model_ready.columns
    assert "label" not in model_ready.columns
    assert TARGET_FIELD in model_ready.columns
    assert model_ready[TARGET_FIELD].tolist() == [0, 1, 0]


def test_prior_transaction_features_use_strictly_earlier_dates():
    model_ready, _ = build_model_dataset(_base_rows())

    same_day_rows = model_ready.loc[model_ready["pdate"] == pd.Timestamp("2016-06-01")]
    later_row = model_ready.loc[model_ready["pdate"] == pd.Timestamp("2016-06-05")].iloc[0]

    assert same_day_rows["prior_tx_count"].tolist() == [0, 0]
    assert same_day_rows["is_repeat_customer"].tolist() == [0, 0]
    assert later_row["prior_tx_count"] == 2
    assert later_row["is_repeat_customer"] == 1


def test_build_model_dataset_rejects_invalid_labels():
    invalid = _base_rows()
    invalid.loc[0, "label"] = 2

    with pytest.raises(ValueError, match="invalid label"):
        build_model_dataset(invalid)
