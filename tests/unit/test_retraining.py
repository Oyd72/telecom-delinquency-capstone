import pandas as pd
import pytest

from src.models.retrain_with_new_data import FEATURES, top_fraction_capture, validate_columns


def test_top_fraction_capture_returns_fraction_of_positives():
    y = pd.Series([1, 0, 1, 0, 1]).to_numpy()
    p = pd.Series([0.9, 0.8, 0.7, 0.2, 0.1]).to_numpy()
    result = top_fraction_capture(y, p, fraction=0.4)
    assert result == pytest.approx(1 / 3)


def test_validate_columns_rejects_missing_required_fields():
    frame = pd.DataFrame({feature: [0.0] for feature in FEATURES})
    with pytest.raises(ValueError):
        validate_columns(frame, "test")
