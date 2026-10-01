import pandas as pd
import pytest

from src.monitoring.performance_monitoring import evaluate_labelled_period


def test_labelled_period_returns_expected_keys():
    y = pd.Series([0, 0, 1, 1])
    p = pd.Series([0.1, 0.2, 0.7, 0.8])
    result = evaluate_labelled_period(
        y,
        p,
        0.5,
        period_name="final_holdout_2016_07_14_to_2016_07_23",
        labels_reliable=True,
    )
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["labels_reliable"] is True


def test_unreliable_or_unmatured_labels_are_blocked():
    with pytest.raises(ValueError, match="not permitted"):
        evaluate_labelled_period(
            pd.Series([0, 1, 0]),
            pd.Series([0.1, 0.8, 0.3]),
            0.1751,
            period_name="future_batch_labels_pending",
            labels_reliable=False,
        )
