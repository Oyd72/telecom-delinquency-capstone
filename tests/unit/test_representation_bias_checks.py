import pandas as pd

from src.monitoring.representation_bias_checks import evaluate_representation


def test_representation_diagnostics_use_only_available_operational_slice():
    processed = pd.DataFrame(
        {
            "delinquent_5d": [0, 1, 0, 1],
            "is_repeat_customer": [0, 0, 1, 1],
            "pdate": ["2016-06-01", "2016-06-02", "2016-07-01", "2016-07-02"],
        }
    )
    interim = pd.DataFrame({"pcircle": ["UPW", "UPW", "UPW", "UPW"]})

    table, summary = evaluate_representation(processed, interim)

    assert set(table["borrower_history_group"]) == {"first_time", "returning"}
    assert summary["protected_characteristics_available"] is False
    assert summary["operational_slice_is_protected_attribute"] is False
    assert summary["pcircle_check"]["usable_for_group_comparison"] is False
    assert summary["automatic_bias_conclusion"] is False


def test_representation_diagnostics_require_group_target_and_date_fields():
    processed = pd.DataFrame({"delinquent_5d": [0, 1]})

    try:
        evaluate_representation(processed)
    except ValueError as exc:
        assert "required columns" in str(exc)
    else:
        raise AssertionError("Expected missing required fields to raise ValueError")
