import numpy as np
import pandas as pd

from src.data.clean_interim import _is_fractional, clean_dataframe


def test_is_fractional_distinguishes_integer_like_values():
    series = pd.Series([0, 1.0, 2.5, np.nan, 3.0000000000001])
    mask = _is_fractional(series)

    assert mask.tolist() == [False, False, True, False, False]


def test_clean_dataframe_applies_high_confidence_rules_and_audits_changes():
    raw = pd.DataFrame(
        {
            "msisdn": [111, 222, 222],
            "aon": [-5.0, 600000.0, 600000.0],
            "last_rech_date_ma": [10.0, 2.0, 2.0],
            "last_rech_date_da": [1.0, 1.0, 1.0],
            "fr_ma_rech30": [1.0, 1.0, 1.0],
            "fr_da_rech30": [1.0, 1.0, 1.0],
            "cnt_da_rech30": [1.5, 2.0, 2.0],
            "cnt_loans90": [3.0, 4.0, 4.0],
        }
    )

    cleaned, audit, summary = clean_dataframe(
        raw,
        run_id="test_run",
        run_timestamp="2026-09-15T00:00:00+00:00",
    )

    assert pd.isna(cleaned.iloc[0]["aon"])
    assert pd.isna(cleaned.iloc[1]["aon"])
    assert pd.isna(cleaned.iloc[0]["cnt_da_rech30"])
    assert str(cleaned["cnt_da_rech30"].dtype) == "Int64"
    assert summary["rows_removed"] == 1
    assert summary["output_rows"] == 2
    assert summary["rules"]["aon:negative_duration"] == 1
    assert summary["rules"]["aon:separated_contamination_regime"] == 2
    assert summary["rules"]["cnt_da_rech30:fractional_count"] == 1
    assert summary["rules"]["exact_duplicate:removed"] == 1
    assert "msisdn" not in audit.columns
    assert set(audit["rule_triggered"]) >= {
        "negative_duration",
        "separated_contamination_regime",
        "fractional_count",
        "exact_duplicate",
    }
