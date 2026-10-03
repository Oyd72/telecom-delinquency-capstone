import numpy as np
import pandas as pd

from src.monitoring.drift_metrics import (
    categorical_share_drift,
    ks_statistic,
    numeric_feature_drift,
    population_stability_index,
)


def test_identical_distribution_has_zero_psi_and_ks():
    s = pd.Series([1, 2, 3, 4, 5] * 20)
    assert abs(population_stability_index(s, s)) < 1e-12
    assert ks_statistic(s, s) == 0.0


def test_shifted_distribution_is_detected():
    ref = pd.DataFrame({"x": np.arange(100, dtype=float)})
    cur = pd.DataFrame({"x": np.arange(100, dtype=float) + 100.0})
    result = numeric_feature_drift(ref, cur, ["x"])
    assert result.loc[0, "psi"] > 0
    assert result.loc[0, "ks_statistic"] > 0.5


def test_categorical_share_drift_preserves_all_categories():
    ref = pd.Series(["a", "a", "b"])
    cur = pd.Series(["b", "c", "c"])
    out = categorical_share_drift(ref, cur)
    assert set(out["category"]) == {"a", "b", "c"}
