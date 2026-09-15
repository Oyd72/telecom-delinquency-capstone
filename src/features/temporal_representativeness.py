"""Assess temporal representativeness and drift between development and holdout data.

This diagnostic accompanies Stage 2 feature screening. It does not assume that an
earlier temporal development subset is representative of the later holdout. Instead,
it measures differences explicitly before filter-method results are interpreted.

The script uses the same chronological split logic as ``filter_screening.py`` and
reports:
- target-rate difference;
- missingness difference by predictor;
- Kolmogorov-Smirnov statistics for numeric distributions;
- Population Stability Index (PSI) using development-derived quantile bins;
- changes in the repeat-customer share and prior-transaction distribution;
- a compact visual summary of PSI values.

These measures are diagnostic. No feature or split is rejected automatically from a
single threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


TARGET = "delinquent_5d"
DATE_FIELD = "pdate"
EPSILON = 1e-6
PSI_BINS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--development-fraction",
        type=float,
        default=0.80,
        help="Approximate earlier fraction used for development screening.",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("reports/tables/temporal_representativeness.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/temporal_representativeness_summary.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("reports/figures/temporal_psi.png"),
    )
    return parser.parse_args()


def temporal_development_split(
    df: pd.DataFrame, fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if not 0.5 <= fraction < 1.0:
        raise ValueError("development-fraction must be between 0.5 and 1.0")

    ordered = df.sort_values(DATE_FIELD, kind="stable").copy()
    target_index = max(
        0,
        min(len(ordered) - 1, int(np.floor(len(ordered) * fraction)) - 1),
    )
    cutoff = ordered.iloc[target_index][DATE_FIELD]
    development = ordered.loc[ordered[DATE_FIELD] <= cutoff].copy()
    holdout = ordered.loc[ordered[DATE_FIELD] > cutoff].copy()

    if development.empty or holdout.empty:
        raise ValueError("Temporal split produced an empty development or holdout set.")
    return development, holdout, cutoff


def _psi_from_development_bins(
    development: pd.Series, holdout: pd.Series, bins: int = PSI_BINS
) -> float | None:
    """Compute PSI with bin edges learned only from non-missing development values."""
    dev = pd.to_numeric(development, errors="coerce").dropna().to_numpy(dtype=float)
    hold = pd.to_numeric(holdout, errors="coerce").dropna().to_numpy(dtype=float)
    if len(dev) == 0 or len(hold) == 0:
        return None

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(dev, quantiles))
    if len(edges) < 2:
        return 0.0

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    dev_counts, _ = np.histogram(dev, bins=edges)
    hold_counts, _ = np.histogram(hold, bins=edges)

    dev_pct = dev_counts / max(dev_counts.sum(), 1)
    hold_pct = hold_counts / max(hold_counts.sum(), 1)
    dev_pct = np.clip(dev_pct, EPSILON, None)
    hold_pct = np.clip(hold_pct, EPSILON, None)

    return float(np.sum((hold_pct - dev_pct) * np.log(hold_pct / dev_pct)))


def _ks_statistic(development: pd.Series, holdout: pd.Series) -> tuple[float | None, float | None]:
    dev = pd.to_numeric(development, errors="coerce").dropna()
    hold = pd.to_numeric(holdout, errors="coerce").dropna()
    if dev.empty or hold.empty:
        return None, None
    result = ks_2samp(dev, hold, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.data)

    required = {DATE_FIELD, TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[DATE_FIELD] = pd.to_datetime(df[DATE_FIELD], errors="coerce")
    if df[DATE_FIELD].isna().any():
        raise ValueError("Unparseable pdate values found in model-ready data.")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must contain only 0/1 values.")

    features = [column for column in df.columns if column not in {DATE_FIELD, TARGET}]
    development, holdout, cutoff = temporal_development_split(
        df, args.development_fraction
    )

    rows: list[dict] = []
    for feature in features:
        dev = development[feature]
        hold = holdout[feature]
        ks_stat, ks_pvalue = _ks_statistic(dev, hold)
        psi = _psi_from_development_bins(dev, hold)

        rows.append(
            {
                "feature": feature,
                "development_missing_rate": float(dev.isna().mean()),
                "holdout_missing_rate": float(hold.isna().mean()),
                "missing_rate_difference": float(hold.isna().mean() - dev.isna().mean()),
                "development_median": float(pd.to_numeric(dev, errors="coerce").median()),
                "holdout_median": float(pd.to_numeric(hold, errors="coerce").median()),
                "development_mean": float(pd.to_numeric(dev, errors="coerce").mean()),
                "holdout_mean": float(pd.to_numeric(hold, errors="coerce").mean()),
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "psi": psi,
            }
        )

    diagnostics = pd.DataFrame(rows).sort_values(
        ["psi", "ks_statistic"], ascending=[False, False], na_position="last", kind="stable"
    )

    dev_target_rate = float(development[TARGET].mean())
    hold_target_rate = float(holdout[TARGET].mean())
    repeat_feature = "is_repeat_customer"
    prior_feature = "prior_tx_count"

    summary = {
        "input_rows": int(len(df)),
        "development_rows": int(len(development)),
        "holdout_rows": int(len(holdout)),
        "development_fraction_requested": float(args.development_fraction),
        "temporal_cutoff_inclusive": cutoff.date().isoformat(),
        "development_date_min": development[DATE_FIELD].min().date().isoformat(),
        "development_date_max": development[DATE_FIELD].max().date().isoformat(),
        "holdout_date_min": holdout[DATE_FIELD].min().date().isoformat(),
        "holdout_date_max": holdout[DATE_FIELD].max().date().isoformat(),
        "development_delinquency_rate": dev_target_rate,
        "holdout_delinquency_rate": hold_target_rate,
        "delinquency_rate_difference": float(hold_target_rate - dev_target_rate),
        "development_repeat_customer_share": (
            float(development[repeat_feature].mean()) if repeat_feature in development else None
        ),
        "holdout_repeat_customer_share": (
            float(holdout[repeat_feature].mean()) if repeat_feature in holdout else None
        ),
        "development_prior_tx_median": (
            float(development[prior_feature].median()) if prior_feature in development else None
        ),
        "holdout_prior_tx_median": (
            float(holdout[prior_feature].median()) if prior_feature in holdout else None
        ),
        "highest_psi_features": diagnostics.head(10)[
            ["feature", "psi", "ks_statistic", "missing_rate_difference"]
        ].to_dict(orient="records"),
        "automatic_split_rejection": False,
        "interpretation_note": (
            "Temporal development data are not assumed representative. PSI, KS, missingness, "
            "target-rate, and repeat-customer differences are reported as diagnostics. "
            "Observed drift should be interpreted substantively rather than hidden by a random split."
        ),
    }

    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)

    diagnostics.to_csv(args.table, index=False)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_data = diagnostics.dropna(subset=["psi"]).sort_values("psi", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(5, len(plot_data) * 0.28)))
    ax.barh(plot_data["feature"], plot_data["psi"])
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_ylabel("Feature")
    ax.set_title("Temporal distribution drift: development vs holdout")
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Representativeness table written to {args.table}")
    print(f"PSI figure written to {args.figure}")
    print(f"Representativeness summary written to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
