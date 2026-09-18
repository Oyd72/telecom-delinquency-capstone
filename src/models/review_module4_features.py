"""Review the Module 4 candidate feature set for leakage and modelling readiness.

This audit is deliberately performed before new Module 4 model fitting. It checks:
- the agreed 12-feature working specification from Module 3;
- absence of identifiers and the source target from predictors;
- treatment of pdate as a temporal control rather than a predictor;
- missingness, uniqueness, and simple target association;
- suspicious near-deterministic numeric relationships with the target;
- whether column names contain obvious direct demographic/protected-trait terms.

The name-based fairness screen is only a safeguard. It does not establish whether a
feature is or is not an indirect proxy; that remains a modelling and interpretation task.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"
TABLE_PATH = PROJECT_ROOT / "reports/tables/module4_feature_leakage_review.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports/tables/module4_feature_leakage_review.json"

TARGET = "delinquent_5d"
TEMPORAL_CONTROL = "pdate"

WORKING_FEATURES = [
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

FORBIDDEN_PREDICTORS = {
    "msisdn",
    "label",
    TARGET,
    TEMPORAL_CONTROL,
}

# Obvious direct demographic/protected-trait terms only. This is intentionally
# conservative and is not used to infer protected characteristics.
DIRECT_FAIRNESS_TERMS = {
    "age",
    "gender",
    "sex",
    "race",
    "ethnicity",
    "religion",
    "disability",
    "nationality",
    "marital",
    "pregnancy",
}


def _name_hits(column: str) -> list[str]:
    lowered = column.lower()
    return sorted(term for term in DIRECT_FAIRNESS_TERMS if term in lowered)


def audit_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {TARGET, TEMPORAL_CONTROL, *WORKING_FEATURES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required model-ready columns: {missing}")

    unexpected_forbidden = sorted(
        col for col in FORBIDDEN_PREDICTORS if col in WORKING_FEATURES
    )
    if unexpected_forbidden:
        raise AssertionError(
            f"Forbidden fields included as predictors: {unexpected_forbidden}"
        )

    if "msisdn" in df.columns or "label" in df.columns:
        raise AssertionError(
            "Identifier/source target unexpectedly present in model-ready data."
        )

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target contains values other than 0/1.")

    rows = []
    for feature in WORKING_FEATURES:
        series = df[feature]
        numeric = pd.to_numeric(series, errors="coerce")
        corr = numeric.corr(df[TARGET]) if numeric.notna().sum() > 1 else float("nan")
        hits = _name_hits(feature)

        rows.append(
            {
                "feature": feature,
                "role": "candidate_predictor",
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
                "unique_non_null": int(series.nunique(dropna=True)),
                "pearson_target_correlation": (
                    None if pd.isna(corr) else float(corr)
                ),
                "abs_target_correlation": (
                    None if pd.isna(corr) else float(abs(corr))
                ),
                "near_deterministic_target_flag": (
                    False if pd.isna(corr) else bool(abs(corr) >= 0.95)
                ),
                "direct_fairness_name_hits": ";".join(hits),
            }
        )

    review = pd.DataFrame(rows)

    numeric_target_flags = review.loc[
        review["near_deterministic_target_flag"], "feature"
    ].tolist()
    fairness_name_flags = review.loc[
        review["direct_fairness_name_hits"].ne(""), "feature"
    ].tolist()

    duplicate_feature_pairs = []
    feature_df = df[WORKING_FEATURES]
    for i, left in enumerate(WORKING_FEATURES):
        for right in WORKING_FEATURES[i + 1 :]:
            if feature_df[left].equals(feature_df[right]):
                duplicate_feature_pairs.append([left, right])

    summary = {
        "rows": int(len(df)),
        "target": TARGET,
        "candidate_feature_count": len(WORKING_FEATURES),
        "candidate_features": WORKING_FEATURES,
        "temporal_control": TEMPORAL_CONTROL,
        "temporal_control_used_as_predictor": TEMPORAL_CONTROL in WORKING_FEATURES,
        "identifier_present_in_model_ready_data": "msisdn" in df.columns,
        "source_label_present_in_model_ready_data": "label" in df.columns,
        "forbidden_predictors_in_feature_set": unexpected_forbidden,
        "near_deterministic_target_flags": numeric_target_flags,
        "exact_duplicate_feature_pairs": duplicate_feature_pairs,
        "direct_fairness_name_flags": fairness_name_flags,
        "fairness_screen_note": (
            "No conclusion about indirect proxy behaviour is drawn from column names. "
            "Potential indirect effects will be reviewed during modelling and interpretation."
        ),
        "missingness_max_rate": float(review["missing_rate"].max()),
        "status": "pass"
        if not (
            unexpected_forbidden
            or numeric_target_flags
            or duplicate_feature_pairs
            or fairness_name_flags
        )
        else "review_required",
    }

    return review, summary


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Model-ready dataset not found: {DATA_PATH.relative_to(PROJECT_ROOT)}"
        )

    df = pd.read_csv(DATA_PATH)
    review, summary = audit_features(df)

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(TABLE_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Module 4 feature/leakage review completed.")
    print()
    print(f"Rows reviewed: {summary['rows']:,}")
    print(f"Candidate predictors: {summary['candidate_feature_count']}")
    print(
        "Temporal control used as predictor:",
        summary["temporal_control_used_as_predictor"],
    )
    print(
        "Identifier present in model-ready data:",
        summary["identifier_present_in_model_ready_data"],
    )
    print(
        "Source label present in model-ready data:",
        summary["source_label_present_in_model_ready_data"],
    )
    print(
        "Near-deterministic target flags:",
        summary["near_deterministic_target_flags"],
    )
    print("Exact duplicate feature pairs:", summary["exact_duplicate_feature_pairs"])
    print("Direct fairness-name flags:", summary["direct_fairness_name_flags"])
    print(f"Maximum missingness rate: {summary['missingness_max_rate']:.4%}")
    print(f"Status: {summary['status']}")
    print()
    print(
        "Note: indirect fairness/proxy effects cannot be established from column names "
        "and will be reviewed during modelling."
    )
    print(
        f"Review table written to: {TABLE_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Summary written to: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
