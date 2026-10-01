"""Final-project monitoring dashboard for the telecom delinquency prototype.

This application is intentionally separate from the stakeholder/XAI dashboard.
It presents governance-facing monitoring evidence using the development reference and
the chronologically later final holdout. No post-23-July records are used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONITORING_DIR = PROJECT_ROOT / "reports" / "monitoring"

STATUS_PATH = MONITORING_DIR / "latest_monitoring_status.json"
FEATURE_DRIFT_PATH = MONITORING_DIR / "feature_drift_final_holdout.csv"
RECONCILED_PATH = MONITORING_DIR / "reconciled_drift_evidence.csv"
RECONCILED_SUMMARY_PATH = MONITORING_DIR / "reconciled_drift_summary.json"
POPULATION_SUMMARY_PATH = MONITORING_DIR / "monitoring_population_summary.json"
EVIDENTLY_HTML_PATH = MONITORING_DIR / "evidently_final_holdout_drift.html"


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_monitoring_evidence() -> dict:
    required = [
        STATUS_PATH,
        FEATURE_DRIFT_PATH,
        RECONCILED_PATH,
        RECONCILED_SUMMARY_PATH,
        POPULATION_SUMMARY_PATH,
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing monitoring evidence. Run the monitoring scripts first: "
            + ", ".join(missing)
        )

    return {
        "status": read_json(STATUS_PATH),
        "feature_drift": pd.read_csv(FEATURE_DRIFT_PATH),
        "reconciled": pd.read_csv(RECONCILED_PATH),
        "reconciled_summary": read_json(RECONCILED_SUMMARY_PATH),
        "population": read_json(POPULATION_SUMMARY_PATH),
    }


st.set_page_config(
    page_title="Telecom delinquency monitoring",
    page_icon="📈",
    layout="wide",
)

st.title("Telecom delinquency model monitoring")
st.caption(
    "Governance-facing monitoring for the academic prototype. "
    "This dashboard is separate from the stakeholder/XAI application."
)

try:
    evidence = load_monitoring_evidence()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

status = evidence["status"]
reconciled = evidence["reconciled"]
reconciled_summary = evidence["reconciled_summary"]

tabs = st.tabs(
    [
        "Model health",
        "Data & prediction drift",
        "Performance & calibration",
        "Fairness & robustness",
    ]
)

with tabs[0]:
    st.subheader("Model and monitoring status")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model version", str(status["artifact_version"]))
    c2.metric("Operating threshold", f"{float(status['operating_threshold']):.2%}")
    c3.metric("Monitoring mode", "Historical demonstration")
    c4.metric("Overall drift status", str(status["overall_drift_status"]).upper())

    st.markdown("#### Monitoring demonstration")
    period_table = pd.DataFrame(
        [
            {
                "Period": "Development reference",
                "Dates": status["reference_period"],
                "Rows": status["reference_rows"],
                "Role": "Reference distribution",
            },
            {
                "Period": "Final holdout",
                "Dates": status["comparison_period"],
                "Rows": status["comparison_rows"],
                "Role": "Later comparison + labelled evaluation",
            },
        ]
    )
    st.dataframe(period_table, width="stretch", hide_index=True)

    st.info(
        "**Why these periods?** The final project demonstrates monitoring using data already "
        "established in the model-development workflow. It does not assume that a separate "
        "untouched future dataset must exist. In operational use, the same monitoring logic "
        "would be applied to future scored batches as they arrive."
    )

    st.markdown("#### Model traceability")
    st.code(
        f"Model: {status['model_name']}\n"
        f"Artifact version: {status['artifact_version']}\n"
        f"SHA-256: {status['artifact_sha256']}",
        language="text",
    )

with tabs[1]:
    st.subheader("Data and prediction drift")
    st.write(
        "The project compares its own PSI implementation with Evidently AI on the same "
        "development-reference and final-holdout populations. Agreement is treated as "
        "convergent evidence; disagreement remains visible as method-sensitive evidence."
    )

    counts = reconciled_summary["counts"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confirmed drift", counts["confirmed"])
    c2.metric("Project-only signals", counts["project_only"])
    c3.metric("Evidently-only signals", counts["evidently_only"])
    c4.metric("No drift", counts["no_drift"])

    display = reconciled.copy()
    fig = px.bar(
        display.melt(
            id_vars=["column", "evidence_classification"],
            value_vars=["project_psi", "evidently_psi"],
            var_name="method",
            value_name="psi",
        ),
        x="column",
        y="psi",
        color="method",
        barmode="group",
        title="PSI by monitored feature and prediction score",
        labels={"column": "", "psi": "PSI", "method": "Method"},
    )
    fig.add_hline(y=0.10, line_dash="dash", annotation_text="0.10 drift threshold")
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        display[
            [
                "column",
                "column_type",
                "project_psi",
                "evidently_psi",
                "evidence_classification",
            ]
        ].sort_values(
            ["evidence_classification", "project_psi"],
            ascending=[True, False],
        ),
        width="stretch",
        hide_index=True,
    )

    confirmed = reconciled_summary["confirmed_columns"]
    method_sensitive = reconciled_summary["method_sensitive_columns"]

    if confirmed:
        st.success(
            "**Confirmed drift:** "
            + ", ".join(confirmed)
            + ". These cross the PSI threshold in both implementations."
        )
    else:
        st.success("No columns cross the PSI threshold in both implementations.")

    if method_sensitive:
        st.info(
            "**Method-sensitive signals:** "
            + ", ".join(method_sensitive)
            + ". These cross the threshold in only one implementation."
        )

    prediction_row = display.loc[
        display["column"] == "calibrated_delinquency_probability"
    ].iloc[0]
    st.markdown("#### Prediction-score comparison")
    p1, p2, p3 = st.columns(3)
    p1.metric(
        "Reference mean risk",
        f"{float(status['prediction_drift']['reference_mean']):.2%}",
    )
    p2.metric(
        "Final-holdout mean risk",
        f"{float(status['prediction_drift']['current_mean']):.2%}",
    )
    p3.metric(
        "Confirmed score drift",
        "Yes" if prediction_row["evidence_classification"] == "confirmed" else "No",
    )

    if EVIDENTLY_HTML_PATH.exists():
        st.caption(
            "A standalone Evidently AI HTML report is also generated in "
            "reports/monitoring/evidently_final_holdout_drift.html."
        )

with tabs[2]:
    st.subheader("Performance and calibration")

    perf = status["final_holdout_performance"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{float(perf['roc_auc']):.3f}")
    c2.metric("Recall", f"{float(perf['recall']):.1%}")
    c3.metric("Precision", f"{float(perf['precision']):.1%}")
    c4.metric("Top-20% capture", f"{float(perf['top20_capture']):.1%}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Brier score", f"{float(perf['brier_score']):.3f}")
    c6.metric("ECE (10 bins)", f"{float(perf['ece_10bin']):.3f}")
    c7.metric("F1", f"{float(perf['f1']):.3f}")

    st.caption(
        "The final holdout has trustworthy project labels, so it can demonstrate both "
        "performance/calibration monitoring and distribution drift."
    )

    st.info(status["future_monitoring_note"])

with tabs[3]:
    st.subheader("Fairness feasibility and operational robustness")

    st.error(
        "**Demographic fairness cannot be demonstrated from this dataset.** "
        "Reliable protected-group attributes are unavailable, so the project does not "
        "infer or manufacture demographic groups."
    )

    st.markdown("#### What the monitoring evidence can support")
    st.write(
        "- Drift in observed model features and prediction scores.\n"
        "- Performance and calibration on the trusted final holdout.\n"
        "- Representation and performance checks across defensible operational segments.\n"
        "- Ongoing visibility of proxy, representation and operational-use risks."
    )

    st.markdown("#### What it cannot support")
    st.write(
        "- Demographic parity, Equalized Odds or disparate-impact claims for protected groups.\n"
        "- A conclusion that the model is fair simply because protected attributes are absent.\n"
        "- Future production performance before trustworthy outcome labels have matured."
    )

    st.markdown("#### Monitoring governance")
    st.write(
        "Fairlearn and Evidently AI are used only where the available evidence supports "
        "their interpretation. Operational segments remain explicitly labelled as robustness "
        "or representation checks, not demographic fairness groups."
    )

    st.caption(status["fairness_note"])
