"""Final-project monitoring dashboard for the telecom delinquency prototype.

This application is intentionally separate from the stakeholder/XAI dashboard.
It presents governance-facing monitoring evidence and keeps labelled performance
monitoring distinct from the post-23-July 2016 label-free diagnostic period.
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
FEATURE_DRIFT_PATH = MONITORING_DIR / "feature_drift_post_23_july.csv"
RECONCILED_PATH = MONITORING_DIR / "reconciled_drift_evidence.csv"
RECONCILED_SUMMARY_PATH = MONITORING_DIR / "reconciled_drift_summary.json"
POPULATION_SUMMARY_PATH = MONITORING_DIR / "monitoring_population_summary.json"
EVIDENTLY_HTML_PATH = MONITORING_DIR / "evidently_post_23_july_drift.html"


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
feature_drift = evidence["feature_drift"]
reconciled = evidence["reconciled"]
reconciled_summary = evidence["reconciled_summary"]
population = evidence["population"]

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
    c3.metric("Monitoring mode", status["monitoring_mode"].replace("_", " ").title())
    c4.metric("Overall drift status", str(status["overall_drift_status"]).upper())

    st.markdown("#### Monitoring periods")
    period_table = pd.DataFrame(
        [
            {
                "Period": "Development reference",
                "Dates": status["reference_period"],
                "Rows": status["reference_rows"],
                "Outcome labels": "Used as development reference",
            },
            {
                "Period": "Final labelled evaluation",
                "Dates": status["labelled_evaluation_period"],
                "Rows": status["labelled_holdout_rows"],
                "Outcome labels": "Trusted for project evaluation",
            },
            {
                "Period": "Post-23-July diagnostic",
                "Dates": status["monitoring_period"],
                "Rows": status["diagnostic_rows"],
                "Outcome labels": "Not treated as ground truth",
            },
        ]
    )
    st.dataframe(period_table, width="stretch", hide_index=True)

    st.warning(
        "**Post-23-July boundary.** The later all-successful outcome regime is unexplained. "
        "Those labels are not treated as ground truth. This period is used only for "
        "feature, population, missingness and prediction-score drift diagnostics."
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
        "reference and diagnostic populations. Agreement is treated as convergent evidence; "
        "disagreement remains visible as method-sensitive evidence."
    )

    counts = reconciled_summary["counts"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confirmed drift", counts["confirmed"])
    c2.metric("Project-only signals", counts["project_only"])
    c3.metric("Evidently-only signals", counts["evidently_only"])
    c4.metric("No drift", counts["no_drift"])

    display = reconciled.copy()
    display["project_psi"] = display["project_psi"].astype(float)
    display["evidently_psi"] = display["evidently_psi"].astype(float)

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

    st.success(
        "**Confirmed drift:** "
        + ", ".join(confirmed)
        + ". These columns cross the PSI threshold in both implementations."
    )
    st.info(
        "**Method-sensitive signals:** "
        + ", ".join(method_sensitive)
        + ". These cross the threshold only in the project implementation and are "
        "therefore not presented as independently confirmed drift."
    )

    prediction_row = display.loc[
        display["column"] == "calibrated_delinquency_probability"
    ].iloc[0]
    st.markdown("#### Prediction-score shift")
    p1, p2, p3 = st.columns(3)
    p1.metric(
        "Reference mean risk",
        f"{float(status['prediction_drift']['reference_mean']):.2%}",
    )
    p2.metric(
        "Diagnostic mean risk",
        f"{float(status['prediction_drift']['current_mean']):.2%}",
    )
    p3.metric(
        "Confirmed score drift",
        "Yes" if prediction_row["evidence_classification"] == "confirmed" else "No",
    )

    if EVIDENTLY_HTML_PATH.exists():
        st.caption(
            "A standalone Evidently AI HTML report is also generated in "
            "reports/monitoring/evidently_post_23_july_drift.html."
        )

with tabs[2]:
    st.subheader("Performance and calibration")

    perf = status["final_labelled_holdout_performance"]
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
        "These performance metrics are calculated only on the labelled final holdout "
        "(14–23 July 2016) using the frozen operating threshold."
    )

    st.error(
        "**Performance monitoring after 23 July is not available.** "
        "The later outcome labels are not treated as reliable ground truth. "
        "Accordingly, no accuracy, precision, recall, calibration, top-risk capture or "
        "outcome-based fairness metric is reported for that period."
    )

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
        "- Representation and performance checks across defensible operational segments.\n"
        "- Ongoing visibility of proxy, representation and operational-use risks."
    )

    st.markdown("#### What it cannot support")
    st.write(
        "- Demographic parity, Equalized Odds or disparate-impact claims for protected groups.\n"
        "- A conclusion that the model is fair simply because protected attributes are absent.\n"
        "- Post-23-July outcome-based fairness comparisons, because those labels are not trusted."
    )

    st.markdown("#### Monitoring governance")
    st.write(
        "Fairlearn and Evidently AI are used only where the available evidence supports "
        "their interpretation. Operational segments remain explicitly labelled as robustness "
        "or representation checks, not demographic fairness groups."
    )

    st.caption(status["fairness_note"])
