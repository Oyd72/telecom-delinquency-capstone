"""Final-project monitoring dashboard for the telecom delinquency prototype.

This application is intentionally separate from the stakeholder/XAI dashboard.
It presents governance-facing monitoring evidence using the development reference and
the chronologically later final holdout. No post-23-July records are used.

The drift view keeps the project's custom monitoring evidence and Evidently AI evidence
separate. Agreement and disagreement are interpreted transparently rather than collapsed
into a single combined drift classification.
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
POPULATION_SUMMARY_PATH = MONITORING_DIR / "monitoring_population_summary.json"
EVIDENTLY_JSON_PATH = MONITORING_DIR / "evidently_final_holdout_drift.json"
EVIDENTLY_HTML_PATH = MONITORING_DIR / "evidently_final_holdout_drift.html"

DRIFT_THRESHOLD = 0.10
PREDICTION_COLUMN = "calibrated_delinquency_probability"


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def extract_evidently_drift(payload: dict) -> pd.DataFrame:
    rows = []
    for metric in payload.get("metrics", []):
        config = metric.get("config") or {}
        if config.get("type") != "evidently:metric_v2:ValueDrift":
            continue
        column = config.get("column")
        value = metric.get("value")
        threshold = float(config.get("threshold", DRIFT_THRESHOLD))
        if column is None or not isinstance(value, (int, float)):
            continue
        rows.append(
            {
                "column": str(column),
                "evidently_psi": float(value),
                "evidently_threshold": threshold,
                "evidently_drift": float(value) >= threshold,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def load_monitoring_evidence() -> dict:
    required = [
        STATUS_PATH,
        FEATURE_DRIFT_PATH,
        POPULATION_SUMMARY_PATH,
        EVIDENTLY_JSON_PATH,
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing monitoring evidence. Run the monitoring scripts first: "
            + ", ".join(missing)
        )

    return {
        "status": read_json(STATUS_PATH),
        "project_drift": pd.read_csv(FEATURE_DRIFT_PATH),
        "population": read_json(POPULATION_SUMMARY_PATH),
        "evidently_raw": read_json(EVIDENTLY_JSON_PATH),
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
project_drift = evidence["project_drift"].copy()
evidently_drift = extract_evidently_drift(evidence["evidently_raw"])

# Add the model-output row to the custom project evidence so the two approaches can be
# compared over the same 13 monitored columns without merging their conclusions.
prediction_row = pd.DataFrame(
    [
        {
            "feature": PREDICTION_COLUMN,
            "psi": float(status["prediction_drift"]["psi"]),
            "ks_statistic": float(status["prediction_drift"]["ks_statistic"]),
            "missingness_change_pp": float(
                status["prediction_drift"]["missingness_change_pp"]
            ),
            "severity": str(status["prediction_drift"]["severity"]),
        }
    ]
)
project_all = pd.concat([project_drift, prediction_row], ignore_index=True)

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
        "Two monitoring approaches are shown separately. The project implementation uses "
        "custom PSI, KS and missingness checks with project-specific severity thresholds. "
        "Evidently AI independently evaluates PSI on the same reference and final-holdout "
        "populations."
    )

    st.markdown("### Project Python monitoring")
    st.caption(
        "Custom monitoring logic implemented in src/monitoring/drift_metrics.py. "
        "Project severity thresholds are design choices for this academic prototype."
    )

    p1, p2, p3 = st.columns(3)
    p1.metric(
        "Features at escalate",
        int((project_drift["severity"] == "escalate").sum()),
    )
    p2.metric(
        "Features at watch",
        int((project_drift["severity"] == "watch").sum()),
    )
    p3.metric(
        "Prediction-score PSI",
        f"{float(status['prediction_drift']['psi']):.3f}",
    )

    project_plot = project_all.copy()
    fig_project = px.bar(
        project_plot,
        x="feature",
        y="psi",
        title="Project PSI by monitored feature and prediction score",
        labels={"feature": "", "psi": "PSI"},
    )
    fig_project.add_hline(
        y=0.10,
        line_dash="dash",
        annotation_text="0.10 watch threshold",
    )
    fig_project.add_hline(
        y=0.25,
        line_dash="dot",
        annotation_text="0.25 escalate threshold",
    )
    st.plotly_chart(fig_project, width="stretch")

    project_table = project_all.rename(
        columns={
            "feature": "column",
            "psi": "project_psi",
            "ks_statistic": "ks_statistic",
            "missingness_change_pp": "missingness_change_pp",
            "severity": "project_severity",
        }
    )
    st.dataframe(
        project_table[
            [
                "column",
                "project_psi",
                "ks_statistic",
                "missingness_change_pp",
                "project_severity",
            ]
        ].sort_values("project_psi", ascending=False),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Evidently AI monitoring")
    st.caption(
        "Independent PSI-based drift evaluation using Evidently AI on the same 13 monitored "
        "columns: 12 model features plus the calibrated prediction score."
    )

    if evidently_drift.empty:
        st.warning("No Evidently ValueDrift metrics were found in the generated report.")
    else:
        e1, e2, e3 = st.columns(3)
        e1.metric(
            "Columns Evidently flags as drifted",
            int(evidently_drift["evidently_drift"].sum()),
        )
        e2.metric(
            "Columns not flagged",
            int((~evidently_drift["evidently_drift"]).sum()),
        )
        score_match = evidently_drift.loc[
            evidently_drift["column"] == PREDICTION_COLUMN
        ]
        score_value = (
            float(score_match.iloc[0]["evidently_psi"])
            if not score_match.empty
            else float("nan")
        )
        e3.metric("Prediction-score PSI", f"{score_value:.3f}")

        fig_evidently = px.bar(
            evidently_drift,
            x="column",
            y="evidently_psi",
            title="Evidently AI PSI by monitored feature and prediction score",
            labels={"column": "", "evidently_psi": "PSI"},
        )
        fig_evidently.add_hline(
            y=DRIFT_THRESHOLD,
            line_dash="dash",
            annotation_text="0.10 Evidently drift threshold",
        )
        st.plotly_chart(fig_evidently, width="stretch")

        evidently_table = evidently_drift.copy()
        evidently_table["evidently_result"] = evidently_table["evidently_drift"].map(
            {True: "drift", False: "no drift"}
        )
        st.dataframe(
            evidently_table[
                [
                    "column",
                    "evidently_psi",
                    "evidently_threshold",
                    "evidently_result",
                ]
            ].sort_values("evidently_psi", ascending=False),
            width="stretch",
            hide_index=True,
        )

    st.markdown("### Interpretation")
    comparison = project_table.merge(
        evidently_drift[["column", "evidently_psi", "evidently_drift"]],
        on="column",
        how="inner",
    )
    comparison["project_drift_at_0_10"] = comparison["project_psi"] >= DRIFT_THRESHOLD

    agreed_drift = comparison.loc[
        comparison["project_drift_at_0_10"] & comparison["evidently_drift"], "column"
    ].tolist()
    project_only = comparison.loc[
        comparison["project_drift_at_0_10"] & ~comparison["evidently_drift"], "column"
    ].tolist()
    evidently_only = comparison.loc[
        ~comparison["project_drift_at_0_10"] & comparison["evidently_drift"], "column"
    ].tolist()

    if agreed_drift:
        st.success(
            "**Agreement on drift:** " + ", ".join(agreed_drift) + ". "
            "Both approaches cross the 0.10 PSI threshold for these columns."
        )
    if project_only:
        st.info(
            "**Project-only signals:** " + ", ".join(project_only) + ". "
            "These cross the 0.10 PSI threshold only in the custom implementation. "
            "The difference is retained visibly because PSI implementations can differ in "
            "binning and edge handling."
        )
    if evidently_only:
        st.info(
            "**Evidently-only signals:** " + ", ".join(evidently_only) + ". "
            "These cross the threshold only in Evidently AI."
        )

    st.markdown("#### Prediction-score comparison")
    p1, p2 = st.columns(2)
    p1.metric(
        "Reference mean risk",
        f"{float(status['prediction_drift']['reference_mean']):.2%}",
    )
    p2.metric(
        "Final-holdout mean risk",
        f"{float(status['prediction_drift']['current_mean']):.2%}",
    )

    if EVIDENTLY_HTML_PATH.exists():
        st.caption(
            "The standalone Evidently AI report is generated at "
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
