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
import plotly.graph_objects as go
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
# compared over the same 13 monitored variables without merging their conclusions.
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
    st.subheader(
        "Model and monitoring status",
        help=(
            "Identifies the exact model artefact being monitored and the historical "
            "periods used to demonstrate the monitoring process."
        ),
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Model version",
        str(status["artifact_version"]),
        help="Version identifier of the frozen model package used for scoring and monitoring.",
    )
    c2.metric(
        "Operating threshold",
        f"{float(status['operating_threshold']):.2%}",
        help=(
            "The probability cut-off used to turn a calibrated risk score into a higher-risk "
            "follow-up flag. It was frozen before the final holdout was evaluated."
        ),
    )
    c3.metric(
        "Monitoring mode",
        "Historical demonstration",
        help=(
            "The monitoring mechanism is demonstrated on historical data. "
            "This does not imply that the model was live in production during this period."
        ),
    )
    c4.metric(
        "Overall drift status",
        str(status["overall_drift_status"]).upper(),
        help=(
            "Project-level alert status derived from the custom PSI and missingness thresholds. "
            "It is a monitoring signal for investigation, not a conclusion that the model has failed."
        ),
    )

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
    st.subheader(
        "Data and prediction drift",
        help=(
            "Drift means that the statistical distribution of model inputs or model "
            "outputs has changed between the reference population and the later comparison population."
        ),
    )
    st.write(
        "Two monitoring approaches are shown separately. The project implementation uses "
        "custom PSI, KS and missingness checks with project-specific severity thresholds. "
        "Evidently AI independently evaluates PSI on the same reference and final-holdout "
        "populations."
    )

    st.subheader(
        "Project Python monitoring",
        help=(
            "The project's own monitoring code calculates Population Stability Index (PSI), "
            "Kolmogorov-Smirnov (KS) statistics and missingness changes for each monitored column."
        ),
    )
    st.caption(
        "Custom monitoring logic implemented in src/monitoring/drift_metrics.py. "
        "Project severity thresholds are design choices for this academic prototype."
    )

    p1, p2, p3 = st.columns(3)
    p1.metric(
        "Variables at escalate",
        int((project_all["severity"] == "escalate").sum()),
        help=(
            "Number of monitored variables, including the calibrated prediction score, whose "
            "custom PSI or missingness change crosses the project's escalation threshold."
        ),
    )
    p2.metric(
        "Variables at watch",
        int((project_all["severity"] == "watch").sum()),
        help=(
            "Number of monitored variables, including the calibrated prediction score, that "
            "cross the project's watch threshold but not the escalation threshold."
        ),
    )
    p3.metric(
        "Prediction-score PSI",
        f"{float(status['prediction_drift']['psi']):.3f}",
        help=(
            "Population Stability Index for the distribution of calibrated model risk scores "
            "between the development reference and final holdout."
        ),
    )

    st.info(
        "**How to read the project PSI thresholds:** "
        "PSI below 0.10 = **OK**; 0.10 to below 0.25 = **WATCH**; "
        "0.25 or above = **ESCALATE**. The dashed line marks the watch threshold; "
        "the dotted line marks the escalation threshold. These are project-specific "
        "monitoring thresholds, not universal standards."
    )

    project_plot = project_all.copy()
    fig_project = px.bar(
        project_plot,
        x="feature",
        y="psi",
        title="Project PSI by monitored variable",
        labels={"feature": "", "psi": "PSI"},
    )
    # Draw threshold lines as plot-area shapes so they span the full width of the
    # first and last bars. Add invisible dummy traces only to provide legend entries.
    fig_project.add_shape(
        type="line",
        xref="paper",
        x0=0,
        x1=1,
        yref="y",
        y0=0.10,
        y1=0.10,
        line=dict(color="#FFB000", dash="dash", width=3),
        layer="above",
    )
    fig_project.add_shape(
        type="line",
        xref="paper",
        x0=0,
        x1=1,
        yref="y",
        y0=0.25,
        y1=0.25,
        line=dict(color="#FF4B4B", dash="dot", width=3),
        layer="above",
    )
    fig_project.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="WATCH threshold (PSI = 0.10)",
            line=dict(color="#FFB000", dash="dash", width=3),
            hoverinfo="skip",
        )
    )
    fig_project.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="ESCALATE threshold (PSI = 0.25)",
            line=dict(color="#FF4B4B", dash="dot", width=3),
            hoverinfo="skip",
        )
    )
    fig_project.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=-0.04,
            title_text="Thresholds",
        ),
        margin=dict(l=230, r=40, t=80, b=90),
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

    st.subheader(
        "Evidently AI monitoring",
        help=(
            "Evidently AI is an independent open-source model-monitoring framework. "
            "Here it calculates PSI on the same reference and final-holdout populations."
        ),
    )
    st.caption(
        "Independent PSI-based drift evaluation using Evidently AI on the same 13 monitored "
        "columns: 12 model inputs plus the calibrated prediction score."
    )

    if evidently_drift.empty:
        st.warning("No Evidently ValueDrift metrics were found in the generated report.")
    else:
        e1, e2, e3 = st.columns(3)
        e1.metric(
            "Variables Evidently flags as drifted",
            int(evidently_drift["evidently_drift"].sum()),
            help=(
                "Number of monitored variables for which Evidently AI reports PSI at or above "
                "its configured drift threshold of 0.10."
            ),
        )
        e2.metric(
            "Variables not flagged",
            int((~evidently_drift["evidently_drift"]).sum()),
            help="Number of monitored variables whose Evidently AI PSI remains below 0.10.",
        )
        score_match = evidently_drift.loc[
            evidently_drift["column"] == PREDICTION_COLUMN
        ]
        score_value = (
            float(score_match.iloc[0]["evidently_psi"])
            if not score_match.empty
            else float("nan")
        )
        e3.metric(
            "Prediction-score PSI",
            f"{score_value:.3f}",
            help=(
                "Evidently AI's PSI for the calibrated prediction-score distribution "
                "between the same reference and final-holdout populations."
            ),
        )

        st.info(
            "**How to read Evidently AI:** a variable is flagged as drifted when Evidently's "
            "PSI is 0.10 or higher in this report. The dashed line marks that threshold."
        )

        fig_evidently = px.bar(
            evidently_drift,
            x="column",
            y="evidently_psi",
            title="Evidently AI PSI by monitored variable",
            labels={"column": "", "evidently_psi": "PSI"},
        )
        fig_evidently.add_shape(
            type="line",
            xref="paper",
            x0=0,
            x1=1,
            yref="y",
            y0=DRIFT_THRESHOLD,
            y1=DRIFT_THRESHOLD,
            line=dict(color="#FFB000", dash="dash", width=3),
            layer="above",
        )
        fig_evidently.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="DRIFT threshold (PSI = 0.10)",
                line=dict(color="#FFB000", dash="dash", width=3),
                hoverinfo="skip",
            )
        )
        fig_evidently.update_layout(
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="right",
                x=-0.04,
                title_text="Threshold",
            ),
            margin=dict(l=230, r=40, t=80, b=90),
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

    st.subheader(
        "Interpretation",
        help=(
            "This section explains where the two monitoring approaches agree and where "
            "their results differ. Differences are kept visible rather than forced into one result."
        ),
    )
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
            "Both approaches cross the 0.10 PSI threshold for these monitored variables."
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

    st.subheader(
        "Prediction-score comparison",
        help=(
            "Compares the average calibrated delinquency probability produced by the model "
            "for the reference population and the final holdout."
        ),
    )
    p1, p2 = st.columns(2)
    p1.metric(
        "Reference mean risk",
        f"{float(status['prediction_drift']['reference_mean']):.2%}",
        help="Average calibrated delinquency probability in the development reference population.",
    )
    p2.metric(
        "Final-holdout mean risk",
        f"{float(status['prediction_drift']['current_mean']):.2%}",
        help="Average calibrated delinquency probability in the chronologically later final holdout.",
    )

    if EVIDENTLY_HTML_PATH.exists():
        st.caption(
            "The standalone Evidently AI report is generated at "
            "reports/monitoring/evidently_final_holdout_drift.html."
        )

with tabs[2]:
    st.subheader(
        "Performance and calibration",
        help=(
            "Performance measures how well the model separates and identifies delinquent cases. "
            "Calibration measures how closely predicted probabilities correspond to observed outcomes."
        ),
    )

    perf = status["final_holdout_performance"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "ROC-AUC",
        f"{float(perf['roc_auc']):.3f}",
        help=(
            "Measures how well the model ranks delinquent cases above non-delinquent cases "
            "across all possible thresholds. 0.5 is random ranking; 1.0 is perfect."
        ),
    )
    c2.metric(
        "Recall",
        f"{float(perf['recall']):.1%}",
        help="Share of all genuinely delinquent cases that the model flags at the frozen threshold.",
    )
    c3.metric(
        "Precision",
        f"{float(perf['precision']):.1%}",
        help="Share of model-flagged cases that actually became delinquent.",
    )
    c4.metric(
        "Top-20% capture",
        f"{float(perf['top20_capture']):.1%}",
        help=(
            "Share of all delinquent cases contained within the 20% of cases that the model "
            "ranks as highest risk."
        ),
    )

    c5, c6, c7 = st.columns(3)
    c5.metric(
        "Brier score",
        f"{float(perf['brier_score']):.3f}",
        help=(
            "Measures the squared error of predicted probabilities. Lower is better; "
            "0 would mean perfectly accurate probability forecasts."
        ),
    )
    c6.metric(
        "ECE (10 bins)",
        f"{float(perf['ece_10bin']):.3f}",
        help=(
            "Expected Calibration Error compares predicted probabilities with observed "
            "outcomes across 10 probability groups. Lower is better."
        ),
    )
    c7.metric(
        "F1",
        f"{float(perf['f1']):.3f}",
        help="Harmonic mean of precision and recall, balancing the two measures in one score.",
    )

    st.caption(
        "Performance and calibration metrics shown below are calculated on the final "
        "holdout using the frozen operating threshold."
    )

    st.info(status["future_monitoring_note"])

with tabs[3]:
    st.subheader(
        "Fairness feasibility and operational robustness",
        help=(
            "Fairness feasibility asks which fairness claims the available data can support. "
            "Operational robustness checks whether performance is stable across defensible non-demographic segments."
        ),
    )

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
