"""Module 5 stakeholder dashboard for five-day telecom delinquency prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboards.dashboard_utils import (  # noqa: E402
    FEATURE_LABELS,
    case_feature_values,
    load_evidence,
    load_model_package,
    local_median_sensitivity,
)
from src.inference.predict_selected_model import predict_frame  # noqa: E402

st.set_page_config(
    page_title="Telecom delinquency stakeholder dashboard",
    page_icon="📊",
    layout="wide",
)

evidence = load_evidence()
metadata = evidence["metadata"]
metrics = evidence["classification"]
shap_summary = evidence["shap"]
counterfactual = evidence["counterfactual"]
fairness = evidence["fairness"]
robustness = evidence["robustness"]
threshold_tradeoff = evidence["threshold_tradeoff"]

threshold = float(counterfactual["classification_threshold"])

st.title("Five-day telecom delinquency")
st.caption("Stakeholder dashboard · Nexford Data Analytics Capstone")

st.warning(
    "**Model status:** validated for academic demonstration, not production-ready. "
    "Performance remained useful on the later holdout, but deterioration exceeded "
    "the pre-defined temporal-stability tolerance. Fresh validation would be required "
    "before operational deployment."
)

tabs = st.tabs(
    ["Overview", "Prediction explorer", "Explainability & what-if", "Fairness & limits"]
)

with tabs[0]:
    st.subheader("What the model is for")
    st.write(
        "The model estimates the probability that a telecom-enabled microcredit "
        "transaction will remain unpaid after five days. It is designed to support "
        "risk prioritisation and repayment follow-up, not to approve or decline credit."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{float(metrics['roc_auc']):.3f}")
    overall_robustness = robustness.loc[robustness["group_type"] == "overall"].iloc[0]
    c2.metric("Top-20% capture", f"{float(overall_robustness['top20_capture']):.1%}")
    c3.metric("Recall at threshold", f"{float(metrics['recall']):.1%}")
    c4.metric("Precision at threshold", f"{float(metrics['precision']):.1%}")
    c5.metric("Observed delinquency", f"{float(metrics['observed_positive_rate']):.1%}")

    st.info(
        "**Business interpretation:** if the team reviews the 20% of cases with the "
        "highest predicted risk, that group contains about 54% of delinquent cases "
        "in the final test period."
    )

    st.markdown("#### Precision and recall trade-off")
    st.write(
        f"At the current operating threshold, the model identifies about "
        f"{float(metrics['recall']):.1%} of delinquent cases, while about "
        f"{float(metrics['precision']):.1%} of flagged cases are actually delinquent. "
        "This means the current threshold favours catching more delinquent cases at the "
        "cost of more unnecessary follow-up flags."
    )
    st.caption(
        "That trade-off supports using the model for prioritisation and review rather "
        "than for automatic credit decisions."
    )

    st.markdown("#### Explore the threshold trade-off")
    st.write(
        "Move the cut-off to see how precision, recall and the share of flagged cases "
        "would have changed on the final holdout. This is a stakeholder scenario view, "
        "not a re-selection of the operating threshold."
    )

    scenario_threshold = st.slider(
        "Illustrative threshold",
        min_value=float(threshold_tradeoff["threshold"].min()),
        max_value=float(threshold_tradeoff["threshold"].max()),
        value=0.18,
        step=0.01,
        format="%.2f",
    )
    scenario_row = threshold_tradeoff.iloc[
        (threshold_tradeoff["threshold"] - scenario_threshold).abs().argsort()[:1]
    ].iloc[0]

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Illustrative cut-off", f"{float(scenario_row['threshold']):.0%}")
    t2.metric("Precision", f"{float(scenario_row['precision']):.1%}")
    t3.metric("Recall", f"{float(scenario_row['recall']):.1%}")
    t4.metric("Cases flagged", f"{float(scenario_row['flagged_rate']):.1%}")

    tradeoff_long = threshold_tradeoff.melt(
        id_vars="threshold",
        value_vars=["precision", "recall"],
        var_name="Metric",
        value_name="Value",
    )
    tradeoff_long["Metric"] = tradeoff_long["Metric"].str.title()
    tradeoff_fig = px.line(
        tradeoff_long,
        x="threshold",
        y="Value",
        color="Metric",
        labels={"threshold": "Illustrative threshold", "Value": "Rate"},
        title="Precision and recall across alternative thresholds",
    )
    tradeoff_fig.add_vline(
        x=threshold,
        line_dash="dash",
        annotation_text="Frozen operating threshold 17.51%",
        annotation_position="top left",
    )
    tradeoff_fig.update_yaxes(tickformat=".0%")
    tradeoff_fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(tradeoff_fig, width="stretch")

    st.caption(
        "The 17.51% operating threshold remains frozen because it was selected on "
        "development-only chronological predictions before the final holdout was opened. "
        "The alternatives above are descriptive holdout scenarios only."
    )

    st.markdown("#### Model boundary")
    st.write(
        "The target is five-day delinquency, not permanent default. The dashboard "
        "does not convert model output into an approval or decline decision."
    )

    with st.expander("Model identity and traceability"):
        st.json(
            {
                "model_name": metadata["model_name"],
                "artifact_version": metadata["artifact_version"],
                "feature_count": metadata["feature_count"],
                "calibration_method": metadata["calibration_method"],
                "artifact_sha256": metadata["artifact_sha256"],
            }
        )

with tabs[1]:
    st.subheader("Prediction explorer")
    st.write(
        "Enter the information available for a transaction and calculate a calibrated "
        "five-day delinquency probability. Blank values are handled by the frozen "
        "training-time median imputer contained in the model package."
    )

    case_option = st.selectbox(
        "Optional starting point",
        ["Blank form", "Low-risk example", "Typical-risk example", "High-risk example"],
        help=(
            "The example values come from the Module 4 SHAP analysis. Only the listed "
            "top contributing fields are prefilled; other fields remain blank."
        ),
    )

    case_map = {
        "Low-risk example": "low_risk",
        "Typical-risk example": "typical_risk",
        "High-risk example": "high_risk",
    }
    defaults = (
        case_feature_values(shap_summary, case_map[case_option])
        if case_option in case_map
        else {}
    )

    feature_values: dict[str, float] = {}
    features = metadata["features"]

    left, right = st.columns(2)
    for idx, feature in enumerate(features):
        container = left if idx % 2 == 0 else right
        default = defaults.get(feature)
        default_text = "" if default is None else str(default)
        raw = container.text_input(
            FEATURE_LABELS.get(feature, feature),
            value=default_text,
            key=f"feature_{feature}",
            help=f"Model field: {feature}",
        )
        try:
            feature_values[feature] = float(raw) if raw.strip() else np.nan
        except ValueError:
            feature_values[feature] = np.nan
            container.caption("Please enter a numeric value or leave the field blank.")

    if st.button("Calculate predicted risk", type="primary"):
        try:
            package = load_model_package()
            frame = pd.DataFrame([feature_values], columns=features)
            result = predict_frame(frame, package).iloc[0]
            calibrated = float(result["calibrated_delinquency_probability"])

            st.metric("Predicted five-day delinquency risk", f"{calibrated:.1%}")
            st.caption(f"Operating threshold: {threshold:.2%}")

            if calibrated >= threshold:
                st.warning(
                    "**Higher-risk follow-up flag.** The score is at or above the "
                    "operating threshold. This is a prioritisation flag, not a credit decision."
                )
            else:
                st.success(
                    "**No higher-risk follow-up flag.** The score is below the operating "
                    "threshold. This does not guarantee on-time repayment."
                )

            st.info(
                "**What does the operating threshold mean?** It is the cut-off used to "
                "turn a predicted probability into an operational follow-up flag. The "
                "17.51% value was selected during development and frozen before the final "
                "holdout was evaluated; it is not a point at which a customer suddenly "
                "becomes objectively high risk."
            )
            st.caption(
                "A lower threshold would generally flag more delinquent cases but also "
                "create more false alarms; a higher threshold would usually improve "
                "precision while missing more delinquent cases. The current threshold "
                "therefore reflects a recall–precision trade-off."
            )

            sensitivity = local_median_sensitivity(frame, package).head(5)
            if sensitivity.empty:
                st.caption(
                    "No case-specific sensitivity is shown because no explicit input values "
                    "were entered. The prediction above was produced from the model's fitted "
                    "median imputation values."
                )
            else:
                st.markdown("#### Which entered values matter most for this prediction?")
                sensitivity["label"] = (
                    sensitivity["feature"].map(FEATURE_LABELS).fillna(sensitivity["feature"])
                )
                sensitivity["direction"] = np.where(
                    sensitivity["risk_change"] >= 0,
                    "pushes this prediction higher",
                    "pushes this prediction lower",
                )
                chart = px.bar(
                    sensitivity.sort_values("abs_risk_change"),
                    x="abs_risk_change",
                    y="label",
                    orientation="h",
                    labels={"abs_risk_change": "Change in predicted risk", "label": ""},
                    title="Local sensitivity compared with the model's training median",
                )
                st.plotly_chart(chart, width="stretch")
                for _, item in sensitivity.iterrows():
                    st.write(
                        f"- **{item['label']}** {item['direction']} when compared with "
                        f"the model's median reference value."
                    )
                st.caption(
                    "This is a one-feature-at-a-time sensitivity check. It is not a causal "
                    "explanation and should not be interpreted as advice to change customer behaviour."
                )
        except FileNotFoundError as exc:
            st.error(str(exc))

with tabs[2]:
    st.subheader("What drives the model")
    top = pd.DataFrame(shap_summary["top_global_features"]).head(8).copy()
    top["label"] = top["feature"].map(FEATURE_LABELS).fillna(top["feature"])

    fig = px.bar(
        top.sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="label",
        orientation="h",
        labels={"mean_abs_shap": "Relative contribution", "label": ""},
        title="Strongest global model drivers",
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "SHAP shows how the fitted model uses features. It does not establish that a "
        "feature causes delinquency or that changing customer behaviour would change repayment."
    )

    st.markdown("#### Borderline predictions can be sensitive to small changes")
    cases = [case for case in counterfactual["cases"] if case.get("counterfactual_found")]
    cards = st.columns(len(cases))
    for card, case in zip(cards, cases):
        change = case["changes"][0]
        with card:
            st.markdown(
                f"**{FEATURE_LABELS.get(change['feature'], change['feature'])}**"
            )
            st.write(f"Input: {change['from']:,.2f} → {change['to']:,.2f}")
            st.write(
                f"Predicted risk: {case['original_score']:.2%} → "
                f"{case['counterfactual_score']:.2%}"
            )

    st.info(
        "**What this means in practice:** for cases very close to the operating threshold, "
        "a small change in one input can move the prediction from one side of the follow-up "
        "cut-off to the other. Borderline cases therefore deserve cautious interpretation."
    )
    st.caption(
        "These are contrastive model checks, not behavioural advice or causal explanations."
    )

with tabs[3]:
    st.subheader("Fairness and robustness")
    st.error(
        "**Demographic fairness cannot be assessed from this dataset.** Reliable protected-group "
        "attributes are not available, so the project does not infer or manufacture substitute "
        "demographic groups."
    )

    tenure = robustness.loc[
        robustness["group_type"] == "account_tenure_quartile"
    ].copy()
    tenure["segment"] = [
        "Shortest tenure",
        "Lower-middle tenure",
        "Upper-middle tenure",
        "Longest tenure",
    ]

    fig = px.bar(
        tenure,
        x="segment",
        y="roc_auc",
        range_y=[0.75, 0.88],
        labels={"segment": "", "roc_auc": "ROC-AUC"},
        title="Model discrimination across account-tenure segments",
    )
    st.plotly_chart(fig, width="stretch")

    st.write(
        "The model remained discriminative in every account-tenure quartile, with ROC-AUC "
        "ranging from about 0.81 to 0.85. Performance improved with longer observed account "
        "history. These are operational robustness results, not demographic fairness results."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### What this can tell us")
        st.write(
            "Whether performance is concentrated in observable operational segments and "
            "whether modest input changes produce unstable scores."
        )
    with right:
        st.markdown("#### What this cannot tell us")
        st.write(
            "Whether outcomes are equitable across sex, age, ethnicity or other protected "
            "demographic groups."
        )

    st.markdown("#### Key limitations")
    for limitation in metadata["known_limitations"]:
        st.write(f"- {limitation}")

    st.caption(fairness["interpretation_caution"])
