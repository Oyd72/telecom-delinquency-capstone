"""Final-project monitoring dashboard scaffold.

This application is intentionally separate from the stakeholder/XAI dashboard.
The first implementation stage is the monitoring calculation layer; the visual interface
will be completed after those calculations and Evidently AI outputs are verified.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Telecom delinquency monitoring", page_icon="📈", layout="wide")
st.title("Telecom delinquency model monitoring")
st.info(
    "Monitoring dashboard scaffold. The final interface will distinguish labelled "
    "performance monitoring from the post-23-July label-free diagnostic period."
)
st.markdown(
    """
### Monitoring boundary
Records after 23 July 2016 may be used for feature, population and prediction-score drift
diagnostics. Their all-successful outcome labels are not treated as ground truth, so this
period must not be used to report supervised model performance or outcome-based fairness.

### Planned views
- Model health
- Data and prediction drift
- Performance and calibration where reliable labels exist
- Fairness feasibility and operational robustness
"""
)
