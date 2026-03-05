import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(layout="wide")

# ------------------------------------------------
# Paths
# ------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

HIST_PATH = ROOT / "clean_50_meters_with_fft_clusters_named.parquet"

RESULTS = Path(__file__).resolve().parent / "results"

DEEpar_FORECAST = RESULTS / "forecast_deepar.parquet"
PATCH_FORECAST = RESULTS / "forecast_patchtst.parquet"

DEEpar_METRICS = RESULTS / "meter_metrics_deepar.parquet"
PATCH_METRICS = RESULTS / "meter_metrics_patchtst.parquet"

GLOBAL_DEEPAR = RESULTS / "global_stats_deepar.parquet"
GLOBAL_PATCH = RESULTS / "global_stats_patchtst.parquet"

# ------------------------------------------------
# Load Data
# ------------------------------------------------

@st.cache_data
def load_data():

    hist = pd.read_parquet(HIST_PATH)

    deepar = pd.read_parquet(DEEpar_FORECAST)
    patch = pd.read_parquet(PATCH_FORECAST)

    metrics_deepar = pd.read_parquet(DEEpar_METRICS)
    metrics_patch = pd.read_parquet(PATCH_METRICS)

    global_deepar = pd.read_parquet(GLOBAL_DEEPAR)
    global_patch = pd.read_parquet(GLOBAL_PATCH)

    return hist, deepar, patch, metrics_deepar, metrics_patch, global_deepar, global_patch


hist, deepar, patch, m_deepar, m_patch, g_deepar, g_patch = load_data()

# ------------------------------------------------
# Header
# ------------------------------------------------

st.title("Smart Meter Multi-Series Demand Forecasting Dashboard")

st.markdown(
"""
Compare **DeepAR** and **PatchTST** models trained from scratch on **50 residential smart meters**  
48-step horizon | 336-step encoder | Global multi-series training
"""
)

st.divider()

# ------------------------------------------------
# Sidebar
# ------------------------------------------------

st.sidebar.header("Controls")

meter_ids = sorted(hist["unique_id"].unique())

selected_meter = st.sidebar.selectbox(
    "Select Household (Meter ID)",
    meter_ids
)

comparison_mode = st.sidebar.checkbox(
    "Enable Model Comparison"
)

# Only show these when comparison is enabled
if comparison_mode:

    aggregation_mode = st.sidebar.radio(
        "Aggregation Mode",
        ["Per Meter", "Average Across All 50 Meters"]
    )

    error_mode = st.sidebar.radio(
        "Error Breakdown",
        ["Overall", "By Hour of Day", "By Day of Week"]
    )

# ------------------------------------------------
# Filter Data
# ------------------------------------------------

hist_meter = hist[hist["unique_id"] == selected_meter]

deepar_meter = deepar[deepar["unique_id"] == selected_meter]
patch_meter = patch[patch["unique_id"] == selected_meter]

forecast_start = patch_meter["timestamp"].min()

hist_plot = hist_meter[hist_meter["timestamp"] < forecast_start]

# ------------------------------------------------
# Forecast Visualization
# ------------------------------------------------

st.subheader("Forecast Visualization ")
st.markdown(
"""
Best Model — PatchTST 
"""
)

fig = go.Figure()

# Historical

fig.add_trace(
    go.Scatter(
        x=hist_plot["timestamp"],
        y=hist_plot["y"],
        name="Historical",
        line=dict(color="black")
    )
)

# PatchTST Forecast ONLY

fig.add_trace(
    go.Scatter(
        x=patch_meter["timestamp"],
        y=patch_meter["prediction"],
        name="PatchTST Forecast",
        line=dict(color="royalblue")
    )
)

fig.update_layout(
    height=500,
    xaxis_title="Time",
    yaxis_title="Electricity Demand"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ------------------------------------------------
# Prediction Mode Metrics (PatchTST Only)
# ------------------------------------------------

if not comparison_mode:

    st.subheader("Model Performance — PatchTST")

    row = m_patch[m_patch["unique_id"] == selected_meter].iloc[0]
    global_row = g_patch.iloc[0]

    st.markdown("### Meter Metrics")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("MAE", f"{row.mae:.2f}")
    c2.metric("RMSE", f"{row.rmse:.2f}")
    c3.metric("sMAPE", f"{row.smape:.2f}")
    c4.metric("MASE-48", f"{row.mase_48:.2f}")
    c5.metric("MASE-336", f"{row.mase_336:.2f}")

    st.markdown("### Global Performance (50 Meters)")

    g1, g2, g3 = st.columns(3)

    g1.metric("Mean MAE", f"{global_row.mae_mean:.2f}")
    g2.metric("Mean RMSE", f"{global_row.rmse_mean:.2f}")
    g3.metric("Mean sMAPE", f"{global_row.smape_mean:.2f}")

# ------------------------------------------------
# Comparison Mode
# ------------------------------------------------

else:

    st.subheader("Model Comparison")

    if aggregation_mode == "Per Meter":

        table = pd.concat([
            m_deepar[m_deepar["unique_id"] == selected_meter],
            m_patch[m_patch["unique_id"] == selected_meter]
        ])

        st.dataframe(
            table[["model","mae","rmse","smape","mase_48","mase_336"]]
        )

    else:

        global_df = pd.concat([g_deepar, g_patch])

        st.dataframe(global_df)

    st.divider()

    # ------------------------------------------------
    # Error Diagnostics
    # ------------------------------------------------

    st.subheader("Error Diagnostics")

    df_error = pd.concat([deepar_meter, patch_meter])

    df_error["abs_error"] = abs(df_error["actual"] - df_error["prediction"])
    df_error["hour"] = pd.to_datetime(df_error["timestamp"]).dt.hour
    df_error["day"] = pd.to_datetime(df_error["timestamp"]).dt.day_name()

    if error_mode == "By Hour of Day":

        err = df_error.groupby(["model", "hour"])["abs_error"].mean().reset_index()

        st.bar_chart(
            err.pivot(index="hour", columns="model", values="abs_error")
        )

    elif error_mode == "By Day of Week":

        err = df_error.groupby(["model", "day"])["abs_error"].mean().reset_index()

        st.bar_chart(
            err.pivot(index="day", columns="model", values="abs_error")
        )

    else:

        st.bar_chart(
            df_error.groupby("model")["abs_error"].mean()
        )

    st.divider()

    # ------------------------------------------------
    # Architecture Explanation
    # ------------------------------------------------

    st.subheader("Model Architecture Comparison")

    arch = pd.DataFrame({

    "Feature":[
    "Architecture",
    "Handles Static Covariates",
    "Multi-horizon Forecasting",
    "Attention Mechanism",
    "Computational Complexity",
    "Interpretability"
    ],

    "DeepAR":[
    "Autoregressive RNN",
    "Yes",
    "Autoregressive",
    "No",
    "Low",
    "Low"
    ],

    "PatchTST":[
    "Transformer",
    "No",
    "Direct",
    "Yes",
    "High",
    "Medium"
    ]

    })

    st.table(arch)