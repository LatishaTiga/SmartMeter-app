from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Smart Meter Multi-Series Demand Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_ORDER = ["TFT", "DeepAR", "PatchTST"]
MODEL_COLORS = {
    "TFT": "#1967D2",
    "DeepAR": "#128A4B",
    "PatchTST": "#D96B00",
}

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# Expected result file locations (user can drop files here)
FORECAST_FILES = {
    "TFT": RESULTS_DIR / "forecast_tft.parquet",
    "DeepAR": RESULTS_DIR / "forecast_deepar.parquet",
    "PatchTST": RESULTS_DIR / "forecast_patchtst.parquet",
}
METER_METRICS_FILE = RESULTS_DIR / "meter_metrics.parquet"
GLOBAL_STATS_FILE = RESULTS_DIR / "global_stats.parquet"
TFT_IMPORTANCE_FILE = RESULTS_DIR / "tft_feature_importance.parquet"


def load_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        alt_csv = path.with_suffix(".csv")
        alt_parquet = path.with_suffix(".parquet")
        if alt_csv.exists():
            path = alt_csv
        elif alt_parquet.exists():
            path = alt_parquet

    if not path.exists():
        return None
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return None


def pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    col_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]
    return None


def standardize_forecast(df: pd.DataFrame, model_name: str) -> pd.DataFrame | None:
    uid_col = pick_first_existing_col(df, ["unique_id", "meter_id", "household_id", "id"])
    ts_col = pick_first_existing_col(df, ["timestamp", "time", "datetime", "date"])
    actual_col = pick_first_existing_col(df, ["actual", "y_true", "y_actual", "y"])
    pred_col = pick_first_existing_col(df, ["prediction", "pred", "y_pred", "forecast"])

    if not all([uid_col, ts_col, actual_col, pred_col]):
        return None

    out = pd.DataFrame(
        {
            "unique_id": df[uid_col].astype(str),
            "timestamp": pd.to_datetime(df[ts_col], errors="coerce"),
            "actual": pd.to_numeric(df[actual_col], errors="coerce"),
            "prediction": pd.to_numeric(df[pred_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp", "actual", "prediction"])
    out["model"] = model_name
    return out.sort_values(["unique_id", "timestamp"])


def standardize_meter_metrics(df: pd.DataFrame) -> pd.DataFrame | None:
    uid_col = pick_first_existing_col(df, ["unique_id", "meter_id", "household_id", "id"])
    model_col = pick_first_existing_col(df, ["model"])
    mae_col = pick_first_existing_col(df, ["mae"])
    rmse_col = pick_first_existing_col(df, ["rmse"])
    smape_col = pick_first_existing_col(df, ["smape", "s_mape"])
    mase48_col = pick_first_existing_col(df, ["mase_48", "mase48", "mase_daily", "mase_daily_48"])
    mase336_col = pick_first_existing_col(df, ["mase_336", "mase336", "mase_weekly", "mase_weekly_336"])
    mape_col = pick_first_existing_col(df, ["mape"])

    if not all([uid_col, model_col, mae_col, rmse_col, smape_col]):
        return None

    out = pd.DataFrame(
        {
            "unique_id": df[uid_col].astype(str),
            "model": df[model_col].astype(str),
            "mae": pd.to_numeric(df[mae_col], errors="coerce"),
            "rmse": pd.to_numeric(df[rmse_col], errors="coerce"),
            "smape": pd.to_numeric(df[smape_col], errors="coerce"),
            "mase_48": pd.to_numeric(df[mase48_col], errors="coerce") if mase48_col else np.nan,
            "mase_336": pd.to_numeric(df[mase336_col], errors="coerce") if mase336_col else np.nan,
            "mape": pd.to_numeric(df[mape_col], errors="coerce") if mape_col else np.nan,
        }
    )
    return out.dropna(subset=["mae", "rmse", "smape"])


def standardize_global_stats(df: pd.DataFrame) -> pd.DataFrame | None:
    model_col = pick_first_existing_col(df, ["model"])
    mape_mean_col = pick_first_existing_col(df, ["mape_mean", "avg_mape", "mean_mape"])
    mape_std_col = pick_first_existing_col(df, ["mape_std", "std_mape"])
    mase_mean_col = pick_first_existing_col(df, ["mase_mean", "avg_mase", "mean_mase"])
    mase_std_col = pick_first_existing_col(df, ["mase_std", "std_mase"])

    if not all([model_col, mape_mean_col, mape_std_col, mase_mean_col, mase_std_col]):
        return None

    out = pd.DataFrame(
        {
            "model": df[model_col].astype(str),
            "mape_mean": pd.to_numeric(df[mape_mean_col], errors="coerce"),
            "mape_std": pd.to_numeric(df[mape_std_col], errors="coerce"),
            "mase_mean": pd.to_numeric(df[mase_mean_col], errors="coerce"),
            "mase_std": pd.to_numeric(df[mase_std_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["mape_mean", "mape_std", "mase_mean", "mase_std"])


def standardize_tft_importance(df: pd.DataFrame) -> pd.DataFrame | None:
    feat_col = pick_first_existing_col(df, ["feature", "variable", "name"])
    imp_col = pick_first_existing_col(df, ["importance", "score", "value"])
    if not feat_col or not imp_col:
        return None

    out = pd.DataFrame(
        {
            "feature": df[feat_col].astype(str),
            "importance": pd.to_numeric(df[imp_col], errors="coerce"),
        }
    ).dropna(subset=["importance"])

    return out.sort_values("importance", ascending=False).head(10)


@st.cache_data(show_spinner=False)
def load_forecast_data() -> tuple[dict[str, pd.DataFrame], list[str]]:
    forecasts: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for model_name in MODEL_ORDER:
        raw = load_table(FORECAST_FILES[model_name])
        if raw is None:
            missing.append(model_name)
            continue
        formatted = standardize_forecast(raw, model_name)
        if formatted is None or formatted.empty:
            missing.append(model_name)
            continue
        forecasts[model_name] = formatted

    return forecasts, missing


def infer_time_step(df: pd.DataFrame) -> pd.Timedelta:
    if df["timestamp"].nunique() < 2:
        return pd.Timedelta(minutes=30)
    diffs = df["timestamp"].sort_values().diff().dropna()
    if diffs.empty:
        return pd.Timedelta(minutes=30)
    return diffs.median()


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(200 * np.mean(np.abs(y_true - y_pred) / denom))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + 1e-8
    return float(100 * np.mean(np.abs(y_true - y_pred) / denom))


def seasonal_naive_mae(series: np.ndarray, period: int) -> float:
    if len(series) <= period:
        return np.nan
    return float(np.mean(np.abs(series[period:] - series[:-period])))


def compute_metrics(series_df: pd.DataFrame, baseline_series: np.ndarray | None = None) -> dict[str, float]:
    y_true = series_df["actual"].to_numpy(dtype=float)
    y_pred = series_df["prediction"].to_numpy(dtype=float)
    base = baseline_series if baseline_series is not None else series_df["actual"].to_numpy(dtype=float)

    metric_mae = mae(y_true, y_pred)
    daily_den = seasonal_naive_mae(base, 48)
    weekly_den = seasonal_naive_mae(base, 336)

    return {
        "mae": metric_mae,
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "mase_48": metric_mae / daily_den if daily_den and not np.isnan(daily_den) else np.nan,
        "mase_336": metric_mae / weekly_den if weekly_den and not np.isnan(weekly_den) else np.nan,
    }


def prepare_series(
    df: pd.DataFrame,
    aggregation_mode: str,
    meter_id: str,
) -> pd.DataFrame:
    if aggregation_mode == "Per Meter":
        out = df[df["unique_id"] == meter_id].copy()
        return out.sort_values("timestamp")

    out = (
        df.groupby("timestamp", as_index=False)[["actual", "prediction"]]
        .mean()
        .assign(unique_id="ALL_METERS")
    )
    return out.sort_values("timestamp")


@st.cache_data(show_spinner=False)
def compute_meter_metrics_from_forecasts(
    forecasts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for model_name, model_df in forecasts.items():
        for meter_id, g in model_df.groupby("unique_id"):
            if len(g) < 50:
                continue
            met = compute_metrics(g.sort_values("timestamp"))
            rows.append(
                {
                    "unique_id": meter_id,
                    "model": model_name,
                    **met,
                }
            )
    return pd.DataFrame(rows)


def build_global_summary(meter_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        meter_metrics.groupby("model", as_index=False)
        .agg(
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            mase_mean=("mase_336", "mean"),
            mase_std=("mase_336", "std"),
        )
        .fillna(0.0)
    )
    return grouped


def make_plot(
    actual_history: pd.DataFrame,
    forecast_window: dict[str, pd.DataFrame],
    compare_mode: bool,
    selected_model: str,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
) -> go.Figure:
    fig = go.Figure()

    if not actual_history.empty:
        fig.add_trace(
            go.Scatter(
                x=actual_history["timestamp"],
                y=actual_history["actual"],
                mode="lines",
                name="Actual",
                line=dict(color="#111111", width=2),
            )
        )

    if compare_mode:
        for model_name in MODEL_ORDER:
            if model_name not in forecast_window:
                continue
            g = forecast_window[model_name]
            fig.add_trace(
                go.Scatter(
                    x=g["timestamp"],
                    y=g["prediction"],
                    mode="lines",
                    name=model_name,
                    line=dict(color=MODEL_COLORS[model_name], width=2.5),
                )
            )
    else:
        if selected_model in forecast_window:
            g = forecast_window[selected_model]
            fig.add_trace(
                go.Scatter(
                    x=g["timestamp"],
                    y=g["prediction"],
                    mode="lines",
                    name=selected_model,
                    line=dict(color=MODEL_COLORS[selected_model], width=3),
                )
            )

    fig.add_vrect(
        x0=forecast_start,
        x1=forecast_end,
        fillcolor="rgba(255, 170, 0, 0.12)",
        line_width=0,
        annotation_text="Forecast Region",
        annotation_position="top left",
    )

    fig.update_layout(
        template="plotly_white",
        height=460,
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(orientation="h", y=1.04, x=0),
        xaxis_title="Time",
        yaxis_title="Demand",
    )
    return fig


forecasts, missing_models = load_forecast_data()

st.title("Smart Meter Multi-Series Demand Forecasting Dashboard")
st.caption(
    "Compare TFT, DeepAR, and PatchTST on 50 residential smart meters  \n"
    "48-step horizon | 336-step encoder | Global multi-series training"
)
st.markdown(
    "This dashboard visualizes forecasts generated using three deep learning architectures "
    "(TFT, DeepAR, PatchTST) trained on 50 harmonized smart meters. Users can inspect "
    "per-meter performance, compare models, and analyze error structure."
)

if not forecasts:
    st.error(
        "No forecast files found. Add files at:\n"
        f"- `{FORECAST_FILES['TFT']}`\n"
        f"- `{FORECAST_FILES['DeepAR']}`\n"
        f"- `{FORECAST_FILES['PatchTST']}`"
    )
    st.stop()

if missing_models:
    st.warning(f"Missing or invalid forecast file(s): {', '.join(missing_models)}")

available_meter_ids = sorted(
    set(pd.concat(list(forecasts.values()), ignore_index=True)["unique_id"].unique())
)

st.sidebar.header("Selection Controls")
meter_id = st.sidebar.selectbox(
    "Select Household (Meter ID)",
    available_meter_ids,
)
selected_model = st.sidebar.selectbox("Select Model", MODEL_ORDER, index=2)
horizon_steps = st.sidebar.slider("Forecast Horizon", min_value=24, max_value=168, value=48)
compare_mode = st.sidebar.checkbox("Enable Side-by-Side Model Comparison")

st.sidebar.markdown("---")
st.sidebar.header("Aggregation Controls")
aggregation_mode = st.sidebar.radio("Aggregation Mode", ["Per Meter", "Average Across All 50 Meters"])
error_mode = st.sidebar.radio("Error Breakdown Mode", ["Overall", "By Hour of Day", "By Day of Week"])

active_models = [m for m in MODEL_ORDER if m in forecasts] if compare_mode else [selected_model]
active_models = [m for m in active_models if m in forecasts]
if not active_models:
    st.error("Selected model(s) are unavailable in the loaded forecast files.")
    st.stop()

prepared = {
    model_name: prepare_series(forecasts[model_name], aggregation_mode, meter_id)
    for model_name in active_models
}
prepared = {k: v for k, v in prepared.items() if not v.empty}

if not prepared:
    st.error("No data available for this meter and aggregation setting.")
    st.stop()

ref_model = active_models[0]
ref_series = prepared[ref_model].sort_values("timestamp")
step = infer_time_step(ref_series)

forecast_end = ref_series["timestamp"].max()
forecast_start = forecast_end - (horizon_steps - 1) * step
history_start = forecast_start - horizon_steps * step

forecast_window: dict[str, pd.DataFrame] = {}
for model_name, model_df in prepared.items():
    window_df = model_df[
        (model_df["timestamp"] >= forecast_start) & (model_df["timestamp"] <= forecast_end)
    ].copy()
    if window_df.empty:
        window_df = model_df.tail(horizon_steps).copy()
    forecast_window[model_name] = window_df

actual_history = ref_series[
    (ref_series["timestamp"] >= history_start) & (ref_series["timestamp"] <= forecast_end)
].copy()
if actual_history.empty:
    actual_history = ref_series.tail(horizon_steps * 2).copy()

st.markdown("## Forecast vs Actual")
fig = make_plot(
    actual_history=actual_history,
    forecast_window=forecast_window,
    compare_mode=compare_mode,
    selected_model=selected_model,
    forecast_start=forecast_start,
    forecast_end=forecast_end,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("## Performance Metrics")
metrics_rows = []
for model_name, g in forecast_window.items():
    if len(g) < 2:
        continue
    baseline = prepared[model_name]["actual"].to_numpy(dtype=float)
    met = compute_metrics(g.sort_values("timestamp"), baseline_series=baseline)
    metrics_rows.append({"Model": model_name, **met})

metrics_df = pd.DataFrame(metrics_rows)

if metrics_df.empty:
    st.info("Not enough points in current window to compute metrics.")
else:
    if compare_mode:
        table = metrics_df.rename(
            columns={
                "mae": "MAE",
                "rmse": "RMSE",
                "smape": "sMAPE",
                "mase_48": "MASE-48",
                "mase_336": "MASE-336",
            }
        )
        st.dataframe(
            table[["Model", "MAE", "RMSE", "sMAPE", "MASE-48", "MASE-336"]].sort_values("MAE"),
            use_container_width=True,
            hide_index=True,
        )
    else:
        single = metrics_df.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("MAE", f"{single['mae']:.4f}")
        c2.metric("RMSE", f"{single['rmse']:.4f}")
        c3.metric("sMAPE", f"{single['smape']:.2f}%")
        c4.metric("MASE (Daily 48)", "NA" if np.isnan(single["mase_48"]) else f"{single['mase_48']:.4f}")
        c5.metric("MASE (Weekly 336)", "NA" if np.isnan(single["mase_336"]) else f"{single['mase_336']:.4f}")

st.markdown("## Error Diagnostics")

if error_mode == "Overall":
    hist_fig = go.Figure()
    for model_name, g in forecast_window.items():
        abs_err = np.abs(g["prediction"].to_numpy(dtype=float) - g["actual"].to_numpy(dtype=float))
        hist_fig.add_trace(
            go.Histogram(
                x=abs_err,
                name=model_name,
                opacity=0.55,
                marker_color=MODEL_COLORS.get(model_name, "#444444"),
                nbinsx=35,
            )
        )
    hist_fig.update_layout(
        barmode="overlay",
        template="plotly_white",
        xaxis_title="Absolute Error",
        yaxis_title="Count",
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(hist_fig, use_container_width=True)

elif error_mode == "By Hour of Day":
    rows = []
    for model_name, g in forecast_window.items():
        g = g.copy()
        g["hour"] = g["timestamp"].dt.hour
        g["abs_error"] = np.abs(g["prediction"] - g["actual"])
        hour_df = g.groupby("hour", as_index=False)["abs_error"].mean()
        hour_df["Model"] = model_name
        rows.append(hour_df)

    chart_df = pd.concat(rows, ignore_index=True)
    hour_fig = px.bar(
        chart_df,
        x="hour",
        y="abs_error",
        color="Model",
        barmode="group",
        color_discrete_map=MODEL_COLORS,
        labels={"hour": "Hour (0-23)", "abs_error": "MAE"},
    )
    hour_fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(hour_fig, use_container_width=True)

else:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    rows = []
    for model_name, g in forecast_window.items():
        g = g.copy()
        g["day"] = g["timestamp"].dt.day_name()
        g["abs_error"] = np.abs(g["prediction"] - g["actual"])
        day_df = g.groupby("day", as_index=False)["abs_error"].mean()
        day_df["day"] = pd.Categorical(day_df["day"], categories=day_order, ordered=True)
        day_df["Model"] = model_name
        rows.append(day_df)

    chart_df = pd.concat(rows, ignore_index=True).sort_values("day")
    day_fig = px.bar(
        chart_df,
        x="day",
        y="abs_error",
        color="Model",
        barmode="group",
        color_discrete_map=MODEL_COLORS,
        labels={"day": "Day of Week", "abs_error": "Mean Error"},
    )
    day_fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(day_fig, use_container_width=True)

if selected_model == "TFT" and not compare_mode:
    st.markdown("## TFT Feature Importance")
    tft_raw = load_table(TFT_IMPORTANCE_FILE)
    if tft_raw is None:
        st.info(
            f"No TFT feature importance file found at `{TFT_IMPORTANCE_FILE}`. "
            "Expected columns include feature and importance."
        )
    else:
        tft_imp = standardize_tft_importance(tft_raw)
        if tft_imp is None or tft_imp.empty:
            st.info("TFT feature importance file loaded, but required columns were not recognized.")
        else:
            tft_fig = px.bar(
                tft_imp.sort_values("importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Blues",
            )
            tft_fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(tft_fig, use_container_width=True)
            st.caption(
                "TFT computes attention-weighted feature importances across encoder and decoder inputs. "
                "Higher importance indicates stronger contribution to forecast generation."
            )

st.markdown("## Model Architecture Comparison")
arch_df = pd.DataFrame(
    [
        ["Architecture", "Attention + LSTM", "Autoregressive RNN", "Transformer"],
        ["Handles Static Covariates", "Yes", "Yes", "No"],
        ["Multi-horizon Forecasting", "Direct", "Autoregressive", "Direct"],
        ["Attention Mechanism", "Yes", "No", "Yes"],
        ["Computational Complexity", "Medium", "Low", "High"],
        ["Interpretability", "High", "Low", "Medium"],
    ],
    columns=["Feature", "TFT", "DeepAR", "PatchTST"],
)
st.dataframe(arch_df, use_container_width=True, hide_index=True)

st.markdown("### Short Architecture Descriptions")
st.markdown(
    "**TFT (Temporal Fusion Transformer)**  \n"
    "Combines LSTM encoders with attention mechanisms and gating layers. Designed for "
    "interpretable multi-horizon forecasting with static and time-varying covariates."
)
st.markdown(
    "**DeepAR**  \n"
    "Autoregressive recurrent neural network that predicts probabilistic sequences one step at a time."
)
st.markdown(
    "**PatchTST**  \n"
    "Pure Transformer architecture that splits time series into patches for efficient long-horizon modeling."
)

st.markdown("## Average Performance Across 50 Meters")
meter_metrics_raw = load_table(METER_METRICS_FILE)
if meter_metrics_raw is not None:
    meter_metrics = standardize_meter_metrics(meter_metrics_raw)
else:
    meter_metrics = None

if meter_metrics is None or meter_metrics.empty:
    meter_metrics = compute_meter_metrics_from_forecasts(forecasts)

global_stats_raw = load_table(GLOBAL_STATS_FILE)
global_stats = (
    standardize_global_stats(global_stats_raw)
    if global_stats_raw is not None
    else None
)
if global_stats is None or global_stats.empty:
    global_stats = build_global_summary(meter_metrics)

global_stats["model"] = pd.Categorical(global_stats["model"], categories=MODEL_ORDER, ordered=True)
global_stats = global_stats.sort_values("model").dropna(subset=["model"])

col_mape, col_mase = st.columns(2)

with col_mape:
    fig_mape = go.Figure(
        data=[
            go.Bar(
                x=global_stats["model"],
                y=global_stats["mape_mean"],
                error_y=dict(type="data", array=global_stats["mape_std"], visible=True),
                marker_color=[MODEL_COLORS.get(m, "#777777") for m in global_stats["model"]],
            )
        ]
    )
    fig_mape.update_layout(
        template="plotly_white",
        title="Average MAPE by Model",
        yaxis_title="MAPE (%)",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_mape, use_container_width=True)

with col_mase:
    fig_mase = go.Figure(
        data=[
            go.Bar(
                x=global_stats["model"],
                y=global_stats["mase_mean"],
                error_y=dict(type="data", array=global_stats["mase_std"], visible=True),
                marker_color=[MODEL_COLORS.get(m, "#777777") for m in global_stats["model"]],
            )
        ]
    )
    fig_mase.update_layout(
        template="plotly_white",
        title="Average MASE by Model",
        yaxis_title="MASE",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_mase, use_container_width=True)

st.dataframe(
    global_stats.rename(
        columns={
            "model": "Model",
            "mape_mean": "MAPE Mean",
            "mape_std": "MAPE Std",
            "mase_mean": "MASE Mean",
            "mase_std": "MASE Std",
        }
    ),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "Global evaluation demonstrates relative robustness of each architecture across "
    "heterogeneous household demand profiles."
)
