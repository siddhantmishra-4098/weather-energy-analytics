"""
dashboard.py
------------------
Streamlit dashboard for the weather-energy-analytics project. Reads the
processed outputs of src/analysis.py (which in turn depend on
src/ingest_weather.py and src/ingest_energy.py having been run first) and
renders three views:

  1. Time series: weather (temperature / wind speed / cloud cover) overlaid
     with German grid variables (load, wind generation, solar generation).
  2. Forecast vs. actual: Open-Meteo forecast weather plotted against the
     historical/reanalysis "actual" weather, plus the MAE/RMSE/bias table
     computed in analysis.py.
  3. Correlation heatmap: weather variables x energy variables.
  4. Energy forecast accuracy: SMARD/ENTSO-E Day-Ahead energy forecasts
     plotted against actuals, plus the MAE/RMSE/bias table.

Run with:  streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

st.set_page_config(page_title="Weather x Energy Analytics -- Germany", layout="wide")


@st.cache_data
def load_data():
    missing = [
        p
        for p in [
            PROCESSED_DIR / "weather_energy_joined.parquet",
            PROCESSED_DIR / "correlation_matrix.csv",
            PROCESSED_DIR / "forecast_accuracy.csv",
            RAW_DIR / "weather_historical.parquet",
            RAW_DIR / "weather_forecast.parquet",
            RAW_DIR / "energy_actuals.parquet",
        ]
        if not p.exists()
    ]
    if missing:
        return None, missing

    joined = pd.read_parquet(PROCESSED_DIR / "weather_energy_joined.parquet")
    corr = pd.read_csv(PROCESSED_DIR / "correlation_matrix.csv", index_col=0)
    accuracy = pd.read_csv(PROCESSED_DIR / "forecast_accuracy.csv", index_col=0)
    weather_hist = pd.read_parquet(RAW_DIR / "weather_historical.parquet")
    weather_fcst = pd.read_parquet(RAW_DIR / "weather_forecast.parquet")
    energy_actuals = pd.read_parquet(RAW_DIR / "energy_actuals.parquet")

    dfs = [joined, weather_hist, weather_fcst, energy_actuals]

    # Energy forecast accuracy depends on ENTSO-E/SMARD forecast data being present --
    # optional, so the rest of the dashboard still works without it.
    energy_accuracy_path = PROCESSED_DIR / "energy_forecast_accuracy.csv"
    energy_fcst_path = RAW_DIR / "energy_forecast.parquet"
    if energy_accuracy_path.exists() and energy_fcst_path.exists():
        energy_accuracy = pd.read_csv(energy_accuracy_path, index_col=0)
        energy_fcst = pd.read_parquet(energy_fcst_path)
        dfs.append(energy_fcst)
    else:
        energy_accuracy = None
        energy_fcst = None

    for df in dfs:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return {
        "joined": joined,
        "corr": corr,
        "accuracy": accuracy,
        "energy_accuracy": energy_accuracy,
        "weather_hist": weather_hist,
        "weather_fcst": weather_fcst,
        "energy_actuals": energy_actuals,
        "energy_fcst": energy_fcst,
    }, []


data, missing_files = load_data()

st.title("Weather x Energy Analytics -- Germany")
st.caption(
    "Weather: Open-Meteo API (Dortmund, NRW) -- Energy: SMARD / Bundesnetzagentur "
    "(or ENTSO-E, if a token is configured) -- NWP: ECMWF open-data (GRIB2 via cfgrib/ecCodes)"
)

if data is None:
    st.error(
        "Processed data not found. Run the pipeline first:\n\n"
        "```\npython src/ingest_weather.py\npython src/ingest_energy.py\npython src/analysis.py\n```\n\n"
        f"Missing files:\n" + "\n".join(f"- `{p}`" for p in missing_files)
    )
    st.stop()

joined = data["joined"]
corr = data["corr"]
accuracy = data["accuracy"]
energy_accuracy = data["energy_accuracy"]
weather_hist = data["weather_hist"]
weather_fcst = data["weather_fcst"]
energy_actuals = data["energy_actuals"]
energy_fcst = data["energy_fcst"]

tab1, tab2, tab3, tab4 = st.tabs(
    ["Weather vs. Energy", "Forecast Accuracy", "Correlation Heatmap", "Energy Forecast Accuracy"]
)

# ---------------------------------------------------------------------------
# Tab 1: time series of weather vs. energy
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Weather vs. German grid variables over time")

    col_a, col_b = st.columns(2)
    weather_var = col_a.selectbox(
        "Weather variable",
        options=["temperature_c", "wind_speed_kmh", "cloud_cover_pct"],
        format_func=lambda c: {
            "temperature_c": "Temperature (°C)",
            "wind_speed_kmh": "Wind speed (km/h)",
            "cloud_cover_pct": "Cloud cover (%)",
        }[c],
    )
    energy_var = col_b.selectbox(
        "Energy variable",
        options=["total_load_mw", "wind_total_mw", "wind_onshore_mw", "wind_offshore_mw", "solar_mw"],
        format_func=lambda c: {
            "total_load_mw": "Total grid load (MW)",
            "wind_total_mw": "Wind generation, total (MW)",
            "wind_onshore_mw": "Wind generation, onshore (MW)",
            "wind_offshore_mw": "Wind generation, offshore (MW)",
            "solar_mw": "Solar generation (MW)",
        }[c],
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=joined["timestamp"], y=joined[weather_var], name=weather_var, yaxis="y1", line=dict(color="#e07b39"))
    )
    fig.add_trace(
        go.Scatter(x=joined["timestamp"], y=joined[energy_var], name=energy_var, yaxis="y2", line=dict(color="#2c6e9e"))
    )
    fig.update_layout(
        xaxis=dict(title="Time (UTC)"),
        yaxis=dict(title=weather_var, side="left"),
        yaxis2=dict(title=energy_var, side="right", overlaying="y"),
        legend=dict(orientation="h", y=1.08),
        height=500,
        margin=dict(t=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"n = {len(joined):,} hourly observations, "
        f"{joined['timestamp'].min()} -> {joined['timestamp'].max()} (UTC)."
    )

# ---------------------------------------------------------------------------
# Tab 2: forecast vs actual
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Open-Meteo forecast vs. actual weather")

    fcst_var = st.selectbox(
        "Variable",
        options=["temperature_c", "wind_speed_kmh", "cloud_cover_pct"],
        format_func=lambda c: {
            "temperature_c": "Temperature (°C)",
            "wind_speed_kmh": "Wind speed (km/h)",
            "cloud_cover_pct": "Cloud cover (%)",
        }[c],
        key="fcst_var",
    )

    actual = weather_hist[["timestamp", fcst_var]].dropna()
    forecast = weather_fcst[["timestamp", fcst_var]].dropna()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=actual["timestamp"], y=actual[fcst_var], name="Actual (historical)", line=dict(color="#333333")))
    fig2.add_trace(go.Scatter(x=forecast["timestamp"], y=forecast[fcst_var], name="Open-Meteo forecast", line=dict(color="#c0392b", dash="dot")))
    fig2.update_layout(height=450, margin=dict(t=30), legend=dict(orientation="h", y=1.08), yaxis_title=fcst_var)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Forecast accuracy (MAE / RMSE / bias), all weather variables:**")
    st.dataframe(accuracy.style.format("{:.3f}"), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: correlation heatmap
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Correlation: weather variables x energy variables")
    fig3 = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels=dict(color="Pearson r"),
    )
    fig3.update_layout(height=450, margin=dict(t=30))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "Computed on hourly data joining Dortmund weather with national German grid data. "
        "A single weather station is a simplification for national totals, but wind-speed <-> "
        "wind-generation and temperature/cloud-cover <-> solar-generation relationships are still visible "
        "because German weather is spatially correlated at synoptic scale."
    )

# ---------------------------------------------------------------------------
# Tab 4: energy forecast vs actual
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Energy Day-Ahead forecast vs. actual")

    if energy_accuracy is None or energy_fcst is None:
        st.info(
            "Not available -- ENTSO-E token not configured and no SMARD forecast data found.\n\n"
            "Set `ENTSOE_API_TOKEN` in `.env` (or just re-run `python src/ingest_energy.py`, which "
            "falls back to SMARD's Day-Ahead forecast automatically), then `python src/analysis.py`."
        )
    else:
        energy_fcst_var = st.selectbox(
            "Variable",
            options=["total_load_mw", "wind_onshore_mw", "wind_offshore_mw", "solar_mw", "wind_total_mw"],
            format_func=lambda c: {
                "total_load_mw": "Total grid load (MW)",
                "wind_onshore_mw": "Wind generation, onshore (MW)",
                "wind_offshore_mw": "Wind generation, offshore (MW)",
                "solar_mw": "Solar generation (MW)",
                "wind_total_mw": "Wind generation, total (MW)",
            }[c],
            key="energy_fcst_var",
        )

        # Plot each series straight from its own dataframe (not an inner join) so the
        # forecast line shows its FULL horizon rather than being clipped to only the
        # timestamps actuals also cover -- same fix already applied to the weather tab.
        energy_actual_series = energy_actuals[["timestamp", energy_fcst_var]].dropna()
        energy_forecast_series = energy_fcst[["timestamp", energy_fcst_var]].dropna()

        fig4 = go.Figure()
        fig4.add_trace(
            go.Scatter(
                x=energy_actual_series["timestamp"],
                y=energy_actual_series[energy_fcst_var],
                name="Actual",
                line=dict(color="#333333"),
            )
        )
        fig4.add_trace(
            go.Scatter(
                x=energy_forecast_series["timestamp"],
                y=energy_forecast_series[energy_fcst_var],
                name="Day-Ahead forecast",
                line=dict(color="#c0392b", dash="dot"),
            )
        )
        fig4.update_layout(height=450, margin=dict(t=30), legend=dict(orientation="h", y=1.08), yaxis_title=energy_fcst_var)
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("**Forecast accuracy (MAE / RMSE / bias), all energy variables:**")
        st.dataframe(energy_accuracy.style.format("{:.3f}"), use_container_width=True)

st.divider()
st.caption(
    "Data sources: Open-Meteo (weather), SMARD/Bundesnetzagentur or ENTSO-E (energy), "
    "ECMWF open-data (GRIB2 NWP fields, see src/decode_grib.py). "
    "Built with Python, pandas, xarray/cfgrib, and Streamlit."
)
