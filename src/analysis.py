"""
analysis.py
------------------
Joins the weather and energy datasets produced by ingest_weather.py /
ingest_energy.py and computes two things a weather-for-energy-markets role
cares about:

1. Correlation between local weather (temperature, wind speed, cloud cover
   at Dortmund, used as an NRW/Germany-representative point) and national
   German grid variables (total load, wind generation, solar generation).
   NOTE: this is a deliberate simplification for a portfolio project -- a
   single weather station is not a true national average, but the physical
   relationships (temperature <-> demand, wind speed <-> wind generation,
   cloud cover <-> solar generation) still show up clearly at national scale
   because German weather is spatially correlated on synoptic scales.

2. Forecast accuracy (MAE, RMSE, bias) of Open-Meteo's forecast weather
   against the historical/reanalysis-backed "actuals" for the same hours,
   per variable.

3. (If data/raw/energy_prices.parquet exists, produced by ingest_price.py)
   residual_load_mw = total_load_mw - wind_total_mw - solar_mw -- the portion
   of demand not covered by variable renewables, which is the standard driver
   of day-ahead price level/volatility -- joined against the day-ahead price
   and correlated with it.

Outputs (all under data/processed/, gitignored, regenerate by re-running):
  - weather_energy_joined.parquet  (tidy joined table, used by dashboard.py)
  - correlation_matrix.csv
  - correlation_heatmap.png  (static plot of the same matrix, for the PDF report)
  - forecast_accuracy.csv
  - weather_energy_price_joined.parquet  (only if energy_prices.parquet exists)
  - price_residual_load_correlation.csv  (only if energy_prices.parquet exists)
  - price_residual_load_scatter.png      (only if energy_prices.parquet exists)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only writes files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_COLS = ["temperature_c", "wind_speed_kmh", "cloud_cover_pct"]
ENERGY_COLS = ["total_load_mw", "wind_onshore_mw", "wind_offshore_mw", "solar_mw", "wind_total_mw"]


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}\n"
            "Run the matching ingest script first (ingest_weather.py / ingest_energy.py)."
        )
    return pd.read_parquet(path)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weather_hist = _require(RAW_DIR / "weather_historical.parquet")
    weather_fcst = _require(RAW_DIR / "weather_forecast.parquet")
    energy = _require(RAW_DIR / "energy_actuals.parquet")
    return weather_hist, weather_fcst, energy


def join_weather_and_energy(weather_hist: pd.DataFrame, energy: pd.DataFrame) -> pd.DataFrame:
    """Inner-join hourly weather (actual) with hourly energy data on timestamp."""
    df = pd.merge(
        weather_hist[["timestamp"] + WEATHER_COLS],
        energy[["timestamp"] + ENERGY_COLS],
        on="timestamp",
        how="inner",
    ).sort_values("timestamp")
    if df.empty:
        raise RuntimeError(
            "Weather and energy datasets do not overlap in time -- "
            "re-run both ingest scripts close together so their windows line up."
        )
    return df


def compute_correlation(joined: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between weather variables and energy variables."""
    corr = joined[WEATHER_COLS + ENERGY_COLS].corr(method="pearson")
    # Keep just the weather x energy block -- that's the relationship we actually care about.
    return corr.loc[WEATHER_COLS, ENERGY_COLS]


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Weather x Energy correlation (Pearson r)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[analysis] Saved plot -> {out_path}")


def compute_forecast_accuracy(weather_hist: pd.DataFrame, weather_fcst: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE / bias of Open-Meteo forecast vs. historical (actual) weather, per variable."""
    merged = pd.merge(
        weather_hist[["timestamp"] + WEATHER_COLS],
        weather_fcst[["timestamp"] + WEATHER_COLS],
        on="timestamp",
        how="inner",
        suffixes=("_actual", "_forecast"),
    )
    if merged.empty:
        raise RuntimeError("No overlapping timestamps between historical and forecast weather data.")

    rows = []
    for col in WEATHER_COLS:
        actual = merged[f"{col}_actual"]
        forecast = merged[f"{col}_forecast"]
        mask = actual.notna() & forecast.notna()
        a, f = actual[mask], forecast[mask]
        error = f - a
        rows.append(
            {
                "variable": col,
                "n_obs": int(mask.sum()),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt((error**2).mean())),
                "bias_mean_error": float(error.mean()),
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def compute_energy_forecast_accuracy(energy_actuals: pd.DataFrame, energy_forecast: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE / bias of SMARD/ENTSO-E Day-Ahead energy forecasts vs. actuals, per variable."""
    merged = pd.merge(
        energy_actuals[["timestamp"] + ENERGY_COLS],
        energy_forecast[["timestamp"] + ENERGY_COLS],
        on="timestamp",
        how="inner",
        suffixes=("_actual", "_forecast"),
    )
    if merged.empty:
        raise RuntimeError("No overlapping timestamps between actual and forecast energy data.")

    rows = []
    for col in ENERGY_COLS:
        actual = merged[f"{col}_actual"]
        forecast = merged[f"{col}_forecast"]
        mask = actual.notna() & forecast.notna()
        a, f = actual[mask], forecast[mask]
        error = f - a
        rows.append(
            {
                "variable": col,
                "n_obs": int(mask.sum()),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt((error**2).mean())),
                "bias_mean_error": float(error.mean()),
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def join_price_and_residual_load(joined: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Join day-ahead price onto the weather+energy table (on timestamp) and add
    residual_load_mw = total_load_mw - wind_total_mw - solar_mw, the demand not
    met by variable renewables -- the usual driver of day-ahead price level."""
    df = pd.merge(
        joined,
        prices[["timestamp", "price_eur_mwh"]],
        on="timestamp",
        how="inner",
    ).sort_values("timestamp")
    if df.empty:
        raise RuntimeError(
            "Weather+energy and price datasets do not overlap in time -- "
            "re-run ingest_price.py close together with the other ingest scripts."
        )
    df["residual_load_mw"] = df["total_load_mw"] - df["wind_total_mw"] - df["solar_mw"]
    return df


def compute_price_residual_correlation(price_joined: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between residual_load_mw and price_eur_mwh."""
    mask = price_joined["residual_load_mw"].notna() & price_joined["price_eur_mwh"].notna()
    x, y = price_joined.loc[mask, "residual_load_mw"], price_joined.loc[mask, "price_eur_mwh"]
    r, p_value = pearsonr(x, y)
    return pd.DataFrame(
        [{"n_obs": int(mask.sum()), "pearson_r": float(r), "p_value": float(p_value)}]
    ).set_index(pd.Index(["residual_load_mw_vs_price_eur_mwh"], name="pair"))


def plot_price_residual_scatter(price_joined: pd.DataFrame, price_corr: pd.DataFrame, out_path: Path) -> None:
    """Scatter of residual_load_mw vs. price_eur_mwh with an OLS fitted line --
    the figure behind the r=0.845-style correlation number, otherwise buried
    in a CSV."""
    mask = price_joined["residual_load_mw"].notna() & price_joined["price_eur_mwh"].notna()
    x = price_joined.loc[mask, "residual_load_mw"].to_numpy(dtype=float)
    y = price_joined.loc[mask, "price_eur_mwh"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept

    row = price_corr.loc["residual_load_mw_vs_price_eur_mwh"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, s=8, alpha=0.35, color="#2c6e9e", label="Hourly observations")
    ax.plot(x_line, y_line, color="#c0392b", linewidth=2, label="OLS fit")
    ax.set_xlabel("Residual load (MW) = total load - wind - solar")
    ax.set_ylabel("Day-ahead price (EUR/MWh)")
    ax.set_title("Residual load vs. day-ahead price")
    ax.annotate(
        f"Pearson r = {row['pearson_r']:.3f}\np = {row['p_value']:.2e}\nn = {int(row['n_obs']):,}",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[analysis] Saved plot -> {out_path}")


def main() -> None:
    try:
        weather_hist, weather_fcst, energy = load_inputs()
        joined = join_weather_and_energy(weather_hist, energy)
        corr = compute_correlation(joined)
        accuracy = compute_forecast_accuracy(weather_hist, weather_fcst)
    except Exception as exc:
        print(f"[analysis] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    joined_path = PROCESSED_DIR / "weather_energy_joined.parquet"
    corr_path = PROCESSED_DIR / "correlation_matrix.csv"
    heatmap_path = PROCESSED_DIR / "correlation_heatmap.png"
    accuracy_path = PROCESSED_DIR / "forecast_accuracy.csv"

    joined.to_parquet(joined_path, index=False)
    corr.to_csv(corr_path)
    accuracy.to_csv(accuracy_path)

    print(f"[analysis] Joined weather+energy rows: {len(joined):,} -> {joined_path}")
    print(f"[analysis] Time range: {joined['timestamp'].min()} -> {joined['timestamp'].max()}")
    print("\n[analysis] Weather x Energy correlation (Pearson r):")
    print(corr.round(3))
    print(f"[analysis] -> {corr_path}")
    plot_correlation_heatmap(corr, heatmap_path)
    print("\n[analysis] Open-Meteo forecast-vs-actual accuracy:")
    print(accuracy.round(3))
    print(f"[analysis] -> {accuracy_path}")

    energy_fcst_path = RAW_DIR / "energy_forecast.parquet"
    if not energy_fcst_path.exists():
        print(
            "\n[analysis] Skipping energy forecast accuracy: "
            f"{energy_fcst_path} not found. Run ingest_energy.py first "
            "(with ENTSOE_API_TOKEN set, or the SMARD fallback, to produce it)."
        )
        return

    try:
        energy_fcst = pd.read_parquet(energy_fcst_path)
        energy_accuracy = compute_energy_forecast_accuracy(energy, energy_fcst)
    except Exception as exc:
        print(f"[analysis] ERROR computing energy forecast accuracy: {exc}", file=sys.stderr)
        sys.exit(1)

    energy_accuracy_path = PROCESSED_DIR / "energy_forecast_accuracy.csv"
    energy_accuracy.to_csv(energy_accuracy_path)
    print("\n[analysis] Energy Day-Ahead forecast-vs-actual accuracy:")
    print(energy_accuracy.round(3))
    print(f"[analysis] -> {energy_accuracy_path}")

    price_path = RAW_DIR / "energy_prices.parquet"
    if not price_path.exists():
        print(
            "\n[analysis] Skipping price / residual-load analysis: "
            f"{price_path} not found. Run ingest_price.py first."
        )
        return

    try:
        prices = pd.read_parquet(price_path)
        price_joined = join_price_and_residual_load(joined, prices)
        price_corr = compute_price_residual_correlation(price_joined)
    except Exception as exc:
        print(f"[analysis] ERROR computing price / residual-load analysis: {exc}", file=sys.stderr)
        sys.exit(1)

    price_joined_path = PROCESSED_DIR / "weather_energy_price_joined.parquet"
    price_corr_path = PROCESSED_DIR / "price_residual_load_correlation.csv"
    price_scatter_path = PROCESSED_DIR / "price_residual_load_scatter.png"
    price_joined.to_parquet(price_joined_path, index=False)
    price_corr.to_csv(price_corr_path)

    print(f"\n[analysis] Weather+energy+price joined rows: {len(price_joined):,} -> {price_joined_path}")
    print("[analysis] Residual load (total_load - wind_total - solar) vs. day-ahead price:")
    print(price_corr.round(3))
    print(f"[analysis] -> {price_corr_path}")
    plot_price_residual_scatter(price_joined, price_corr, price_scatter_path)


if __name__ == "__main__":
    main()
