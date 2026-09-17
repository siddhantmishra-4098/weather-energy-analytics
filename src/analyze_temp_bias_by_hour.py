"""
analyze_temp_bias_by_hour.py
------------------
Breaks down the Open-Meteo temperature forecast error by hour-of-day (UTC)
to check whether the overall warm bias (see forecast_accuracy.csv, produced
by analysis.py) is concentrated at particular times -- e.g. worse around
midday, which would point to the model underestimating cloud cover or
overestimating daytime solar heating.

Inputs:
  - data/raw/weather_historical.parquet  (actual)
  - data/raw/weather_forecast.parquet    (Open-Meteo forecast)

Outputs (data/processed/, gitignored, regenerate by re-running):
  - temp_bias_by_hour.png  (mean bias per hour-of-day, bar chart)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only writes files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TEMP_COL = "temperature_c"


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}\n"
            "Run the matching ingest script first (ingest_weather.py)."
        )
    return pd.read_parquet(path)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    weather_hist = _require(RAW_DIR / "weather_historical.parquet")
    weather_fcst = _require(RAW_DIR / "weather_forecast.parquet")
    return weather_hist, weather_fcst


def compute_bias_by_hour(weather_hist: pd.DataFrame, weather_fcst: pd.DataFrame) -> pd.DataFrame:
    """Inner-join actual vs. forecast temperature on timestamp, then group the
    forecast error (forecast - actual) by hour-of-day (0-23, UTC)."""
    merged = pd.merge(
        weather_hist[["timestamp", TEMP_COL]],
        weather_fcst[["timestamp", TEMP_COL]],
        on="timestamp",
        how="inner",
        suffixes=("_actual", "_forecast"),
    )
    if merged.empty:
        raise RuntimeError("No overlapping timestamps between historical and forecast weather data.")

    merged = merged.dropna(subset=[f"{TEMP_COL}_actual", f"{TEMP_COL}_forecast"])
    merged["error"] = merged[f"{TEMP_COL}_forecast"] - merged[f"{TEMP_COL}_actual"]
    merged["hour"] = pd.to_datetime(merged["timestamp"]).dt.hour

    grouped = merged.groupby("hour")["error"].agg(
        mean_bias="mean",
        mae=lambda e: e.abs().mean(),
        n_obs="count",
    )
    grouped = grouped.reindex(range(24))  # keep every hour present even if n_obs == 0
    grouped.index.name = "hour"
    return grouped.reset_index().sort_values("hour")


def plot_bias_by_hour(table: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(table["hour"], table["mean_bias"], color="#c0392b")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Mean forecast error, forecast - actual (°C)")
    ax.set_title("Open-Meteo temperature forecast bias by hour of day")
    ax.set_xticks(range(24))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[analyze_temp_bias_by_hour] Saved plot -> {out_path}")


def main() -> None:
    try:
        weather_hist, weather_fcst = load_inputs()
        table = compute_bias_by_hour(weather_hist, weather_fcst)
    except Exception as exc:
        print(f"[analyze_temp_bias_by_hour] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[analyze_temp_bias_by_hour] Rows used (overlapping timestamps): {int(table['n_obs'].sum()):,}")
    print("[analyze_temp_bias_by_hour] Temperature forecast bias by hour-of-day (UTC):")
    with pd.option_context("display.max_rows", None):
        print(table.round(3).to_string(index=False))

    out_path = PROCESSED_DIR / "temp_bias_by_hour.png"
    plot_bias_by_hour(table, out_path)


if __name__ == "__main__":
    main()
