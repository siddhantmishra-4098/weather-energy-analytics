"""
analyze_energy_bias_by_hour.py
------------------
Breaks down the Day-Ahead energy forecast error by hour-of-day (UTC), the
same way analyze_temp_bias_by_hour.py does for the Open-Meteo temperature
forecast -- to check whether the overall bias (see energy_forecast_accuracy.csv,
produced by analysis.py) is concentrated at particular times, e.g. load
forecasts running high in the evening ramp, or solar forecasts running low
around midday.

Inputs:
  - data/raw/energy_actuals.parquet   (actual)
  - data/raw/energy_forecast.parquet  (SMARD or ENTSO-E Day-Ahead forecast)
  energy_forecast.parquet is optional output of ingest_energy.py (SMARD
  fallback always produces it; ENTSO-E needs ENTSOE_API_TOKEN) -- if it's
  missing, this script prints a clear message and exits cleanly rather than
  erroring, same as analysis.py's handling of the same file.

Outputs (data/processed/, gitignored, regenerate by re-running):
  - energy_bias_by_hour.png  (heatmap: hour-of-day x variable, color = mean bias)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only writes files, no display needed
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ENERGY_COLS = ["total_load_mw", "wind_onshore_mw", "wind_offshore_mw", "solar_mw", "wind_total_mw"]


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}\n"
            "Run the matching ingest script first (ingest_energy.py)."
        )
    return pd.read_parquet(path)


def compute_bias_by_hour(energy_actuals: pd.DataFrame, energy_forecast: pd.DataFrame) -> pd.DataFrame:
    """Inner-join actual vs. forecast energy data on timestamp, then group each
    variable's forecast error (forecast - actual) by hour-of-day (0-23, UTC)."""
    merged = pd.merge(
        energy_actuals[["timestamp"] + ENERGY_COLS],
        energy_forecast[["timestamp"] + ENERGY_COLS],
        on="timestamp",
        how="inner",
        suffixes=("_actual", "_forecast"),
    )
    if merged.empty:
        raise RuntimeError("No overlapping timestamps between actual and forecast energy data.")

    merged["hour"] = pd.to_datetime(merged["timestamp"]).dt.hour

    tables = []
    for col in ENERGY_COLS:
        sub = merged.dropna(subset=[f"{col}_actual", f"{col}_forecast"]).copy()
        sub["error"] = sub[f"{col}_forecast"] - sub[f"{col}_actual"]
        grouped = sub.groupby("hour")["error"].agg(
            mean_bias="mean",
            mae=lambda e: e.abs().mean(),
            n_obs="count",
        )
        grouped = grouped.reindex(range(24))  # keep every hour present even if n_obs == 0
        grouped.index.name = "hour"
        grouped = grouped.reset_index()
        grouped.insert(0, "variable", col)
        tables.append(grouped)

    return pd.concat(tables, ignore_index=True).sort_values(["variable", "hour"])


def plot_bias_by_hour(table: pd.DataFrame, out_path: Path) -> None:
    """Single heatmap: hour-of-day (x) x variable (y), color = mean bias (MW).

    Each variable is annotated with its own value, since the variables span
    very different generation scales (total_load_mw in the tens of thousands
    of MW vs. wind_offshore_mw in the low thousands) -- a single shared color
    scale keeps the smaller-scale rows visually flat, so the printed numbers
    in each cell are what carries the exact comparison, the color is for
    spotting the sign/shape at a glance.
    """
    pivot = table.pivot(index="variable", columns="hour", values="mean_bias").reindex(ENERGY_COLS)

    fig, ax = plt.subplots(figsize=(12, 4))
    vmax = float(pivot.abs().max().max())
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:,.0f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Mean bias, forecast - actual (MW)")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_title("Day-Ahead energy forecast bias by hour of day")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[analyze_energy_bias_by_hour] Saved plot -> {out_path}")


def main() -> None:
    energy_actuals_path = RAW_DIR / "energy_actuals.parquet"
    energy_forecast_path = RAW_DIR / "energy_forecast.parquet"

    if not energy_forecast_path.exists():
        print(
            f"[analyze_energy_bias_by_hour] Skipping: {energy_forecast_path} not found. "
            "Run ingest_energy.py first (with ENTSOE_API_TOKEN set, or the SMARD fallback, "
            "to produce it)."
        )
        return

    try:
        energy_actuals = _require(energy_actuals_path)
        energy_forecast = _require(energy_forecast_path)
        table = compute_bias_by_hour(energy_actuals, energy_forecast)
    except Exception as exc:
        print(f"[analyze_energy_bias_by_hour] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    for col in ENERGY_COLS:
        sub = table[table["variable"] == col]
        print(f"\n[analyze_energy_bias_by_hour] {col} -- rows used (overlapping timestamps): {int(sub['n_obs'].sum()):,}")
        with pd.option_context("display.max_rows", None):
            print(sub.drop(columns="variable").round(3).to_string(index=False))

    out_path = PROCESSED_DIR / "energy_bias_by_hour.png"
    plot_bias_by_hour(table, out_path)


if __name__ == "__main__":
    main()
