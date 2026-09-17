"""
ingest_weather.py
------------------
Pulls hourly weather data for a German region (default: Dortmund, NRW) from the
Open-Meteo API and saves it to parquet.

Open-Meteo is used for two distinct products, both free and keyless:

1. Historical/reanalysis weather (ERA5-based "archive" API) for the last N days
   -> data/raw/weather_historical.parquet
   This is the closest thing to "ground truth" observed weather and is what we
   correlate against actual energy generation/demand in analysis.py.

2. Forecast weather, as it was actually issued at the time, from Open-Meteo's
   Historical Forecast API -> data/raw/weather_forecast.parquet
   Comparing these forecasts to the historical/observed values lets
   analysis.py compute forecast accuracy (MAE/RMSE), which is one of the
   explicit skills this project demonstrates.

   NOTE (fixed after an earlier bug): this used to call the plain forecast
   endpoint with `past_days`, which does NOT return the forecast as it was
   issued -- Open-Meteo's own docs say `past_days` data is "stitched to the
   previous run without gaps or discontinuities," i.e. it is a continuously
   updated, hindsight-corrected timeseries, not a frozen forecast. That made
   the forecast-accuracy numbers look far better than a real day-ahead
   forecast would be, because both "forecast" and "actual" were effectively
   close to the truth. The Historical Forecast API used below
   (historical-forecast-api.open-meteo.com) returns the forecast run as it
   was archived at the time, which is what forecast-accuracy evaluation
   actually needs.

Variables pulled: 2m temperature (C), 10m wind speed (km/h), cloud cover (%).
Wind speed and cloud cover are the two variables most directly linked to
German wind & solar generation; temperature is the main driver of demand.

No API key is required for Open-Meteo's non-commercial tier. See
https://open-meteo.com/en/docs for full documentation.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# --- paths & config -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

LOCATION_NAME = os.getenv("LOCATION_NAME", "Dortmund")
LAT = float(os.getenv("LOCATION_LAT", "51.5136"))
LON = float(os.getenv("LOCATION_LON", "7.4653"))

HISTORY_DAYS = 90  # keep the pull scoped so it runs fast and stays well under rate limits

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

HOURLY_VARS = "temperature_2m,wind_speed_10m,cloud_cover"

REQUEST_TIMEOUT = 30  # seconds


def _fetch_json(url: str, params: dict) -> dict:
    """GET a URL and return parsed JSON, or raise loudly with the response body.

    We deliberately do NOT catch-and-fall-back-to-fake-data here: if Open-Meteo
    is unreachable or rejects the request, the caller needs to see that.
    """
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Open-Meteo request failed ({resp.status_code}) for {url}\n"
            f"params={params}\nresponse body={resp.text[:500]}"
        ) from exc
    return resp.json()


def _hourly_to_dataframe(payload: dict, location_name: str) -> pd.DataFrame:
    """Flatten an Open-Meteo `hourly` block into a tidy DataFrame."""
    hourly = payload.get("hourly")
    if not hourly:
        raise RuntimeError(f"Open-Meteo response had no 'hourly' block: {payload}")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(
        columns={
            "time": "timestamp",
            "temperature_2m": "temperature_c",
            "wind_speed_10m": "wind_speed_kmh",
            "cloud_cover": "cloud_cover_pct",
        }
    )
    df["location"] = location_name
    df["latitude"] = payload.get("latitude")
    df["longitude"] = payload.get("longitude")
    return df[
        [
            "timestamp",
            "location",
            "latitude",
            "longitude",
            "temperature_c",
            "wind_speed_kmh",
            "cloud_cover_pct",
        ]
    ]


def fetch_historical_weather(lat: float, lon: float, days: int, location_name: str) -> pd.DataFrame:
    """Historical (ERA5 reanalysis-backed) hourly weather for the last `days` days."""
    end = date.today() - timedelta(days=1)  # archive API lags ~a few days behind real-time
    start = end - timedelta(days=days)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
    }
    print(f"[ingest_weather] Fetching historical weather {start} -> {end} for {location_name}...")
    payload = _fetch_json(ARCHIVE_URL, params)
    return _hourly_to_dataframe(payload, location_name)


def fetch_forecast_weather(lat: float, lon: float, days: int, location_name: str) -> pd.DataFrame:
    """Forecast weather as it was actually issued, for the same window as
    fetch_historical_weather, via Open-Meteo's Historical Forecast API.

    This is a fixed-lead-time archive (the forecast run as it stood at the
    time), not the continuously-updated `past_days` data the plain forecast
    endpoint returns. Using the same start/end window as the historical pull
    means every timestamp in weather_forecast.parquet has a matching row in
    weather_historical.parquet, so analysis.py's forecast-vs-actual join
    still lines up the same way it did before.
    """
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
    }
    print(f"[ingest_weather] Fetching archived (as-issued) forecast weather {start} -> {end} for {location_name}...")
    payload = _fetch_json(HISTORICAL_FORECAST_URL, params)
    return _hourly_to_dataframe(payload, location_name)


def main() -> None:
    try:
        hist_df = fetch_historical_weather(LAT, LON, HISTORY_DAYS, LOCATION_NAME)
        fcst_df = fetch_forecast_weather(LAT, LON, HISTORY_DAYS, LOCATION_NAME)
    except Exception as exc:
        print(f"[ingest_weather] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    hist_path = RAW_DIR / "weather_historical.parquet"
    fcst_path = RAW_DIR / "weather_forecast.parquet"
    hist_df.to_parquet(hist_path, index=False)
    fcst_df.to_parquet(fcst_path, index=False)

    print(f"[ingest_weather] Wrote {len(hist_df):,} historical rows -> {hist_path}")
    print(f"[ingest_weather] Wrote {len(fcst_df):,} forecast rows   -> {fcst_path}")
    print(f"[ingest_weather] Historical range: {hist_df['timestamp'].min()} -> {hist_df['timestamp'].max()}")
    print(f"[ingest_weather] Forecast range:   {fcst_df['timestamp'].min()} -> {fcst_df['timestamp'].max()}")


if __name__ == "__main__":
    main()
