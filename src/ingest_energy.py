"""
ingest_energy.py
------------------
Pulls German electricity market data (total grid load + wind/solar generation)
for the last N days.

Two providers are supported:

1. SMARD (Bundesnetzagentur / German grid regulator's transparency platform)
   -- https://www.smard.de -- is the DEFAULT and requires no API key or
   registration at all. Its `chart_data` JSON endpoints are publicly
   documented (see https://github.com/bundesAPI/smard-api) and are what this
   script uses out of the box. Filter IDs used (region "DE", resolution
   "hour"), verified empirically against the live API:
     410  = Stromverbrauch: Gesamt (Netzlast)      -> total_load_mw
     4067 = Stromerzeugung: Wind Onshore (realized) -> wind_onshore_mw
     1225 = Stromerzeugung: Wind Offshore (realized) -> wind_offshore_mw
     4068 = Stromerzeugung: Photovoltaik (realized)  -> solar_mw

   Forecast counterparts (Day-Ahead), verified the same way -- confirmed via
   the smard.de "Marktdaten visualisieren" UI (module IDs 6000411, 2000123,
   2003791, 2000125, each offset from the underlying chart_data filter ID by
   a category prefix) and by checking that each filter's chart_data series
   actually extends ~1-2 days past "now" (a forecast horizon), unlike the
   realized filters above which stop at "now":
     411  = Stromverbrauch: Prognostizierte Netzlast (Day-Ahead) -> total_load_mw
     123  = Stromerzeugung: Wind Onshore Prognose (Day-Ahead)    -> wind_onshore_mw
     3791 = Stromerzeugung: Wind Offshore Prognose (Day-Ahead)   -> wind_offshore_mw
     125  = Stromerzeugung: Photovoltaik Prognose (Day-Ahead)    -> solar_mw

2. ENTSO-E Transparency Platform -- https://transparency.entsoe.eu -- is the
   pan-European standard (and the one named explicitly in most energy-market
   job postings), but it requires a free account plus an emailed request to
   support to activate API access before a token is issued (see
   .env.example for the exact steps). This script will use ENTSO-E instead
   of SMARD automatically if `ENTSOE_API_TOKEN` is set in the environment.

Either way, real data only: if the selected provider's API call fails, the
error is raised/printed and the script exits non-zero -- it never falls back
to synthetic or mocked numbers.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

ENTSOE_TOKEN = os.getenv("ENTSOE_API_TOKEN", "").strip()
ENTSOE_AREA_CODE = os.getenv("ENTSOE_AREA_CODE", "10Y1001A1001A83F")  # Germany/Luxembourg bidding zone

HISTORY_DAYS = 90
FORECAST_HORIZON_DAYS = 7  # how far past "now" to keep pulling published forecasts, mirrors ingest_weather.py's forecast_days
REQUEST_TIMEOUT = 30

# --- SMARD filter IDs (region "DE") ---------------------------------------
SMARD_BASE = "https://www.smard.de/app/chart_data"
SMARD_FILTERS = {
    "total_load_mw": 410,
    "wind_onshore_mw": 4067,
    "wind_offshore_mw": 1225,
    "solar_mw": 4068,
}
SMARD_FORECAST_FILTERS = {
    "total_load_mw": 411,
    "wind_onshore_mw": 123,
    "wind_offshore_mw": 3791,
    "solar_mw": 125,
}
SMARD_REGION = "DE"
SMARD_RESOLUTION = "hour"


# ---------------------------------------------------------------------------
# SMARD (default, no key needed)
# ---------------------------------------------------------------------------

def _smard_series(filter_id: int, start_ms: int, end_ms: int) -> pd.Series:
    """Fetch one SMARD chart_data series and return it as a pd.Series
    indexed by UTC timestamp, clipped to [start_ms, end_ms].

    SMARD splits history into fixed "blocks"; we look up the available block
    start times via index_hour.json, pick the blocks that overlap our
    window, and concatenate their series.
    """
    index_url = f"{SMARD_BASE}/{filter_id}/{SMARD_REGION}/index_{SMARD_RESOLUTION}.json"
    resp = requests.get(index_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    block_starts = resp.json().get("timestamps", [])
    if not block_starts:
        raise RuntimeError(f"SMARD returned no available blocks for filter {filter_id} at {index_url}")

    # Blocks starting at/before our window start, plus every block inside the window.
    before = [t for t in block_starts if t <= start_ms]
    first_block = before[-1] if before else block_starts[0]
    blocks_needed = sorted({t for t in block_starts if first_block <= t <= end_ms} | {first_block})

    frames = []
    for block_ts in blocks_needed:
        data_url = (
            f"{SMARD_BASE}/{filter_id}/{SMARD_REGION}/"
            f"{filter_id}_{SMARD_REGION}_{SMARD_RESOLUTION}_{block_ts}.json"
        )
        r = requests.get(data_url, timeout=REQUEST_TIMEOUT)
        try:
            r.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"SMARD request failed ({r.status_code}) for {data_url}") from exc
        series = r.json().get("series", [])
        if series:
            frames.append(pd.DataFrame(series, columns=["timestamp_ms", "value"]))

    if not frames:
        raise RuntimeError(f"SMARD returned no series data for filter {filter_id} in the requested window")

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="timestamp_ms")
    df = df[(df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)]
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_localize(None)
    return df.set_index("timestamp")["value"]


def fetch_from_smard(days: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    print(f"[ingest_energy] Fetching SMARD data {start} -> {end} (region={SMARD_REGION})...")
    series_by_col = {}
    for col_name, filter_id in SMARD_FILTERS.items():
        print(f"[ingest_energy]   filter {filter_id} -> {col_name}")
        series_by_col[col_name] = _smard_series(filter_id, start_ms, end_ms)

    df = pd.DataFrame(series_by_col)
    df = df.sort_index().reset_index().rename(columns={"index": "timestamp"})
    df["region"] = "DE"
    df["source"] = "SMARD"

    # Drop rows where every measured column is NaN (e.g. not-yet-published near "now").
    value_cols = list(SMARD_FILTERS.keys())
    df = df.dropna(subset=value_cols, how="all")
    df["wind_total_mw"] = df[["wind_onshore_mw", "wind_offshore_mw"]].sum(axis=1, min_count=1)
    return df[["timestamp", "region", "source"] + value_cols + ["wind_total_mw"]]


def fetch_forecast_from_smard(days: int) -> pd.DataFrame:
    """Day-Ahead forecast counterparts of fetch_from_smard's actuals.

    SMARD archives its own past Day-Ahead forecasts, so a window of
    [now - days, now] returns the forecast that was published for each of
    those past days. The window's upper bound is pushed FORECAST_HORIZON_DAYS
    past "now" as well, so the most recently published forecast (which
    extends ~1-2 days ahead of "now") isn't clipped off -- this is the same
    truncation bug fixed in the weather forecast chart, applied here from the
    start.
    """
    end = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=FORECAST_HORIZON_DAYS)
    start = end - timedelta(days=days + FORECAST_HORIZON_DAYS)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    print(f"[ingest_energy] Fetching SMARD forecast data {start} -> {end} (region={SMARD_REGION})...")
    series_by_col = {}
    for col_name, filter_id in SMARD_FORECAST_FILTERS.items():
        print(f"[ingest_energy]   filter {filter_id} -> {col_name}")
        series_by_col[col_name] = _smard_series(filter_id, start_ms, end_ms)

    df = pd.DataFrame(series_by_col)
    df = df.sort_index().reset_index().rename(columns={"index": "timestamp"})
    df["region"] = "DE"
    df["source"] = "SMARD"

    value_cols = list(SMARD_FORECAST_FILTERS.keys())
    df = df.dropna(subset=value_cols, how="all")
    df["wind_total_mw"] = df[["wind_onshore_mw", "wind_offshore_mw"]].sum(axis=1, min_count=1)
    return df[["timestamp", "region", "source"] + value_cols + ["wind_total_mw"]]


# ---------------------------------------------------------------------------
# ENTSO-E (used automatically if ENTSOE_API_TOKEN is set)
# ---------------------------------------------------------------------------

def fetch_from_entsoe(days: int) -> pd.DataFrame:
    """Pull day-ahead-region actual load + wind/solar generation from ENTSO-E.

    Requires the `entsoe-py` package and a valid ENTSOE_API_TOKEN. Any
    failure (bad token, no access, network error) is raised as-is -- we do
    not fall back to fabricated numbers.
    """
    from entsoe import EntsoePandasClient  # imported lazily so SMARD path has no hard dependency

    client = EntsoePandasClient(api_key=ENTSOE_TOKEN)

    end = pd.Timestamp.now(tz="Europe/Berlin")
    start = end - pd.Timedelta(days=days)

    print(f"[ingest_energy] Fetching ENTSO-E data {start} -> {end} (area={ENTSOE_AREA_CODE})...")

    load = client.query_load(ENTSOE_AREA_CODE, start=start, end=end)
    # query_generation returns a wide DataFrame with a column per production type
    # (and, for some types, an "Actual Aggregated"/"Actual Consumption" sub-level).
    gen = client.query_generation(ENTSOE_AREA_CODE, start=start, end=end, psr_type=None)
    if isinstance(gen.columns, pd.MultiIndex):
        gen.columns = [c[0] if isinstance(c, tuple) else c for c in gen.columns]

    def _col(df: pd.DataFrame, name: str) -> pd.Series:
        matches = [c for c in df.columns if name.lower() in str(c).lower()]
        if not matches:
            return pd.Series(dtype=float)
        return df[matches[0]]

    out = pd.DataFrame(index=load.index)
    out["total_load_mw"] = load.iloc[:, 0] if isinstance(load, pd.DataFrame) else load
    out["wind_onshore_mw"] = _col(gen, "Wind Onshore")
    out["wind_offshore_mw"] = _col(gen, "Wind Offshore")
    out["solar_mw"] = _col(gen, "Solar")
    out = out.reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.tz_convert("UTC").dt.tz_localize(None)
    out["region"] = "DE"
    out["source"] = "ENTSO-E"
    out["wind_total_mw"] = out[["wind_onshore_mw", "wind_offshore_mw"]].sum(axis=1, min_count=1)
    return out[
        ["timestamp", "region", "source", "total_load_mw", "wind_onshore_mw", "wind_offshore_mw", "solar_mw", "wind_total_mw"]
    ]


def fetch_forecast_from_entsoe(days: int) -> pd.DataFrame:
    """Day-Ahead forecast counterpart of fetch_from_entsoe's actuals.

    Requires the `entsoe-py` package and a valid ENTSOE_API_TOKEN. Any
    failure (bad token, no access, network error) is raised as-is -- we do
    not fall back to fabricated numbers.
    """
    from entsoe import EntsoePandasClient  # imported lazily so SMARD path has no hard dependency

    client = EntsoePandasClient(api_key=ENTSOE_TOKEN)

    end = pd.Timestamp.now(tz="Europe/Berlin") + pd.Timedelta(days=FORECAST_HORIZON_DAYS)
    start = end - pd.Timedelta(days=days + FORECAST_HORIZON_DAYS)

    print(f"[ingest_energy] Fetching ENTSO-E forecast data {start} -> {end} (area={ENTSOE_AREA_CODE})...")

    load_fcst = client.query_load_forecast(ENTSOE_AREA_CODE, start=start, end=end)
    # query_wind_and_solar_forecast returns a wide DataFrame with a column per production type.
    gen_fcst = client.query_wind_and_solar_forecast(ENTSOE_AREA_CODE, start=start, end=end, psr_type=None)
    if isinstance(gen_fcst.columns, pd.MultiIndex):
        gen_fcst.columns = [c[0] if isinstance(c, tuple) else c for c in gen_fcst.columns]

    def _col(df: pd.DataFrame, name: str) -> pd.Series:
        matches = [c for c in df.columns if name.lower() in str(c).lower()]
        if not matches:
            return pd.Series(dtype=float)
        return df[matches[0]]

    out = pd.DataFrame(index=load_fcst.index)
    out["total_load_mw"] = load_fcst.iloc[:, 0] if isinstance(load_fcst, pd.DataFrame) else load_fcst
    out["wind_onshore_mw"] = _col(gen_fcst, "Wind Onshore")
    out["wind_offshore_mw"] = _col(gen_fcst, "Wind Offshore")
    out["solar_mw"] = _col(gen_fcst, "Solar")
    out = out.reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.tz_convert("UTC").dt.tz_localize(None)
    out["region"] = "DE"
    out["source"] = "ENTSO-E"
    out["wind_total_mw"] = out[["wind_onshore_mw", "wind_offshore_mw"]].sum(axis=1, min_count=1)
    return out[
        ["timestamp", "region", "source", "total_load_mw", "wind_onshore_mw", "wind_offshore_mw", "solar_mw", "wind_total_mw"]
    ]


def main() -> None:
    try:
        if ENTSOE_TOKEN:
            print("[ingest_energy] ENTSOE_API_TOKEN found -> using ENTSO-E Transparency Platform.")
            df = fetch_from_entsoe(HISTORY_DAYS)
            fcst_df = fetch_forecast_from_entsoe(HISTORY_DAYS)
        else:
            print("[ingest_energy] No ENTSOE_API_TOKEN in .env -> using SMARD (no key required).")
            df = fetch_from_smard(HISTORY_DAYS)
            fcst_df = fetch_forecast_from_smard(HISTORY_DAYS)
    except Exception as exc:
        print(f"[ingest_energy] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    out_path = RAW_DIR / "energy_actuals.parquet"
    fcst_path = RAW_DIR / "energy_forecast.parquet"
    df.to_parquet(out_path, index=False)
    fcst_df.to_parquet(fcst_path, index=False)

    print(f"[ingest_energy] Wrote {len(df):,} actual rows -> {out_path}")
    print(f"[ingest_energy] Range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(df.select_dtypes("number").describe().T[["count", "mean", "min", "max"]])

    print(f"[ingest_energy] Wrote {len(fcst_df):,} forecast rows -> {fcst_path}")
    print(f"[ingest_energy] Forecast range: {fcst_df['timestamp'].min()} -> {fcst_df['timestamp'].max()}")
    print(fcst_df.select_dtypes("number").describe().T[["count", "mean", "min", "max"]])


if __name__ == "__main__":
    main()
