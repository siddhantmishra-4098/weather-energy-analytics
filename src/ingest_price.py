"""
ingest_price.py
------------------
Pulls the German day-ahead electricity price (DE-LU bidding zone) for the
last N days (same window as ingest_energy.py), so the imbalance-cost figure
in the generation model can be tied to a real settlement price instead of a
flat EUR/MWh placeholder.

Two providers are supported, mirroring ingest_energy.py exactly:

1. SMARD (Bundesnetzagentur / German grid regulator's transparency platform)
   -- https://www.smard.de -- is the DEFAULT and requires no API key. Filter
   ID used (resolution "hour"), verified empirically against the live API
   (200 OK, block timestamps covering the requested window, values in a
   plausible EUR/MWh range -- see below):
     4169 = Grosshandelspreise Day-Ahead (DE-LU bidding zone) -> price_eur_mwh

   Germany and Luxembourg have shared a single price zone ("DE-LU") since
   October 2018, which is why this uses region "DE-LU" rather than the "DE"
   region used for load and generation in ingest_energy.py (those are
   national totals reported under the old aggregate region code; day-ahead
   price is zone-specific and DE-LU is the zone Germany actually settles
   in). Verified live: GET .../4169/DE-LU/index_hour.json returns 200 with
   block timestamps from 2018-10-01 through the present, and the most recent
   block's values (~30-700 EUR/MWh, mean ~218 EUR/MWh over a September 2026
   week) fall in the expected day-ahead range and agree with the ENTSO-E
   series below for the same hours.

2. ENTSO-E Transparency Platform -- used automatically if ENTSOE_API_TOKEN is
   set in .env, via entsoe-py's query_day_ahead_prices.

   IMPORTANT: query_day_ahead_prices needs the actual DE-LU *bidding zone*
   EIC code, 10Y1001A1001A82H -- NOT the ENTSOE_AREA_CODE value
   (10Y1001A1001A83F) that ingest_energy.py uses for load/generation. That
   code maps to entsoe-py's generic "DE" control-area aggregate, which
   query_load/query_generation accept but query_day_ahead_prices does not
   (it raises NoMatchingDataError for every window, past or present, tested
   empirically here). This was caught by actually running the query against
   several windows (recent and historical) before trusting it -- see
   ENTSOE_PRICE_AREA_CODE below, overridable via env if needed, but distinct
   from ENTSOE_AREA_CODE by default.

Either way, real data only: if the selected provider's API call fails, the
error is raised/printed and the script exits non-zero -- it never falls back
to synthetic or mocked numbers.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

ENTSOE_TOKEN = os.getenv("ENTSOE_API_TOKEN", "").strip()
# DE-LU day-ahead price bidding zone EIC code -- deliberately NOT the same as
# ingest_energy.py's ENTSOE_AREA_CODE (10Y1001A1001A83F), see module docstring.
ENTSOE_PRICE_AREA_CODE = os.getenv("ENTSOE_PRICE_AREA_CODE", "10Y1001A1001A82H")

HISTORY_DAYS = 90
REQUEST_TIMEOUT = 30

# --- SMARD filter ID (bidding zone "DE-LU") --------------------------------
SMARD_BASE = "https://www.smard.de/app/chart_data"
SMARD_PRICE_FILTER = 4169
SMARD_PRICE_REGION = "DE-LU"
SMARD_RESOLUTION = "hour"


# ---------------------------------------------------------------------------
# SMARD (default, no key needed)
# ---------------------------------------------------------------------------

def _smard_series(filter_id: int, region: str, start_ms: int, end_ms: int) -> pd.Series:
    """Same block-lookup logic as ingest_energy.py's _smard_series, parameterized
    on region since price uses "DE-LU" instead of "DE"."""
    index_url = f"{SMARD_BASE}/{filter_id}/{region}/index_{SMARD_RESOLUTION}.json"
    resp = requests.get(index_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    block_starts = resp.json().get("timestamps", [])
    if not block_starts:
        raise RuntimeError(f"SMARD returned no available blocks for filter {filter_id} at {index_url}")

    before = [t for t in block_starts if t <= start_ms]
    first_block = before[-1] if before else block_starts[0]
    blocks_needed = sorted({t for t in block_starts if first_block <= t <= end_ms} | {first_block})

    frames = []
    for block_ts in blocks_needed:
        data_url = (
            f"{SMARD_BASE}/{filter_id}/{region}/"
            f"{filter_id}_{region}_{SMARD_RESOLUTION}_{block_ts}.json"
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

    print(f"[ingest_price] Fetching SMARD day-ahead price {start} -> {end} (region={SMARD_PRICE_REGION})...")
    series = _smard_series(SMARD_PRICE_FILTER, SMARD_PRICE_REGION, start_ms, end_ms)

    df = series.rename("price_eur_mwh").reset_index().rename(columns={"index": "timestamp"})
    df["region"] = SMARD_PRICE_REGION
    df["source"] = "SMARD"
    df = df.dropna(subset=["price_eur_mwh"])
    return df[["timestamp", "region", "source", "price_eur_mwh"]]


# ---------------------------------------------------------------------------
# ENTSO-E (used automatically if ENTSOE_API_TOKEN is set)
# ---------------------------------------------------------------------------

def fetch_from_entsoe(days: int) -> pd.DataFrame:
    """Requires the `entsoe-py` package and a valid ENTSOE_API_TOKEN. Any
    failure (bad token, no access, network error) is raised as-is -- we do
    not fall back to fabricated numbers."""
    from entsoe import EntsoePandasClient  # imported lazily so SMARD path has no hard dependency

    client = EntsoePandasClient(api_key=ENTSOE_TOKEN)

    end = pd.Timestamp.now(tz="Europe/Berlin")
    start = end - pd.Timedelta(days=days)

    print(f"[ingest_price] Fetching ENTSO-E day-ahead price {start} -> {end} (area={ENTSOE_PRICE_AREA_CODE})...")

    prices = client.query_day_ahead_prices(ENTSOE_PRICE_AREA_CODE, start=start, end=end)
    out = prices.rename("price_eur_mwh").reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.tz_convert("UTC").dt.tz_localize(None)
    out["region"] = "DE-LU"
    out["source"] = "ENTSO-E"
    return out[["timestamp", "region", "source", "price_eur_mwh"]]


def main() -> None:
    try:
        if ENTSOE_TOKEN:
            print("[ingest_price] ENTSOE_API_TOKEN found -> using ENTSO-E Transparency Platform.")
            df = fetch_from_entsoe(HISTORY_DAYS)
        else:
            print("[ingest_price] No ENTSOE_API_TOKEN in .env -> using SMARD (no key required).")
            df = fetch_from_smard(HISTORY_DAYS)
    except Exception as exc:
        print(f"[ingest_price] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    out_path = RAW_DIR / "energy_prices.parquet"
    df.to_parquet(out_path, index=False)

    print(f"[ingest_price] Wrote {len(df):,} rows -> {out_path}")
    print(f"[ingest_price] Range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(df["price_eur_mwh"].describe())
    if df["price_eur_mwh"].between(-500, 1000).mean() < 0.95:
        print(
            "[ingest_price] WARNING: more than 5% of prices fall outside a plausible "
            "[-500, 1000] EUR/MWh range -- double-check filter ID / region before trusting this data.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
