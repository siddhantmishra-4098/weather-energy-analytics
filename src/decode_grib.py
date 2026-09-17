"""
decode_grib.py
------------------
Downloads a real forecast field in native GRIB2 format from ECMWF's open-data
feed (https://data.ecmwf.int / open-data.ecmwf.int, via the official
`ecmwf-opendata` client) and decodes it with cfgrib + xarray -- i.e. ecCodes
under the hood. This is the part of the project that specifically exercises
NWP file formats rather than a JSON/REST weather API.

ECMWF open-data serves whole global fields (no server-side spatial
subsetting for the free feed), so the workflow is:
  1. Download one GRIB2 message: 2m temperature ("2t") from the latest
     available IFS 0.25-degree forecast run, at two lead times (analysis
     step 0h and a forecast step 24h ahead).
  2. Decode with `xarray.open_dataset(..., engine="cfgrib")`.
  3. Crop the global grid down to a Central Europe bounding box (client-side,
     with xarray) so plotting is fast and the result is relevant to the
     Germany-focused analysis in this project.
  4. Plot the cropped field and save a PNG, and print the value at the grid
     point nearest to our reference location (Dortmund) as a sanity check.

No API key is required -- ECMWF open-data is public under CC-BY-4.0
(attribution required, which is why every plot/README mentions ECMWF).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only writes files, no display needed
import matplotlib.pyplot as plt
import xarray as xr
from dotenv import load_dotenv
from ecmwf.opendata import Client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

LOCATION_NAME = os.getenv("LOCATION_NAME", "Dortmund")
LAT = float(os.getenv("LOCATION_LAT", "51.5136"))
LON = float(os.getenv("LOCATION_LON", "7.4653"))

# Central Europe bounding box used to crop the global GRIB field for plotting.
BBOX = {"lat_min": 42.0, "lat_max": 58.0, "lon_min": -5.0, "lon_max": 20.0}

GRIB_PARAM = "2t"  # 2m temperature -- the variable most directly tied to power demand
FORECAST_STEP_HOURS = 24


def download_grib(step: int, target: Path) -> Path:
    """Download one GRIB2 field from the latest available ECMWF open-data run."""
    client = Client()  # defaults: source="ecmwf", model="ifs", resol="0p25"
    print(f"[decode_grib] Requesting param={GRIB_PARAM} step={step}h from ECMWF open-data (latest run)...")
    try:
        client.retrieve(type="fc", stream="oper", step=step, param=GRIB_PARAM, target=str(target))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download GRIB2 data from ECMWF open-data: {exc}\n"
            "Check connectivity to https://data.ecmwf.int / object storage mirrors."
        ) from exc
    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(f"ECMWF open-data download produced no data at {target}")
    print(f"[decode_grib] Downloaded {target} ({target.stat().st_size / 1024:.1f} KB)")
    return target


def decode_and_crop(grib_path: Path) -> xr.Dataset:
    """Decode a GRIB2 file with cfgrib (ecCodes) and crop to BBOX."""
    print(f"[decode_grib] Decoding {grib_path.name} with cfgrib/ecCodes...")
    ds = xr.open_dataset(grib_path, engine="cfgrib")

    # ECMWF open-data longitudes run 0..360 or -180..180 depending on field;
    # normalize to -180..180 so our bounding box logic is simple either way.
    if float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")

    cropped = ds.sel(
        latitude=slice(BBOX["lat_max"], BBOX["lat_min"]),  # latitude is descending in this grid
        longitude=slice(BBOX["lon_min"], BBOX["lon_max"]),
    )
    return cropped


def plot_field(ds: xr.Dataset, var: str, title: str, out_path: Path) -> None:
    data = ds[var] - 273.15  # Kelvin -> Celsius for temperature
    fig, ax = plt.subplots(figsize=(8, 6))
    im = data.plot(ax=ax, cmap="RdYlBu_r", add_colorbar=True, cbar_kwargs={"label": "2m temperature (°C)"})
    ax.scatter([LON], [LAT], color="black", marker="*", s=120, zorder=5, label=LOCATION_NAME)
    ax.legend(loc="upper right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[decode_grib] Saved plot -> {out_path}")


def nearest_point_value(ds: xr.Dataset, var: str, lat: float, lon: float) -> float:
    point = ds[var].sel(latitude=lat, longitude=lon, method="nearest")
    return float(point.values) - 273.15


def main() -> None:
    try:
        analysis_path = download_grib(0, RAW_DIR / "ecmwf_2t_step00.grib2")
        forecast_path = download_grib(FORECAST_STEP_HOURS, RAW_DIR / f"ecmwf_2t_step{FORECAST_STEP_HOURS:02d}.grib2")

        ds_analysis = decode_and_crop(analysis_path)
        ds_forecast = decode_and_crop(forecast_path)
    except Exception as exc:
        print(f"[decode_grib] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    run_time = str(ds_analysis["time"].values)[:16].replace("T", " ")

    plot_field(
        ds_analysis,
        "t2m",
        f"ECMWF IFS 2m temperature -- analysis (run {run_time}Z, step 0h)",
        PROCESSED_DIR / "grib_2t_step00.png",
    )
    plot_field(
        ds_forecast,
        "t2m",
        f"ECMWF IFS 2m temperature -- +{FORECAST_STEP_HOURS}h forecast (run {run_time}Z)",
        PROCESSED_DIR / f"grib_2t_step{FORECAST_STEP_HOURS:02d}.png",
    )

    t_now = nearest_point_value(ds_analysis, "t2m", LAT, LON)
    t_fcst = nearest_point_value(ds_forecast, "t2m", LAT, LON)
    print(f"[decode_grib] {LOCATION_NAME} nearest-gridpoint 2m temp: "
          f"{t_now:.1f}C (step 0h) -> {t_fcst:.1f}C (step {FORECAST_STEP_HOURS}h)")
    print("[decode_grib] Data source: ECMWF open-data, CC-BY-4.0 -- https://www.ecmwf.int/en/forecasts/datasets/open-data")


if __name__ == "__main__":
    main()
