"""Every filesystem path the project uses.

All paths are absolute, derived from this file's own location, so the code
behaves identically whether it is run from ``notebooks/``, ``src/`` or the
repository root. Nothing here should ever be a relative literal.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- data ------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ZIP_DIR = DATA_DIR / "zipped"
CACHE_DIR = DATA_DIR / "cached"
PROCESSED_DIR = DATA_DIR / "processed"
DERIVED_DIR = DATA_DIR / "derived"

# --- outputs ---------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# --- taxi ------------------------------------------------------------------
TAXI_RAW_ZIP = ZIP_DIR / "nyc-taxi-trip-duration.zip"
TAXI_RAW_CSV = RAW_DIR / "train.csv"
TAXI_CACHE_PICKLE = CACHE_DIR / "taxi_cache.pkl"
TAXI_PROCESSED_CSV = PROCESSED_DIR / "taxi_clean.csv"

# Resumable checkpoint for the OSRM route distances. Absolute, so the
# pipeline resumes from the same file no matter which directory it runs in.
ROUTE_DIST_PARQUET = DERIVED_DIR / "with_route_dist.parquet"

# --- weather ---------------------------------------------------------------
WEATHER_RAW_ZIP = ZIP_DIR / "nyc-taxi-wunderground-weather.zip"
WEATHER_RAW_CSV_NAME = "weatherdata.csv"
WEATHER_RAW_CSV1 = RAW_DIR / WEATHER_RAW_CSV_NAME
WEATHER_RAW_CSV2 = RAW_DIR / "weather2_raw.csv"
WEATHER_CACHE_PICKLE = CACHE_DIR / "weather_cache.pkl"
WEATHER_PROCESSED_CSV = PROCESSED_DIR / "weather_clean.csv"

# --- merged ----------------------------------------------------------------
# Single definition. This previously existed three times: here (pointing at
# raw/), in merge_pipeline (pointing at processed/) and hardcoded again in
# data_io.load_taxi_weather_data.
MERGED_CSV = PROCESSED_DIR / "taxi_weather.csv"

_MANAGED_DIRS = (
  RAW_DIR, PROCESSED_DIR, ZIP_DIR, CACHE_DIR, DERIVED_DIR,
  MODELS_DIR, FIGURES_DIR, ARTIFACTS_DIR,
)

for _p in _MANAGED_DIRS:
  _p.mkdir(parents=True, exist_ok=True)
