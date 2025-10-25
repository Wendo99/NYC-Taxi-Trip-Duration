from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[
  2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

MERGED_CSV = RAW_DIR / "taxi_weather.csv"

ZIP_DIR = DATA_DIR / "zipped"
TAXI_RAW_ZIP = ZIP_DIR / 'nyc-taxi-trip-duration.zip'

TAXI_RAW_CSV = RAW_DIR / "train.csv"
CACHE_DIR = DATA_DIR / "cached"
TAXI_CACHE_PICKLE = CACHE_DIR / "taxi_cache.pkl"
PROCESSED_DIR = DATA_DIR / "processed"
TAXI_PROCESSED_CSV = PROCESSED_DIR / "taxi_clean.csv"

WEATHER_RAW_ZIP = Path(
    f"{ZIP_DIR}/nyc-taxi-wunderground-weather.zip")  # original Kaggle zip
WEATHER_RAW_CSV_NAME = "weatherdata.csv"
WEATHER_RAW_CSV1 = Path(
    f"{RAW_DIR}/weatherdata.csv")  # original Kaggle zip
WEATHER_RAW_CSV2 = Path(
    f"{RAW_DIR}/weather2_raw.csv")
WEATHER_CACHE_PICKLE = Path(f"{CACHE_DIR}/weather_cache.pkl")
WEATHER_PROCESSED_CSV = Path(f"{PROCESSED_DIR}/weather_clean.csv")

for _p in (RAW_DIR, PROCESSED_DIR, ZIP_DIR, CACHE_DIR):
  _p.mkdir(parents=True, exist_ok=True)
