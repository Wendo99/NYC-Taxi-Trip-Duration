from pathlib import Path

# Project paths # -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[
  2]
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ZIP_DIR = DATA_DIR / "zipped"
CACHE_DIR = DATA_DIR / "cached"

MERGED_CSV = RAW_DIR / "taxi_weather.csv"

for _p in (RAW_DIR, PROCESSED_DIR, ZIP_DIR, CACHE_DIR):
  _p.mkdir(parents=True, exist_ok=True)

# Modelling / evaluation defaults # -------------------------------------------

SPLIT_RATIO = 0.20
RANDOM_SEED = 42
