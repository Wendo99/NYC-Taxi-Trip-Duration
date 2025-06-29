from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from constants.core_c import ZIP_DIR, RAW_DIR, CACHE_DIR, PROCESSED_DIR

# File locations ------------------------------------------------------------
TAXI_RAW_ZIP = ZIP_DIR / 'nyc-taxi-trip-duration.zip'
TAXI_RAW_CSV = RAW_DIR / "train.csv"
TAXI_CACHE_PICKLE = CACHE_DIR / "taxi_cache.pkl"
TAXI_PROCESSED_CSV = PROCESSED_DIR / "taxi_clean.csv"

RAW_TIME_COL = "pickup_datetime"


# ------------------------------------------------------------


@dataclass(frozen=True)
class PassengerLimits:
  min_passengers: int = 1
  max_passengers: int = 6


@dataclass(frozen=True)
class TripDurationLimits:
  min_sec: int = 60
  max_sec: int = 3 * 3600


@dataclass(frozen=True)
class GeoBounds:
  min_lon: float = -74.05
  max_lon: float = -73.73
  min_lat: float = 40.59
  max_lat: float = 40.90
  cols: ClassVar[list[str]] = [
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
  ]


# ------------------------------------------------------------

n_pickup_clusters = 5
n_dropoff_clusters = 8
batch_size = 10_000
