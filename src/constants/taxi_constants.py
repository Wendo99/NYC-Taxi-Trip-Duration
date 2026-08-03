from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

# String Const. ----------------------------------------------------------------
TIME_REF_COL = "pickup_datetime"

PICKUP_COORDS = ['pickup_latitude',
                 'pickup_longitude']
DROPOFF_COORDS = ['dropoff_latitude', 'dropoff_longitude']

# Mini-Batch K-Means Cluster ---------------------------------------------------
ENABLE_MB = True
N_PICKUP_CLUSTERS = 5
N_DROPOFF_CLUSTERS = 12
CLUSTER_BATCH_SIZE = 100_000

# HDBC Cluster ---------------------------------------------------
ENABLE_HDBC = False
PICKUP_MIN_CLUSTER_SIZE = 10
PICKUP_MIN_SAMPLES = 2
# Previously written with trailing commas, which silently made these tuples
# ((10,) and (2,)) instead of ints.
DROPOFF_MIN_CLUSTER_SIZE = 10
DROPOFF_MIN_SAMPLES = 2

# Airport bounding boxes -------------------------------------------------------
# (min, max) per axis. Bounds are validated at import time because an inverted
# pair silently yields a flag that is always 0 — which is exactly what happened
# to LaGuardia, whose latitudes used to read (40.774, 40.765).
JFK_LON = (-73.837, -73.745)
JFK_LAT = (40.622, 40.675)
LAGUARDIA_LON = (-73.894, -73.861)
LAGUARDIA_LAT = (40.765, 40.774)

AIRPORT_BOXES = {
  "jfk": (JFK_LON, JFK_LAT),
  "laguardia": (LAGUARDIA_LON, LAGUARDIA_LAT),
}

for _name, (_lon, _lat) in AIRPORT_BOXES.items():
  if _lon[0] >= _lon[1] or _lat[0] >= _lat[1]:
    raise ValueError(
        f"{_name}: bounding box must be (min, max) per axis, got "
        f"lon={_lon}, lat={_lat}")


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
