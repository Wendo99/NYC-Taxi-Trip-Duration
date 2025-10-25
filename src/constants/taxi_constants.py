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
DROPOFF_MIN_CLUSTER_SIZE = 10,
DROPOFF_MIN_SAMPLES = 2,


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
