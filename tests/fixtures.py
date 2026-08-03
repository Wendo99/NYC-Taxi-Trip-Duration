"""Small deterministic frames used by the characterisation tests.

Everything here is synthetic and hand-checkable: no Kaggle credentials, no
OSRM server, no reading of the multi-hundred-megabyte processed CSVs. The
point is to pin *behaviour*, not to reproduce the real dataset.
"""
from __future__ import annotations

import pandas as pd

# Two well-known NYC reference points, used so the distance and airport
# assertions stay meaningful rather than arbitrary.
TIMES_SQUARE = (-73.9855, 40.7580)
JFK = (-73.7900, 40.6450)
LAGUARDIA = (-73.8740, 40.7700)


def taxi_frame() -> pd.DataFrame:
  """Six trips covering the branches the taxi features care about.

  Row 0  ordinary midtown trip, weekday afternoon
  Row 1  midtown -> JFK, morning rush
  Row 2  midtown -> LaGuardia, evening rush
  Row 3  group trip at night
  Row 4  New Year's Day (US federal holiday), early morning
  Row 5  out-of-bounds coordinates and an over-long duration (outlier row)
  """
  return pd.DataFrame({
    "id": [f"id{i}" for i in range(6)],
    "vendor_id": [1, 2, 1, 2, 1, 2],
    "pickup_datetime": pd.to_datetime([
      "2016-03-15 14:30:00",  # Tuesday afternoon
      "2016-03-15 07:15:00",  # AM rush
      "2016-03-15 17:45:00",  # PM rush
      "2016-03-19 23:10:00",  # Saturday night
      "2016-01-01 04:05:00",  # holiday, early morning
      "2016-06-30 12:00:00",
    ]),
    "dropoff_datetime": pd.to_datetime([
      "2016-03-15 14:45:00", "2016-03-15 08:00:00", "2016-03-15 18:30:00",
      "2016-03-19 23:25:00", "2016-01-01 04:20:00", "2016-06-30 14:00:00",
    ]),
    "passenger_count": [1, 2, 1, 4, 1, 3],
    "pickup_longitude": [
      TIMES_SQUARE[0], TIMES_SQUARE[0], TIMES_SQUARE[0],
      TIMES_SQUARE[0], TIMES_SQUARE[0], -75.5,
    ],
    "pickup_latitude": [
      TIMES_SQUARE[1], TIMES_SQUARE[1], TIMES_SQUARE[1],
      TIMES_SQUARE[1], TIMES_SQUARE[1], 41.9,
    ],
    "dropoff_longitude": [
      -73.9700, JFK[0], LAGUARDIA[0], -73.9600, -73.9900, -75.6,
    ],
    "dropoff_latitude": [
      40.7600, JFK[1], LAGUARDIA[1], 40.7700, 40.7500, 41.8,
    ],
    "store_and_fwd_flag": ["N", "N", "Y", "N", "N", "N"],
    "trip_duration": [900, 2700, 2700, 900, 900, 7200],
  })


def weather_frame() -> pd.DataFrame:
  """Four hourly observations in imperial units, as the raw Kaggle feed is."""
  return pd.DataFrame({
    "timestamp": [
      "2016-03-15 14:00:00", "2016-03-15 14:30:00",
      "2016-03-15 15:00:00", "2016-01-01 04:00:00",
    ],
    "temp": [50.0, 52.0, 55.0, 30.0],            # Fahrenheit
    "windspeed": [10.0, 12.0, 8.0, 20.0],        # mph
    "precip": [0.0, 0.1, 0.0, 0.2],              # inches
    "pressure": [29.92, 29.90, 29.95, 30.10],    # inches Hg
    "dailyprecip": ["0.0", "T", "0.0", "0.5"],   # 'T' == trace
    "dailysnow": ["0.0", "0.0", "0.0", "T"],
    "humidity": [55.0, 57.0, 50.0, 70.0],
    "fog": [0, 0, 0, 1],
    "rain": [0, 1, 0, 1],
    "snow": [0, 0, 0, 1],
    "conditions": ["Clear", "Light Rain", "Partly Cloudy", "Light Snow"],
  })
