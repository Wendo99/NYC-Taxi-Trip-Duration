"""Feature engineering for the taxi trips.

Each function takes a DataFrame and returns a copy with columns added, so the
pipeline in ``taxi_pipeline`` reads as a chain of transformations. Note
``add_time_features`` here derives *pickup*-based features; the same-named
function in ``weather_utilities`` derives observation-hour features. Both
produce ``hour_of_year``, which is the join key between the two datasets, so
they must stay in step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import get_calendar
from sklearn.cluster import MiniBatchKMeans

from nyc_taxi.features.distance_utilities import _check_columns


# ------------------------------------------------------------------ #
def add_us_holiday_flag(
    df: pd.DataFrame,
    dt_col: str,
    flag_col: str = "is_holiday",
) -> pd.DataFrame:
  """
  Add boolean *flag_col* that is **True** on U.S. federal holidays.

  Parameters
  ----------
  df : DataFrame
      Input taxi records.
  dt_col : str
      Name of the datetime column (must be timezone-aware or in local NYC time).
  flag_col : str, default ``"is_holiday"``
      Name of the new column.

  Returns
  -------
  DataFrame
      Copy of *df* with one additional boolean column.
  """
  _check_columns(df, [dt_col])
  df = df.copy()

  df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

  # Looked up through the public registry: pandas' __all__ for
  # pandas.tseries.holiday lists get_calendar but not the calendar classes
  # themselves, so importing USFederalHolidayCalendar directly reaches past
  # the module's declared API. Same class, same holidays.
  cal = get_calendar("USFederalHolidayCalendar")
  holidays = cal.holidays(
      start=df[dt_col].min().normalize(),
      end=df[dt_col].max().normalize(),
  )
  df[flag_col] = df[dt_col].dt.normalize().isin(holidays).astype("int8")
  return df


# Time-of-day bands, inclusive on both ends, as hours 0-23. These encode the
# NYC traffic rhythm that notebooks/taxi.ipynb section 4 measures: a trough
# around 05:00 and a peak at 18:00.
EARLY_MORNING_HOURS = (3, 5)
RUSH_AM_HOURS = (6, 8)
RUSH_PM_HOURS = (16, 18)
NIGHT_STARTS_AT = 22        # 22:00 onwards ...
NIGHT_ENDS_BEFORE = 3       # ... through to 02:59
SATURDAY = 5                # dayofweek: Monday=0
HOURS_PER_DAY = 24


def add_time_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
  """Derive hour, weekday, month, hour-of-year and the time-of-day flags.

  ``hour_of_year`` is the join key to the weather data, so it must match
  ``weather_utilities.add_time_features``.
  """
  _check_columns(df, [dt_col])
  df = df.copy()
  ts = df[dt_col].dt

  df["pickup_hour"] = ts.hour.astype("int8")
  df["pickup_weekday"] = ts.dayofweek.astype("int8")
  df["pickup_month"] = ts.month.astype("int8")
  df["hour_of_year"] = (
      (ts.dayofyear - 1) * HOURS_PER_DAY + ts.hour).astype("int16")
  df["is_weekend"] = (ts.dayofweek >= SATURDAY).astype("int8")

  hour = df["pickup_hour"]
  df["is_early_morning"] = hour.between(*EARLY_MORNING_HOURS, 'both').astype(
      "int8")
  df["is_rush_am"] = hour.between(*RUSH_AM_HOURS, 'both').astype("int8")
  df["is_rush_pm"] = hour.between(*RUSH_PM_HOURS, 'both').astype("int8")
  df["is_night"] = (
      (hour < NIGHT_ENDS_BEFORE) | (hour >= NIGHT_STARTS_AT)
  ).astype("int8")
  return df


def add_trip_duration_features(df: pd.DataFrame) -> pd.DataFrame:
  """Add duration in minutes and log-seconds (← skew reduction)."""
  _check_columns(df, ["trip_duration"])
  df = df.copy()
  df["trip_duration_min"] = (df["trip_duration"] / 60.0).astype("float32")
  df["trip_duration_log"] = np.log1p(df["trip_duration"]).astype("float32")
  return df


def add_store_and_fwd_flag(
    df: pd.DataFrame,
    src_col: str = "store_and_fwd_flag",
    dest_col: str = "store_and_fwd_flag_bin",
) -> pd.DataFrame:
  """
  Convert ``"Y"``/``"N"`` string flag to 1/0 tiny integer.
  Unknown values become NaN (Int8 can hold that).
  """
  _check_columns(df, [src_col])
  df = df.copy()
  mapping = {"Y": 1, "N": 0}
  df[dest_col] = df[src_col].map(mapping).astype("Int8")
  return df


def create_geo_clusters(df, feature_cols, prefix, n_clusters, random_state,
    batch_size):
  """Assign each row to a MiniBatchKMeans cluster over *feature_cols*.

  Gives the model a coarse notion of neighbourhood that raw coordinates do
  not; the clusters are one-hot encoded downstream.
  """
  coords = df[feature_cols]
  kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                           random_state=random_state,
                           batch_size=batch_size)
  cluster_labels = kmeans.fit_predict(coords)
  df[f'{prefix}_cluster'] = pd.Series(cluster_labels, index=df.index).astype(
      np.int32)
  return df


GROUP_TRIP_MIN_PASSENGERS = 2


def create_is_group_trip(df):
  """Flag trips carrying two or more passengers."""
  df["is_group_trip"] = (
      df["passenger_count"] >= GROUP_TRIP_MIN_PASSENGERS).astype("int8")
  return df


def _within_box(lon, lat, lon_bounds, lat_bounds) -> np.ndarray:
  """Vectorised point-in-bounding-box test."""
  return (
      (lon_bounds[0] <= lon) & (lon <= lon_bounds[1]) &
      (lat_bounds[0] <= lat) & (lat <= lat_bounds[1])
  )


def add_airport_flags(df: pd.DataFrame, name: str, lon_bounds, lat_bounds
    ) -> pd.DataFrame:
  """Flag trips starting or ending inside an airport bounding box.

  Adds ``is_{name}_pick`` and ``is_{name}_drop``. Replaces the former
  ``get_jfk_flag`` / ``get_la_gua`` pair, which were identical apart from
  their constants.
  """
  _check_columns(df, ["pickup_longitude", "pickup_latitude",
                      "dropoff_longitude", "dropoff_latitude"])
  for role in ("pickup", "dropoff"):
    inside = _within_box(
        df[f"{role}_longitude"].to_numpy(),
        df[f"{role}_latitude"].to_numpy(),
        lon_bounds, lat_bounds)
    suffix = "pick" if role == "pickup" else "drop"
    df[f"is_{name}_{suffix}"] = inside.astype("int8")
  return df
