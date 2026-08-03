from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.cluster import MiniBatchKMeans

from utilities.distance_utilities import _check_columns


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

  cal = USFederalHolidayCalendar()
  holidays = cal.holidays(
      start=df[dt_col].min().normalize(),
      end=df[dt_col].max().normalize(),
  )
  df[flag_col] = df[dt_col].dt.normalize().isin(holidays).astype("int8")
  return df


def add_time_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
  """Derive hour, weekday, month, hour-of-year and weekend flag."""
  _check_columns(df, [dt_col])
  df = df.copy()
  ts = df[dt_col].dt

  df["pickup_hour"] = ts.hour.astype("int8")
  df["pickup_weekday"] = ts.dayofweek.astype("int8")
  df["pickup_month"] = ts.month.astype("int8")
  df["hour_of_year"] = ((ts.dayofyear - 1) * 24 + ts.hour).astype("int16")
  df["is_weekend"] = (ts.dayofweek >= 5).astype("int8")

  df["is_early_morning"] = df["pickup_hour"].between(3, 5, 'both').astype(
      "int8")

  df["is_rush_am"] = df["pickup_hour"].between(6, 8, 'both').astype("int8")

  df["is_rush_pm"] = df["pickup_hour"].between(16, 18, 'both').astype("int8")

  df["is_night"] = (
      (df["pickup_hour"] < 3) |
      (df["pickup_hour"] >= 22)
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
  coords = df[feature_cols]
  kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                           random_state=random_state,
                           batch_size=batch_size)
  cluster_labels = kmeans.fit_predict(coords)
  df[f'{prefix}_cluster'] = pd.Series(cluster_labels, index=df.index).astype(
      np.int32)
  return df


def create_is_group_trip(df):
  df["is_group_trip"] = (df["passenger_count"] >= 2).astype("int8")
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
