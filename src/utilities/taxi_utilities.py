from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.array_api import int32
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
      int32)
  return df


def create_is_group_trip(df):
  df["is_group_trip"] = (df["passenger_count"] >= 2).astype("int8")
  return df


def get_la_gua(df):
  # Convert columns to NumPy arrays for pickups
  lon = df["pickup_longitude"].to_numpy()
  lat = df["pickup_latitude"].to_numpy()
  # Define La Guardia boundaries (approximate)
  la_lon = (-73.894, -73.861)
  la_lat = (40.774, 40.765)
  # Vectorized flag for La Guardia pickups
  df["is_laguardia_pick"] = (
      (la_lon[0] <= lon) & (lon <= la_lon[1]) &
      (la_lat[0] <= lat) & (lat <= la_lat[1])
  ).astype("int8")
  # Repeat for drop-offs (update lon/lat arrays)
  lon = df["dropoff_longitude"].to_numpy()
  lat = df["dropoff_latitude"].to_numpy()
  df["is_laguardia_drop"] = (
      (la_lon[0] <= lon) & (lon <= la_lon[1]) &
      (la_lat[0] <= lat) & (lat <= la_lat[1])
  ).astype("int8")
  return df


def get_jfk_flag(df):
  # Convert columns to NumPy arrays
  lon = df["pickup_longitude"].to_numpy()
  lat = df["pickup_latitude"].to_numpy()
  # Define JFK boundaries
  jfk_lon = (-73.837, -73.745)  # widen by 0.01° east-west
  jfk_lat = (40.622, 40.675)  # widen by 0.01° south-north
  # Vectorized flag for JFK pickups
  df["is_jfk_pick"] = (
      (jfk_lon[0] <= lon) & (lon <= jfk_lon[1]) &
      (jfk_lat[0] <= lat) & (lat <= jfk_lat[1])
  ).astype("int8")
  # Repeat for drop-offs (update lon/lat arrays)
  lon = df["dropoff_longitude"].to_numpy()
  lat = df["dropoff_latitude"].to_numpy()
  df["is_jfk_drop"] = (
      (jfk_lon[0] <= lon) & (lon <= jfk_lon[1]) &
      (jfk_lat[0] <= lat) & (lat <= jfk_lat[1])
  ).astype("int8")
  return df
