from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
import requests
from numpy.typing import ArrayLike
from tqdm.asyncio import tqdm

EARTH_RADIUS_KM: float = 6_378.137
OSRM_URL = "http://localhost:5001/route/v1/driving/"


def haversine(
    lat1: ArrayLike,
    lon1: ArrayLike,
    lat2: ArrayLike,
    lon2: ArrayLike,
    radius: float = EARTH_RADIUS_KM
) -> np.ndarray:
  """
  Vectorised great‑circle distance between two points using the
  haversine formula.
  Parameters
  ----------
  lat1, lon1, lat2, lon2
      Coordinate pairs in **decimal degrees**.  Can be floats, NumPy arrays,
      or pandas Series – they are converted with ``np.asarray`` and support
      broadcasting.
  radius
      Sphere radius in kilometres.  Defaults to the WGS‑84 equatorial value
      (6378.137km).
      Returns
  -------
  np.ndarray
      Distance(s) in kilometres with the same broadcasted shape as the
      inputs.
  """
  # Ensure NumPy arrays for broadcasting then convert to radians
  lat1, lon1, lat2, lon2 = map(
      np.radians, map(np.asarray, (lat1, lon1, lat2, lon2))
  )
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
      dlon / 2.0
  ) ** 2
  return 2.0 * radius * np.arcsin(np.sqrt(a))


def osrm_distance_km(pick_lon, pick_lat, drop_lon, drop_lat):
  url = (f"{OSRM_URL}"
         f"{pick_lon},{pick_lat};{drop_lon},{drop_lat}"
         "?overview=false")
  r = requests.get(url, timeout=2)
  if r.ok and r.json()["code"] == "Ok":
    return r.json()["routes"][0]["distance"] / 1000.0  # metres → km
  return np.nan


def _check_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
  missing = [c for c in cols if c not in df.columns]
  if missing:
    raise KeyError(f"Missing column(s) {missing}")


def add_haversine(df: pd.DataFrame) -> pd.DataFrame:
  """Add haversine distance in km + log1p(km)."""
  needed = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
  ]
  _check_columns(df, needed)
  df = df.copy()
  df["hav_dist_km"] = haversine(
      df["pickup_latitude"],
      df["pickup_longitude"],
      df["dropoff_latitude"],
      df["dropoff_longitude"],
  ).astype("float32")
  df["hav_dist_km_log"] = np.log1p(df["hav_dist_km"]).astype("float32")
  return df


def add_route_distance(df):
  tqdm.pandas()
  out = "../data/derived/with_route_dist.parquet"
  if os.path.exists(out):
    done = pd.read_parquet(out)
    start_ix = done.shape[0]
    df_remaining = df.iloc[start_ix:].copy()
  else:
    start_ix = 0
    df_remaining = df.copy()
    done = pd.DataFrame(columns=df.columns.tolist() + ["route_distance_km"])
  tqdm.pandas()
  chunk = 50_000
  for i in range(0, len(df_remaining), chunk):
    sub = df_remaining.iloc[i:i + chunk]
    sub["route_distance_km"] = sub.progress_apply(
        lambda row: osrm_distance_km(
            row.pickup_longitude, row.pickup_latitude,
            row.dropoff_longitude, row.dropoff_latitude
        ), axis=1
    ).astype("float32")

    done = pd.concat([done, sub], axis=0)
    done.to_parquet(out, index=False)
  df = done
  df['route_distance_log_km'] = np.log1p(df['route_distance_km'])
  return df
