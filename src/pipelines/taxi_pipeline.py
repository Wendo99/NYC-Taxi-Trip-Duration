from __future__ import annotations

from pathlib import Path

import pandas as pd

import constants.modell_constants as modelling_constants
import constants.path_file_constants
import constants.taxi_constants as taxi_constants
import data_io
import utilities.cluster_utilities as cluster_utilities
import utilities.distance_utilities as distance_utilities
import utilities.shared_utilities as shared_utilities
import utilities.taxi_utilities as taxi_utilities


def build_taxi_dataset(save_csv: bool = False) -> pd.DataFrame:
  """
  Build the cleaned+ feature‑enriched NYC‑taxi DataFrame.

  Parameters
  ----------
  save_csv : bool, default ``False``
      When *True*, the function writes the resulting dataset to the path
      defined in ``constants.taxi_const.TAXI_PROCESSED_CSV`` (overwriting any
      existing file).

  Returns
  -------
  pandas.DataFrame
      A copy of the raw taxi data after basic cleaning, outlier flagging,
      feature engineering (time‑utilities, Haversine distance, holiday flag,
      etc.) and ready for modelling.

  Notes
  -----
  * The transformation is **pure** except for the optional CSV write side
    effect triggered by ``save_csv=True``.
  * The function does **not** mutate the cached raw DataFrame returned by
    :pyfunc:`data_io.load_taxi_data`; all operations are performed on a copy.
  """

  # 1 Raw Ingest  ----------------------------------------

  df = data_io.load_taxi_data()

  # 2 convert datetime obj to datetime dtype ----------------------------------

  df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors="coerce"
                                         )
  df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'],
                                          errors="coerce"
                                          )

  # 3 Outlier-Flags / Clipping  ----------------------------------------
  outlier = (
    ("passenger_count", "passenger_count_invalid",
     taxi_constants.PassengerLimits.min_passengers,
     taxi_constants.PassengerLimits.max_passengers),
    ("pickup_longitude", "pickup_longitude_invalid",
     taxi_constants.GeoBounds.min_lon,
     taxi_constants.GeoBounds.max_lon),
    ("pickup_latitude", "pickup_latitude_invalid",
     taxi_constants.GeoBounds.min_lat, taxi_constants.GeoBounds.max_lat),
    ("dropoff_longitude", "dropoff_longitude_invalid",
     taxi_constants.GeoBounds.min_lon, taxi_constants.GeoBounds.max_lon),
    ("dropoff_latitude", "dropoff_latitude_invalid",
     taxi_constants.GeoBounds.min_lat, taxi_constants.GeoBounds.max_lat),
    ("trip_duration", "trip_duration_outlier",
     taxi_constants.TripDurationLimits.min_sec,
     taxi_constants.TripDurationLimits.max_sec)
  )

  df = shared_utilities.flag_and_clip(df, outlier)

  # 4 feature creation ----------------------------------------
  df = taxi_utilities.add_store_and_fwd_flag(df)
  df = taxi_utilities.add_us_holiday_flag(df, 'pickup_datetime')
  df = taxi_utilities.add_trip_duration_features(df)
  df = taxi_utilities.create_is_group_trip(df)
  df = distance_utilities.add_haversine(df)
  df = distance_utilities.add_route_distance(df)
  df = taxi_utilities.add_time_features(df, taxi_constants.TIME_REF_COL)
  df = taxi_utilities.get_jfk_flag(df)
  df = taxi_utilities.get_la_gua(df)

  if taxi_constants.ENABLE_MB:
    df = taxi_utilities.create_geo_clusters(df, ['pickup_longitude',
                                                 'pickup_latitude'],
                                            'pickup',
                                            taxi_constants.N_PICKUP_CLUSTERS,
                                            modelling_constants.RANDOM_STATE,
                                            taxi_constants.CLUSTER_BATCH_SIZE)
    df = taxi_utilities.create_geo_clusters(df, ['dropoff_longitude',
                                                 'dropoff_latitude'],
                                            'dropoff',
                                            taxi_constants.N_DROPOFF_CLUSTERS,
                                            modelling_constants.RANDOM_STATE,
                                            taxi_constants.CLUSTER_BATCH_SIZE)

  if taxi_constants.ENABLE_HDBC:
    df = cluster_utilities.add_hdbc_clusters(df, "pickup",
                                             taxi_constants.PICKUP_COORDS,
                                             taxi_constants.PICKUP_MIN_CLUSTER_SIZE,
                                             taxi_constants.PICKUP_MIN_SAMPLES)
    df = cluster_utilities.add_hdbc_clusters(df, "dropoff",
                                             taxi_constants.DROPOFF_COORDS,
                                             taxi_constants.DROPOFF_MIN_CLUSTER_SIZE,
                                             taxi_constants.DROPOFF_MIN_SAMPLES)

  # 5 write CSV ---------------------------------------------------
  if save_csv:
    Path(
        constants.path_file_constants.TAXI_PROCESSED_CSV).parent.mkdir(
        parents=True,
        exist_ok=True)
    df.to_csv(constants.path_file_constants.TAXI_PROCESSED_CSV, index=False)

  return df
