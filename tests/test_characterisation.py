"""Characterisation tests: pin current pipeline behaviour before refactoring.

These are not specification tests. They record what the code *does today* so
that a refactor which accidentally changes behaviour fails loudly. Where a
value is pinned because of a known bug, it is marked with ``xfail`` rather
than asserted as correct, so that fixing the bug turns the test green instead
of red.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import constants.taxi_constants as tc
from constants import weather_constants as w
from fixtures import weather_frame, taxi_frame
from utilities import distance_utilities as du
from utilities import shared_utilities as su
from utilities import taxi_utilities as tu
from utilities import weather_utilities as wu

OUTLIER_SPEC = (
  ("passenger_count", "passenger_count_invalid",
   tc.PassengerLimits.min_passengers, tc.PassengerLimits.max_passengers),
  ("pickup_longitude", "pickup_longitude_invalid",
   tc.GeoBounds.min_lon, tc.GeoBounds.max_lon),
  ("pickup_latitude", "pickup_latitude_invalid",
   tc.GeoBounds.min_lat, tc.GeoBounds.max_lat),
  ("dropoff_longitude", "dropoff_longitude_invalid",
   tc.GeoBounds.min_lon, tc.GeoBounds.max_lon),
  ("dropoff_latitude", "dropoff_latitude_invalid",
   tc.GeoBounds.min_lat, tc.GeoBounds.max_lat),
  ("trip_duration", "trip_duration_outlier",
   tc.TripDurationLimits.min_sec, tc.TripDurationLimits.max_sec),
)

# Column order produced by build_taxi_dataset, minus the clustering step
# (MiniBatchKMeans needs more rows than a six-row fixture provides).
EXPECTED_TAXI_COLUMNS = [
  "id", "vendor_id", "pickup_datetime", "dropoff_datetime", "passenger_count",
  "pickup_longitude", "pickup_latitude", "dropoff_longitude",
  "dropoff_latitude", "store_and_fwd_flag", "trip_duration",
  "passenger_count_invalid", "pickup_longitude_invalid",
  "pickup_latitude_invalid", "dropoff_longitude_invalid",
  "dropoff_latitude_invalid", "trip_duration_outlier",
  "store_and_fwd_flag_bin", "is_holiday", "trip_duration_min",
  "trip_duration_log", "is_group_trip", "hav_dist_km", "hav_dist_km_log",
  "pickup_hour", "pickup_weekday", "pickup_month", "hour_of_year",
  "is_weekend", "is_early_morning", "is_rush_am", "is_rush_pm", "is_night",
  "is_jfk_pick", "is_jfk_drop", "is_laguardia_pick", "is_laguardia_drop",
]

EXPECTED_TAXI_DTYPES = {
  "trip_duration_min": "float32", "trip_duration_log": "float32",
  "hav_dist_km": "float32", "hav_dist_km_log": "float32",
  "store_and_fwd_flag_bin": "Int8", "is_holiday": "int8",
  "is_group_trip": "int8", "pickup_hour": "int8", "pickup_weekday": "int8",
  "pickup_month": "int8", "hour_of_year": "int16", "is_weekend": "int8",
  "is_early_morning": "int8", "is_rush_am": "int8", "is_rush_pm": "int8",
  "is_night": "int8", "is_jfk_pick": "int8", "is_jfk_drop": "int8",
  "is_laguardia_pick": "int8", "is_laguardia_drop": "int8",
  "trip_duration_outlier": "int8",
}


@pytest.fixture(scope="module")
def taxi_features() -> pd.DataFrame:
  """The taxi feature chain exactly as build_taxi_dataset applies it."""
  df = su.flag_and_clip(taxi_frame(), OUTLIER_SPEC)
  df = tu.add_store_and_fwd_flag(df)
  df = tu.add_us_holiday_flag(df, "pickup_datetime")
  df = tu.add_trip_duration_features(df)
  df = tu.create_is_group_trip(df)
  df = du.add_haversine(df)
  df = tu.add_time_features(df, tc.TIME_REF_COL)
  df = tu.get_jfk_flag(df)
  df = tu.get_la_gua(df)
  return df


# --------------------------------------------------------------- taxi shape
def test_taxi_columns_and_order(taxi_features):
  assert list(taxi_features.columns) == EXPECTED_TAXI_COLUMNS


def test_taxi_dtypes(taxi_features):
  actual = {c: str(taxi_features[c].dtype) for c in EXPECTED_TAXI_DTYPES}
  assert actual == EXPECTED_TAXI_DTYPES


def test_row_count_preserved(taxi_features):
  assert len(taxi_features) == 6


# ------------------------------------------------------------ taxi features
def test_trip_duration_features(taxi_features):
  assert taxi_features["trip_duration_min"].tolist() == [
    15.0, 45.0, 45.0, 15.0, 15.0, 120.0]
  np.testing.assert_allclose(
      taxi_features["trip_duration_log"],
      np.log1p([900, 2700, 2700, 900, 900, 7200]), rtol=1e-6)


def test_store_and_fwd_and_group_flags(taxi_features):
  assert taxi_features["store_and_fwd_flag_bin"].tolist() == [0, 0, 1, 0, 0, 0]
  assert taxi_features["is_group_trip"].tolist() == [0, 1, 0, 1, 0, 1]


def test_us_holiday_flag_detects_new_years_day(taxi_features):
  # Row 4 is 2016-01-01; every other row is an ordinary day.
  assert taxi_features["is_holiday"].tolist() == [0, 0, 0, 0, 1, 0]


def test_time_features(taxi_features):
  assert taxi_features["pickup_hour"].tolist() == [14, 7, 17, 23, 4, 12]
  assert taxi_features["pickup_weekday"].tolist() == [1, 1, 1, 5, 4, 3]
  assert taxi_features["is_weekend"].tolist() == [0, 0, 0, 1, 0, 0]
  assert taxi_features["is_rush_am"].tolist() == [0, 1, 0, 0, 0, 0]
  assert taxi_features["is_rush_pm"].tolist() == [0, 0, 1, 0, 0, 0]
  assert taxi_features["is_night"].tolist() == [0, 0, 0, 1, 0, 0]
  assert taxi_features["is_early_morning"].tolist() == [0, 0, 0, 0, 1, 0]
  assert taxi_features["hour_of_year"].tolist() == [
    1790, 1783, 1793, 1895, 4, 4356]


def test_haversine_distance(taxi_features):
  np.testing.assert_allclose(
      taxi_features["hav_dist_km"],
      [1.325793, 20.747200, 9.495442, 2.531202, 0.968033, 0.0], rtol=1e-5)
  np.testing.assert_allclose(
      taxi_features["hav_dist_km_log"],
      np.log1p(taxi_features["hav_dist_km"]), rtol=1e-5)


def test_outlier_flagging_and_clipping(taxi_features):
  # Row 5 sits outside the NYC bounding box on every coordinate.
  assert taxi_features["pickup_longitude_invalid"].tolist() == [
    0, 0, 0, 0, 0, 1]
  assert taxi_features["pickup_latitude_invalid"].tolist() == [
    0, 0, 0, 0, 0, 1]
  # ...and its coordinates are clipped back onto the boundary.
  assert taxi_features.loc[5, "pickup_longitude"] == tc.GeoBounds.min_lon
  assert taxi_features.loc[5, "pickup_latitude"] == tc.GeoBounds.max_lat


# ------------------------------------------------------------ airport flags
def test_jfk_dropoff_detected(taxi_features):
  """Row 1 ends at JFK; nothing else does."""
  assert taxi_features["is_jfk_drop"].tolist() == [0, 1, 0, 0, 0, 0]
  assert taxi_features["is_jfk_pick"].tolist() == [0, 0, 0, 0, 0, 0]


@pytest.mark.xfail(
    strict=True,
    reason="BUG: taxi_utilities.get_la_gua uses la_lat=(40.774, 40.765) — "
           "the lower bound exceeds the upper bound, so the comparison can "
           "never be true and the flag is 0 for every row.")
def test_laguardia_dropoff_detected(taxi_features):
  """Row 2 ends at LaGuardia and should be flagged, mirroring JFK."""
  assert taxi_features["is_laguardia_drop"].tolist() == [0, 0, 1, 0, 0, 0]


def test_laguardia_currently_always_zero(taxi_features):
  """Pins the broken behaviour so the refactor cannot change it silently.

  Delete this test together with the xfail above once the bounds are fixed.
  """
  assert taxi_features["is_laguardia_pick"].sum() == 0
  assert taxi_features["is_laguardia_drop"].sum() == 0


# ----------------------------------------------------------------- weather
@pytest.fixture(scope="module")
def weather_features() -> pd.DataFrame:
  df = weather_frame()
  df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
  df = wu.clean_trace_values(df, w.RAIN_TRACE_INCH, ["dailyprecip"])
  df = wu.clean_trace_values(df, w.SNOW_TRACE_INCH, ["dailysnow"])
  df = wu.convert_units(df)
  df = wu.add_time_features(df, "datetime")
  df = wu.split_precip_into_rain_and_snow(df)
  df["windspeed_kph_sqrt"] = np.sqrt(df["windspeed_kph"])
  return df


def test_unit_conversions(weather_features):
  np.testing.assert_allclose(
      weather_features["temp_c"],
      [10.0, 11.111111, 12.777778, -1.111111], rtol=1e-5)
  np.testing.assert_allclose(
      weather_features["windspeed_kph"],
      [16.09340, 19.31208, 12.87472, 32.18680], rtol=1e-5)
  np.testing.assert_allclose(
      weather_features["pressure_hpa"],
      [1013.207888, 1012.530610, 1014.223805, 1019.303390], rtol=1e-6)


def test_trace_values_replaced(weather_features):
  """'T' becomes the configured trace constant, then converts to mm."""
  np.testing.assert_allclose(
      weather_features["daily_precip_mm"], [0.0, 0.254, 0.0, 12.7], rtol=1e-6)
  np.testing.assert_allclose(
      weather_features["daily_snow_mm"], [0.0, 0.0, 0.0, 2.54], rtol=1e-6)


def test_precip_split_by_condition(weather_features):
  """precip_mm is routed to rain or snow by the free-text 'conditions'."""
  np.testing.assert_allclose(
      weather_features["rain_mm"], [0.0, 2.54, 0.0, 0.0], rtol=1e-6)
  np.testing.assert_allclose(
      weather_features["snow_mm"], [0.0, 0.0, 0.0, 5.08], rtol=1e-6)


def test_weather_time_features(weather_features):
  assert weather_features["hour_of_day"].tolist() == [14, 14, 15, 4]
  assert weather_features["hour_of_year"].tolist() == [1790, 1790, 1791, 4]


def test_hourly_aggregation_collapses_to_distinct_hours(weather_features):
  agg = wu.aggregate_weather_hourly(weather_features)
  assert len(agg) == 3  # the two 14:00 observations merge
  assert agg["datetime_hour"].astype(str).tolist() == [
    "2016-01-01 04:00:00", "2016-03-15 14:00:00", "2016-03-15 15:00:00"]


def test_ordinal_classification(weather_features):
  cls = wu.classify_weather_data(weather_features)
  assert cls["temp_class"].tolist() == ["cool", "mild", "mild", "cold"]
  assert cls["temp_code"].tolist() == [1, 2, 2, 0]
  assert cls["humidity_class"].tolist() == [
    "normal", "normal", "dry", "normal"]
  assert cls["fog_class"].tolist() == ["no_fog", "no_fog", "no_fog", "fog"]
  assert cls["cloud_class"].tolist() == [
    "clear", "unknown", "partly_cloudy", "unknown"]


# ------------------------------------------------------------- merge key
def test_hour_of_year_is_a_valid_join_key(taxi_features, weather_features):
  """Both pipelines must derive hour_of_year identically or the merge is wrong.

  The 14:30 taxi trip and the 14:00 weather observation share hour 1790.
  """
  assert taxi_features.loc[0, "hour_of_year"] == 1790
  assert weather_features.loc[0, "hour_of_year"] == 1790
