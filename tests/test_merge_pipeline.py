"""Tests for the taxi/weather join.

The join is the step that produces the modelling dataset, and it carries a
``validate="many_to_one"`` contract: many taxi trips per weather hour, never
the reverse. A duplicated weather hour would silently multiply taxi rows, so
that contract is worth pinning.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nyc_taxi.pipelines.merge_pipeline import merge_taxi_weather


@pytest.fixture
def taxi() -> pd.DataFrame:
  """Three trips across two distinct weather hours."""
  return pd.DataFrame({
    "id": ["a", "b", "c"],
    "hour_of_year": [1790, 1790, 1791],
    "trip_duration_log": [6.8, 7.9, 6.5],
  })


@pytest.fixture
def weather() -> pd.DataFrame:
  return pd.DataFrame({
    "hour_of_year": [1790, 1791, 1792],
    "temp_c": [10.0, 12.5, 13.0],
    "rain_mm": [0.0, 2.54, 0.0],
  })


def test_merge_keeps_every_taxi_row(taxi, weather):
  """A left join: no trip may be dropped because weather is missing."""
  merged = merge_taxi_weather(taxi, weather)
  assert len(merged) == len(taxi)
  assert merged["id"].tolist() == ["a", "b", "c"]


def test_merge_attaches_the_matching_hour(taxi, weather):
  merged = merge_taxi_weather(taxi, weather).set_index("id")
  assert merged.loc["a", "temp_c"] == 10.0   # hour 1790
  assert merged.loc["b", "temp_c"] == 10.0   # same hour, second trip
  assert merged.loc["c", "temp_c"] == 12.5   # hour 1791


def test_unmatched_taxi_hour_yields_nan_not_a_dropped_row(taxi, weather):
  taxi = pd.concat([taxi, pd.DataFrame({
    "id": ["d"], "hour_of_year": [9999], "trip_duration_log": [7.0]})],
                   ignore_index=True)
  merged = merge_taxi_weather(taxi, weather)
  assert len(merged) == 4
  assert pd.isna(merged.loc[merged["id"] == "d", "temp_c"]).all()


def test_duplicate_weather_hour_is_rejected(taxi, weather):
  """many_to_one must fail loudly rather than duplicate taxi rows."""
  dupe = pd.concat([weather, weather.iloc[[0]]], ignore_index=True)
  with pytest.raises(pd.errors.MergeError):
    merge_taxi_weather(taxi, dupe)


def test_missing_join_key_raises_keyerror(taxi, weather):
  without_key = weather.drop(columns=["hour_of_year"])
  with pytest.raises(KeyError, match="must exist in both frames"):
    merge_taxi_weather(taxi, without_key)


