"""Unit conversion, cleaning and ordinal classification of weather data.

The raw Wunderground feed is imperial, irregularly sampled, and describes sky
state as free text. This module converts to metric, resolves trace
precipitation markers, aggregates to an hourly grid, and maps both numeric
measurements and the free-text ``conditions`` onto ordinal classes with paired
``*_class`` (label) and ``*_code`` (integer) columns.

See ``notebooks/weather.ipynb`` for the known limits of the source data.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nyc_taxi.config import weather_constants

# Re-exported: these moved to the constants layer to break a circular import,
# but the weather notebook imports them from here. The redundant ``as`` form
# is the conventional way to mark a deliberate re-export (PEP 484), and unlike
# an ``__all__`` entry it does not narrow this module's public surface — an
# ``__all__`` here made every other import from this module look undeclared.
from nyc_taxi.config.weather_constants import (
  OrdinalScale as OrdinalScale,  # noqa: PLC0414
)


def fahrenheit_to_celsius(df, col, new_col):
  df[new_col] = (df[col] - 32) * 5 / 9
  return df


def miles_to_kilometers(df, col, new_col):
  df[new_col] = df[col] * weather_constants.MPH_TO_KPH
  return df


def inch_to_millimeters(df, col, new_col):
  df[new_col] = df[col] * weather_constants.INCH_TO_MM
  return df


def inch_mercury_to_hpa(df, col, new_col):
  df[new_col] = df[col] * weather_constants.IN_TO_HPA
  return df


def clean_trace_and_convert(df, cols, val, trace='T'):
  """Replace the trace marker ('T') with *val*, then coerce to numeric."""
  for col in cols:
    df[col] = df[col].replace(trace, val)
    df[col] = pd.to_numeric(df[col], errors='coerce')
  return df


def split_precip_into_rain_and_snow(df):
  """Split precip_mm into rain_mm / snow_mm using the free-text conditions."""
  df = df.copy()
  df['rain_mm'] = df['precip_mm'].where(
      df['conditions'].str.contains('Rain', na=False), 0)
  df['snow_mm'] = df['precip_mm'].where(
      df['conditions'].str.contains('Snow', na=False), 0)
  return df


def _cat_codes(s: pd.Series) -> pd.Series:
  return s.astype("category").cat.codes


def classify_and_code(
    df: pd.DataFrame,
    src_col: str,
    scale,
    dst_prefix: str,
) -> None:
  """
  Classify *src_col* with *scale* and append two new columns:

  * ``{dst_prefix}_class`` – string label
  * ``{dst_prefix}_code``  – ordered category code (int8)

  Works in‑place to keep memory footprint low.
  """
  cls_col = f"{dst_prefix}_class"
  code_col = f"{dst_prefix}_code"
  df[cls_col] = classify_ordinal(df[src_col], scale)
  df[code_col] = _cat_codes(df[cls_col])


COND_TO_CLOUD = {
  "Clear": "clear",
  "Scattered Clouds": "scattered_clouds",
  "Partly Cloudy": "partly_cloudy",
  "Mostly Cloudy": "mostly_cloudy",
  "Overcast": "overcast",
}
COND_TO_HAZE = {"Haze": "haze"}
COND_TO_FREEZING = {
  "Light Freezing Rain": "light_freezing_rain",
  "Light Freezing Fog": "light_freezing_fog",
}


def classify_from_conditions(df: pd.DataFrame, dst_prefix: str,
    label_map: dict, default: str, code_map: dict) -> None:
  """Derive ``{dst_prefix}_class`` / ``_code`` from the free-text conditions.

  Shared by the cloud, haze and freezing classifiers, which previously
  existed as three copies of this body.

  Note the lookup is on the exact string, so a label carrying trailing
  whitespace falls through to *default* — see ``notebooks/weather.ipynb``
  section 2.2.
  """
  cls_col = f"{dst_prefix}_class"
  df[cls_col] = df["conditions"].map(label_map).fillna(default)
  df[f"{dst_prefix}_code"] = df[cls_col].map(code_map)


def classify_fog(df):
  """Fog comes from a binary column rather than the conditions text."""
  df["fog_class"] = np.where(df["fog"] == 1, "fog", "no_fog")
  df["fog_code"] = df["fog_class"].map(weather_constants.FOG_MAP)


def classify_weather_data(df):
  """Add every ``*_class`` / ``*_code`` pair in one pass.

  The order of the calls below fixes the column order of the processed
  dataset, so changing it changes the output schema.

  These were previously nine one-line wrapper functions (``classify_temp``,
  ``classify_clouds``, ...), each a single call with fixed constants and none
  used anywhere else. Inlining them puts what is classified, from which
  column, onto one screen.
  """
  df = df.copy()
  wc = weather_constants

  classify_and_code(df, "temp_c", wc.TEMP_SCALE, "temp")
  classify_and_code(df, "windspeed_kph", wc.WIND_SCALE, "windspeed")
  classify_and_code(df, "humidity", wc.HUMIDITY_SCALE, "humidity")
  classify_fog(df)
  classify_from_conditions(df, "freezing", COND_TO_FREEZING,
                           "no_freezing_rain_fog", wc.FREEZING_MAP)
  classify_from_conditions(df, "cloud", COND_TO_CLOUD, "unknown", wc.CLOUD_MAP)
  classify_from_conditions(df, "hazy", COND_TO_HAZE, "no_haze", wc.HAZE_MAP)
  classify_and_code(df, "pressure_hpa", wc.PRESSURE_SCALE, "pressure")
  classify_and_code(df, "rain_mm", wc.RAIN_SCALE, "rain")
  classify_and_code(df, "snow_mm", wc.SNOW_SCALE, "snow")
  return df


def aggregate_weather_hourly(df):
  """
  Aggregates weather observations to hourly level.

  - Averages continuous variables
  - Uses 'first' or 'max' for daily totals
  - Uses 'max' for binary or flag columns
  - Applies mode for categorical 'conditions'

  Parameters:
      df (pd.DataFrame): Weather data with datetime_hour column

  Returns:
      pd.DataFrame: Hourly aggregated weather data
  """
  return (
    df.groupby('datetime_hour').agg({
      'temp_c': 'mean',
      'windspeed_kph': 'mean',
      'humidity': 'mean',
      'pressure_hpa': 'mean',
      'daily_precip_mm': 'first',
      'daily_snow_mm': 'first',
      'rain_mm': 'mean',
      'snow_mm': 'mean',
      'windspeed_kph_sqrt': 'mean',
      'fog': 'max',
      'rain': 'max',
      'snow': 'max',
      'conditions': lambda x: x.mode().iloc[0] if not x.mode().empty else
      x.iloc[0]
    })
    .reset_index()
  )


def convert_units(df: pd.DataFrame) -> pd.DataFrame:
  """
  Convert all imperial columns to metric equivalents *in place*.

  Expected imperial columns (if present):
  - temp           → temp_c
  - windspeed    → windspeed_kph
  - precip        → precip_mm
  - pressure    → pressure_hpa
  """
  df = df.copy()

  if "temp" in df.columns and "temp_c" not in df.columns:
    df = fahrenheit_to_celsius(df, "temp", "temp_c")

  if "windspeed" in df.columns and "windspeed_kph" not in df.columns:
    df = miles_to_kilometers(df, "windspeed", "windspeed_kph")

  if "precip" in df.columns and "precip_mm" not in df.columns:
    df = inch_to_millimeters(df, "precip", "precip_mm")

  if "pressure" in df.columns and "pressure_hpa" not in df.columns:
    df = inch_mercury_to_hpa(df, "pressure", "pressure_hpa")

  if "dailyprecip" in df.columns and "daily_precip_mm" not in df.columns:
    df = inch_to_millimeters(df, "dailyprecip", "daily_precip_mm")

  if "dailysnow" in df.columns and "daily_snow_mm" not in df.columns:
    df = inch_to_millimeters(df, "dailysnow", "daily_snow_mm")

  return df


def clean_trace_values(
    df: pd.DataFrame,
    trace_inch: float,
    cols: list[str] | None = None,
    trace_symbol: str = "T",
) -> pd.DataFrame:
  """
   Replace trace precipitation symbol ('T') by *trace_mm* numeric value.
  """
  if cols is None:
    cols = ["dailyprecip",
            "dailysnow"]

  return clean_trace_and_convert(df.copy(), cols=cols, val=trace_inch,
                                 trace=trace_symbol)


def add_time_features(df, datetime_col):
  """Floor *datetime_col* to the hour and derive the hour-based features.

  ``hour_of_year`` produced here is the join key to the taxi data.
  """
  df['datetime_hour'] = df[datetime_col].dt.floor('h')
  return _add_time_features_from_hour(df)


def _add_time_features_from_hour(df: pd.DataFrame) -> pd.DataFrame:
  """
  Adds time-based features derived from 'datetime_hour':
  - hour_of_day: hour [0–23]
  - hour_of_year: absolute hour count since year start

  ``hour_of_year`` is the join key against the taxi dataset, so this must stay
  in step with ``taxi_utilities.add_time_features``.

  Parameters:
      df (pd.DataFrame): DataFrame with 'datetime_hour' column

  Returns:
      pd.DataFrame: DataFrame with the new time features
  """
  df = df.copy()
  df['hour_of_day'] = df['datetime_hour'].dt.hour
  df['day_of_year'] = df['datetime_hour'].dt.dayofyear
  df['hour_of_year'] = ((df['day_of_year'] - 1) * 24) + df['hour_of_day']
  df.drop(columns=['day_of_year'], inplace=True)
  return df


def add_weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
  """
  Create interaction features between rainfall/snowfall and time/weekend flags.

  - rain_rush_am:   rain_mm × is_rush_am
  - rain_rush_pm:   rain_mm × is_rush_pm
  - snow_weekend:   snow_mm × is_weekend
  """
  df = df.copy()
  df["rain_rush_am"] = df["rain_mm"] * df["is_rush_am"]
  df["rain_rush_pm"] = df["rain_mm"] * df["is_rush_pm"]
  df["snow_weekend"] = df["snow_mm"] * df["is_weekend"]
  return df


def classify_ordinal(series, scale: OrdinalScale) -> Any:
  """Map a numeric series onto the ordinal labels of *scale*."""
  to_labels = np.vectorize(scale.label, otypes=[object])
  return to_labels(series)
