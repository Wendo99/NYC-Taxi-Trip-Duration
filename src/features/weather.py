from __future__ import annotations

import pandas as pd

from constants.weather_c import (
  TEMP_SCALE,
  WIND_SCALE,
  HUMIDITY_SCALE,
  PRESSURE_SCALE,
  RAIN_SCALE,
  SNOW_SCALE,
  CLOUD_MAP, HAZE_MAP, FREEZING_MAP, FOG_MAP, INCH_TO_MM, MPH_TO_KPH,
  INHG_TO_HPA
)
from features.utils import classify_ordinal


def fahrenheit_to_celsius(df, col, new_col):
  df[new_col] = (df[col] - 32) * 5 / 9
  return df


def miles_to_kilometers(df, col, new_col):
  df[new_col] = df[col] * MPH_TO_KPH
  return df


def inch_to_millimeters(df, col, new_col):
  df[new_col] = df[col] * INCH_TO_MM
  return df


def inch_mercury_to_hpa(df, col, new_col):
  df[new_col] = df[col] * INHG_TO_HPA
  return df


def clean_trace_and_convert(df, cols, val, trace='T'):
  for col in cols:
    df[col] = df[col].replace(trace, val)
    df[col] = pd.to_numeric(df[col], errors='coerce')
  return df


def split_precip_into_rain_and_snow(df):
  df = df.copy()
  df['rain_mm'] = df['precip_mm'].where(
      df['conditions'].str.contains('Rain', na=False), 0)
  df['snow_mm'] = df['precip_mm'].where(
      df['conditions'].str.contains('Snow', na=False), 0)
  return df


def _ensure_category_codes(df: pd.DataFrame, cls_col: str,
    code_col: str) -> None:
  df[code_col] = df[cls_col].astype("category").cat.codes


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


def classify_temp(df: pd.DataFrame) -> None:
  classify_and_code(df, "temp_c", TEMP_SCALE, "temp")


def classify_windspeed(df: pd.DataFrame) -> None:
  classify_and_code(df, "windspeed_kph", WIND_SCALE, "windspeed")


def classify_humidity(df: pd.DataFrame) -> None:
  classify_and_code(df, "humidity", HUMIDITY_SCALE, "humidity")


def classify_pressure(df: pd.DataFrame) -> None:
  classify_and_code(df, "pressure_hpa", PRESSURE_SCALE, "pressure")


def classify_rain(df: pd.DataFrame) -> None:
  classify_and_code(df, "rain_mm", RAIN_SCALE, "rain")


def classify_snow(df: pd.DataFrame) -> None:
  classify_and_code(df, "snow_mm", SNOW_SCALE, "snow")


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


def classify_clouds(df):
  df["cloud_class"] = (
    df["conditions"].map(COND_TO_CLOUD).fillna("unknown")
  )
  df["cloud_code"] = df["cloud_class"].map(CLOUD_MAP)


def classify_haze(df):
  df["hazy_class"] = (
    df["conditions"].map(COND_TO_HAZE).fillna("no_haze")
  )
  df["hazy_code"] = df["hazy_class"].map(HAZE_MAP)


def classify_freezing(df):
  df["freezing_class"] = (
    df["conditions"].map(COND_TO_FREEZING).fillna("no_freezing_rain_fog")
  )
  df["freezing_code"] = df["freezing_class"].map(FREEZING_MAP)


def classify_fog(df):
  df["fog_class"] = df["fog"].apply(lambda x: "fog" if x == 1 else "no_fog")
  df["fog_code"] = df["fog_class"].map(FOG_MAP)


def classify_weather_data(df):
  df = df.copy()
  classify_temp(df)
  classify_windspeed(df)
  classify_humidity(df)
  classify_fog(df)
  classify_freezing(df)
  classify_clouds(df)
  classify_haze(df)
  classify_pressure(df)
  classify_rain(df)
  classify_snow(df)
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


def timestamp_to_datetime(df, col, new_col):
  df[new_col] = pd.to_datetime(df[col])
  return df


def add_time_features(df, datetime_col):
  df['datetime_hour'] = df[datetime_col].dt.floor('h')
  return _add_time_features_from_hour(df)


def _add_time_features_from_hour(df: pd.DataFrame) -> pd.DataFrame:
  """
  Adds time-based features derived from 'datetime_hour':
  - hour_of_day: hour [0–23]
  - hour_of_year: absolute hour count since year start

  Parameters:
      df (pd.DataFrame): DataFrame with 'datetime_hour' column

  Returns:
      pd.DataFrame: DataFrame with new time features
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
