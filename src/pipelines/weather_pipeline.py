from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import constants.path_file_constants
from constants import weather_constants as w
from data_io import load_weather_data
from src.utilities import shared_utilities as feat_utils
from src.utilities import weather_utilities as feat_weather

log = logging.getLogger(__name__)


def build_weather_dataset(
    save_csv: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
  """Return a fully processed hourly weather DataFrame.

  Parameters
  ----------
  save_csv : bool, default ``False``
      If ``True`` the resulting data-set is written to
      ``constants.weather_c.WEATHER_PROCESSED_CSV``.
  verbose : bool, default ``False``
      Print intermediate shapes / timings.
  """

  # 1. Ensure raw data is loaded/cached   ----------
  load_weather_data()

  # 1a. Read both raw CSVs   ----------
  csv1 = pd.read_csv(Path(constants.path_file_constants.WEATHER_RAW_CSV1))
  csv2 = pd.read_csv(Path(constants.path_file_constants.WEATHER_RAW_CSV2))
  df = pd.concat([csv1, csv2], ignore_index=True)

  df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")

  # 2. Replace trace values   ----------
  df = feat_weather.clean_trace_values(df, w.RAIN_TRACE_INCH, ["dailyprecip"])
  df = feat_weather.clean_trace_values(df, w.SNOW_TRACE_INCH, ["dailysnow"])

  # 3. Unit conversion   ----------
  df = feat_weather.convert_units(df)

  # 4. Interpolate   ----------
  df = (
    df.set_index("datetime")
    .sort_index()
  )
  df["windspeed_kph"] = df["windspeed_kph"].interpolate(method="time")
  df["pressure_hpa"] = df["pressure_hpa"].interpolate(method="time")
  df = df.reset_index()  # brings 'datetime' back as a column

  outlier = (("windspeed_kph",
              "windspeed_outliers",
              w.WindLimits.kph_min,
              w.WindLimits.kph_max), ("daily_snow_mm",
                                      "daily_snow_outliers",
                                      w.DailySnowLimit.mm_min,
                                      w.DailySnowLimit.mm_max))

  # 5. Flag + clip outliers   ----------
  df = feat_utils.flag_and_clip(df, outlier)

  # 6. Feature engineering   ----------
  df = feat_weather.add_time_features(df, "datetime")
  df = feat_weather.split_precip_into_rain_and_snow(df)
  df["windspeed_kph_sqrt"] = np.sqrt(df["windspeed_kph"])

  # 7. Aggregate   ----------
  df = feat_weather.aggregate_weather_hourly(df)

  if "datetime" not in df.columns and "datetime_hour" in df.columns:
    df = df.rename(columns={"datetime_hour": "datetime"})
  df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

  # recompute time‐based utilities
  df = feat_weather.add_time_features(df, datetime_col="datetime")

  # 8. Ordinal classifications   ----------
  df = feat_weather.classify_weather_data(df)

  # ── Override hour_of_year as a continuous count from the first timestamp ──
  # first_ts = df["datetime"].min().floor("h")
  # df["hour_of_year"] = (
  #     (df["datetime"] - first_ts) // pd.Timedelta(hours=1)
  # ).astype(int)

  #   ----------
  df = df.sort_values("datetime_hour").reset_index(drop=True)

  # 9. Persist if requested
  if save_csv:
    Path(constants.path_file_constants.PROCESSED_DIR).mkdir(parents=True,
                                                            exist_ok=True)
    df.to_csv(constants.path_file_constants.WEATHER_PROCESSED_CSV, index=False)
    if verbose:
      log.info("Written              : %s",
               constants.path_file_constants.WEATHER_PROCESSED_CSV)

  if verbose:
    log.info("Final shape          : %s", df.shape)

  return df
