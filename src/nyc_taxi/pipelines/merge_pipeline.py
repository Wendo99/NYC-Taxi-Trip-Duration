"""Join the processed taxi and weather datasets on the shared hour key."""
from __future__ import annotations

import pandas as pd

from nyc_taxi.config.path_file_constants import (
  MERGED_CSV,
  TAXI_PROCESSED_CSV,
  WEATHER_PROCESSED_CSV,
)
from nyc_taxi.features.weather_utilities import add_weather_interactions
from nyc_taxi.frames import read_frame


def merge_taxi_weather(
    taxi_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
    on: str = "hour_of_year",
    suffixes: tuple[str, str] = ("", "_wx"),
) -> pd.DataFrame:
  """
  Merge taxi and weather frames on the specified key.

  Parameters
  ----------
  taxi_df, weather_df
      If None, they're loaded from the processed CSV paths.
      Merge strategy (default "left": keep all taxi rows).
  on
      Join key, expected to be present in both frames.
  suffixes
      Column-name suffixes passed to :func:`pandas.merge`.

  Returns
  -------
  pd.DataFrame
      Combined data.
  """
  taxi = read_frame(TAXI_PROCESSED_CSV) if taxi_df is None else taxi_df
  weather = (
    read_frame(WEATHER_PROCESSED_CSV) if weather_df is None else weather_df
  )

  if on not in taxi.columns or on not in weather.columns:
    raise KeyError(f"join key {on!r} must exist in both frames")

  return pd.merge(
      taxi,
      weather,
      how="left",
      on=on,
      validate="many_to_one",
      suffixes=suffixes,
  )


def build_merged_dataset(save_csv: bool = False) -> pd.DataFrame:
  """Join the processed taxi and weather sets and add interaction features."""
  df = merge_taxi_weather()
  df = add_weather_interactions(df)
  if save_csv:
    df.to_csv(MERGED_CSV, index=False)
  return df
