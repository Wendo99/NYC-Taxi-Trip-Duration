from __future__ import annotations

from pathlib import Path

import pandas as pd

from constants.core_c import PROCESSED_DIR
from constants.taxi_c import TAXI_PROCESSED_CSV
from constants.weather_c import WEATHER_PROCESSED_CSV
from features.weather import add_weather_interactions

MERGED_CSV = Path(PROCESSED_DIR) / "taxi_weather.csv"


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
  taxi_df = pd.read_csv(TAXI_PROCESSED_CSV) if taxi_df is None else taxi_df
  weather_df = (
    pd.read_csv(WEATHER_PROCESSED_CSV) if weather_df is None else weather_df
  )

  if on not in taxi_df.columns or on not in weather_df.columns:
    raise KeyError(f"join key {on!r} must exist in both frames")

  return pd.merge(
      taxi_df,
      weather_df,
      how="left",
      on=on,
      validate="many_to_one",
      suffixes=suffixes,
  )


def build_merged_dataset(save_csv: bool = False) -> pd.DataFrame:
  df = merge_taxi_weather()
  df = add_weather_interactions(df)
  if save_csv:
    MERGED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MERGED_CSV, index=False)
  return df
