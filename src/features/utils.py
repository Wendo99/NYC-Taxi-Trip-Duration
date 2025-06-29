from __future__ import annotations

from typing import Any, Tuple, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from constants.weather_c import OrdinalScale

# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #
__all__: list[str] = [
  "split_train_test",
  "timestamp_to_datetime",
  "add_time_features",
  "flag_and_clip",
  "classify_ordinal",
]


def split_train_test(
    df: pd.DataFrame,
    test_size: float | int = 0.2,
    random_state: int | None = None,
    stratify_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  Split a *DataFrame* into train‑ and test‑sets **while preserving the DataFrame
  structure** (no conversion to NumPy arrays).

  Parameters
  ----------
  df : pd.DataFrame
      Source data.
  test_size : float | int, default 0.2
      Passed to ``sklearn.model_selection.train_test_split``.
  random_state : int | None, default None
      Seed for deterministic shuffling.
  stratify_col : str | None, default None
      Column name to stratify on.  If ``None`` no stratification is used.

  Returns
  -------
  Tuple[pd.DataFrame, pd.DataFrame]
      ``(train_df, test_df)``, both copied from the original so that
      in‑place modifications do **not** touch the source.
  """
  # ---- basic validation -------------------------------------------------
  if isinstance(test_size, float):
    if not (0.0 < test_size < 1.0):
      raise ValueError("`test_size` as float must be in the interval (0, 1).")
  elif isinstance(test_size, int):
    if not (1 <= test_size < len(df)):
      raise ValueError(
          "`test_size` as int must be at least 1 and smaller than the dataset."
      )
  else:
    raise TypeError("`test_size` must be float or int.")

  if stratify_col is not None and df[stratify_col].isna().any():
    raise ValueError(
        f"Column {stratify_col!r} contains NaNs – cannot stratify.")

  stratify_arr = df[stratify_col] if stratify_col else None

  train_idx, test_idx = train_test_split(
      df.index,
      test_size=test_size,
      random_state=random_state,
      stratify=stratify_arr,
  )
  return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def timestamp_to_datetime(
    df: pd.DataFrame,
    cols: str | Sequence[str],
    new_cols: str | Sequence[str] | None = None,
    *,
    errors: str = "coerce",
) -> pd.DataFrame:
  """
  Convert one or multiple timestamp columns to ``datetime64[ns]``.

  Parameters
  ----------
  df : pd.DataFrame
  cols : str or list[str]
      Source column(s) containing timestamps.
  new_cols : str or list[str] or None, default None
      Destination column name(s).  If *None*, overwrite ``cols``.
  errors : {'raise', 'coerce', 'ignore'}, default 'coerce'
      How to handle parsing errors (forwarded to ``pd.to_datetime``).

  Returns
  -------
  pd.DataFrame
      A *copy* of the input with the converted columns.
  """
  if isinstance(cols, str):
    cols = [cols]
  if new_cols is None:
    new_cols = cols
  elif isinstance(new_cols, str):
    new_cols = [new_cols]

  out = df.copy()
  for src, dest in zip(cols, new_cols):
    out[dest] = pd.to_datetime(out[src], errors=errors)
  return out


def add_time_features(
    df: pd.DataFrame,
    time_col: str,
    *,
    drop_source: bool = False,
) -> pd.DataFrame:
  """
  Expand a datetime column into hour / weekday / month / day-of‑year / hour‑of‑year.

  Parameters
  ----------
  df : pd.DataFrame
  time_col : str
      Name of the datetime column to expand.
  drop_source : bool, default False
      Whether to drop *time_col* in the returned frame.

  Returns
  -------
  pd.DataFrame
  """
  dti = pd.to_datetime(df[time_col])
  out = df.copy()
  out["hour"] = dti.dt.hour
  out["weekday"] = dti.dt.day_ofweek
  out["month"] = dti.dt.month
  doy = dti.dt.dayofyear
  out["hour_of_year"] = (doy - 1) * 24 + out["hour"]
  if drop_source:
    out = out.drop(columns=[time_col])
  return out


def flag_and_clip(
    df: pd.DataFrame,
    col: str,
    flag_name: str,
    lower: float | int,
    upper: float | int,
) -> pd.DataFrame:
  out = df.copy()
  out[flag_name] = ((out[col] < lower) | (out[col] > upper)).astype('int8')
  out[col] = out[col].clip(lower=lower, upper=upper)
  return out


def classify_ordinal(series, scale: OrdinalScale) -> Any:
  import numpy as np

  to_labels = np.vectorize(scale.label, otypes=[object])
  return to_labels(series)
