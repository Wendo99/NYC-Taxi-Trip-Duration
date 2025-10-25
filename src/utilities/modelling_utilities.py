from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from constants.features_constants import GEO_PICK, GEO_DROP, \
  RES_TABLE_EXCLUDE_FEATURES, FEATURES


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


def fill_res_col(features: dict, exclude: set | None = None) -> set:
  """Return a set of feature keys from the provided dictionary, including GEO_PICK and GEO_DROP,
  and excluding the provided utilities.

  Args:
      features: Dictionary containing feature definitions.
      exclude: An optional set of utilities to exclude.
  """
  # Start with keys from utilities.
  result = set(features.keys())
  # Add keys from GEO_PICK and GEO_DROP.
  result.update(GEO_PICK)
  result.update(GEO_DROP)
  # Remove any excluded utilities.
  if exclude:
    result.difference_update(exclude)
  return result


excluded = set(RES_TABLE_EXCLUDE_FEATURES.keys())
RES_COL = fill_res_col(FEATURES, exclude=excluded)
