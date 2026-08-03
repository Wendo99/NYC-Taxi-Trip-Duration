"""Cleaning helpers used by both the taxi and the weather pipeline."""
import pandas as pd


def flag_and_clip(
    df: pd.DataFrame,
    outlier
) -> pd.DataFrame:
  """Flag out-of-range values, then clip them to the boundary.

  *outlier* is an iterable of ``(source_col, flag_col, low, high)``. For each
  entry a 0/1 ``flag_col`` records whether the value was out of range, and
  ``source_col`` is clipped into ``[low, high]``.

  Rows are never dropped. The row count stays stable and the flag itself
  becomes a feature the model can use — which is why the extreme values in
  the processed data sit exactly on the configured limits.
  """
  out = df.copy()
  for src, flag, lo, hi in outlier:
    out[flag] = ((out[src] < lo) | (out[src] > hi)).astype('int8')
    out[src] = out[src].clip(lower=lo, upper=hi)
  return out
