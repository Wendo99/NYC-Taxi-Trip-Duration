import pandas as pd


def flag_and_clip(
    df: pd.DataFrame,
    outlier
) -> pd.DataFrame:
  out = df.copy()
  for src, flag, lo, hi in outlier:
    out[flag] = ((out[src] < lo) | (out[src] > hi)).astype('int8')
    out[src] = out[src].clip(lower=lo, upper=hi)
  return out
