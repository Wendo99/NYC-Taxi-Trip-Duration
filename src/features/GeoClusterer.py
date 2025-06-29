from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.validation import check_is_fitted


class GeoClusterer(BaseEstimator, TransformerMixin):
  """
  Mini-batch K-means clustering for latitude/longitude pairs.

  Parameters
  ----------
  cols : list[str]
      Names of the two numeric columns (lat, lon) to cluster.
  n_clusters : int, default 40
  random_state : int, default 42
  batch_size : int, default 10_000
  """

  def __init__(
      self,
      cols: List[str],
      n_clusters: int = 40,
      random_state: int = 42,
      batch_size: int = 10_000,
  ) -> None:
    self.cols = cols
    self.n_clusters = n_clusters
    self.random_state = random_state
    self.batch_size = batch_size
    self.km_: MiniBatchKMeans | None = None

  # ------------------------------------------------------------------ #
  def fit(self, x: pd.DataFrame, y=None):  # noqa: N802 (sklearn API)
    if not all(c in x.columns for c in self.cols):
      missing = set(self.cols) - set(x.columns)
      raise KeyError(f"GeoClusterer: missing column(s) {missing}")

    if x[self.cols].isna().any().any():
      raise ValueError("GeoClusterer: NaNs in coordinate columns")

    self.km_ = MiniBatchKMeans(
        n_clusters=self.n_clusters,
        random_state=self.random_state,
        batch_size=self.batch_size,
    ).fit(x[self.cols])

    return self

  # ------------------------------------------------------------------ #
  def transform(self, x: pd.DataFrame) -> pd.DataFrame:  # noqa: N802
    check_is_fitted(self, "km_")
    clusters = self.km_.predict(x[self.cols])
    return pd.DataFrame({"cluster": clusters}, index=x.index)
