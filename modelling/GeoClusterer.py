import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans


class GeoClusterer(BaseEstimator, TransformerMixin):
  """Adds one integer column `cluster` determined by MiniBatchKMeans."""

  def __init__(self, cols, n_clusters=40, random_state=42, batch_size=10_000):
    self.cols = cols
    self.n_clusters = n_clusters
    self.random_state = random_state
    self.batch_size = batch_size

  def fit(self, x, y=None):
    self.km_ = MiniBatchKMeans(
        n_clusters=self.n_clusters,
        random_state=self.random_state,
        batch_size=self.batch_size
    ).fit(x[self.cols])
    return self

  def transform(self, x):
    clusters = self.km_.predict(x[self.cols])
    # return DataFrame to preserve column name for ColumnTransformer
    return pd.DataFrame({'cluster': clusters}, index=x.index)
