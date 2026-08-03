"""Column casting helpers and the sklearn ColumnTransformer."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from nyc_taxi.config.features_constants import (
  CAT_ALL,
  GEO_DROP,
  GEO_PICK,
  NUM_ALL,
)


def cast_features(df: pd.DataFrame, features, dtype: str) -> pd.DataFrame:
  """Cast every column of *features* present in *df* to *dtype*.

  Columns missing from *df* are skipped, which is what lets one feature list
  drive datasets built with different optional features enabled.

  Replaces the former ``feature_to_fp32`` / ``feature_to_category`` /
  ``feature_to_bool`` / ``feature_to_int8`` quadruplet, which differed only
  in the dtype string.
  """
  for col in features:
    if col in df.columns:
      df[col] = df[col].astype(dtype)
  return df


def feature_to_fp32(df: pd.DataFrame, features) -> pd.DataFrame:
  """Convenience wrapper kept because the notebooks call it by name."""
  return cast_features(df, features, "float32")


def feature_to_category(df: pd.DataFrame, features) -> pd.DataFrame:
  """Convenience wrapper kept because the notebooks call it by name."""
  return cast_features(df, features, "category")


def build_preprocessor() -> ColumnTransformer:
  """Numeric impute+scale, ordinal encoding, one-hot for the geo clusters.

  ``memory=None`` is passed explicitly on each inner pipeline, matching
  ``models_factory``: transformer caching is deliberately off, because the
  cache would have to be invalidated by hand whenever the feature lists in
  ``features_constants`` change.
  """
  num_pipe = Pipeline([
    ("imputer", KNNImputer(missing_values=np.nan)),
    ("scale", StandardScaler()),
  ], memory=None)

  cat_pipe = Pipeline([
    ("encoder",
     OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
  ], memory=None)

  geo_pipe = Pipeline([
    ("onHot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
  ], memory=None)

  return ColumnTransformer([
    ("nums", num_pipe, NUM_ALL),
    ("cats", cat_pipe, CAT_ALL),
    ("geo_pick", geo_pipe, GEO_PICK),
    ("geo_drop", geo_pipe, GEO_DROP),
  ], remainder="drop")
