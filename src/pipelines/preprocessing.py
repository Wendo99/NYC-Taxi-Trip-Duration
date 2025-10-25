import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, \
  OneHotEncoder
from sklearn.preprocessing import StandardScaler

from constants.features_constants import NUM_ALL, CAT_ALL, GEO_PICK, \
  GEO_DROP


def feature_to_fp32(df, features):
  for col in features:
    if col in df.columns:
      df[col] = df[col].astype('float32')
  return df


def feature_to_category(df, features):
  for col in features:
    if col in df.columns:
      df[col] = df[col].astype('category')
  return df


def feature_to_bool(df, features):
  for col in features:
    if col in df.columns:
      df[col] = df[col].astype('bool')
  return df


def feature_to_int8(df, features):
  for col in features:
    if col in df.columns:
      df[col] = df[col].astype('int8')
  return df


def build_preprocessor() -> ColumnTransformer:
  num_pipe = Pipeline([
    ('imputer', KNNImputer(missing_values=np.nan)),
    ("scale", StandardScaler()),
  ])

  cat_pipe = Pipeline([
    ("encoder",
     OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
  ])

  geo_pipe = Pipeline([
    ("onHot",
     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
  ])

  preprocessor = ColumnTransformer([
    ("nums", num_pipe, NUM_ALL),
    ("cats", cat_pipe, CAT_ALL),
    ("geo_pick", geo_pipe, GEO_PICK),
    ("geo_drop", geo_pipe, GEO_DROP)
  ], remainder="drop")

  return preprocessor
