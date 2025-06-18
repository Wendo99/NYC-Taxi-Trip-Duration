import pandas as pd
from IPython.core.display_functions import display
# sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from modelling.modelling_config import CV_FOLDS, RIDGE_ALPHA, RANDOM_SEED, \
  LASSO_ALPHA, N_PICKUP_CLUSTERS, KMEANS_BATCH_SIZE, \
  N_DROPOFF_CLUSTERS

LOG_RMSE_MEAN_ = 'log-RMSE (mean)'


# --- Pipelines
def make_linear_pipeline(model_type='linreg', preprocessing=None):
  if model_type == 'ridge':
    model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)
  elif model_type == 'lasso':
    model = Lasso(alpha=LASSO_ALPHA, random_state=RANDOM_SEED,
                  )
  else:
    model = LinearRegression()
  return Pipeline([('pre', preprocessing), ('model', model)], memory=None)


def cat_base_pipelining():
  return Pipeline([
    ('encoder',
     OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
  ], memory=None)


def num_base_pipelining():
  return make_pipeline(
      StandardScaler(), memory=None)


def bool_base_pipelining():
  return make_pipeline(FunctionTransformer(), memory=None)


def geo_base_pipelining(n_clusters, random_state, batch_size):
  return make_pipeline(MiniBatchKMeans(n_clusters, random_state=random_state,
                                       batch_size=batch_size), memory=None)


def create_geo_clusters(df, feature_cols, prefix, n_clusters, random_state,
    batch_size):
  coords = df[feature_cols]
  kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                           random_state=random_state,
                           batch_size=batch_size)
  cluster_labels = kmeans.fit_predict(coords)
  df[f'{prefix}_cluster'] = pd.Series(cluster_labels, index=df.index).astype(
      'category')
  return df


def get_display_models_results(models, x_train, y_train, cv_folds=CV_FOLDS):
  results = []

  for name, model in models.items():
    scores = cross_val_score(model, x_train, y_train,
                             scoring="neg_root_mean_squared_error",
                             cv=cv_folds)
    rmse_scores = -scores
    results.append({
      "Model": name,
      LOG_RMSE_MEAN_: rmse_scores.mean(),
      "log-RMSE (std)": rmse_scores.std()
    })

  results_df = pd.DataFrame(results).sort_values(LOG_RMSE_MEAN_)
  return results_df


def compare_models_results(results, seconds=False):
  import matplotlib.pyplot as plt
  import numpy as np

  if seconds and LOG_RMSE_MEAN_ in results.columns:
    results = results.copy()
    results["RMSE (sec)"] = np.expm1(results[LOG_RMSE_MEAN_]).round(3)

  results = results.sort_values(LOG_RMSE_MEAN_)
  results.plot(x="Model", y=LOG_RMSE_MEAN_, kind="barh", legend=False,
               figsize=(8, 4))
  plt.xlabel("log-RMSE (lower is better)")
  plt.title("Model Performance")
  plt.tight_layout()
  plt.show()

  if seconds:
    display(results)


# Feature selection utility
def select_features(df: pd.DataFrame, feature_groups: dict) -> pd.DataFrame:
  """Returns a DataFrame with only the selected features."""
  cols = sum(feature_groups.values(), [])  # flatten all feature lists
  return df[cols]


# Preprocessing pipeline builder
def make_preprocessing_pipeline(feature_groups: dict) -> ColumnTransformer:
  return ColumnTransformer([
    ('num', num_base_pipelining(), feature_groups.get('num', [])),
    ('cat', cat_base_pipelining(), feature_groups.get('cat', [])),
    ('geo_pick',
     geo_base_pipelining(N_PICKUP_CLUSTERS, RANDOM_SEED, KMEANS_BATCH_SIZE),
     feature_groups.get('geo_pick', [])),
    ('geo_drop',
     geo_base_pipelining(N_DROPOFF_CLUSTERS, RANDOM_SEED, KMEANS_BATCH_SIZE),
     feature_groups.get('geo_drop', [])),
    ('bool', bool_base_pipelining(), feature_groups.get('bool', [])),
  ])


import numpy as np
from sklearn.preprocessing import FunctionTransformer

EARTH_RADIUS_KM = 6371.0088


def _haversine_array(x):
  # ensure numpy array and use only the first 4 columns in the expected order
  x = np.asarray(x, dtype=float)[:, :4]

  lat1, lon1, lat2, lon2 = np.radians(x.T)
  dlat = lat2 - lat1
  dlon = lon2 - lon1

  a = np.sin(dlat / 2.0) ** 2 + \
      np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
  c = 2 * np.arcsin(np.sqrt(a))
  km = EARTH_RADIUS_KM * c
  return km.reshape(-1, 1)


def _haversine_feature_names(_=None) -> list[str]:
  """Return a single output name so ColumnTransformer can introspect."""
  return ["hav_dist_km"]


def build_haversine_transformer() -> FunctionTransformer:
  """
  Picklable transformer that adds 'hav_dist_km' (great-circle distance).
  """
  return FunctionTransformer(
      func=_haversine_array,
      feature_names_out=_haversine_feature_names,
      validate=False,
      check_inverse=False,
  )
