import pandas as pd
from IPython.core.display_functions import display
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from modelling.modelling_config import CV_FOLDS, RIDGE_ALPHA, RANDOM_SEED, \
  LASSO_ALPHA, MAX_ITER, N_PICKUP_CLUSTERS, KMEANS_BATCH_SIZE, \
  N_DROPOFF_CLUSTERS

LOG_RMSE_MEAN_ = 'log-RMSE (mean)'


# --- Pipelines
def make_linear_pipeline(model_type='linreg', preprocessing=None):
  if model_type == 'ridge':
    model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)
  elif model_type == 'lasso':
    model = Lasso(alpha=LASSO_ALPHA, random_state=RANDOM_SEED,
                  max_iter=MAX_ITER)
  else:
    model = LinearRegression()
  return Pipeline([('pre', preprocessing), ('model', model)], memory=None)


def cat_base_pipelining():
  return 'passthrough'


def num_base_pipelining():
  return make_pipeline(
      StandardScaler(), memory=None)


def bool_base_pipelining():
  return 'passthrough'


def geo_base_pipelining(n_clusters, random_state, batch_size):
  return make_pipeline(MiniBatchKMeans(n_clusters, random_state=random_state,
                                       batch_size=batch_size), memory=None)


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
  display(results_df)
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
