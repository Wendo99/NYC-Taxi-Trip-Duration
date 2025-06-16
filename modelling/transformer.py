import pandas as pd
from IPython.core.display_functions import display
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from modelling.modelling_config import CV_FOLDS, RIDGE_ALPHA, RANDOM_SEED, \
  LASSO_ALPHA, MAX_ITER


# --- Pipelines
def make_linear_pipeline(model_type='linreg', preprocessing=None):
  if model_type == 'ridge':
    model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)
  elif model_type == 'lasso':
    model = Lasso(alpha=LASSO_ALPHA, random_state=RANDOM_SEED,
                  max_iter=MAX_ITER)
  else:
    model = LinearRegression()
  return Pipeline([('pre', preprocessing), ('model', model)])


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
      "log-RMSE (mean)": rmse_scores.mean(),
      "log-RMSE (std)": rmse_scores.std()
    })

  results_df = pd.DataFrame(results).sort_values("log-RMSE (mean)")
  display(results_df)
  return results_df


def compare_models_results(results, seconds=False):
  import matplotlib.pyplot as plt
  import numpy as np

  if seconds and 'log-RMSE (mean)' in results.columns:
    results = results.copy()
    results["RMSE (sec)"] = np.expm1(results["log-RMSE (mean)"]).round(3)

  results = results.sort_values("log-RMSE (mean)")
  results.plot(x="Model", y="log-RMSE (mean)", kind="barh", legend=False,
               figsize=(8, 4))
  plt.xlabel("log-RMSE (lower is better)")
  plt.title("Model Performance")
  plt.tight_layout()
  plt.show()

  if seconds:
    display(results)

  # --- Model evaluation
  # def get_display_models_results(models, x_train, y_train, cv_folds=CV_FOLDS):
  #   results = []
  #
  #   for name, model in models.items():
  #     scores = cross_val_score(model, x_train, y_train,
  #                              scoring="neg_root_mean_squared_error", cv=cv_folds)
  #     rmse_scores = -scores
  #     results.append({
  #       "Model": name,
  #       "log-RMSE (mean)": rmse_scores.mean(),
  #       "log-RMSE (std)": rmse_scores.std()
  #     })
  #
  #   results_df = pd.DataFrame(results).sort_values(by="log-RMSE (mean)")
  #
  #   display(results_df)
  #
  #   return results_df

  # def convert_logrmse_to_seconds(results_df):
  #   """
  #   Converts log-RMSE values to approximate RMSE in seconds.
  #     Parameters:
  #         results_df (pd.DataFrame): DataFrame with 'log-RMSE (mean)' column.
  #     Returns:
  #         pd.DataFrame: DataFrame with additional 'RMSE (sec, approx)' column.
  #     """
  #   df = results_df.copy()
  #   df["RMSE (sec, approx)"] = np.expm1(df["log-RMSE (mean)"])
  #   df["RMSE (sec, approx)"] = df["RMSE (sec, approx)"].round(2)
  #   display(df)
  #   return df
  #
  #
  # def compare_models_results(models_results):
  #   sns.barplot(x="log-RMSE (mean)", y="Model", data=models_results)
  #   plt.title("Model comparison based on log-RMSE")
  #   plt.xlabel("log-RMSE (error measure)")
  #   plt.tight_layout()
  #   plt.show()
