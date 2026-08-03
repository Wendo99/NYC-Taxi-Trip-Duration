"""Model fitting, cross-validation, hyper-parameter search and error tables."""
from __future__ import annotations

import json
from typing import List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from nyc_taxi.config.modell_constants import RANDOM_STATE, param_spaces
from nyc_taxi.config.path_file_constants import ARTIFACTS_DIR, MODELS_DIR
from pipelines.models_factory import build_model
from utilities.modelling_utilities import RES_COL

# Name of the estimator step inside the pipelines built by models_factory.
MODEL_STEP = "model"
PREPROCESSOR_STEP = "preprocessor"


def get_feature_names_safe(preprocessor, original_cols):
  """Best-effort feature names out of a fitted ColumnTransformer."""
  names = []
  for name, trans, cols, _ in preprocessor._iter(
      fitted=True,
      column_as_labels=False,
      skip_empty_columns=True,
      skip_drop=True
  ):
    if trans == "passthrough":
      names.extend(
          [original_cols[c] if isinstance(c, int) else c for c in cols])
    elif hasattr(trans, "steps"):
      last_step = trans.steps[-1][1]
      if last_step == "passthrough" or last_step == "identity":
        names.extend(
            [original_cols[c] if isinstance(c, int) else c for c in cols])
      elif hasattr(last_step, "get_feature_names_out"):
        try:
          names.extend(last_step.get_feature_names_out(cols))
        except Exception:
          names.extend([f"{name}__{c}" for c in cols])
      else:
        names.extend([f"{name}__{c}" for c in cols])
    elif hasattr(trans, "get_feature_names_out"):
      try:
        names.extend(trans.get_feature_names_out(cols))
      except Exception:
        names.extend([f"{name}__{c}" for c in cols])
    else:
      names.extend([f"{name}__{c}" for c in cols])
  return np.asarray(names)


def _importance_table(features, values, top_n: int,
    value_col: str = "importance") -> pd.DataFrame:
  """Rank *features* by *values* and add relative/cumulative percentages.

  This block previously existed in four places (once in each of the three
  ``top_*_features`` helpers, and twice inside ``top_generic_features``).
  """
  values = np.asarray(values, dtype=float)
  order = np.argsort(values)[::-1][:top_n]

  df = pd.DataFrame({
    "feature": np.asarray(features)[order],
    value_col: values[order],
  }).reset_index(drop=True)

  total = df[value_col].sum()
  rel = df[value_col] / total if total else df[value_col] * 0.0
  df["rel_importance"] = (rel * 100).round(2)
  df["cum_importance"] = (rel.cumsum() * 100).round(2)
  return df


def _require_pipeline(modell) -> Tuple[object, object]:
  """Return (estimator, preprocessor) from a fitted project pipeline."""
  if not isinstance(modell, Pipeline):
    raise TypeError("'modell' must be a sklearn Pipeline")
  if MODEL_STEP not in modell.named_steps:
    raise ValueError(f"pipeline has no step named {MODEL_STEP!r}")
  return modell.named_steps[MODEL_STEP], modell.named_steps[PREPROCESSOR_STEP]


def top_linreg_features(modell, x_train, top_n: int = 20) -> pd.DataFrame:
  """Rank features of a linear model by absolute coefficient."""
  estimator, preprocessor = _require_pipeline(modell)
  if not hasattr(estimator, "coef_"):
    raise AttributeError("model has no attribute 'coef_'")

  features = get_feature_names_safe(preprocessor, x_train.columns)
  return _importance_table(
      features, np.abs(estimator.coef_.ravel()), top_n, value_col="abs_coef")


def top_tree_features(
    modell,
    x_train,
    top_n: int = 20,
    xgb_importance: str = "gain",
    as_dataframe: bool = True,
) -> Union[pd.DataFrame, Tuple[List[str], np.ndarray]]:
  """Rank features of a tree ensemble by impurity or gain importance."""
  estimator, preprocessor = _require_pipeline(modell)
  features = get_feature_names_safe(preprocessor, x_train.columns)

  if hasattr(estimator, "feature_importances_"):
    importance = estimator.feature_importances_
  elif estimator.__class__.__name__.startswith("XGB"):
    booster = estimator.get_booster()
    score_dict = booster.get_score(importance_type=xgb_importance)
    importance = np.zeros(len(features))
    for k, v in score_dict.items():
      importance[int(k[1:])] = v
  else:
    raise TypeError("The model type is not supported.")

  if not as_dataframe:
    order = np.argsort(importance)[::-1][:top_n]
    return np.asarray(features)[order].tolist(), importance[order]

  return _importance_table(features, importance, top_n)


def top_generic_features(
    modell,
    x_train,
    y_train,
    top_n: int = 20,
    scorer=None,
    random_state: int = RANDOM_STATE,
    n_repeats: int = 5,
    subsample: int | None = None,
) -> pd.DataFrame:
  """Rank features for any estimator, falling back to permutation importance.

  Uses native importances or coefficients when the estimator exposes them,
  and only pays for permutation importance otherwise.
  """
  if scorer is None:
    scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

  estimator, preprocessor = _require_pipeline(modell)
  names = get_feature_names_safe(preprocessor, x_train.columns)

  if hasattr(estimator, "feature_importances_"):
    return _importance_table(names, estimator.feature_importances_, top_n)

  if hasattr(estimator, "coef_"):
    return _importance_table(
        names, np.abs(estimator.coef_.ravel()), top_n, value_col="abs_coef")

  if subsample is not None and subsample < len(x_train):
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(x_train), size=subsample, replace=False)
    x_perm = x_train.iloc[idx]
    y_perm = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]
  else:
    x_perm, y_perm = x_train, y_train

  result = permutation_importance(
      modell, x_perm, y_perm,
      n_repeats=n_repeats,
      scoring=scorer,
      random_state=random_state,
      n_jobs=-1,
  )
  return _importance_table(names, result.importances_mean, top_n)


def search_hyperparameters(modell_name: str, preprocessor, x_train, y_train,
    n_iter, save: bool = False):
  """Randomised search over ``param_spaces[modell_name]``; prints the best.

  With ``save=True`` the winning params and estimator are written to
  ``artifacts/`` so a long search does not have to be repeated.
  """
  search_modell = RandomizedSearchCV(
      estimator=build_model(modell_name, preprocessor),
      param_distributions=param_spaces[modell_name],
      n_iter=n_iter,
      cv=5,
      scoring="neg_root_mean_squared_error",
      random_state=RANDOM_STATE,
      n_jobs=-1,
      verbose=1,
      refit=True
  )
  search_modell.fit(x_train, y_train)

  print(f"Best {modell_name} CV score log-RMSE:", -search_modell.best_score_)
  print(f"Best {modell_name} hyper-parameters:", search_modell.best_params_)

  if save:
    save_search_results(search_modell, modell_name)
  return search_modell


def save_search_results(search, name: str) -> None:
  """Persist the best params and estimator of a completed search."""
  (ARTIFACTS_DIR / f"{name}_best_params.json").write_text(
      json.dumps(search.best_params_, indent=2))
  joblib.dump(search.best_estimator_, ARTIFACTS_DIR / f"{name}_model.joblib")


def cv_train(modell_name: str, model_pipe, x_train, y_train):
  """Report mean/std log-RMSE over a 3-fold cross-validation."""
  log_rmses = -cross_val_score(model_pipe, x_train, y_train,
                               scoring="neg_root_mean_squared_error",
                               cv=3)
  scores = pd.Series(log_rmses)
  print(f"{modell_name} Log-RMSE (mean): {scores.mean():.6f}")
  print(f"{modell_name} Log-RMSE (std): {scores.std():.6f}")
  return scores


def fit_save_model(model_name, preprocessor, x_train, y_train,
    retrain: bool = False, model_dir=None):
  """Fit and cache a model, or load the cached one when it already exists."""
  model_dir = MODELS_DIR if model_dir is None else model_dir
  model_dir.mkdir(parents=True, exist_ok=True)
  model_path = model_dir / f"{model_name.lower()}.joblib"

  if model_path.exists() and not retrain:
    return joblib.load(model_path)

  modell = build_model(model_name, preprocessor)
  modell.fit(x_train, y_train)
  joblib.dump(modell, model_path)
  return modell


def get_res_errors(modell, x_train, y_train):
  """Attach predictions and residuals to a copy of the feature frame."""
  y_pred = modell.predict(x_train)
  res = y_train - y_pred

  df_err = x_train.copy()
  df_err["y_true_log"] = y_train
  df_err["y_pred_log"] = y_pred
  df_err["residual"] = res
  df_err["abs_res"] = res.abs()
  return df_err


def rmse(y_true, y_pred):
  """Root mean squared error.

  Thin wrapper over sklearn so notebooks and the residual helpers below share
  one definition.
  """
  return root_mean_squared_error(y_true, y_pred)


def rmse_by_group(df, col):
  """RMSE per level of *col*, worst first."""
  return (df
          .groupby(col, observed=True)
          .agg(rmse=("y_true_log",
                     lambda y: rmse(y, df.loc[y.index, "y_pred_log"])))
          .rmse
          .sort_values(ascending=False))


def list_res_errors(df_err, model_name: str):
  """Print the per-segment error table for every residual grouping column."""
  print(model_name)
  for col in RES_COL:
    print(f"\n=== {col} ===")
    print(rmse_by_group(df_err, col).head(10))


def list_errors_10_bins(df_err, model_name: str, col):
  """Print RMSE across ten equal-frequency bins of a continuous column."""
  df_err["dist_bin"] = pd.qcut(df_err[col], q=10, labels=False)
  print(f"{model_name} – {col}")
  print(rmse_by_group(df_err, "dist_bin"))
