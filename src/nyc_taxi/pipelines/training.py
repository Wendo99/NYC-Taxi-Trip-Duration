"""Model fitting, cross-validation, hyperparameter search and error tables."""
from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from nyc_taxi.config.modell_constants import RANDOM_STATE, param_spaces
from nyc_taxi.config.path_file_constants import ARTIFACTS_DIR, MODELS_DIR
from nyc_taxi.pipelines.models_factory import build_model

# Name of the estimator step inside the pipelines built by models_factory.
MODEL_STEP = "model"
PREPROCESSOR_STEP = "preprocessor"


def get_feature_names_safe(preprocessor, original_cols=None) -> np.ndarray:
  """Output feature names of a fitted ColumnTransformer, prefixes stripped.

  ``ColumnTransformer.get_feature_names_out`` prepends the transformer name
  (``nums__hav_dist_km_log``); the importance tables read better without it,
  so the ``<transformer>__`` prefix is removed.

  This replaces a hand-rolled walk over the private ``preprocessor._iter``,
  which reimplemented — with several fallbacks — what sklearn has exposed
  publicly since 1.0. Verified to produce identical names, in identical
  order, for this project's preprocessor.

  Parameters
  ----------
  preprocessor
      A *fitted* ColumnTransformer.
  original_cols
      Unused; kept so existing call sites need no change.
  """
  del original_cols  # kept for backwards compatibility
  names = preprocessor.get_feature_names_out()
  return np.asarray([str(n).split("__", 1)[-1] for n in names])


def importance_table(features, values, top_n: int,
    value_col: str = "importance") -> pd.DataFrame:
  """Rank *features* by *values* and add relative/cumulative percentages.

  This block previously existed in four places (once in each of the three
  ``top_*_features`` helpers, and twice inside ``top_generic_features``).

  Percentages are shares of the importance of **all** features, not just the
  ``top_n`` displayed. Normalizing over the displayed subset instead made
  ``rel_importance`` depend on ``top_n`` — the same feature read 52.96 % at
  ``top_n=40`` and 62.88 % at ``top_n=15`` — and made ``cum_importance``
  always end at 100 %, hiding how much of the model the table leaves out.
  """
  values = np.asarray(values, dtype=float)
  total = values.sum()
  order = np.argsort(values)[::-1][:top_n]

  df = pd.DataFrame({
    "feature": np.asarray(features)[order],
    value_col: values[order],
  }).reset_index(drop=True)

  rel = df[value_col] / total if total else df[value_col] * 0.0
  df["rel_importance"] = (rel * 100).round(2)
  df["cum_importance"] = (rel.cumsum() * 100).round(2)
  return df


def _require_pipeline(modell) -> tuple[Any, ColumnTransformer]:
  """Return (estimator, preprocessor) from a fitted project pipeline.

  The estimator is typed ``Any`` on purpose: it may be any of the regressors
  in ``models_factory``, and the callers below duck-type on ``coef_``,
  ``feature_importances_`` or ``get_booster`` after an explicit ``hasattr``
  guard.
  """
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
  return importance_table(
      features, np.abs(estimator.coef_.ravel()), top_n, value_col="abs_coef")


def top_tree_features(
    modell,
    x_train,
    top_n: int = 20,
    xgb_importance: str = "gain",
) -> pd.DataFrame:
  """Rank features of a tree ensemble by impurity or gain importance.

  Always returns a DataFrame. The former ``as_dataframe=False`` branch, which
  returned a ``(names, values)`` tuple, made the return type depend on an
  argument value; it had no callers, and the same data is available as
  ``df["feature"].tolist()`` / ``df["importance"].to_numpy()``.
  """
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

  return importance_table(features, importance, top_n)


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
    return importance_table(names, estimator.feature_importances_, top_n)

  if hasattr(estimator, "coef_"):
    return importance_table(
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
  return importance_table(names, result.importances_mean, top_n)


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


def list_errors_10_bins(df_err, model_name: str, col):
  """Print RMSE across ten equal-frequency bins of a continuous column."""
  df_err["dist_bin"] = pd.qcut(df_err[col], q=10, labels=False)
  print(f"{model_name} – {col}")
  print(rmse_by_group(df_err, "dist_bin"))
