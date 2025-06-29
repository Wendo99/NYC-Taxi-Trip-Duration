from __future__ import annotations

import os
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV, \
  cross_val_score
from sklearn.pipeline import Pipeline

from constants import core_c, features_c
from constants.modelling_c import RANDOM_STATE, param_spaces
from pipelines.merge_pipeline import build_merged_dataset
from pipelines.models import build_model


def load_taxi_weather_data(recompute=False):
  csv = core_c.PROCESSED_DIR / "taxi_weather.csv"
  if not csv.exists() or recompute:
    df = build_merged_dataset(save_csv=True)
  else:
    df = pd.read_csv(csv)
  return df


def get_feature_names_safe(preprocessor, original_cols):
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


def top_linreg_features(modell, x_train, top_n: int = 20) -> (
    pd.DataFrame):
  """
        :param top_n:
        :param x_train:
        :param modell:
    """
  if not isinstance(modell, Pipeline):
    raise TypeError("'pipe' must be a sklearn pipeline")
  if "modell" not in modell.named_steps:
    raise ValueError("Last pipeline step must be called 'modell'")
  nam_steps = modell.named_steps["modell"]
  if not hasattr(nam_steps, "coef_"):
    raise AttributeError("modell has no attribute 'coef_'")

  features = get_feature_names_safe(modell.named_steps["preprocessor"],
                                    x_train.columns)
  coefs = nam_steps.coef_.ravel()
  abscoef = np.abs(coefs)

  order = np.argsort(abscoef)[::-1][:top_n]
  df = pd.DataFrame(
      {"feature": features[order],
       "abs_coef": abscoef[order]}
  ).reset_index(drop=True)

  total = df["abs_coef"].sum()

  df["rel_importance"] = (df["abs_coef"] / total)
  df["cum_importance"] = df["rel_importance"].cumsum()

  df["rel_importance"] = (df["rel_importance"] * 100).round(2)
  df["cum_importance"] = (df["cum_importance"] * 100).round(2)

  return df


def top_tree_features(
    modell,
    x_train,
    top_n: int = 20,
    xgb_importance: str = "gain",
    as_dataframe: bool = True,
) -> Union[pd.DataFrame, Tuple[List[str], np.ndarray]]:
  """

  :param as_dataframe:
  :param xgb_importance:
  :param top_n:
  :param x_train:
  :param modell:
  """

  if not isinstance(modell, Pipeline):
    raise TypeError("'pipe' must be a sklearn pipeline")

  if "modell" not in modell.named_steps:
    raise ValueError("Last pipeline step must be called 'modell'")

  nam_steps = modell.named_steps["modell"]
  features = get_feature_names_safe(modell.named_steps["preprocessor"],
                                    x_train.columns)

  if hasattr(nam_steps, "feature_importances_"):
    importance = nam_steps.feature_importances_

  elif nam_steps.__class__.__name__.startswith("XGB"):
    booster = nam_steps.get_booster()
    score_dict = booster.get_score(importance_type=xgb_importance)
    importance = np.zeros(len(features))
    for k, v in score_dict.items():
      idx = int(k[1:])
      importance[idx] = v

  else:
    raise TypeError("The modell type is not supported.")

  order = np.argsort(importance)[::-1][:top_n]

  df = pd.DataFrame({"feature": features[order], "importance": importance[
    order]}).reset_index(drop=True)

  total = df["importance"].sum()

  df["rel_importance"] = (df["importance"] / total)
  df["cum_importance"] = df["rel_importance"].cumsum()

  df["rel_importance"] = (df["rel_importance"] * 100).round(2)
  df["cum_importance"] = (df["cum_importance"] * 100).round(2)

  if as_dataframe:
    return df
  else:
    return features[order].tolist(), importance[order]


log_rmse = make_scorer(root_mean_squared_error,
                       greater_is_better=False)


def cv_report(modell: str, x_train, y_train):
  pipe = build_model(modell)

  cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
  scores = cross_validate(
      pipe,
      X=x_train,
      y=y_train,
      cv=cv,
      scoring=make_scorer(root_mean_squared_error,
                          greater_is_better=False),
      return_train_score=False,
      n_jobs=-1,
  )

  mean_log_rmse = -scores["test_score"].mean()
  std_log_rmse = scores["test_score"].std()
  print(modell + f"3-fold CV: {mean_log_rmse:.6f} ± {std_log_rmse:.6f}")


def top_generic_features(
    modell,
    x_train,
    y_train,
    top_n: int = 20,
    scorer=None,
    random_state: int = 42,
    n_repeats: int = 5,
    subsample: int | None = None,
) -> pd.DataFrame:
  """

  :param subsample:
  :param n_repeats:
  :param random_state:
  :param scorer:
  :param top_n:
  :param y_train:
  :param x_train:
  :param modell:
  """
  if scorer is None:
    scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

  nam_steps = modell.named_steps["modell"]
  pre = modell.named_steps["preprocessor"]

  if hasattr(nam_steps, "feature_importance_"):
    names = get_feature_names_safe(pre, x_train.columns)
    imps = nam_steps.feature_importances_
    order = np.argsort(imps)[::-1][:top_n]

    df = pd.DataFrame({"feature": names[order],
                       "importance": imps[order]}).reset_index(drop=True)

    total = df["importance"].sum()

    df["rel_importance"] = (df["importance"] / total)
    df["cum_importance"] = df["rel_importance"].cumsum()

    df["rel_importance"] = (df["rel_importance"] * 100).round(2)
    df["cum_importance"] = (df["cum_importance"] * 100).round(2)

    return df

  if hasattr(nam_steps, "coef_"):
    names = get_feature_names_safe(pre, x_train.columns)
    coefs = nam_steps.coef_.ravel()
    order = np.argsort(np.abs(coefs))[::-1][:top_n]

    df = pd.DataFrame({"feature": names[order],
                       "abs_coef": np.abs(coefs[order])}).reset_index(
        drop=True)

    total = df["abs_coef"].sum()

    df["rel_importance"] = (df["abs_coef"] / total)
    df["cum_importance"] = df["rel_importance"].cumsum()

    df["rel_importance"] = (df["rel_importance"] * 100).round(2)
    df["cum_importance"] = (df["cum_importance"] * 100).round(2)

    return df

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
  names = get_feature_names_safe(pre, x_train.columns)
  imps = result.importances_mean
  order = np.argsort(imps)[::-1][:top_n]

  df = pd.DataFrame({"feature": names[order], "importance": imps[
    order]}).reset_index(drop=True)

  total = df["importance"].sum()

  df["rel_importance"] = (df["importance"] / total)
  df["cum_importance"] = df["rel_importance"].cumsum()

  df["rel_importance"] = (df["rel_importance"] * 100).round(2)
  df["cum_importance"] = (df["cum_importance"] * 100).round(2)

  return df


import json, joblib, pathlib

RESULTS_DIR = pathlib.Path("../artifacts")
RESULTS_DIR.mkdir(exist_ok=True)


def save_best(search, name):
  (RESULTS_DIR / f"{name}_best_params.json").write_text(
      json.dumps(search.best_params_, indent=2)
  )
  joblib.dump(search.best_estimator_, RESULTS_DIR / f"{name}_model.job")


def search_hyperparameters(modell_name: str, x_train, y_train, n_iter):
  pipeline = build_model(modell_name)

  search_modell = RandomizedSearchCV(
      estimator=pipeline,
      param_distributions=param_spaces[modell_name],
      n_iter=n_iter,
      cv=3,
      scoring="neg_root_mean_squared_error",
      random_state=RANDOM_STATE,
      n_jobs=-1,
      verbose=1,
      refit=False
  )

  search_modell.fit(x_train, y_train)

  print("Best " + modell_name + " CV score log-RMSE:",
        -search_modell.best_score_)
  print("Best " + modell_name + " hyper-parameters:",
        search_modell.best_params_)


def cv_train(modell_name: str, model_pipe, x_train, y_train):
  pipe = model_pipe
  log_rmses = -cross_val_score(pipe, x_train, y_train,
                               scoring="neg_root_mean_squared_error",
                               cv=3)
  print(
      modell_name + f" Log-RMSE (mean): {pd.Series(log_rmses).mean():.6f}")
  print(
      modell_name + f" Log-RMSE (std): {pd.Series(log_rmses).std():.6f}")


def fit_save_model(model_name, preprocessor, x_train, y_train,
    retrain=False, model_dir="../models"):
  os.makedirs(model_dir, exist_ok=True)
  model_path = os.path.join(model_dir, f"{model_name.lower()}.joblib")
  if os.path.exists(model_path) and not retrain:
    modell = joblib.load(model_path)
  else:
    modell = build_model(model_name, preprocessor)
    modell.fit(x_train, y_train)
    joblib.dump(modell, model_path)
  return modell


def get_res_errors(modell, x_train, y_train):
  y_pred = modell.predict(x_train)
  res = y_train - y_pred

  df_err = x_train.copy()
  df_err["y_true_log"] = y_train
  df_err["y_pred_log"] = y_pred
  df_err["residual"] = res
  df_err["abs_res"] = res.abs()
  return df_err


def rmse(y_true, y_pred):
  return np.sqrt(np.mean((y_true - y_pred) ** 2))


def rmse_by_group(df, col):
  return (df
          .groupby(col, observed=True)
          .agg(rmse=("y_true_log",
                     lambda y: rmse(y,
                                    df.loc[y.index, "y_pred_log"])))
          .rmse
          .sort_values(ascending=False))


def list_res_errors(df_err, model_name: str):
  print(model_name)

  for col in features_c.RES_COL:
    print(f"\n=== {col} ===")
    print(rmse_by_group(df_err, col).head(10))


def list_errors_10_bins(df_err, model_name: str, col):
  df_err["dist_bin"] = pd.qcut(df_err[col], q=10, labels=False)
  print(model_name + ' – ' + col)
  print(rmse_by_group(df_err, "dist_bin"))
