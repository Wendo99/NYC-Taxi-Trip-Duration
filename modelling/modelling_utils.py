from collections import OrderedDict
from typing import Iterable

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from modelling.modelling_config import RANDOM_SEED, RF_ESTIMATORS, MAX_RF_DEPTH, \
  RF_N_JOBS, MIN_RF_SAMPLES_LEAF, MAX_RF_FEATURES, \
  LGBM_ESTIMATORS, LGBM_LEARNING_RATE, LGBM_MAX_DEPTH, XGB_ESTIMATORS, \
  XGB_LEARNING_RATE, XGB_MAX_DEPTH, XGB_VERBOSITY, DT_MAX_DEPTH, CPP_ALPHA, \
  DT_MAX_FEATURES, DT_MIN_SAMPLES_LEAF, DT_MIN_SAMPLES_SPLIT, \
  LINREG_FIT_INTERCEPT, LINREG_POSITIVE, RIDGE_ALPHA, LASSO_ALPHA, \
  LASSO_SELECTION, LASSO_FIT_INTERCEPT, LGBM_SUBSAMPLE, LGBM_COLSAMPLE_BYTREE, \
  LGBM_VERBOSE, XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE, XGB_N_JOBS, \
  XGB_TREE_METHOD, RF_BOOTSTRAP, XGB_GAMMA


def feature_to_category(df, features):
  for col in features:
    if col in df.columns:
      df[col] = df[col].astype('category')
  return df


def feature_as_bool(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    inplace: bool = False
) -> pd.DataFrame:
  """
  Cast *cols* to pandas BooleanDtype (nullable) so you can keep `NaN`
  sentinels if they exist.

  Parameters
  ----------
  df : DataFrame
  cols : list[str] | Index | any iterable of column names
  inplace : bool, default False
      If True, mutate *df*; otherwise work on a copy and return it.

  Returns
  -------
  DataFrame
      With the requested columns cast to BooleanDtype.
  """
  if not inplace:
    df = df.copy(deep=False)

  for col in cols:
    # silently skip missing cols instead of raising KeyError
    if col in df.columns:
      df[col] = df[col].astype("boolean")  # keeps NaNs

  return df


_BASE_MODELS = OrderedDict({
  # linear family
  # ---------- linear family -------------------------------------------------
  "LinearRegression": (
    LinearRegression,
    dict(
        fit_intercept=LINREG_FIT_INTERCEPT,
        positive=LINREG_POSITIVE,
    ),),
  "Ridge": (
    Ridge,
    dict(
        alpha=RIDGE_ALPHA,  # reuse same intercept choice
    ),),
  "Lasso": (
    Lasso,
    dict(
        alpha=LASSO_ALPHA,
        selection=LASSO_SELECTION,
        fit_intercept=LASSO_FIT_INTERCEPT,
    ),),

  # ---------- tree / ensemble family ---------------------------------------
  "DecisionTree": (
    DecisionTreeRegressor,
    dict(
        max_depth=DT_MAX_DEPTH,
        ccp_alpha=CPP_ALPHA,
        max_features=DT_MAX_FEATURES,
        min_samples_leaf=DT_MIN_SAMPLES_LEAF,
        min_samples_split=DT_MIN_SAMPLES_SPLIT,
        random_state=RANDOM_SEED,
        n_jobs=RF_N_JOBS,
    ),
  ),
  "RandomForest": (
    RandomForestRegressor,
    dict(
        n_estimators=RF_ESTIMATORS,
        max_depth=MAX_RF_DEPTH,
        random_state=RANDOM_SEED,
        n_jobs=RF_N_JOBS,
        min_samples_leaf=MIN_RF_SAMPLES_LEAF,
        max_features=MAX_RF_FEATURES,
        bootstrap=RF_BOOTSTRAP,
    ),
  ),
  "LightGBM": (
    LGBMRegressor,
    dict(
        n_estimators=LGBM_ESTIMATORS,
        learning_rate=LGBM_LEARNING_RATE,
        max_depth=LGBM_MAX_DEPTH,
        subsample=LGBM_SUBSAMPLE,
        feature_fraction=LGBM_COLSAMPLE_BYTREE,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=LGBM_VERBOSE,
        bagging_fraction=LGBM_BAGGING_FRACTION,
        bagging_freq=LGBM_BAGGING_FREQ,
        num_leaves=LGBM_NUM_LEAVES
    ),
  ),
  "XGBoost": (
    XGBRegressor,
    dict(
        n_estimators=XGB_ESTIMATORS,
        learning_rate=XGB_LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        random_state=RANDOM_SEED,
        n_jobs=XGB_N_JOBS,
        tree_method=XGB_TREE_METHOD,
        verbosity=XGB_VERBOSITY,
        gamma=XGB_GAMMA
    ),
  ),
})


def make_all_models(pre) -> dict[str, Pipeline]:
  """Return one pipeline per baseline model (linear + tree)."""
  return {
    name: Pipeline(
        steps=[("pre", pre), ("model", est(**params))],
        memory=None)
    for name, (est, params) in _BASE_MODELS.items()
  }
