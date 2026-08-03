"""Construction of the candidate regressors, each wrapped in a pipeline.

Estimators are built lazily so only the requested model is instantiated, and
so an unusable optional dependency affects only its own entry — XGBoost needs
a system OpenMP runtime that the wheel does not bundle.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from sklearn.pipeline import Pipeline

import nyc_taxi.config.modell_constants as model_constants
from nyc_taxi.config.modell_constants import RANDOM_STATE

XGB_INSTALL_HINT = (
  "XGBoost is an optional dependency of this project. Install it with "
  "`uv sync --extra xgb`. On macOS the OpenMP runtime is required as well: "
  "`brew install libomp`. All other models run without it."
)


def _build_xgboost():
  # Imported lazily so a missing XGBoost only affects the XGBoost model
  # instead of this whole module. Note the broad except: when libomp is
  # absent, xgboost raises XGBoostError at import time, not ImportError.
  try:
    from xgboost import XGBRegressor  # noqa: PLC0415
  except Exception as exc:
    raise ImportError(f"{XGB_INSTALL_HINT}\n\nOriginal error: {exc}") from exc

  return XGBRegressor(random_state=RANDOM_STATE,
                      n_jobs=model_constants.N_JOBS,
                      colsample_bytree=model_constants.COLSAMPLE_BYTREE,
                      gamma=model_constants.GAMMA,
                      learning_rate=model_constants.LEARNING_RATE,
                      min_child_weight=model_constants.MIN_CHILD_WEIGHT,
                      reg_alpha=model_constants.REG_ALPHA,
                      reg_lambda=model_constants.REG_LAMBDA,
                      subsample=model_constants.SUBSAMPLE,
                      max_depth=model_constants.X_MAX_DEPTH,
                      n_estimators=model_constants.X_N_ESTIMATORS,
                      )


def build_model(model_name: str, preprocessor: ColumnTransformer | None):
  """Return the named estimator, wrapped in a pipeline if given one.

  ``preprocessor`` was annotated ``None`` rather than ``ColumnTransformer |
  None``, so every caller passing a real transformer looked like a type error.
  """
  # Values are builders, not instances, so only the requested model is created.
  modell_builder = {
    "LinearRegression": lambda: LinearRegression(n_jobs=model_constants.N_JOBS),
    "Ridge": lambda: Ridge(
        random_state=RANDOM_STATE,
        alpha=model_constants.R_ALPHA
    ),
    "RandomForest": lambda: RandomForestRegressor(random_state=RANDOM_STATE,
                                                  n_jobs=model_constants.N_JOBS,
                                                  max_features=model_constants.RF_MAX_FEATURES,
                                                  min_samples_leaf=model_constants.RF_MIN_SAMPLES_LEAF,
                                                  ),
    'XGBoost': _build_xgboost,
    'Bayes': BayesianRidge,  # already a zero-arg builder

  }
  modell = modell_builder[model_name]()

  if preprocessor is not None:
    return Pipeline([
      ("preprocessor", preprocessor),
      ("model", modell)
    ], memory=None)
  return modell
