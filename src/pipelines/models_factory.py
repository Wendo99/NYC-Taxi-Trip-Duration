from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import constants.modell_constants as model_constants
from constants.modell_constants import RANDOM_STATE


def build_model(model_name, preprocessor: None):
  modell_builder = {
    "LinearRegression": LinearRegression(n_jobs=model_constants.N_JOBS),
    "Ridge": Ridge(
        random_state=RANDOM_STATE,
        alpha=model_constants.R_ALPHA
    ),
    "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE,
                                          n_jobs=model_constants.N_JOBS,
                                          max_features=model_constants.RF_MAX_FEATURES,
                                          min_samples_leaf=model_constants.RF_MIN_SAMPLES_LEAF,
                                          ),
    'XGBoost': XGBRegressor(random_state=RANDOM_STATE,
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
                            ),
    'Bayes': BayesianRidge(
    ),

  }
  modell = modell_builder[model_name]

  if preprocessor is not None:
    return Pipeline([
      ("preprocessor", preprocessor),
      ("model", modell)
    ], memory=None)
  return modell
