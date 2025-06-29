from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from constants.modelling_c import RANDOM_STATE


def build_model(model_name, preprocessor: None):
  modell_builder = {
    "LinearRegression": LinearRegression(n_jobs=-1),
    "Ridge": Ridge(
        # alpha=R_ALPHA,
        random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE,
                                          n_jobs=-1,
                                          # n_estimators=80,
                                          # max_samples=0.2,
                                          # bootstrap=True,
                                          max_features=1.0,
                                          min_samples_leaf=1,
                                          # n_estimators=RF_N_ESTIMATORS,
                                          # min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                                          # max_features=RF_MAX_FEATURES
                                          ),
    'XGBoost': XGBRegressor(random_state=RANDOM_STATE,
                            # learning_rate=X_LEARNING_RATE,
                            # max_depth=X_MAX_DEPTH,
                            # n_estimators=X_N_ESTIMATORS,
                            n_jobs=-1,
                            ),
    'Bayes': BayesianRidge(
    ),

  }
  modell = modell_builder[model_name]

  if preprocessor is not None:
    return Pipeline([
      ("preprocessor", preprocessor),
      ("modell", modell)
    ], memory=None)
  return modell
