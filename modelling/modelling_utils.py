from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from modelling.modelling_config import RANDOM_SEED, DT_MAX_DEPTH, CPP_ALPHA, \
  RF_ESTIMATORS, MAX_RF_DEPTH, RF_N_JOBS, MIN_RF_SAMPLES_LEAF, MAX_RF_FEATURES, \
  LGBM_ESTIMATORS, LGBM_LEARNING_RATE, LGBM_MAX_DEPTH, XGB_ESTIMATORS, \
  XGB_LEARNING_RATE, XGB_MAX_DEPTH, XGB_VERBOSITY
from modelling.transformer import make_linear_pipeline


def feature_to_category(df, features):
  for col in features:
    if col in df.columns:
      df[col] = df[col].astype('category')
  return df


def feature_as_bool(df, features):
  for col in features:
    df[col] = df[col].astype(bool)
  return df


def make_all_models(preprocessor: ColumnTransformer) -> dict:
  return {
    'LinearRegression': make_linear_pipeline('linreg', preprocessor),
    'Ridge': make_linear_pipeline('ridge', preprocessor),
    'Lasso': make_linear_pipeline('lasso', preprocessor),
    'DecisionTree': Pipeline([
      ('pre', preprocessor),
      ('model', DecisionTreeRegressor(
          random_state=RANDOM_SEED,
          max_depth=DT_MAX_DEPTH,
          ccp_alpha=CPP_ALPHA
      ))
    ], memory=None),
    'RandomForest': Pipeline([
      ('pre', preprocessor),
      ('model', RandomForestRegressor(
          n_estimators=RF_ESTIMATORS,
          max_depth=MAX_RF_DEPTH,
          random_state=RANDOM_SEED,
          n_jobs=RF_N_JOBS,
          min_samples_leaf=MIN_RF_SAMPLES_LEAF,
          max_features=MAX_RF_FEATURES
      ))
    ], memory=None),
    'LightGBM': Pipeline([
      ('pre', preprocessor),
      ('model', LGBMRegressor(
          n_estimators=LGBM_ESTIMATORS,
          learning_rate=LGBM_LEARNING_RATE,
          max_depth=LGBM_MAX_DEPTH,
          random_state=RANDOM_SEED
      ))
    ], memory=None),
    'XGBoost': Pipeline([
      ('pre', preprocessor),
      ('model', XGBRegressor(
          n_estimators=XGB_ESTIMATORS,
          learning_rate=XGB_LEARNING_RATE,
          max_depth=XGB_MAX_DEPTH,
          random_state=RANDOM_SEED,
          verbosity=XGB_VERBOSITY
      ))
    ], memory=None)
  }
