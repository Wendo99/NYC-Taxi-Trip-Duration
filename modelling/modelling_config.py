# modelling_config.py  – suggested skeleton
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCfg:
  n_estimators: int
  max_depth: int | None
  learning_rate: float = 0.1
  subsample: float | None = None
  colsample_bytree: float | None = None
  extra: dict = None  # model-specific catch-all


RANDOM_SEED = 42
CV_FOLDS = 3

KMEANS = dict(
    n_pickup_clusters=5,
    n_dropoff_clusters=4,
    batch_size=10_000,
)

MODEL_CFGS: dict[str, ModelCfg] = {
  "RandomForest": ModelCfg(
      n_estimators=200, max_depth=10, learning_rate=0.0,  # LR ignored
      extra=dict(min_samples_leaf=5, max_features="sqrt", n_jobs=-1),
  ),
  "LightGBM": ModelCfg(
      n_estimators=400, max_depth=6, learning_rate=0.05,
      subsample=0.8, colsample_bytree=0.8,
      extra=dict(verbose=-1),
  ),
  "XGBoost": ModelCfg(
      n_estimators=600, max_depth=8, learning_rate=0.05,
      subsample=0.8, colsample_bytree=0.8,
      extra=dict(tree_method="hist", n_jobs=-1, verbosity=0),
  ),
}

# random seed
RANDOM_SEED = 42

CV_FOLDS = 3

# Number of clusters used for geo_pick and geo_drop features (KMeans)
N_PICKUP_CLUSTERS = 5
N_DROPOFF_CLUSTERS = 4
KMEANS_BATCH_SIZE = 10000

# Number of trees for RandomForest and comparable models (conservative starting value)
DEFAULT_ESTIMATORS = 10

# decision tree
DT_MAX_DEPTH = 10
DT_MAX_FEATURES = None
DT_MIN_SAMPLES_LEAF = 9
DT_MIN_SAMPLES_SPLIT = 6
CPP_ALPHA = 0.0

# RandomForest parameters
RF_ESTIMATORS = 10
MAX_RF_DEPTH = 10
MIN_RF_SAMPLES_LEAF = 5
MAX_RF_FEATURES = 'sqrt'
RF_N_JOBS = -1
RF_BOOTSTRAP = True
RF_MIN_SAMPLES_SPLIT = 6

# Linreg parameters
LINREG_POSITIVE = False
LINREG_FIT_INTERCEPT = True

# Ridge parameters
RIDGE_ALPHA = 20.0

# Lasso parameters
LASSO_ALPHA = 0.1
LASSO_SELECTION = 'cyclic'
LASSO_FIT_INTERCEPT = True

# LightGBM parameters
LGBM_ESTIMATORS = 600
LGBM_MAX_DEPTH = 8
LGBM_SUBSAMPLE = 0.8
LGBM_COLSAMPLE_BYTREE = 0.8
LGBM_LEARNING_RATE = 0.1
LGBM_VERBOSE = -1
LGBM_FEATURE_FRACTION = 0.8
LGBM_BAGGING_FRACTION = 0.8
LGBM_BAGGING_FREQ = 0
LGBM_NUM_LEAVES = 63

# XGBoost
XGB_ESTIMATORS = 300
XGB_MAX_DEPTH = 9
XGB_LEARNING_RATE = 0.05
XGB_N_JOBS = -1
XGB_TREE_METHOD = 'hist'
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_VERBOSITY = 0
XGB_GAMMA = 0.0
