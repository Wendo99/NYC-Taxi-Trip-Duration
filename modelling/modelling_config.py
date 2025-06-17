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
CPP_ALPHA = 0.001

# RandomForest parameters
RF_ESTIMATORS = 10
MAX_RF_DEPTH = 10
MIN_RF_SAMPLES_LEAF = 5
MAX_RF_FEATURES = 'sqrt'
RF_N_JOBS = -1

# Ridge parameters
RIDGE_ALPHA = 1.0

# Lasso parameters
LASSO_ALPHA = 0.01
MAX_ITER = 5000

# LightGBM parameters
LGBM_ESTIMATORS = 50
LGBM_MAX_DEPTH = 10
LGBM_SUBSAMPLE = 0.8
LGBM_COLSAMPLE_BYTREE = 0.8
LGBM_LEARNING_RATE = 0.01

# XGBoost
XGB_ESTIMATORS = 50
XGB_MAX_DEPTH = 8
XGB_LEARNING_RATE = 0.1
XGB_N_JOBS = -1
XGB_TREE_METHOD = 'hist'
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_VERBOSITY = 0
