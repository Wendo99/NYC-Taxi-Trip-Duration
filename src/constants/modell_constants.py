from __future__ import annotations

from scipy.stats import uniform, randint

# global settings # -----------------------------------------------------------

CV_FOLDS: int = 3

# test train set settings ---------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# general  ---------------------------------------------------
N_JOBS = -1

# Ridge ---------------------------------------------------

R_ALPHA = 1.3110318597055903

# Random Forest ---------------------------------------------------

# RF_N_ESTIMATORS = 142
RF_MIN_SAMPLES_LEAF = 1
RF_MAX_FEATURES = 1.0

# XGBoost ---------------------------------------------------
COLSAMPLE_BYTREE = 0.8003377500251196
GAMMA = 0.8572009075316447
LEARNING_RATE = 0.07508884729488528
MIN_CHILD_WEIGHT = 2
REG_ALPHA = 2.497327922401265
REG_LAMBDA = 0.21233911067827616
SUBSAMPLE = 0.6636424704863906
X_MAX_DEPTH = 8
X_N_ESTIMATORS = 187

# ---------------------------------------------------

param_spaces = {
  "LinearRegression": {

  },
  "Ridge": {
    "model__alpha": uniform(0.01, 20.0),
  },
  "RandomForest": {
    "model__n_estimators": randint(50, 150),
  },
  "XGBoost": {
    "model__max_depth": randint(4, 12),
    "model__n_estimators": randint(100, 300),
    "model__learning_rate": uniform(0.01, 0.10),
    "model__subsample": uniform(0.5, 0.9),
    "model__colsample_bytree": uniform(0.5, 0.9),
    "model__min_child_weight": randint(1, 12),
    "model__gamma": uniform(0, 6),
    "model__reg_alpha": uniform(0, 3),
    "model__reg_lambda": uniform(0, 1),
  },
  "Bayes": {
    "model__alpha_1": uniform(1e-6, 1e-4),
    "model__lambda_1": uniform(1e-6, 1e-4)
  }
}
