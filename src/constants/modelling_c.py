from __future__ import annotations

from scipy.stats import uniform, randint

# global settings # -----------------------------------------------------------

RANDOM_STATE: int = 42
CV_FOLDS: int = 3

# test train set settings ---------------------------------------------------

TEST_SIZE = 0.20

# Ridge ---------------------------------------------------

R_ALPHA = 7.50080237694725

# Random Forest ---------------------------------------------------

RF_N_ESTIMATORS = 142
RF_MIN_SAMPLES_LEAF = 5
RF_MAX_FEATURES = None

# XGBoost ---------------------------------------------------


X_LEARNING_RATE = 0.09599404067363206
X_MAX_DEPTH = 9
X_N_ESTIMATORS = 70

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
    "model__max_depth": [3, 6, 9],
    "model__n_estimators": randint(50, 150),
    "model__learning_rate": uniform(0.01, 0.1),
  },
  "Bayes": {
    "model__alpha_1": uniform(1e-6, 1e-4),
    "model__lambda_1": uniform(1e-6, 1e-4)
  }
}
