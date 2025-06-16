NUM_ALL = ['passenger_count', 'hav_dist_km_log', 'temp_c',
           'windspeed_kph_sqrt', 'humidity',
           'pressure_hpa', 'rain_mm', 'snow_mm',
           'daily_precip_mm', 'daily_snow_mm']

CAT_ALL = ['pickup_weekday', 'pickup_month', 'vendor_id',
           'pickup_hour',
           'hour_of_year',
           'hour_of_day', 'temp_code',
           'windspeed_code', 'humidity_code', 'fog_code',
           'freezing_code',
           'cloud_code',
           'hazy_code',
           'pressure_code', 'rain_code', 'snow_code']

GEO_PICK = ['pickup_longitude', 'pickup_latitude']

GEO_DROP = ['dropoff_longitude', 'dropoff_latitude']

BOOL_ALL = [
  'store_and_fwd_flag_bin', 'fog', 'rain', 'snow', 'cloud_missing_flag',
  'windspeed_outliers', 'daily_snow_outliers', 'passenger_count_invalid',
  'same_location_long_trip', 'trip_duration_outlier'
]

WEATHER_ALL = [
  'temp_c', 'humidity', 'pressure_hpa', 'windspeed_kph_sqrt',
  'rain_mm', 'snow_mm', 'daily_precip_mm', 'daily_snow_mm',
  'fog', 'rain', 'snow', 'cloud_code', 'rain_code', 'snow_code',
  'temp_code', 'windspeed_code', 'pressure_code', 'hazy_code',
  'freezing_code', 'humidity_code', 'fog_code', 'cloud_missing_flag',
  'windspeed_outliers', 'daily_snow_outliers'
]

attr_to_drop = [
  'pickup_datetime', 'dropoff_datetime', 'datetime_hour', 'windspeed_kph',
  'hav_dist_km']

num_select_lin_attr = ['passenger_count', 'hav_dist_km_log', 'temp_c',
                       'windspeed_kph_sqrt', 'humidity',
                       'pressure_hpa']

cat_select_lin_attr = ['pickup_weekday', 'pickup_month', 'vendor_id',
                       'pickup_hour',
                       'hour_of_year',
                       'hour_of_day', 'temp_code',
                       'windspeed_code', 'humidity_code', 'fog_code',
                       'freezing_code',
                       'cloud_code',
                       'hazy_code',
                       'pressure_code', 'rain_code', 'snow_code']

bool_select_lin_attr = ['store_and_fwd_flag_bin', 'fog', 'rain', 'snow',
                        'cloud_missing_flag']

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
DT_MAX_DEPTH = 8
CPP_ALPHA = 0.001

# RandomForest parameters
RF_ESTIMATORS = 10
MAX_RF_DEPTH = 10
MIN_RF_SAMPLES_LEAF = 5
MAX_RF_FEATURES = 'sqrt'

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

# XGBoost
XGB_ESTIMATORS = 50
XGB_MAX_DEPTH = 8
XGB_LEARNING_RATE = 0.1
XGB_N_JOBS = -1
XGB_TREE_METHOD = 'hist'
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
