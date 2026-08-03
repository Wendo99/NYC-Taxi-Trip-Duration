"""Which columns the model sees, and which are deliberately withheld.

``FEATURES`` is the allow-list: name -> (group, flag), where the group decides
how ``build_preprocessor`` treats the column. ``NUM_ALL`` and ``CAT_ALL`` are
derived from it, so adding a feature here is enough to wire it through.

``DROPPED_FEATURES`` is the deny-list, and matters more than it looks. It holds
raw identifiers, the target in its other units, and ``dropoff_datetime`` — from
which ``trip_duration`` is exactly recoverable by subtraction (see
``notebooks/taxi.ipynb``, section 2). Anything in it must never reach a model.

``INVALID_COLS`` are the flags written by ``flag_and_clip``; rows with any of
them set are excluded before training.
"""
TARGET = "trip_duration_log"

FEATURES = {
  # --- NUMERIC
  "hav_dist_km_log": ("num", False),

  # --- CATEGORICAL
  "is_rush_am": ("cat", False),
  "is_rush_pm": ("cat", False),
  "is_night": ("cat", False),
  'is_early_morning': ("cat", False),
  "is_holiday": ("cat", False),
  "pickup_weekday": ("cat", False),
  "vendor_id": ("cat", False),

  'is_jfk_drop': ("cat", False),
  "pickup_month": ("cat", False),
  "passenger_count": ("num", False),
  "temp_code": ("cat", True),

  "temp_c": ("num", True),

  "pressure_code": ("cat", True),
  "hour_of_day": ("cat", False),
  "rain": ("cat", True),

  "dropoff_longitude": ("num", False),
  "dropoff_latitude": ("num", False),
  "pickup_longitude": ("num", False),
  "pickup_latitude": ("num", False),

  'daily_precip_mm': ("num", False),
  'pickup_cluster': ("geo", False),
  'dropoff_cluster': ("geo", False),
  "snow_weekend": ("cat", True),
  "humidity": ("num", True),

}

DROPPED_FEATURES = {
  "trip_duration_outlier": ("cat", False),
  "pickup_longitude_invalid": ("cat", False),
  "passenger_count_invalid": ("cat", False),
  "pickup_latitude_invalid": ("cat", False),
  "dropoff_longitude_invalid": ("cat", False),
  "dropoff_latitude_invalid": ("cat", False),
  "rain_mm": ("num", True),
  "pressure_hpa": ("num", True),
  "cloud_code": ("cat", True),
  'daily_snow_mm': ("num", False),
  "snow_code": ("cat", True),
  "windspeed_kph_sqrt": ("num", True),
  "snow_mm": ("num", True),
  "humidity_code": ("cat", True),
  "windspeed_code": ("cat", True),
  "snow": ("cat", True),
  "rain_code": ("cat", True),
  "hour_of_year": ("cat", False),
  "hazy_code": ("cat", True),
  "rain_rush_am": ("cat", False),
  "is_jfk_pick": ("cat", False),
  "conditions": ("cat", True),
  "pickup_datetime": ("cat", False),
  "dropoff_datetime": ("cat", False),
  'is_laguardia_drop': ("cat", False),
  'is_laguardia_pick': ("cat", False),
  "freezing_code": ("cat", True),
  "is_group_trip": ("cat", False),
  "is_weekend": ("cat", False),
  "datetime": ("cat", False),
  "datetime_hour": ("cat", False),
  "pickup_hour": ("cat", False),

  "rain_rush_pm": ("cat", False),
  "store_and_fwd_flag_bin": ("cat", False),
  "fog": ("cat", True),
  'fog_code': ("cat", True),

  "temp_class": ("cat", True),
  "windspeed_class": ("cat", True),
  "humidity_class": ("cat", True),
  "pressure_class": ("cat", True),
  "rain_class": ("cat", True),
  "snow_class": ("cat", True),
  "fog_class": ("cat", True),
  "freezing_class": ("cat", True),
  "cloud_class": ("cat", True),
  "hazy_class": ("cat", True),
  "trip_duration": ("num", False),
  "trip_duration_min": ("num", False),
  "windspeed_kph": ("num", True),
  "store_and_fwd_flag": ("cat", False),
  "id": ("cat", False),
  "hav_dist_km": ("num", False),
  'route_distance_km': ("num", False),
  'route_distance_log_km': ("num", False)
}

INVALID_COLS = [
  "pickup_longitude_invalid",
  "pickup_latitude_invalid",
  "dropoff_longitude_invalid",
  "dropoff_latitude_invalid",
  "passenger_count_invalid",
  "trip_duration_outlier"
]

# One-hot encoded by build_preprocessor; the HDBSCAN variants
# (pickup_cluster_hdb / dropoff_cluster_hdb) had equivalents here but nothing
# referenced them — ENABLE_HDBC is off and MiniBatchKMeans supplies the
# clusters the model actually uses.
GEO_PICK = ["pickup_cluster"]
GEO_DROP = ["dropoff_cluster"]

NUM_ALL = [f for f, (g, _) in FEATURES.items() if g == "num"]
CAT_ALL = [f for f, (g, _) in FEATURES.items() if g == "cat"]
