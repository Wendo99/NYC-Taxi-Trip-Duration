TARGET = "trip_duration_log"

FEATURES = {
  # --- NUMERIC
  'route_distance_log_km': ("num", False),
  "hav_dist_km_log": ("num", False),
  "temp_c": ("num", True),
  "dropoff_longitude": ("num", False),
  "dropoff_latitude": ("num", False),
  "pickup_latitude": ("num", False),
  "pickup_longitude": ("num", False),
  'daily_precip_mm': ("num", False),
  "rain_mm": ("num", True),
  "passenger_count": ("num", False),
  "pressure_hpa": ("num", True),

  # --- CATEGORICAL
  "rain": ("cat", True),
  "is_rush_am": ("cat", False),
  "is_rush_pm": ("cat", False),
  "is_night": ("cat", False),
  'is_early_morning': ("cat", False),
  "is_holiday": ("cat", False),
  "trip_duration_outlier": ("cat", False),
  "pickup_weekday": ("cat", False),
  "vendor_id": ("cat", False),
  "pickup_month": ("cat", False),
  "temp_code": ("cat", True),
  "hour_of_day": ("cat", False),
  "hour_of_year": ("cat", False),
  "is_jfk_pick": ("cat", False),
  'is_jfk_drop': ("cat", False),

}

NUM_ALL = [f for f, (g, _) in FEATURES.items() if g == "num"]
CAT_ALL = [f for f, (g, _) in FEATURES.items() if g == "cat"]

GEO_PICK = ["pickup_cluster"]
GEO_DROP = ["dropoff_cluster"]


# python
def fill_res_col(features: dict, exclude: set | None = None) -> set:
  """Return a set of feature keys from the provided dictionary, including GEO_PICK and GEO_DROP,
  and excluding the provided features.

  Args:
      features: Dictionary containing feature definitions.
      exclude: An optional set of features to exclude.
  """
  # Start with keys from features.
  result = set(features.keys())
  # Add keys from GEO_PICK and GEO_DROP.
  result.update(GEO_PICK)
  result.update(GEO_DROP)
  # Remove any excluded features.
  if exclude:
    result.difference_update(exclude)
  return result


EXCLUDE_FEATURES = {
  'route_distance_log_km': ("num", False),
  "hav_dist_km_log": ("num", False),
}

# Exclude features using keys from DROPPED_FEATURES.
excluded = set(EXCLUDE_FEATURES.keys())
RES_COL = fill_res_col(FEATURES, exclude=excluded)

DROPPED_FEATURES = {
  "windspeed_kph_sqrt": ("num", True),
  "humidity": ("num", True),
  "snow_mm": ("num", True),
  'daily_snow_mm': ("num", False),
  "hazy_code": ("cat", True),
  "rain_rush_am": ("cat", False),
  "snow_code": ("cat", True),
  "snow_weekend": ("cat", True),
  "passenger_count_invalid": ("cat", False),
  "pickup_longitude_invalid": ("cat", False),
  "humidity_code": ("cat", True),
  "rain_code": ("cat", True),
  "snow": ("cat", True),
  "rain_rush_pm": ("cat", False),
  "freezing_code": ("cat", True),
  "store_and_fwd_flag_bin": ("cat", False),
  "is_group_trip": ("cat", False),
  "fog": ("cat", True),
  "is_weekend": ("cat", False),
  'fog_code': ("cat", True),
  "cloud_code": ("cat", True),
  "conditions": ("cat", True),
  "windspeed_code": ("cat", True),
  "pressure_code": ("cat", True),
  "pickup_latitude_invalid": ("cat", False),
  "dropoff_longitude_invalid": ("cat", False),
  "dropoff_latitude_invalid": ("cat", False),
  "pickup_cluster_hdb": ("num", False),
  "dropoff_cluster_hdb": ("num", False),
  "id": ("cat", False),
  "pickup_datetime": ("cat", False),
  "dropoff_datetime": ("cat", False),
  "datetime": ("cat", False),
  "datetime_hour": ("cat", False),
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
  "hav_dist_km": ("num", False),
  "windspeed_kph": ("num", True),
  "store_and_fwd_flag": ("cat", False),
  "pickup_hour": ("cat", False),
  'route_distance_km': ("num", False),
}
