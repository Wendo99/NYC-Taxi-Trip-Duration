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

NUM_NO_WEATHER = ['passenger_count', 'hav_dist_km_log']

CAT_NO_WEATHER = ['pickup_weekday', 'pickup_month', 'vendor_id',
                  'pickup_hour',
                  'hour_of_year',
                  'hour_of_day', 'temp_code']

BOOL_NO_WEATHER = [
  'store_and_fwd_flag_bin', 'passenger_count_invalid',
  'same_location_long_trip', 'trip_duration_outlier'
]

FEATURE_GROUPS_ALL = {
  'num': NUM_ALL,
  'cat': CAT_ALL,
  'geo_pick': GEO_PICK,
  'geo_drop': GEO_DROP,
  'bool': BOOL_ALL
}

FEATURE_GROUPS_NO_WEATHER = {
  'num': NUM_NO_WEATHER,
  'cat': CAT_NO_WEATHER,
  'geo_pick': GEO_PICK,
  'geo_drop': GEO_DROP,
  'bool': BOOL_NO_WEATHER
}
