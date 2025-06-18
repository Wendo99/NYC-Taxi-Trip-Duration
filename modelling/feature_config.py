_FEATURES = {
  # column_name        logical_group      weather?
  "passenger_count": ("num", False),
  "temp_c": ("num", True),
  "windspeed_kph_sqrt": ("num", True),
  "humidity": ("num", True),
  "pressure_hpa": ("num", True),
  "rain_mm": ("num", True),
  "snow_mm": ("num", True),
  "daily_precip_mm": ("num", True),
  "daily_snow_mm": ("num", True),

  "pickup_weekday": ("cat", False),
  "pickup_month": ("cat", False),
  "vendor_id": ("cat", False),
  "pickup_hour": ("cat", False),
  "hour_of_year": ("cat", False),
  "hour_of_day": ("cat", False),
  "temp_code": ("cat", True),
  "windspeed_code": ("cat", True),
  "humidity_code": ("cat", True),
  "fog_code": ("cat", True),
  "freezing_code": ("cat", True),
  "cloud_code": ("cat", True),
  "hazy_code": ("cat", True),
  "pressure_code": ("cat", True),
  "rain_code": ("cat", True),
  "snow_code": ("cat", True),

  "store_and_fwd_flag_bin": ("bool", False),
  "fog": ("bool", True),
  "rain": ("bool", True),
  "snow": ("bool", True),
  "cloud_missing_flag": ("bool", True),
  'is_holiday': ("bool", False)
}

# --- 2. programmatically build the old lists ------------------
NUM_ALL = [f for f, (g, _) in _FEATURES.items() if g == "num"]
CAT_ALL = [f for f, (g, _) in _FEATURES.items() if g == "cat"]
BOOL_ALL = [f for f, (g, _) in _FEATURES.items() if g == "bool"]

NUM_NO_WEATHER = [f for f, (g, w) in _FEATURES.items() if g == "num" and not w]
CAT_NO_WEATHER = [f for f, (g, w) in _FEATURES.items() if g == "cat" and not w]
BOOL_NO_WEATHER = [f for f, (g, w) in _FEATURES.items() if
                   g == "bool" and not w]

FEATURE_GROUPS_ALL = {
  "num": NUM_ALL,
  "cat": CAT_ALL,
  "bool": BOOL_ALL
}
FEATURE_GROUPS_NO_WEATHER = {
  "num": NUM_NO_WEATHER,
  "cat": CAT_NO_WEATHER,
  "bool": BOOL_NO_WEATHER
}
