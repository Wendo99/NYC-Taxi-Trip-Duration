import numpy as np

from features.distances import haversine


# Weather

def add_classified_weather_features(df):
  classify_rain(df)
  classify_snow(df)
  classify_clouds(df)
  classify_haze(df)
  classify_freezing(df)
  classify_fog(df)
  classify_temp(df)
  classify_windspeed(df)
  classify_humidity(df)
  classify_pressure(df)
  return df


# rain

def classify_rain_label(x):
  if abs(x - 0.0) < 1e-9:
    return 'no_rain'
  elif x < 0.25:
    return 'trace_rain'
  elif x < 2.5:
    return 'light_rain'
  elif x < 10.0:
    return 'moderate_rain'
  elif x >= 10:
    return 'heavy_rain'
  elif x >= 50:
    return 'very_heavy_rain'
  return None


def classify_rain(df):
  """
  Classifies and codes hourly precipitation levels.

  Adds two columns:
  - rain_class: label (e.g. 'light_rain')
  - rain_code: ordinal code (e.g. 1)
  """
  rain_mapping = {
    'no_rain': 0,
    'trace_rain': 1,
    'light_rain': 2,
    'moderate_rain': 3,
    'heavy_rain': 4,
    'very_heavy_rain': 5
  }

  df['rain_class'] = df['precip_mm'].apply(classify_rain_label)
  df['rain_code'] = df['rain_class'].map(rain_mapping)


# snow

def classify_snow(df):
  snow_mapping = {
    'no_snow': 0,
    'light_snow': 1,
    "snow": 2,
    'heavy_snow': 3
  }
  df['snow_class'] = df['conditions'].apply(
      classify_snow_label)
  df['snow_code'] = df['snow_class'].map(
      snow_mapping)


def classify_snow_label(x):
  if x == "Light Snow":
    return 'light_snow'
  elif x == "Snow":
    return "snow"
  elif x == "Heavy Snow":
    return 'heavy_snow'
  else:
    return 'no_snow'


# clouds

def classify_clouds(df):
  cloud_mapping = {
    "unknown": 0,
    "clear": 1,
    'scattered_clouds': 2,
    'partly_cloudy': 3,
    'mostly_cloudy': 4,
    "overcast": 5
  }
  df['cloud_class'] = df['conditions'].apply(
      classify_cloud_label)
  df['cloud_code'] = df['cloud_class'].map(
      cloud_mapping)


def classify_cloud_label(x):
  if x == "Clear":
    return "clear"  # 0–10%
  elif x == "Scattered Clouds":
    return 'scattered_clouds'  # ~25–50%
  elif x == "Partly Cloudy":
    return 'partly_cloudy'  # ~20–60%
  elif x == "Mostly Cloudy":
    return 'mostly_cloudy'  # ~60–90%
  elif x == "Overcast":
    return "overcast"  # >90%
  else:
    return 'unknown'


# haze

def classify_haze(df):
  haze_mapping = {
    'no_haze': 0,
    "haze": 1
  }
  df['hazy_class'] = df['conditions'].apply(
      classify_haze_label)
  df['hazy_code'] = df['hazy_class'].map(
      haze_mapping)


def classify_haze_label(x):
  return "haze" if x == "Haze" else 'no_haze'


# freezing

def classify_freezing(df):
  freezing_mapping = {
    "unknown": 0,
    'light_freezing_rain': 1,
    'light_freezing_fog': 2
  }
  df['freezing_class'] = df['conditions'].apply(
      classify_freezing_label)
  df['freezing_code'] = df['freezing_class'].map(
      freezing_mapping)


def classify_freezing_label(x):
  if x == "Light Freezing Fog":
    return 'light_freezing_fog'
  elif x == "Light Freezing Rain":
    return 'light_freezing_rain'
  else:
    return 'unknown'


# fog

def classify_fog(df):
  fog_mapping = {
    "fog": 1,
    "no_fog": 0
  }
  df['fog_class'] = df['fog'].apply(
      classify_fog_label)
  df['fog_code'] = df['fog_class'].map(fog_mapping)


def classify_fog_label(x):
  if x == 1:
    return "fog"
  elif x == 0:
    return 'no_fog'
  return 'unknown'


# temp

def classify_temp(df):
  temp_mapping = {
    'very_cold': 0,
    "cold": 1,
    "cool": 2,
    "mild": 3,
    "warm": 4,
    "hot": 5
  }
  df['temp_class'] = df['temp_c'].apply(
      classify_temp_label)
  df['temp_code'] = df['temp_class'].map(
      temp_mapping)


def classify_temp_label(t):
  if t <= -5:
    return 'very_cold'
  elif t <= 5:
    return "cold"
  elif t <= 15:
    return "cool"
  elif t <= 20:
    return "mild"
  elif t <= 25:
    return "warm"
  elif t > 25:
    return "hot"
  else:
    return 'unknown'


# windspeed
def classify_windspeed(df):
  windspeed_mapping = {
    "calm": 0,
    'light_air': 1,
    'light_breeze': 2,
    'light_wind': 3,
    'moderate_wind': 4,
    "fresh_wind": 5,
    "strong_wind": 6,
    "stiff_wind": 7,
    "stormy_wind": 8,
    "storm": 9,
    "heavy_storm": 10,
    "hurricane_like_storm": 11,
    "hurricane": 12
  }
  df['windspeed_class'] = df['windspeed_kph'].apply(
      classify_wind_label)
  df['windspeed_code'] = df['windspeed_class'].map(
      windspeed_mapping)


def classify_wind_label(speed):
  if abs(speed - 0.0) < 1e-9:
    return "calm"
  elif speed < 5:
    return 'light_air'
  elif speed < 10:
    return 'light_breeze'
  elif speed < 20:
    return 'light_wind'
  elif speed < 30:
    return 'moderate_wind'
  elif speed < 40:
    return 'fresh_wind'
  elif speed < 50:
    return 'strong_wind'
  elif speed < 65:
    return 'stiff_wind'
  elif speed < 75:
    return 'stormy_wind'
  elif speed < 90:
    return 'storm'
  elif speed < 105:
    return 'heavy_storm'
  elif speed < 120:
    return 'hurricane_like_storm'
  elif speed >= 120:
    return 'hurricane'
  else:
    return 'unknown'


def classify_humidity(df):
  humidity_mapping = {
    'very_dry': 0,
    'dry': 1,
    'normal': 2,
    'wet': 3,
    'very_wet': 4
  }
  df['humidity_class'] = df['humidity'].apply(
      classify_humidity_label)

  df['humidity_code'] = df['humidity_class'].map(
      humidity_mapping)


def classify_humidity_label(h):
  if h <= 30:
    return 'very_dry'
  elif h <= 50:
    return 'dry'
  elif h <= 70:
    return 'normal'
  elif h <= 85:
    return 'wet'
  elif h > 85:
    return 'very_wet'
  else:
    return 'unknown'


def classify_pressure(df):
  pressure_mapping = {
    'very_low': 0,
    "low": 1,
    "normal": 2,
    "high": 3,
    'very_high': 4
  }
  df['pressure_class'] = df['pressure_hPa'].apply(
      classify_pressure_label)
  df['pressure_code'] = df['pressure_class'].map(
      pressure_mapping)


def classify_pressure_label(p):
  if p <= 980:
    return 'very_low'
  elif p <= 1000:
    return "low"
  elif p <= 1020:
    return "normal"
  elif p <= 1030:
    return "high"
  elif p > 1030:
    return "very_high"
  else:
    return 'unknown'


def add_same_location_flag(df, precision=5):
  df['is_same_location'] = (
                               df['pickup_latitude'].round(precision) == df[
                             'dropoff_latitude'].round(precision)
                           ) & (
                               df['pickup_longitude'].round(precision) == df[
                             'dropoff_longitude'].round(precision)
                           )
  return df
