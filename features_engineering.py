import numpy as np
import pandas as pd


def add_weather_features(df):
  add_correct_measurements_weather_features(df)
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


def add_correct_measurements_weather_features(df):
  df['temp_c'] = (df['temp'] - 32) * 5 / 9
  df['windspeed_kph'] = df['windspeed'] * 1.60934
  df['precip_mm'] = df['precip'] * 25.4
  df['pressure_hPa'] = df['pressure'] * 33.8639
  df['precip_daily_mm'] = df['dailyprecip'] * 25.4
  df['daily_snow_mm'] = df['dailysnow'] * 25.4


def classify_rain(df):
  rain_mapping = {
    'no_rain': 0,
    'light_rain': 1,
    'moderate_rain': 2,
    'heavy_rain': 3,
    'very_heavy_rain': 4,
    'extreme_rain': 5
  }
  df['rain_class'] = df['precip_mm'].apply(
      classify_rain_label)
  df['rain_code'] = df['rain_class'].map(
      rain_mapping)


def classify_rain_label(x):
  if x >= 30:
    return 'extreme_rain'
  elif x >= 15:
    return 'very_heavy_rain'
  elif x >= 7.5:
    return 'heavy_rain'
  elif x >= 2.5:
    return 'moderate_rain'
  elif x > 0:
    return 'light_rain'
  else:
    return 'no_rain'


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
    return "unknown"


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


def classify_freezing(df):
  freezing_mapping = {
    "none": 0,
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
    return "none"


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
  return None


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
  if t < -5:
    return 'very_cold'
  elif t < 5:
    return "cold"
  elif t < 15:
    return "cool"
  elif t < 20:
    return "mild"
  elif t < 25:
    return "warm"
  else:
    return "hot"


def classify_windspeed(df):
  windspeed_mapping = {
    "calm": 0,
    'light_air': 1,
    'light_breeze': 2,
    'moderate_breeze': 3,
    'strong_breeze': 4,
    "stormy": 5
  }
  df['windspeed_class'] = df['windspeed_kph'].apply(
      classify_wind_label)
  df['windspeed_code'] = df['windspeed_class'].map(
      windspeed_mapping)


def classify_wind_label(speed):
  if speed < 1:
    return "calm"
  elif speed < 12:
    return 'light_air'
  elif speed < 29:
    return 'light_breeze'
  elif speed < 50:
    return 'moderate_breeze'
  elif speed < 75:
    return 'strong_breeze'
  else:
    return "stormy"


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
  else:
    return 'very_wet'


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
  if p < 980:
    return 'very_low'
  elif p < 1000:
    return "low"
  elif p < 1020:
    return "normal"
  elif p < 1030:
    return "high"
  else:
    return 'very_high'


def haversine(lat1, lon1, lat2, lon2):
  r = 6378.135  # Earth's radius in km

  # Convert latitude and longitude to radians
  lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

  # Calculate the difference between the two coordinates
  dlat = lat2 - lat1
  dlon = lon2 - lon1

  # Apply the haversine formula
  a = (np.sin(dlat / 2)) ** 2 + (np.cos(lat1) * np.cos(lat2)) * (
    np.sin(dlon / 2)) ** 2
  c = 2 * r * np.arcsin(np.sqrt(a))

  # Return the distance
  return c


def vincenty(lat1, lon1, lat2, lon2):
  # Convert latitude and longitude to radians
  lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

  # Calculate the difference between the two coordinates
  dlat = lat2 - lat1
  dlon = lon2 - lon1

  # Apply the Vincenty formula
  a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
      dlon / 2) ** 2
  c = 2 * np.atan2(np.sqrt(a), np.sqrt(1 - a))

  # Calculate the ellipsoid parameters
  f = 1 / 298.257223563  # flattening of the Earth's ellipsoid
  b = (1 - f) * 6378.135  # semi-minor axis of the Earth's ellipsoid

  # Return the distance
  return c * b


def add_weather_time_features(df, timestamp_col='timestamp'):
  df['datetime'] = pd.to_datetime(df[timestamp_col], errors='coerce')
  df['datetime_hour'] = df['datetime'].dt.floor('h')
  df['hour_of_day'] = df['datetime_hour'].dt.hour
  df['day_of_year'] = df['datetime_hour'].dt.dayofyear
  df['hour_of_year'] = ((df['day_of_year'] - 1) * 24) + df['hour_of_day']
  df.drop(columns=['day_of_year'], inplace=True)
  return df


def interpolate_columns(df, cols, time_index='datetime_hour'):
  df = df.set_index(time_index)
  for col in cols:
    df[col] = df[col].interpolate(method='time')
  return df.reset_index()


def add_taxi_time_features(df, col='pickup_datetime'):
  df[col] = pd.to_datetime(df[col])
  df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
  df['pickup_hour'] = df[col].dt.hour
  df['pickup_weekday'] = df[col].dt.dayofweek.map({
    0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
  })
  df['pickup_month'] = df[col].dt.month
  df['day_of_year'] = df[col].dt.dayofyear
  df['hour_of_year'] = ((df['day_of_year'] - 1) * 24) + df['pickup_hour']
  return df.drop(columns=['day_of_year'])


def add_taxi_distance_features(df):
  df['hav_dist_km'] = haversine(df['pickup_latitude'], df['pickup_longitude'],
                                df['dropoff_latitude'],
                                df['dropoff_longitude'])
  return df

def add_trip_duration_features(df):
  df['trip_duration_min'] = df['trip_duration'] / 60
  df['trip_duration_log'] = np.log1p(df['trip_duration'])
  return df

def add_same_location_flag(df, precision=5):
  df['is_same_location'] = (
                               df['pickup_latitude'].round(precision) == df[
                             'dropoff_latitude'].round(precision)
                           ) & (
                               df['pickup_longitude'].round(precision) == df[
                             'dropoff_longitude'].round(precision)
                           )
  return df