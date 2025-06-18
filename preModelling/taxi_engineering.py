import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def add_us_holiday_flag(df, datetime_col='pickup_datetime',
    flag_col='is_holiday'):
  """
  Add a boolean column that is True on US federal holidays.
  """
  df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
  cal = USFederalHolidayCalendar()
  holidays = cal.holidays(
      start=df[datetime_col].min().normalize(),
      end=df[datetime_col].max().normalize()
  )
  df[flag_col] = df[datetime_col].dt.normalize().isin(holidays)
  return df


def add_taxi_time_features(df, col):
  df['pickup_hour'] = df[col].dt.hour
  df['pickup_weekday'] = df[col].dt.dayofweek
  df['pickup_month'] = df[col].dt.month
  df['day_of_year'] = df[col].dt.dayofyear
  df['hour_of_year'] = ((df['day_of_year'] - 1) * 24) + df['pickup_hour']
  return df.drop(columns=['day_of_year'])


def add_trip_duration_features(df):
  df['trip_duration_min'] = df['trip_duration'] / 60
  df['trip_duration_log'] = np.log1p(df['trip_duration'])
  return df


def add_taxi_distance_features(df):
  df['hav_dist_km'] = haversine(df['pickup_latitude'], df['pickup_longitude'],
                                df['dropoff_latitude'],
                                df['dropoff_longitude'])
  return df


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
