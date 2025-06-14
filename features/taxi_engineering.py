import numpy as np

from features.distances import haversine


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
