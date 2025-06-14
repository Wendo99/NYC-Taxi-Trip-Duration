import pandas as pd


def timestamp_to_datetime(df, col, new_col):
  df[new_col] = pd.to_datetime(df[col])
  return df


def add_time_features(df, datetime_col):
  df['datetime_hour'] = df[datetime_col].dt.floor('h')
  return add_time_features_datetime_hour(df)


def add_time_features_datetime_hour(df):
  """
  Adds time-based features derived from 'datetime_hour':
  - hour_of_day: hour [0–23]
  - hour_of_year: absolute hour count since year start

  Parameters:
      df (pd.DataFrame): DataFrame with 'datetime_hour' column

  Returns:
      pd.DataFrame: DataFrame with new time features
  """
  df = df.copy()
  df['hour_of_day'] = df['datetime_hour'].dt.hour
  df['day_of_year'] = df['datetime_hour'].dt.dayofyear
  df['hour_of_year'] = ((df['day_of_year'] - 1) * 24) + df['hour_of_day']
  df.drop(columns=['day_of_year'], inplace=True)
  return df
