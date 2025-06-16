import pandas as pd
from sklearn.model_selection import train_test_split


def merge_taxi_weather(taxi_df, weather_df):
  return pd.merge(taxi_df, weather_df, how='left', on='hour_of_year',
                  validate='many_to_one')


def split_train_test(df, test_size, random_state, stratify_col=None):
  """
  Split a DataFrame into training and test sets.

  Parameters:
      df (DataFrame): The input DataFrame.
      test_size (float): Proportion for test split.
      random_state (int): Random seed.
      stratify_col (str, optional): Column name to stratify on.

  Returns:
      DataFrame, DataFrame: train_set, test_set
  """
  return train_test_split(
      df, test_size=test_size, random_state=random_state,
      stratify=df[stratify_col] if stratify_col else None
  )


def timestamp_to_datetime(df, col, new_col):
  df[new_col] = pd.to_datetime(df[col])
  return df


def add_time_features(df, datetime_col):
  df['datetime_hour'] = df[datetime_col].dt.floor('h')
  return add_time_features_datetime_hour(df)


def add_time_features_datetime_hour(df):
  """
  Adds time-based preModelling derived from 'datetime_hour':
  - hour_of_day: hour [0–23]
  - hour_of_year: absolute hour count since year start

  Parameters:
      df (pd.DataFrame): DataFrame with 'datetime_hour' column

  Returns:
      pd.DataFrame: DataFrame with new time preModelling
  """
  df = df.copy()
  df['hour_of_day'] = df['datetime_hour'].dt.hour
  df['day_of_year'] = df['datetime_hour'].dt.dayofyear
  df['hour_of_year'] = ((df['day_of_year'] - 1) * 24) + df['hour_of_day']
  df.drop(columns=['day_of_year'], inplace=True)
  return df
