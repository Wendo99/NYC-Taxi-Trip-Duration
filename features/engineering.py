import pandas as pd


def timestamp_to_datetime(df, col, new_col):
  df[new_col] = pd.to_datetime(df[col])
  return df


def add_time_features(df, datetime_col):
  df['datetime_hour'] = df[datetime_col].dt.floor('h')
  df['hour_of_day'] = df['datetime_hour'].dt.hour
  df['day_of_year'] = df['datetime_hour'].dt.dayofyear
  df['hour_of_year'] = ((df['day_of_year'] - 1) * 24) + df['hour_of_day']
  df.drop(columns=['day_of_year'], inplace=True)
  return df
