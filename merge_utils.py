import pandas as pd


def merge_taxi_weather(taxi_df, weather_df):
  taxi_df['hour_of_year'] = taxi_df['hour_of_year'].astype(int)
  weather_df['hour_of_year'] = weather_df['hour_of_year'].astype(int)
  assert weather_df['hour_of_year'].is_unique
  return pd.merge(taxi_df, weather_df, how='left', on='hour_of_year',
                  validate='many_to_one')
