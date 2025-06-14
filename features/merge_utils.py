import pandas as pd

def merge_taxi_weather(taxi_df, weather_df):
  return pd.merge(taxi_df, weather_df, how='left', on='hour_of_year',
                  validate='many_to_one')
