import pandas as pd


def add_weather_time_features(df, timestamp_col='timestamp'):
  df['datetime'] = pd.to_datetime(df[timestamp_col], errors='coerce')
  df['datetime_hour'] = df['datetime'].dt.floor('h')
  df['hour_of_day'] = df['datetime_hour'].dt.hour
  df['day_of_year'] = df['datetime_hour'].dt.dayofyear
  df['hour_of_year'] = ((df['day_of_year'] - 1) * 24) + df['hour_of_day']
  df.drop(columns=['day_of_year'], inplace=True)
  return df


def aggregate_weather_hourly(df):
  return (
    df.groupby('datetime_hour')
    .agg({
      'temp': 'mean',
      'windspeed': 'mean',
      'humidity': 'mean',
      'precip': 'sum',
      'pressure': 'mean',
      'dailyprecip': 'first',
      'dailysnow': 'first',
      'fog': 'max',
      'rain': 'max',
      'snow': 'max',
      'conditions': lambda x: x.mode().iloc[0] if not x.mode().empty else
      x.iloc[0]
    })
    .reset_index()
  )


def clean_trace_and_convert(df, cols, val, trace='T', ):
  for col in cols:
    df[col] = df[col].replace(trace, val)
    df[col] = pd.to_numeric(df[col], errors='coerce')
  return df


def add_metric_measurements(df):
  df['dailyprecip'] = pd.to_numeric(df['dailyprecip'], errors='coerce')
  df['dailysnow'] = pd.to_numeric(df['dailysnow'], errors='coerce')
  df['temp_c'] = (df['temp'] - 32) * 5 / 9
  df['windspeed_kph'] = df['windspeed'] * 1.60934
  df['precip_mm'] = df['precip'] * 25.4
  df['pressure_hPa'] = df['pressure'] * 33.8639
  df['precip_daily_mm'] = df['dailyprecip'] * 25.4
  df['daily_snow_mm'] = df['dailysnow'] * 25.4
