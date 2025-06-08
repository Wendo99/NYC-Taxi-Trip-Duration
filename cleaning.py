import pandas as pd


def interpolate_time_series(df, cols, index_col='datetime_hour'):
  df = df.copy().set_index(index_col)
  for col in cols:
    df[col] = df[col].interpolate(method='time')
  return df.reset_index()


def clean_trace_and_convert(df, cols, trace='T', val=0.001):
  df = replace_trace_values(df, cols, trace=trace, val=val)
  df = convert_to_float(df, cols)
  return df


def replace_trace_values(df, cols, trace='T', val=0.001):
  for col in cols:
    df[col] = df[col].replace(trace, val)
  return df


def convert_to_float(df, cols):
  for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
  return df


def remove_invalid_passenger_counts(df, valid_range):
  return df[df['passenger_count'].between(*valid_range)]


def remove_invalid_locations(df, lat_range=(40.47, 41.0),
    lon_range=(-74.3, -73.6)):
  return df[
    df['pickup_latitude'].between(*lat_range) &
    df['dropoff_latitude'].between(*lat_range) &
    df['pickup_longitude'].between(*lon_range) &
    df['dropoff_longitude'].between(*lon_range)
    ]


def remove_unrealistic_durations(df, min_seconds=60, max_seconds=3 * 3600):
  return df[df['trip_duration'].between(min_seconds, max_seconds)]


def remove_suspicious_same_location_trips(df, duration_threshold=300,
    precision=5):
  same_location = (
                      df['pickup_latitude'].round(precision) == df[
                    'dropoff_latitude'].round(precision)
                  ) & (
                      df['pickup_longitude'].round(precision) == df[
                    'dropoff_longitude'].round(precision)
                  )
  return df[~(same_location & (df['trip_duration'] > duration_threshold))]


def flag_outliers(df, lat_range=(40.47, 41.0), lon_range=(-74.3, -73.6),
    valid_passenger_range=(1, 6),
    min_duration=60, max_duration=3 * 3600,
    duration_threshold=300, precision=5):
  df = df.copy()

  # Flagge ungültige Passagieranzahl
  df['flag_invalid_passenger'] = ~df['passenger_count'].between(
      *valid_passenger_range)

  # Flagge GPS außerhalb NYC
  df['flag_invalid_location'] = ~(
      df['pickup_latitude'].between(*lat_range) &
      df['dropoff_latitude'].between(*lat_range) &
      df['pickup_longitude'].between(*lon_range) &
      df['dropoff_longitude'].between(*lon_range)
  )

  # Flagge unrealistische Dauer
  df['flag_unrealistic_duration'] = ~df['trip_duration'].between(min_duration,
                                                                 max_duration)

  # Flagge Pickup = Dropoff bei langer Dauer
  same_location = (
                      df['pickup_latitude'].round(precision) == df[
                    'dropoff_latitude'].round(precision)
                  ) & (
                      df['pickup_longitude'].round(precision) == df[
                    'dropoff_longitude'].round(precision)
                  )
  df['flag_same_location_long'] = same_location & (
      df['trip_duration'] > duration_threshold)

  return df
