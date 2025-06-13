import pandas as pd





def flag_and_clip_precipitation(df: pd.DataFrame, col='precip_mm',
    clip_threshold_low=0.0, clip_threshold_high=20.0):
  """
  Flags and clips extreme precipitation values in the DataFrame.

  Parameters:
      df (pd.DataFrame): The input DataFrame.
      col (str): The name of the precipitation column.
      clip_threshold_low (float): Minimum allowed precipitation in mm.
      clip_threshold_high (float): Maximum allowed precipitation in mm.

  Returns:
      pd.DataFrame: DataFrame with new boolean column 'precip_is_outlier' and clipped values.
  """
  df = df.copy()
  df['precip_is_outlier'] = (df[col] < clip_threshold_low) | (
      df[col] > clip_threshold_high)
  df[col] = df[col].clip(lower=clip_threshold_low, upper=clip_threshold_high)
  return df


def flag_and_clip_pressure(df: pd.DataFrame, col: str = 'pressure_hPa',
    clip_threshold_low: float = 980.0,
    clip_threshold_high: float = 1040.0) -> pd.DataFrame:
  """
  Flags and clips extreme pressure values in the DataFrame.

  Parameters:
      df (pd.DataFrame): The input DataFrame.
      col (str): The name of the pressure column.
      clip_threshold_low (float): Minimum allowed pressure in hPa.
      clip_threshold_high (float): Maximum allowed pressure in hPa.

  Returns:
      pd.DataFrame: DataFrame with new boolean column 'pressure_is_outlier' and clipped values.
  """
  df = df.copy()
  df['pressure_is_outlier'] = (df[col] < clip_threshold_low) | (
      df[col] > clip_threshold_high)
  df[col] = df[col].clip(lower=clip_threshold_low, upper=clip_threshold_high)
  return df


def flag_and_clip_daily_precip(df: pd.DataFrame, col='precip_daily_mm',
    clip_threshold_low=0.0, clip_threshold_high=80.0):
  """
  Flags and clips extreme daily precipitation values in the DataFrame.

  Parameters:
      df (pd.DataFrame): The input DataFrame.
      col (str): The name of the daily precipitation column.
      clip_threshold_low (float): Minimum allowed precipitation in mm.
      clip_threshold_high (float): Maximum allowed precipitation in mm.

  Returns:
      pd.DataFrame: DataFrame with new boolean column 'precip_daily_is_outlier' and clipped values.
  """
  df = df.copy()
  df['precip_daily_is_outlier'] = (df[col] < clip_threshold_low) | (
      df[col] > clip_threshold_high)
  df[col] = df[col].clip(lower=clip_threshold_low, upper=clip_threshold_high)
  return df


def flag_and_clip_daily_snow(df: pd.DataFrame, col='daily_snow_mm',
    clip_threshold=50.0):
  """
  Flags and clips extreme daily snowfall values in the DataFrame.

  Parameters:
      df (pd.DataFrame): The input DataFrame.
      col (str): The name of the daily snow column.
      clip_threshold (float): Maximum allowed snow depth in mm.

  Returns:
      pd.DataFrame: DataFrame with new boolean column 'daily_snow_is_outlier' and clipped values.
  """
  df = df.copy()
  df['daily_snow_is_outlier'] = df[col] > clip_threshold
  df[col] = df[col].clip(upper=clip_threshold)
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
