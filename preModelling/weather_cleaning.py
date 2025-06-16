import pandas as pd


def get_outliers(df, col):
  q1 = df[col].quantile(0.25)
  q3 = df[col].quantile(0.75)
  iqr = q3 - q1
  # Define outlier bounds
  lower_bound = q1 - 1.5 * iqr
  upper_bound = q3 + 1.5 * iqr
  temp = df.copy()
  # Detect outliers
  return temp[
    (temp[col] < lower_bound) | (
        temp[col] > upper_bound)]


def flag_and_clip(df, col, new_col, lower_threshold, upper_threshold):
  df = df.copy()
  df[new_col] = (df[col] < lower_threshold) | (
      df[col] > upper_threshold)
  df[col] = df[col].clip(lower=lower_threshold, upper=upper_threshold)
  return df


def interpolate_time_series(df, feature, index_col, method):
  df = df.copy().set_index(index_col)
  df[feature] = df[feature].interpolate(method)
  df = df.reset_index()  # Reassign the DataFrame after resetting the index
  return df


def clean_trace_and_convert(df, cols, val, trace='T', ):
  for col in cols:
    df[col] = df[col].replace(trace, val)
    df[col] = pd.to_numeric(df[col], errors='coerce')
  return df


def correct_trace_rain(row):
  if row['precip_mm'] == 0 and (row['rain'] == 1 or row['snow'] == 1):
    return True
  return None


def fahrenheit_to_celsius(df, col, new_col):
  df[new_col] = (df[col] - 32) * 5 / 9
  return df


def miles_to_kilometers(df, col, new_col):
  df[new_col] = df[col] * 1.60934
  return df


def inch_to_millimeters(df, col, new_col):
  df[new_col] = df[col] * 25.4
  return df


def inch_mercury_to_hpa(df, col, new_col):
  df[new_col] = df[col] * 33.8639
  return df


def to_numeric(df, col, new_col):
  df[new_col] = pd.to_numeric(df[col], errors='coerce')
  return df


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
