from preModelling.data_config import NYC_LAT_MIN, NYC_LAT_MAX, NYC_LON_MIN, \
  NYC_LON_MAX


def flag_invalid_coords(df):
  df = df.copy()
  df['pickup_coord_invalid'] = (
      (df['pickup_latitude'] < NYC_LAT_MIN) | (
      df['pickup_latitude'] > NYC_LAT_MAX) |
      (df['pickup_longitude'] < NYC_LON_MIN) | (df['pickup_longitude'] >
                                                NYC_LON_MAX)
  )
  df['dropoff_coord_invalid'] = (
      (df['dropoff_latitude'] < NYC_LAT_MIN) | (
      df['dropoff_latitude'] > NYC_LAT_MAX) |
      (df['dropoff_longitude'] < NYC_LON_MIN) | (
          df['dropoff_longitude'] > NYC_LON_MAX)
  )
  return df


def flag_geographic_outliers(df):
  """
  Flags geographic anomalies:
  - pickup or dropoff outside NYC bounds
  - pickup and dropoff at (nearly) same location with long duration
  Adds boolean columns:

  - 'pickup_coord_invalid'
  - 'dropoff_coord_invalid'
  - 'same_location_long_trip'
  """
  df = df.copy()

  df['pickup_coord_invalid'] = (
      (df['pickup_latitude'] < NYC_LAT_MIN) | (
      df['pickup_latitude'] > NYC_LAT_MAX) |
      (df['pickup_longitude'] < NYC_LON_MIN) | (
          df['pickup_longitude'] > NYC_LON_MAX)
  )

  df['dropoff_coord_invalid'] = (
      (df['dropoff_latitude'] < NYC_LAT_MIN) | (
      df['dropoff_latitude'] > NYC_LAT_MAX) |
      (df['dropoff_longitude'] < NYC_LON_MIN) | (
          df['dropoff_longitude'] > NYC_LON_MAX)
  )

  df['same_location_long_trip'] = (
      (df['pickup_latitude'].round(5) == df['dropoff_latitude'].round(5)) &
      (df['pickup_longitude'].round(5) == df['dropoff_longitude'].round(5)) &
      (df['trip_duration'] > 300)
  )
  return df


def encode_store_and_fwd_flag(df, col='store_and_fwd_flag',
    new_col='store_and_fwd_flag_bin'):
  df = df.copy()
  mapping = {'N': 0, 'Y': 1}
  df[new_col] = df[col].map(mapping)
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
