import numpy as np


def haversine(lat1, lon1, lat2, lon2):
  r = 6378.135  # Earth's radius in km

  # Convert latitude and longitude to radians
  lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

  # Calculate the difference between the two coordinates
  dlat = lat2 - lat1
  dlon = lon2 - lon1

  # Apply the haversine formula
  a = (np.sin(dlat / 2)) ** 2 + (np.cos(lat1) * np.cos(lat2)) * (
    np.sin(dlon / 2)) ** 2
  c = 2 * r * np.arcsin(np.sqrt(a))

  # Return the distance
  return c


def vincenty(lat1, lon1, lat2, lon2):
  # Convert latitude and longitude to radians
  lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

  # Calculate the difference between the two coordinates
  dlat = lat2 - lat1
  dlon = lon2 - lon1

  # Apply the Vincenty formula
  a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
      dlon / 2) ** 2
  c = 2 * np.atan2(np.sqrt(a), np.sqrt(1 - a))

  # Calculate the ellipsoid parameters
  f = 1 / 298.257223563  # flattening of the Earth's ellipsoid
  b = (1 - f) * 6378.135  # semi-minor axis of the Earth's ellipsoid

  # Return the distance
  return c * b
