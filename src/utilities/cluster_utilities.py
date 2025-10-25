import numpy as np
import pandas as pd
from sklearn.cluster._hdbscan import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

import constants.taxi_constants as taxi_constants


def get_geo_mask(df):
  mask = (
      (df["pickup_latitude"] > taxi_constants.GeoBounds.min_lat) &
      (df["pickup_latitude"] < taxi_constants.GeoBounds.max_lat) &
      (df["pickup_longitude"] > taxi_constants.GeoBounds.min_lon) &
      (df["pickup_longitude"] < taxi_constants.GeoBounds.max_lon) &
      (df["dropoff_latitude"] > taxi_constants.GeoBounds.min_lat) &
      (df["dropoff_latitude"] < taxi_constants.GeoBounds.max_lat) &
      (df["dropoff_longitude"] > taxi_constants.GeoBounds.min_lon) &
      (df["dropoff_longitude"] < taxi_constants.GeoBounds.max_lon)
  )
  return mask


def add_hdbc_clusters(df, cluster_type, coord, min_cluster_size, min_samples):
  cluster_labels = run_hdbscan_sample(coords_deg=coord, min_cluster_size=min_cluster_size,min_samples=min_samples)
  df[cluster_type + "_cluster_hdb"] = -1
  df.loc[get_geo_mask(df), cluster_type + "_cluster_hdb"] = cluster_labels
  return df


def run_hdbscan_sample(coords_deg,
    min_cluster_size=150,
    min_samples=None,
    sample_size=100_000,
    seed=42):
  n = len(coords_deg)
  rng = np.random.default_rng(seed)
  samp_ix = rng.choice(n, min(n, sample_size), replace=False)
  samp = coords_deg[samp_ix]
  coords_rad = np.radians(samp)

  clusterer = hdbscan.HDBSCAN(
      min_cluster_size=min_cluster_size,
      min_samples=min_samples or min_cluster_size // 10,
      metric="haversine",
      cluster_selection_method="leaf",
  )
  clusterer.fit(coords_rad)

  full_lbl = np.full(n, -1, dtype=int)
  full_lbl[samp_ix] = clusterer.labels_

  core_mask = clusterer.labels_ >= 0
  if core_mask.any():
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(coords_rad[core_mask])
    full_coords_rad = np.radians(coords_deg)
    dist, idx = nbrs.kneighbors(full_coords_rad, return_distance=True)
    nn_lbl = clusterer.labels_[core_mask][idx.ravel()]
    try:
      threshold = clusterer.minimum_spanning_tree_.max()
    except AttributeError:
      threshold = np.inf
    mask = (full_lbl == -1) & (dist.ravel() <= threshold)
    full_lbl[mask] = nn_lbl[mask]

  return full_lbl


def optimize_cluster_params(df, cluster_type='pickup', param_grid=None,
    sample_size=10_000,
    seed=42,
    min_silhouette_threshold=0.2):
  """
  Optimizes clustering parameters using silhouette score.
  Applies a minimum silhouette score threshold to filter out poor results.

  Args:
      df (pd.DataFrame): DataFrame with coordinates.
      cluster_type (str): 'pickup' or 'dropoff'.
      param_grid (list): List of dicts with parameters.
      sample_size (int): Sample size for evaluation.
      seed (int): Random seed.
      min_silhouette_threshold (float): Minimum acceptable silhouette score.

  Returns:
      dict: Best parameter combination and a DataFrame summary.
  """
  mask = get_geo_mask(df)

  if cluster_type == "pickup":
    coords = df.loc[mask, ["pickup_latitude", "pickup_longitude"]].to_numpy()
  elif cluster_type == "dropoff":
    coords = df.loc[mask, ["dropoff_latitude", "dropoff_longitude"]].to_numpy()

  if param_grid is None:
    # Expanded grid for NYC cab pickup coordinates
    param_grid = [{
      'min_cluster_size': size,
      'min_samples': max(2, size // 10)
    } for size in
      [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900
       ]]

  df_sample = df.sample(n=sample_size, random_state=seed)
  coords = df_sample[coords].to_numpy()

  results = []
  best_score = -1
  best_params = None

  for params in param_grid:
    labels = run_hdbscan_sample(coords,
                                min_cluster_size=params['min_cluster_size'],
                                min_samples=params['min_samples'],
                                sample_size=sample_size,
                                seed=seed)
    mask = labels != -1
    unique_labels = np.unique(labels[mask])
    if len(unique_labels) <= 1:
      score = -1
    else:
      try:
        score = silhouette_score(coords[mask], labels[mask], random_state=42)
        if score < min_silhouette_threshold:
          score = -1
      except Exception:
        score = -1
    results.append({
      'min_cluster_size': params['min_cluster_size'],
      'min_samples': params['min_samples'],
      'silhouette_score': score,
      'n_clusters': len(unique_labels)
    })
    if score > best_score:
      best_score = score
      best_params = params.copy()

  results_df = pd.DataFrame(results)
  return {'best_params': best_params, 'results': results_df}
