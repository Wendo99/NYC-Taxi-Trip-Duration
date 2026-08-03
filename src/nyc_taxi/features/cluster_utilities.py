"""HDBSCAN-based geo clustering (optional; enabled via ENABLE_HDBC)."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors

import nyc_taxi.config.taxi_constants as taxi_constants


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
    seed=42,
    max_assign_dist: float | None = None):
  """Cluster a sample with HDBSCAN, then label the rest by nearest core point.

  Parameters
  ----------
  max_assign_dist
      Optional haversine cutoff, in radians, beyond which a point is left
      unlabelled (-1) instead of inheriting its nearest core point's cluster.
      ``None`` (the default) assigns regardless of distance.
  """
  n = len(coords_deg)
  rng = np.random.default_rng(seed)
  samp_ix = rng.choice(n, min(n, sample_size), replace=False)
  samp = coords_deg[samp_ix]
  coords_rad = np.radians(samp)

  clusterer = HDBSCAN(
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
    # This previously read `clusterer.minimum_spanning_tree_.max()` inside a
    # try/except AttributeError. That attribute belongs to the standalone
    # `hdbscan` package, not sklearn's port, which exposes only labels_,
    # probabilities_ and n_features_in_ — so the except branch fired every
    # time and the cutoff was always inf. The cutoff is now an explicit
    # parameter; the default keeps the behaviour that was actually in effect.
    threshold = np.inf if max_assign_dist is None else max_assign_dist
    mask = (full_lbl == -1) & (dist.ravel() <= threshold)
    full_lbl[mask] = nn_lbl[mask]

  return full_lbl


