"""Raw-data acquisition: download from Kaggle, unpack, cache as pickle.

Every loader follows the same shape — return the pickle cache if it exists,
otherwise download, extract, read the CSV and write the cache — so that shape
lives once in :func:`_load_with_cache`.

Kaggle credentials are never read from this repository. ``api.authenticate()``
picks them up from ``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME`` /
``KAGGLE_KEY`` environment variables.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Callable
from zipfile import ZipFile

import pandas as pd
from kaggle import KaggleApi

from nyc_taxi.config import path_file_constants as paths
from nyc_taxi.frames import read_frame
from nyc_taxi.pipelines.merge_pipeline import build_merged_dataset

log = logging.getLogger(__name__)

_AUTH_HINT = (
  "Kaggle API authentication failed. Ensure that ~/.kaggle/kaggle.json "
  "exists and is properly configured, or that KAGGLE_USERNAME and "
  "KAGGLE_KEY are set."
)


def _authenticated_api() -> KaggleApi:
  """Return an authenticated Kaggle client, or raise with a usable hint."""
  api = KaggleApi()
  try:
    api.authenticate()
  except Exception as exc:
    raise RuntimeError(_AUTH_HINT) from exc
  return api


def download_kaggle_competition(api, competition_name, path):
  api.competition_download_files(competition_name, path=path)


def download_kaggle_dataset(api, dataset_slug, path):
  api.dataset_download_files(dataset_slug, path=str(path), unzip=False)


def extract_inner_zips(zip_path, data_dir, required_inner_zips):
  """Unpack the competition archive if any expected inner zip is missing."""
  existing_inner_zips = {z.name for z in data_dir.glob("*.zip")}
  if required_inner_zips - existing_inner_zips:
    with ZipFile(zip_path, "r") as outer_zip:
      outer_zip.extractall(data_dir)


def extract_csv_from_zip(zip_path: Path, csv_name: str, target_dir: Path):
  """Extract a single named CSV, tolerating a nested path inside the archive."""
  target_dir.mkdir(parents=True, exist_ok=True)

  with ZipFile(zip_path) as zf:
    names = zf.namelist()

    if csv_name in names:
      member = csv_name
    else:
      matches = [n for n in names if Path(n).name == csv_name]
      if not matches:
        raise KeyError(
            f"{csv_name!r} not found in archive. Contains: {names[:5]} …")
      member = matches[0]

    zf.extract(member, path=target_dir)
    (target_dir / member).rename(target_dir / csv_name)


def extract_csv_from_inner_zips(data_dir, extracted_dir):
  """Extract every CSV from every zip in *data_dir*, skipping existing files."""
  extracted_dir.mkdir(parents=True, exist_ok=True)
  for inner_zip in data_dir.glob("*.zip"):
    with ZipFile(inner_zip, "r") as zip_ref:
      for member in zip_ref.namelist():
        if member.endswith(".csv"):
          target_file = extracted_dir / Path(member).name
          if not target_file.is_file():
            with zip_ref.open(member) as src, open(target_file, "wb") as dst:
              dst.write(src.read())


def _load_with_cache(pkl_path: Path, csv_path: Path,
    fetch: Callable[[], None]) -> pd.DataFrame:
  """Return the cached frame, else run *fetch*, read *csv_path* and cache it.

  *fetch* is responsible for making ``csv_path`` exist; it is skipped entirely
  on a cache hit, so no network access happens once the pickle is present.
  """
  if pkl_path.is_file():
    with open(pkl_path, "rb") as f:
      return pickle.load(f)

  fetch()

  if not csv_path.is_file():
    raise FileNotFoundError(f"'{csv_path}' was not found - unpacking failed.")

  df = read_frame(csv_path)

  pkl_path.parent.mkdir(parents=True, exist_ok=True)
  with open(pkl_path, "wb") as f:
    pickle.dump(df, f)

  return df


def load_taxi_data() -> pd.DataFrame:
  """Raw NYC taxi trips, downloading from the Kaggle competition if needed."""
  log.info("Loading NYC taxi raw data …")

  def fetch() -> None:
    if not paths.TAXI_RAW_ZIP.is_file():
      download_kaggle_competition(
          _authenticated_api(), "nyc-taxi-trip-duration", paths.ZIP_DIR)
    extract_inner_zips(
        paths.TAXI_RAW_ZIP, paths.ZIP_DIR,
        {"train.zip", "test.zip", "sample_submission.zip"})
    extract_csv_from_inner_zips(paths.ZIP_DIR, paths.RAW_DIR)

  return _load_with_cache(paths.TAXI_CACHE_PICKLE, paths.TAXI_RAW_CSV, fetch)


def load_weather_data() -> pd.DataFrame:
  """Raw NYC weather observations from the Wunderground Kaggle dataset."""
  log.info("Loading NYC weather raw data …")

  def fetch() -> None:
    if not paths.WEATHER_RAW_ZIP.is_file():
      download_kaggle_dataset(
          _authenticated_api(), "pschale/nyc-taxi-wunderground-weather",
          paths.ZIP_DIR)
      log.info("Weather ZIP downloaded.")
    if not paths.WEATHER_RAW_CSV1.is_file():
      extract_csv_from_zip(
          paths.WEATHER_RAW_ZIP, paths.WEATHER_RAW_CSV1.name, paths.RAW_DIR)
      log.info("Weather CSV extracted.")

  return _load_with_cache(
      paths.WEATHER_CACHE_PICKLE, paths.WEATHER_RAW_CSV1, fetch)


def load_taxi_weather_data(recompute: bool = False) -> pd.DataFrame:
  """The merged taxi+weather modeling set, rebuilding it when absent."""
  if recompute or not paths.MERGED_CSV.exists():
    return build_merged_dataset(save_csv=True)
  return read_frame(paths.MERGED_CSV)
