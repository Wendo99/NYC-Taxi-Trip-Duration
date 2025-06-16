import pickle
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

import preModelling.data_config as cf


def download_kaggle_competition(api, competition_name, path):
  try:
    api.authenticate()
  except Exception as e:
    raise RuntimeError("Kaggle API authentication failed. "
                       "Ensure that ~/.kaggle/kaggle.json exists and is properly configured.") from e
  api.competition_download_files(competition_name, path=path)


def download_kaggle_dataset(api, dataset_slug, path):
  """Authenticate and download dataset files from Kaggle."""
  try:
    api.authenticate()
  except Exception as e:
    raise RuntimeError("Kaggle API authentication failed. "
                       "Ensure that ~/.kaggle/kaggle.json exists and is properly configured.") from e
  api.dataset_download_files(dataset_slug, path=str(path), unzip=False)


def extract_inner_zips(zip_path, data_dir, required_inner_zips):
  existing_inner_zips = {z.name for z in data_dir.glob("*.zip")}
  missing_inner_zips = required_inner_zips - existing_inner_zips

  if missing_inner_zips:
    with ZipFile(zip_path, 'r') as outer_zip:
      outer_zip.extractall(data_dir)


def extract_csv_from_inner_zips(data_dir, extracted_dir):
  extracted_dir.mkdir(parents=True, exist_ok=True)
  for inner_zip in data_dir.glob("*.zip"):
    with ZipFile(inner_zip, 'r') as zip_ref:
      for member in zip_ref.namelist():
        if member.endswith(".csv"):
          target_file = extracted_dir / Path(member).name
          if not target_file.is_file():
            with zip_ref.open(member) as src, open(target_file, "wb") as dst:
              dst.write(src.read())


def extract_csv_from_zip(zip_path, csv_name, target_dir):
  """Extract a specific CSV file from a ZIP archive to a target directory."""
  target_dir.mkdir(parents=True, exist_ok=True)
  with ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extract(csv_name, path=target_dir)


def load_taxi_data():
  """Load NYC taxi trip data with optional caching and Kaggle download."""
  download_file_name = "nyc-taxi-trip-duration.zip"
  data_dir = Path(cf.PACKAGE_DIR)
  extracted_dir = Path(cf.TAXI_DATA_RAW_DIR)
  csv_path = extracted_dir / "train.csv"
  pkl_path = extracted_dir / "train.pkl"

  zip_path = data_dir / download_file_name

  # Step 0: Fast CSV access via cache (if available)
  if pkl_path.is_file():
    with open(pkl_path, "rb") as f:
      # Load from cache
      return pickle.load(f)

  # Step 1: Download, only if ZIP is still missing
  if not zip_path.is_file():
    data_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    download_kaggle_competition(api, "nyc-taxi-trip-duration", data_dir)

  # Step 2: Unpack ZIP only if inner ZIPs are missing
  inner_zip_names = {"train.zip", "test.zip", "sample_submission.zip"}
  extract_inner_zips(zip_path, data_dir, inner_zip_names)

  # Step 3: Extract only missing CSVs from inner ZIPs
  extract_csv_from_inner_zips(data_dir, extracted_dir)

  # Step 4: Load CSV and save pkl cache
  if not csv_path.is_file():
    raise FileNotFoundError(
        f"'{csv_path}' was not found - unpacking failed.")

  df = pd.read_csv(csv_path)
  with open(pkl_path, "wb") as f:
    pickle.dump(df, f)  # type: ignore[arg-type]

  return df


def load_weather_data():
  """Load NYC weather data with optional caching and Kaggle download."""
  dataset_slug = "pschale/nyc-taxi-wunderground-weather"
  zip_name = "nyc-taxi-wunderground-weather.zip"
  csv_name = "weatherdata.csv"

  data_dir = Path(cf.WEATHER_DATA_RAW_DIR)
  package_dir = Path(cf.PACKAGE_DIR)

  zip_path = package_dir / zip_name
  csv_path = data_dir / csv_name
  pkl_path = data_dir / "weather_data_raw.pkl"

  # Step 0: Use cache if available
  if pkl_path.is_file():
    with open(pkl_path, "rb") as f:
      return pickle.load(f)

  # Step 1: Download ZIP if missing
  if not zip_path.is_file():
    package_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    download_kaggle_dataset(api, dataset_slug, package_dir)

  # Step 2: Extract CSV if missing
  if not csv_path.is_file():
    extract_csv_from_zip(zip_path, csv_name, data_dir)

  # Step 3: Load CSV and write cache
  if not csv_path.is_file():
    raise FileNotFoundError(
        f"{csv_path} was not found - unpacking failed.")

  df = pd.read_csv(csv_path)
  with open(pkl_path, "wb") as f:
    pickle.dump(df, f)  # type: ignore[arg-type]

  return df
