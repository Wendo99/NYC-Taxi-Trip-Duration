import pickle
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def load_taxi_data():
  download_file_name = "nyc-taxi-trip-duration.zip"
  data_dir = Path("data_packages")
  extracted_dir = Path("data")
  csv_path = extracted_dir / "train.csv"
  pkl_path = extracted_dir / "train.pkl"

  zip_path = data_dir / download_file_name

  # Step 0: Fast CSV access via cache (if available)
  if pkl_path.is_file():
    with open(pkl_path, "rb") as f:
      return pickle.load(f)

  # Step 1: Download, only if ZIP is still missing
  if not zip_path.is_file():
    data_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files("nyc-taxi-trip-duration", path=data_dir)

  # Step 2: Unpack ZIP only if inner ZIPs are missing
  inner_zip_names = {"train.zip", "test.zip", "sample_submission.zip"}
  existing_inner_zips = {z.name for z in data_dir.glob("*.zip")}
  missing_inner_zips = inner_zip_names - existing_inner_zips

  if missing_inner_zips:
    with ZipFile(zip_path, 'r') as outer_zip:
      outer_zip.extractall(data_dir)

  # Step 3: Extract only missing CSVs from inner ZIPs
  extracted_dir.mkdir(parents=True, exist_ok=True)
  for inner_zip in data_dir.glob("*.zip"):
    with ZipFile(inner_zip, 'r') as zip_ref:
      for member in zip_ref.namelist():
        if member.endswith(".csv"):
          target_file = extracted_dir / Path(member).name
          if not target_file.is_file():
            zip_ref.extract(member, path=extracted_dir)

  # Step 4: Load CSV and save pkl cache
  if not csv_path.is_file():
    raise FileNotFoundError(
        f"'{csv_path}' wurde nicht gefunden – Entpackung fehlgeschlagen.")

  df = pd.read_csv(csv_path)
  with open(pkl_path, "wb") as f:
    pickle.dump(df, f)

  return df


def load_weather_data():
  dataset_slug = "pschale/nyc-taxi-wunderground-weather"
  zip_name = "nyc-taxi-wunderground-weather.zip"
  csv_name = "weatherdata.csv"

  data_dir = Path("data")
  package_dir = Path("data_packages")

  zip_path = package_dir / zip_name
  csv_path = data_dir / csv_name
  pkl_path = data_dir / "weatherdata.pkl"

  # Schritt 0: Cache verwenden
  if pkl_path.is_file():
    with open(pkl_path, "rb") as f:
      return pickle.load(f)

  # Schritt 1: ZIP herunterladen, wenn sie nicht existiert
  if not zip_path.is_file():
    package_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(package_dir), unzip=False)

  # Schritt 2: Entpacken, falls CSV noch nicht existiert
  if not csv_path.is_file():
    data_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "r") as zip_ref:
      zip_ref.extract(csv_name, path=data_dir)

  # Schritt 3: CSV laden und Cache schreiben
  if not csv_path.is_file():
    raise FileNotFoundError(
      f"{csv_path} wurde nicht gefunden – Entpackung fehlgeschlagen.")

  df = pd.read_csv(csv_path)
  with open(pkl_path, "wb") as f:
    pickle.dump(df, f)

  return df
