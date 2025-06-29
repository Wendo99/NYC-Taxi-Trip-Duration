from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from constants.core_c import ZIP_DIR, RAW_DIR, CACHE_DIR, PROCESSED_DIR

# ------------------------------------------------------------------ #
WEATHER_RAW_ZIP = Path(
    f"{ZIP_DIR}/nyc-taxi-wunderground-weather.zip")  # original Kaggle zip
WEATHER_RAW_CSV_NAME = "weatherdata.csv"
WEATHER_RAW_CSV1 = Path(
    f"{RAW_DIR}/weatherdata.csv")  # original Kaggle zip
WEATHER_RAW_CSV2 = Path(
    f"{RAW_DIR}/weather2_raw.csv")

WEATHER_CACHE_PICKLE = Path(f"{CACHE_DIR}/weather_cache.pkl")
WEATHER_PROCESSED_CSV = Path(f"{PROCESSED_DIR}/weather_clean.csv")

# Physical reference values ---------------------------------------------------

RAIN_TRACE_INCH = 0.01  # inch
SNOW_TRACE_INCH = 0.10  # inch

INCH_TO_MM = 25.4
MPH_TO_KPH = 1.60934
INHG_TO_HPA = 33.8639


# Ordinal scales --------------------------------------------------------------


def _make_map(labels: Sequence[str], *, unknown_code: int | None = None,
    start: int = 0) -> dict[str, int]:
  """Return mapping ``label -> ordinal_code`` (plus optional 'unknown' key)."""
  mapping: dict[str, int] = {lbl: i for i, lbl in
                             enumerate(labels, start=start)}
  if unknown_code is not None:
    mapping["unknown"] = unknown_code
  return mapping


@dataclass(frozen=True)
class OrdinalScale:
  """Simple container for thresholds + labels (low .. high)."""

  thresholds: Sequence[float]
  labels: Sequence[str]

  def label(self, value: float | int | None) -> str:
    """Return text label for a single numeric value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
      # NaN or missing value
      return "unknown"
    for thr, lab in zip(self.thresholds, self.labels):
      if value <= thr:
        return lab
    return self.labels[-1]

  # temperature °C -------------------------------------------------------------


TEMP_SCALE = OrdinalScale(
    thresholds=(-10, 0, 10, 20, 30),
    labels=("very_cold", "cold", "cool", "mild", "warm", "hot"),
)

# wind speed kph -------------------------------------------------------------
WIND_SCALE = OrdinalScale(
    thresholds=(5, 10, 20, 30, 40, 50, 65, 75, 90, 105, 120),
    labels=(
      "calm",
      "light_air",
      "light_breeze",
      "light_wind",
      "moderate_wind",
      "fresh_wind",
      "strong_wind",
      "stiff_wind",
      "stormy_wind",
      "storm",
      "heavy_storm",
      "hurricane_like_storm",
      "hurricane",
    ),
)


@dataclass(frozen=True)
class WindLimits:
  kph_min: float = 0.0
  kph_max: float = 120.0


#  humidity % --------------------------------------------------------
HUMIDITY_SCALE = OrdinalScale(
    thresholds=(30, 50, 70, 85),
    labels=("very_dry", "dry", "normal", "wet", "very_wet"),
)

#  pressure hPa ------------------------------------------------------
PRESSURE_SCALE = OrdinalScale(
    thresholds=(980, 1000, 1020, 1030),
    labels=("very_low", "low", "normal", "high", "very_high"),
)

# rain mm / h --------------------------------------------------------
RAIN_SCALE = OrdinalScale(
    thresholds=(0.5, 2.5, 7.6, 50),
    labels=("no_rain", "light_rain", "moderate_rain", "heavy_rain",
            "very_heavy_rain")
)

# snow mm / h --------------------------------------------------------

SNOW_SCALE = OrdinalScale(
    thresholds=(0.5, 1.0, 5.0, 10.0),
    labels=("no_snow", "light_snow", "moderate_snow", "heavy_snow",
            "very_heavy_snow"),
)


@dataclass(frozen=True)
class DailySnowLimit:
  mm_min: float = 0.0
  mm_max: float = 600.0


# clouds
CLOUD_LABELS = ("clear", "scattered_clouds", "partly_cloudy",
                "mostly_cloudy", "overcast")
CLOUD_MAP = _make_map(CLOUD_LABELS, unknown_code=-1)

# haze
HAZE_LABELS = ("no_haze", "haze")
HAZE_MAP = _make_map(HAZE_LABELS)

# freezing rain / fog
FREEZING_LABELS = ("no_freezing_rain_fog",
                   "light_freezing_rain",
                   "light_freezing_fog")
FREEZING_MAP = _make_map(FREEZING_LABELS)

# fog
FOG_LABELS = ("no_fog", "fog")
FOG_MAP = _make_map(FOG_LABELS, unknown_code=-1)
