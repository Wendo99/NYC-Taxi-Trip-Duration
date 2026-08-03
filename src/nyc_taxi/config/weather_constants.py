"""Weather reference values, unit factors and ordinal scales.

``OrdinalScale`` and ``make_map`` live here rather than in
``utilities.weather_utilities`` because the scales below are built from them
at import time. Defining them in utilities made constants depend on
utilities — a cycle that was previously worked around by importing
``weather_constants`` inside ten separate function bodies.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OrdinalScale:
  """Thresholds plus the labels they map onto, ordered low → high.

  ``labels`` is deliberately one longer than ``thresholds``: N cut points
  divide the number line into N+1 bands, and the final label is the
  open-ended one above the highest threshold.
  """

  thresholds: Sequence[float]
  labels: Sequence[str]

  def label(self, value: float | None) -> str:
    """Return the text label for a single numeric value."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
      return "unknown"
    # strict=False is required, not an oversight: labels is longer than
    # thresholds by one, and that trailing label is returned below.
    for thr, lab in zip(self.thresholds, self.labels, strict=False):
      if value <= thr:
        return lab
    return self.labels[-1]


def make_map(labels: Sequence[str], *, unknown_code: int | None = None,
    start: int = 0) -> dict[str, int]:
  """Return a ``label -> ordinal_code`` mapping, plus optional 'unknown'."""
  mapping: dict[str, int] = {lbl: i
                             for i, lbl in enumerate(labels, start=start)}
  if unknown_code is not None:
    mapping["unknown"] = unknown_code
  return mapping


# Physical reference values ---------------------------------------------------

RAIN_TRACE_INCH = 0.01  # inch
SNOW_TRACE_INCH = 0.10  # inch

INCH_TO_MM = 25.4
MPH_TO_KPH = 1.60934
IN_TO_HPA = 33.8639


@dataclass(frozen=True)
class DailySnowLimit:
  mm_min: float = 0.0
  mm_max: float = 600.0


@dataclass(frozen=True)
class WindLimits:
  kph_min: float = 0.0
  kph_max: float = 120.0


def _ordinal_scale(thresholds, labels):
  return OrdinalScale(thresholds=thresholds, labels=labels)


def _make_map(labels, unknown_code=None):
  return make_map(labels, unknown_code=unknown_code)


# temperature °C -------------------------------------------------------------


TEMP_SCALE = _ordinal_scale(
    thresholds=(-10, 0, 10, 20, 30),
    labels=("very_cold", "cold", "cool", "mild", "warm", "hot"),
)

# wind speed kph -------------------------------------------------------------
WIND_SCALE = _ordinal_scale(
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

#  humidity % --------------------------------------------------------
HUMIDITY_SCALE = _ordinal_scale(
    thresholds=(30, 50, 70, 85),
    labels=("very_dry", "dry", "normal", "wet", "very_wet"),
)

#  pressure hPa ------------------------------------------------------
PRESSURE_SCALE = _ordinal_scale(
    thresholds=(980, 1000, 1020, 1030),
    labels=("very_low", "low", "normal", "high", "very_high"),
)

# rain mm / h --------------------------------------------------------
RAIN_SCALE = _ordinal_scale(
    thresholds=(0.5, 2.5, 7.6, 50),
    labels=("no_rain", "light_rain", "moderate_rain", "heavy_rain",
            "very_heavy_rain")
)

# snow mm / h --------------------------------------------------------

SNOW_SCALE = _ordinal_scale(
    thresholds=(0.5, 1.0, 5.0, 10.0),
    labels=("no_snow", "light_snow", "moderate_snow", "heavy_snow",
            "very_heavy_snow"),
)

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
