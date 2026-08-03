"""Exploratory plots for the raw taxi data (used by notebooks/taxi.ipynb)."""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import pyplot as plt

from nyc_taxi.config import taxi_constants


def _thousands_axis(ax) -> None:
  """Format the y axis with thousands separators, no scientific offset."""
  ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
  ax.yaxis.get_offset_text().set_visible(False)


def _annotate_bars(ax, bars, values, headroom: float = 0.03,
    rotation: int = 0) -> None:
  """Write each bar's value above it and leave room for the labels.

  Extracted from plot_passenger_counts / plot_trip_day / plot_trip_hour,
  which each carried their own copy of this loop.
  """
  values = np.asarray(values)
  top = values.max() if len(values) else 0
  offset = top * headroom if top > 0 else 0.1
  for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
            f"{int(val):,}", ha="center", va="bottom", fontsize=9,
            rotation=rotation, clip_on=False)


def plot_rides_date(pickup_counts, dropoff_counts):
  fig, ax = plt.subplots(figsize=(12, 5))
  pickup_counts.plot(ax=ax, label="Pick_ups", color='green', alpha=0.6)
  dropoff_counts.plot(ax=ax, label="Drop_offs", color='blue', alpha=0.6)

  ax.set_title("Daily Pickup and Dropoff Counts")
  ax.set_xlabel("Date")
  ax.set_ylabel("Number of Rides")
  ax.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()


def plot_passenger_counts(pc):
  bins = np.arange(pc.min() - 0.5, pc.max() + 1.5, 1.0)

  fig, ax = plt.subplots(figsize=(8, 5))
  counts, edges, patches = ax.hist(pc, bins=bins, color='skyblue',
                                   edgecolor='black', alpha=0.8)

  ax.ticklabel_format(style='plain', axis='y')  # disable 1e6 offset
  _thousands_axis(ax)
  _annotate_bars(ax, patches, counts, headroom=0.01)

  ax.set_xlabel("Number of Passengers per ride")
  ax.set_ylabel("Number of Rides")
  ax.set_title("Distribution of passenger_count")
  ax.set_xticks(range(pc.min(), pc.max() + 1))
  ax.grid(axis='y', alpha=0.3)
  plt.tight_layout()
  plt.show()


def plot_geo_distr(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):
  plt.figure(figsize=(6, 6))
  plt.scatter(pickup_lon, pickup_lat, s=0.5, alpha=0.1,
              label='pickup')
  plt.scatter(dropoff_lon, dropoff_lat, s=0.5, alpha=0.1,
              label='dropoff')
  plt.xlim(taxi_constants.GeoBounds.min_lon, taxi_constants.GeoBounds.max_lon)
  plt.ylim(taxi_constants.GeoBounds.min_lat, taxi_constants.GeoBounds.max_lat)
  plt.xlabel("Longitude")
  plt.ylabel("Latitude")
  plt.legend(loc='upper right')
  plt.title("Geographic Distribution of Taxi Pickups and Dropoffs")
  plt.grid(True)
  plt.tight_layout()
  plt.show()


def plot_trips_month(df):
  if 'value' in df.columns:
    monthly = df.set_index('pickup_datetime').resample('ME')['value'].sum()
  else:
    monthly = df.set_index('pickup_datetime').resample('ME').size()

  # plotting with improved labels, title size and date formatting
  fig, ax = plt.subplots(figsize=(10, 4))
  ax.bar(monthly.index, monthly.values, width=20, align='center',
         color='skyblue', edgecolor='black', alpha=0.85)
  ax.set_ylim(0, monthly.max() * 1.15)
  ax.set_title('Monthly Counts', fontsize=18, fontweight='bold')
  ax.set_xlabel('Month', fontsize=14)
  ax.set_ylabel('Count', fontsize=14)

  # format x axis for dates
  ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
  plt.xticks(rotation=45, ha='right', fontsize=10)
  plt.tight_layout()
  plt.show()

def plot_trip_day(df):
  day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
               'Saturday', 'Sunday']
  dow = df['pickup_datetime'].dt.day_name()
  counts = dow.value_counts().reindex(day_order).fillna(0).astype(int)

  fig, ax = plt.subplots(figsize=(9, 4))
  bars = ax.bar(day_order, counts.values, color='skyblue', edgecolor='black',
                alpha=0.85)

  ax.set_ylim(0, counts.max() * 1.15 if counts.max() > 0 else 1)

  _thousands_axis(ax)
  _annotate_bars(ax, bars, counts.values)

  ax.set_xlabel('Day of Week')
  ax.set_ylabel('Number of Trips')
  ax.set_title('Number of Taxi Trips by Day of Week')
  plt.tight_layout()
  plt.show()


def plot_trip_hour(df):
  # aggregate by hour (0-23), ensure all hours present
  hours = df['pickup_datetime'].dt.hour.dropna().astype(int)
  counts = hours.value_counts().reindex(range(24),
                                        fill_value=0).sort_index().astype(int)

  # plot
  fig, ax = plt.subplots(figsize=(10, 6))
  bars = ax.bar(counts.index, counts.values, color='skyblue', edgecolor='black',
                alpha=0.85)

  # increase vertical headroom so labels clear the top border
  ax.set_ylim(0, counts.max() * 1.30 if counts.max() > 0 else 1)

  _thousands_axis(ax)
  # larger offset and rotated labels, since 24 bars sit close together
  _annotate_bars(ax, bars, counts.values, headroom=0.08, rotation=45)

  ax.set_xlim(left=-0.75, right=23.75)
  ax.set_xticks(range(24))
  ax.set_xticklabels([f'{h}:00' for h in range(24)], rotation=45, ha='right')
  ax.set_xlabel('Hour of Day')
  ax.set_ylabel('Number of Trips')
  ax.set_title('Number of Taxi Trips by Hour (0–23)')
  ax.grid(axis='y', alpha=0.3)

  # reserve extra top margin so annotations don't bump into the figure border
  plt.tight_layout(rect=[0, 0, 1, 0.94])
  plt.show()
