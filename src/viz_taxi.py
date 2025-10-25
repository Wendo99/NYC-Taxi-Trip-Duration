import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

from constants import taxi_constants


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

  # annotate counts above each bar
  offset = max(counts) * 0.01
  for rect, count in zip(patches, counts):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    ax.text(x, y + offset, int(count), ha='center', va='bottom', fontsize=9)

  # remove scientific notation and show thousands separators
  ax.ticklabel_format(style='plain', axis='y')  # disable 1e6 offset
  ax.yaxis.set_major_formatter(
    mticker.StrMethodFormatter('{x:,.0f}'))  # e.g. 1,234,567
  ax.yaxis.get_offset_text().set_visible(False)

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
    monthly = df.set_index('pickup_datetime').resample('M')['value'].sum()
  else:
    monthly = df.set_index('pickup_datetime').resample('M').size()

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

  # use StrMethodFormatter for thousands separators; do NOT call ticklabel_format
  ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
  ax.yaxis.get_offset_text().set_visible(False)

  offset = counts.max() * 0.03 if counts.max() > 0 else 0.1
  for bar, val in zip(bars, counts.values):
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    ax.text(x, y + offset, f'{int(val):,}', ha='center', va='bottom',
            fontsize=9)

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

  # format y axis with thousands separators
  ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
  ax.yaxis.get_offset_text().set_visible(False)

  # larger annotation offset and disable clipping so labels are fully visible
  offset = counts.max() * 0.08 if counts.max() > 0 else 0.8
  for bar, val in zip(bars, counts.values):
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    ax.text(x, y + offset, f'{int(val):,}', rotation=45, ha='center',
            va='bottom',
            fontsize=9, clip_on=False)

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

