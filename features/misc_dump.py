#%% md
# Filter haversine in combination
#%%
# filtered = taxi_weather[
#   (taxi_weather['trip_duration_min'].between(*tc.TYPICAL_DURATION_MIN)) &
#   (taxi_weather['hav_dist_km'].between(*tc.TYPICAL_DISTANCE_KM)) &
#   (taxi_weather['pickup_hour'].between(*tc.TYPICAL_HOURS)) &
#   # (taxi_weather['pickup_weekday'].between(*tc.TYPICAL_WEEKDAYS)) &
#   (taxi_weather['passenger_count'].between(*tc.TYPICAL_PASSENGERS)) &
#   (taxi_weather['pickup_longitude'].between(*tc.LON_RANGE)) &
#   (taxi_weather['dropoff_longitude'].between(*tc.LON_RANGE)) &
#   (taxi_weather['pickup_latitude'].between(*tc.LAT_RANGE)) &
#   (taxi_weather['dropoff_latitude'].between(*tc.LAT_RANGE))
#   ]
#
# filtered.plot.scatter(x="hav_dist_km", y="trip_duration_log", alpha=0.7, grid=True)
# plt.xlabel("hav_dist_km")
# plt.ylabel("Trip Duration")
# plt.title("scatter: Weekday, Distance, Time")
# plt.show()
#
# filtered.plot.hexbin(
#     x="hav_dist_km", y="trip_duration_log",
#     gridsize=50, cmap='plasma'
# )
# plt.xlabel("hav_dist_km")
# plt.ylabel("Trip Duration")
# plt.title("Hexbin: Weekday, Distance, Time")
# plt.grid(True)
# plt.show()
# # TODO: Add zone-based filtering (e.g. by longitude/latitude clusters or external zone map)