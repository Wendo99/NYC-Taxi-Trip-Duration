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


## cluster - pickup

# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
#
# inertias = []
# K = range(2, 16)  # von 2 bis 20 Cluster testen
#
# for k in K:
#   kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
#   kmeans.fit(coords)
#   inertias.append(kmeans.inertia_)
#
# plt.figure(figsize=(8, 4))
# plt.plot(K, inertias, 'o-')
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Inertia (Measure of inertia)")
# plt.title("Elbow method: KMeans on GPS coordinates")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# import folium
# from sklearn.cluster import MiniBatchKMeans
# import matplotlib.cm as cm
#
# kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=10000)
# pickup_labels = kmeans.fit_predict(coords)
# taxi_weather['pickup_cluster'] = pickup_labels
# centers = kmeans.cluster_centers_
#
# # Interaktive Karte
# m_pickup = folium.Map(location=[40.75, -73.97], zoom_start=11)
#
# colors = cm.tab10.colors  # bis zu 10 unterschiedliche Farben
# sample = taxi_weather.sample(1000, random_state=42)
#
# for _, row in sample.iterrows():
#   cluster = int(row['pickup_cluster'])
#   folium.CircleMarker(
#       location=[row['pickup_latitude'], row['pickup_longitude']],
#       radius=3,
#       color=colors[cluster % len(colors)],
#       fill=True,
#       fill_opacity=0.6,
#       weight=0.5
#   ).add_to(m_pickup)
#
# # Clusterzentren markieren
# for c in centers:
#   folium.Marker(location=[c[1], c[0]], icon=folium.Icon(color='black')).add_to(m_pickup)
#
# m_pickup  # zeigt Karte im Notebook

## cluster - dropoff

# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
#
# inertias = []
# K = range(2, 16)  # von 2 bis 20 Cluster testen
#
# for k in K:
#   kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
#   kmeans.fit(coords)
#   inertias.append(kmeans.inertia_)
#
# plt.figure(figsize=(8, 4))
# plt.plot(K, inertias, 'o-')
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Inertia (Measure of inertia)")
# plt.title("Elbow method: KMeans on GPS coordinates")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# import folium
# from sklearn.cluster import MiniBatchKMeans
#
# kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=10000)
# dropoff_labels = kmeans.fit_predict(coords)
# taxi_weather['dropoff_cluster'] = dropoff_labels
# centers = kmeans.cluster_centers_
#
# # Interaktive Karte
# m_dropoff = folium.Map(location=[40.75, -73.97], zoom_start=11)
#
# colors = cm.tab10.colors  # bis zu 10 unterschiedliche Farben
# sample = taxi_weather.sample(1000, random_state=42)
#
# for _, row in sample.iterrows():
#   cluster = int(row['dropoff_cluster'])
#   folium.CircleMarker(
#       location=[row['dropoff_latitude'], row['dropoff_longitude']],
#       radius=3,
#       color=colors[cluster % len(colors)],
#       fill=True,
#       fill_opacity=0.6,
#       weight=0.5
#   ).add_to(m_dropoff)
#
# # Clusterzentren markieren
# for c in centers:
#   folium.Marker(location=[c[1], c[0]], icon=folium.Icon(color='black')).add_to(m_dropoff)
#
# m_dropoff  # zeigt Karte im Notebook
