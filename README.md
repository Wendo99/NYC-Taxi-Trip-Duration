https://www.kaggle.com/c/nyc-taxi-trip-duration/overview

### TODO

- Exploration of Weather
    - Check out Outliers
    - Maybe Capping
- Exploration of Taxi
    - Some trip durations extend to more than 3 hours
    - Some trip durations are less than 60 secs
    - Some trip locations are probably outside of New York
    - Some trips do end at start location after 5 min trip
    - Some trips do have more than 6 passengers
    - Remove or filter coordinates outside the approximate bounds of NYC
    - Optionally define bounding boxes for boroughs or airports for further geographic feature engineering
    - Optionally bin passenger_count into simplified categories: solo, small group (2–4), full load (5–6), unknown (0,>6)
    - Use the log-transformed trip_duration_log as target variable for regression models.
    - Use pickup_hourofyear for weather-data joins.
  
### Long term Exploration

- Welche Merkmale (z. B. Startzeitpunkt, geografische Lage) sind am aussagekräftigsten für die Vorhersage der
  Reisedauer?
- Mit welchen statistischen Modellen lässt sich die Reisedauer am zuverlässigsten auf Basis der vorhandenen Merkmale
  prognostizieren?
- Lässt sich die Vorhersagequalität durch die Integration von Wetterdaten (Temperatur, Luftfeuchtigkeit,
  Windgeschwindigkeit, Niederschlagsmenge) verbessern?

### NYC Taxi Trips Duration Data

- Not up to date

**File descriptions**

* **train.csv** - the training set (contains 1458644 trip records)
* **test.csv** - the testing set (contains 625134 trip records)
* **sample_submission.csv** - a sample submission file in the correct format

**Data fields**

* **id** - a unique identifier for each trip
* **vendor_id** - a code indicating the provider associated with the trip record
* **pickup_datetime** - date and time when the meter was engaged
* **dropoff_datetime** - date and time when the meter was disengaged
* **passenger_count** - the number of passengers in the vehicle (driver entered value)
* **pickup_longitude** - the longitude where the meter was engaged
* **pickup_latitude** - the latitude where the meter was engaged
* **dropoff_longitude** - the longitude where the meter was disengaged
* **dropoff_latitude** - the latitude where the meter was disengaged
* **store_and_fwd_flag** - This flag indicates whether the trip record was held in vehicle memory before sending to
  the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and
  forward trip
* **trip_duration** - duration of the trip in seconds

### 2016 Jan-June NYC Weather, hourly

- Not up to date

**Data fields for 1/1/16 - 7/1/16**

* Time/date stamp (d/m/y h:m)
* temperature (degrees F)
* windspeed (mph)
* Relative Humidity (%)
* Precipitation during last hour (inches)
* Pressure (inches of mercury)
* Description of Condition (string, eg "Overcast")
* Total precipitation during the day
* Total Snow during the day
* current conditions include fog (boolean)
* currently raining (boolean)
* currently snowing (boolean)

