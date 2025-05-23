https://www.kaggle.com/c/nyc-taxi-trip-duration/overview

# Actual Notes

- Bearbeiten der Wetterdaten
***

- Welche Merkmale (z. B. Startzeitpunkt, geografische Lage) sind am aussagekräftigsten für die Vorhersage der
  Reisedauer?
- Mit welchen statistischen Modellen lässt sich die Reisedauer am zuverlässigsten auf Basis der vorhandenen Merkmale
  prognostizieren?
- Lässt sich die Vorhersagequalität durch die Integration von Wetterdaten (Temperatur, Luftfeuchtigkeit,
  Windgeschwindigkeit, Niederschlagsmenge) verbessern?

# Workflow
## Get the Data

## NYC Taxi Trips Duration

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

## 2016 Jan-June NYC Weather, hourly

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

## Analyze and Explore Data

### Weatherdata

**How are the weather datapoints distributed?**

Maßnahmen

Vor dem Mergen mit Taxi-Daten müssen wir:

1. [ ] Zeitstempel vereinheitlichen (z.B. floor('h'))
2. [ ] Nur repräsentative Messzeitpunkte behalten (z.B. minute == 51)
3. [ ] Evtl. fehlende Stunden interpolieren oder ignorieren (abhängig vom Modell)


**Minutenverteilung prüfen**

Maßnahmen

* Wir können und sollten ausschließlich Zeilen mit minute == 51 verwenden.
  * Damit stellen wir sicher, dass wir eine konsistente Messung pro Stunde verwenden.
  * Restliche Minuten werden ignoriert, da sie inkonsistent und potenziell redundant sind.

**Dubletten prüfen**

Maßnahmen

Vor der Aggregation auf Stundenebene sollten wir:

1. [ ] doppelte Einträge aggregieren, z.B. Mittelwert oder dominanten Zustand wählen.
2. [ ] alternativ: nur den ersten Eintrag behalten (vereinfachte Lösung, aber potenzieller Informationsverlust).

**Wetterdaten sinnvoll auf Stundenbasis aggregieren**