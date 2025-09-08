# New York City Taxi Trip Duration – Project

- Framing the Problem
    - Goal: Based on individual trip attributes, our goal is to build and
      train a model that predicts the total ride duration of taxi trips in New
      York City.
- EDA
  - Which characteristics (e.g. start time, geographical location) are most meaningful for predicting
    trip duration?
  - Which statistical models can be used to predict the journey duration most reliably on the basis of
    the existing characteristics?
  - Can the forecast quality be improved by integrating weather data (temperature, humidity, wind
    speed, precipitation)?
    
### Project Structure Overview

- `/data/`
    - `/data/*` – Will contain downloaded or saved data files. 
- `/src/`
    - `/src/constants/*` – Constants – e.g. Strings, Parameters etc.
    - `/src/features/*` – Functions for Pre-Training – e.g. calculating the haversine distance, feature engineering of taxi and weather data and utilities
    - `/src/pipeline/*` – Functions for Pre-Training DataSet-Pipelines and Modell-Training
- `/notebooks/
    - `/notebooks/*` – Contains alle Jupyter Notebooks used in EDA-/Engineering/Merge- and Modelling-Phase
        1. Taxi
        2. Weather
        3. Merge
        4. Modelling
     
***

### Data Description – New York City Taxi Trip Duration

Kaggle-Competition: https://www.kaggle.com/c/nyc-taxi-trip-duration/overview

NYC TLC: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

**File descriptions**

- **train.csv** - the training set (contains 1458644 trip records)
- **test.csv** - the testing set (contains 625134 trip records)
- **sample_submission.csv** - a sample submission file in the correct format

**Data fields**

- **id** - a unique identifier for each trip
- **vendor_id** - a code indicating the provider associated with the trip record
- **pickup_datetime** - date and time when the meter was engaged
- **dropoff_datetime** - date and time when the meter was disengaged
- **passenger_count** - the number of passengers in the vehicle (driver entered
  value)
- **pickup_longitude** - the longitude where the meter was engaged
- **pickup_latitude** - the latitude where the meter was engaged
- **dropoff_longitude** - the longitude where the meter was disengaged
- **dropoff_latitude** - the latitude where the meter was disengaged
- **store_and_fwd_flag** - This flag indicates whether the trip record was held
  in vehicle memory before sending to the vendor because the vehicle did not
  have a connection to the server
    - Y=store and forward;
    - N=not a store and forward trip
- **trip_duration** - duration of the trip in seconds

### 2016 Jan-June NYC Weather, hourly

https://www.kaggle.com/datasets/pschale/nyc-taxi-wunderground-weather

- Time/date stamp (d/m/y h:m)
- temperature (degrees F)
- windspeed (mph)
- Relative Humidity (%)
- Precipitation during last hour (inches)
- Pressure (inches of mercury)
- Description of Condition (string, eg "Overcast")
- Total precipitation during the day
- Total Snow during the day
- current conditions include fog (boolean)
- currently raining (boolean)
- currently snowing (boolean)

