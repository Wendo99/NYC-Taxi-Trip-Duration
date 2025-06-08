# modelling
TEST_SIZE = 0.2
RANDOM_STATE = 42

TYPICAL_HOURS = (5, 18)
TYPICAL_WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
TYPICAL_DISTANCE_KM = (1, 20)
TYPICAL_DURATION_MIN = (5, 120)
TYPICAL_PASSENGERS = (1, 6)

LAT_RANGE = (40.47, 41.0)
LON_RANGE = (-74.3, -73.6)


# weather

WEATHER_FEATURES_TO_IMPUTE = ['windspeed_kph', 'pressure_hPa']
WEATHER_PATH = "data/weather_data.csv"
WEATHER_RAW_PATH = "data/weather_data_raw.csv"
WEATHER_COLUMNS_TO_DROP = ['temp', 'windspeed', 'pressure', 'precip',
                           'dailyprecip', 'dailysnow']

# taxi

TAXI_PATH = "data/taxi_data.csv"
TAXI_FLAGGED_PATH = "data/taxi_data_filtered.csv"

NYC_LAT_MIN = 40.47
NYC_LAT_MAX = 41.0
NYC_LON_MIN = -74.3
NYC_LON_MAX = -73.6

TAXI_WEATHER_PATH = "data/taxi_weather_data.csv"