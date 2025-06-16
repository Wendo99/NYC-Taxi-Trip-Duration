# PATHs
WEATHER_DATA_RAW_DIR = "../data/weather_data_raw"
WEATHER_DATA_DIR = "../data"
PACKAGE_DIR = "../data_packages"

WEATHER_DATA_RAW_SAVE = WEATHER_DATA_RAW_DIR + "/weather_data_raw.csv"
WEATHER_DATA_SAVE = WEATHER_DATA_DIR + "/weather.csv"
WEATHER_DATA_SAVE_WITH_CLASS = WEATHER_DATA_DIR + "/weather_data_classes.csv"

WEATHER_DATA_PART_2 = '../gitData/weather_data_new_rows.csv'

TAXI_DATA_RAW_DIR = "../data/taxi_data_raw"
TAXI_DATA_DIR = "../data"
TAXI_DATA_RAW_SAVE = TAXI_DATA_RAW_DIR + "/taxi_data_raw.csv"
TAXI_DATA_SAVE = TAXI_DATA_DIR + "/taxi.csv"

TAXI_WEATHER_DATA_SAVE = "../data/taxi_weather_raw.csv"

TAXI_PATH = "../data/taxi_data.csv"
TAXI_FLAGGED_PATH = "data/taxi_data_filtered.csv"

# Test Train Set Parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
LAT_RANGE = (40.47, 41.0)
LON_RANGE = (-74.3, -73.6)
NYC_LAT_MIN = 40.5
NYC_LAT_MAX = 41.0
NYC_LON_MIN = -74.3
NYC_LON_MAX = -73.6
TYPICAL_HOURS = (5, 18)
TYPICAL_WEEKDAYS = [0,1,2,3]
TYPICAL_DISTANCE_KM = (1, 20)
TYPICAL_DURATION_MIN = (5, 120)
TYPICAL_PASSENGERS = (1, 6)
WEATHER_COLUMNS_TO_DROP = ['temp', 'windspeed', 'pressure', 'precip',
                           'dailyprecip', 'dailysnow']
WEATHER_FEATURES_TO_IMPUTE = ['windspeed_kph', 'pressure_hPa']
