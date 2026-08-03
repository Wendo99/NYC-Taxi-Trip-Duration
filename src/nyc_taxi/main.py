"""Build every processed dataset, in dependency order.

Run as a script: the taxi and weather tables are produced independently, then
merged on ``hour_of_year``. Each stage writes to ``data/processed/``.
"""
from nyc_taxi.pipelines import merge_pipeline, taxi_pipeline, weather_pipeline

SAVE_TAXI = True
SAVE_WEATHER = True
MERGE = True

if __name__ == '__main__':
  taxi_pipeline.build_taxi_dataset(SAVE_TAXI)
  weather_pipeline.build_weather_dataset(SAVE_WEATHER)
  merge_pipeline.build_merged_dataset(MERGE)
