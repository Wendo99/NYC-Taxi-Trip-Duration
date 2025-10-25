import pipelines.merge_pipeline as merge_pipeline
import pipelines.taxi_pipeline as taxi_pipeline
import pipelines.weather_pipeline as weather_pipeline

SAVE_TAXI = True
SAVE_WEATHER = True
MERGE = True

taxi_pipeline.build_taxi_dataset(SAVE_TAXI)
weather_pipeline.build_weather_dataset(SAVE_WEATHER)
merge_pipeline.build_merged_dataset(MERGE)
