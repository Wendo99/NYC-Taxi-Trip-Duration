# NYC Taxi Trip Duration Prediction

A machine learning coursework project predicting the duration of New York City yellow-cab trips using 1,458,644 records from January–June 2016, enriched with hourly weather observations. Alongside modelling, the project implements a modular Python package for data acquisition, cleaning, validation, feature engineering and evaluation.

**Headline result:** XGBoost achieved 0.3288 RMSE on `log1p(trip_duration)` on a held-out set of 288,397 trips, 54.6% lower than the mean-prediction baseline of 0.7247. The metric is on the transformed target scale, not in seconds.

## Quick start

**Requirements:** [uv](https://docs.astral.sh/uv/), the Kaggle API credentials needed to download the input data, and (for XGBoost on some macOS systems) OpenMP (`brew install libomp`). The repository pins Python 3.11 in `.python-version`; `pyproject.toml` permits Python 3.11–3.13.

```bash
git clone https://github.com/Wendo99/NYC-Taxi-Trip-Duration.git
cd NYC-Taxi-Trip-Duration
uv sync --extra dev --extra xgb
uv run pytest
uv run python -m nyc_taxi.main
```

The last command runs the taxi, weather and merge pipelines in dependency order. Its principal outputs are:

- `data/processed/taxi_clean.csv` — processed taxi records and features
- `data/processed/weather_clean.csv` — processed hourly weather observations
- `data/processed/taxi_weather.csv` — merged table

The package creates directories for processed data, caches and other outputs. Most generated outputs are Git-ignored. Data download requires Kaggle access; local pickle caches are reused once available.

**Kaggle authentication:** Configure the [Kaggle API](https://github.com/Kaggle/kaggle-api) using `~/.kaggle/kaggle.json` or the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables. Accept the [competition rules](https://www.kaggle.com/competitions/nyc-taxi-trip-duration) if Kaggle requires this before download. Never commit API credentials.

To explore the analysis without running the pipeline, view the committed notebook outputs: [taxi](notebooks/taxi.ipynb), [weather](notebooks/weather.ipynb), then [modeling](notebooks/modeling.ipynb).

## Architecture and data flow

```text
Kaggle taxi + weather data
          |
          v
  data_io.py: download, extract, cache
          |
          +----------------------------+
          |                            |
          v                            v
  taxi_pipeline.py             weather_pipeline.py
  flag, clean, engineer        convert, aggregate, classify
          |                            |
          +-------------+--------------+
                        v
                merge_pipeline.py
              validated left join
                        |
                        v
                processed CSV files
                        |
                        v
              notebooks and models
```

| Component | Responsibility |
| --- | --- |
| [`config/`](src/nyc_taxi/config/) | Central configuration of filesystem paths, feature lists, cleaning bounds and hyperparameters |
| [`data_io.py`](src/nyc_taxi/data_io.py) | Download, extraction and caching of raw data |
| [`features/`](src/nyc_taxi/features/) | Taxi, weather, geographic and distance feature functions |
| [`pipelines/`](src/nyc_taxi/pipelines/) | Dataset construction, merging, model training and feature importance |
| [`visuals/`](src/nyc_taxi/visuals/) | Exploratory plots and model diagnostics |
| [`frames.py`](src/nyc_taxi/frames.py) | DataFrame loading helpers |
| [`main.py`](src/nyc_taxi/main.py) | Run the processing stages |
| [`tests/`](tests/) | Automated tests for transformations, interfaces and regression behaviour |

The importable `nyc_taxi` package uses a conventional `src/` layout with a build backend defined in [`pyproject.toml`](pyproject.toml). Dependency resolution is recorded in [`uv.lock`](uv.lock). Paths are resolved by locating the project-level `pyproject.toml` rather than assuming a particular working directory.

## Testing and data validation

The repository contains 69 pytest tests, including characterization tests, tests of DataFrame shapes and types, standalone module imports, joins, and model-related helpers. Run them with:

```bash
uv sync --extra dev
uv run pytest
```

Three particularly important data contracts are:

1. **No target leakage.** `dropoff_datetime` is not a predictor because trip duration is the difference between dropoff and pickup times, and the dropoff timestamp is unavailable at prediction time. This equality was observed for all 1,458,644 raw trips.
2. **Explicit invalid-record handling.** Configurable bounds flag and clip out-of-range values. The modelling notebook excludes 16,660 flagged trips (1.14%) before splitting the data, leaving 1,441,984 trips for modelling; the flags are not model features.
3. **Safe data integration.** `merge_taxi_weather` retains taxi records with unmatched weather observations and validates a `many_to_one` relationship. Duplicate weather keys raise an error rather than silently multiplying rows.

For example, the merge function can be exercised with synthetic data without downloading the full dataset:

```python
import pandas as pd
from nyc_taxi.pipelines.merge_pipeline import merge_taxi_weather

trips = pd.DataFrame({"id": ["a", "b"], "hour_of_year": [10, 11]})
weather = pd.DataFrame({"hour_of_year": [10], "temp_c": [12.0]})
combined = merge_taxi_weather(trips, weather)
print(combined[["id", "temp_c"]])
# Both trips remain; the missing weather value for "b" is NaN.
```

The tests are intended to guard against regressions. Characterization tests record current behaviour and should not all be interpreted as assertions of ideal behaviour.

## Results

The held-out test set comprises 288,397 trips excluded from training and model selection. The target is `log1p(trip_duration)`, consistent with the competition's RMSLE formulation.

| Held-out test evaluation | Log-RMSE | Reduction vs. baseline |
| --- | ---: | ---: |
| Predict the training-set mean | 0.7247 | — |
| **XGBoost (trained on approximately 1.15 million rows)** | **0.3288** | **54.6%** |

Model selection used three-fold cross-validation on the same 300,000-row subsample for each candidate:

| Model | Cross-validation log-RMSE | Reported variation (±) | Fit time |
| --- | ---: | ---: | ---: |
| **XGBoost** | **0.3404** | 0.0007 | 4 s |
| Random Forest | 0.3450 | 0.0008 | 61 s |
| Linear Regression | 0.4228 | 0.0008 | 1 s |

The reported training/test gap was 0.0045. This is encouraging for that evaluation, but it does not establish performance on later or operational data.

### Research questions and findings

**Which characteristics matter most?** Haversine distance (`hav_dist_km_log`) accounted for 52.96% of the reported XGBoost feature gain. Other important signals included pickup/dropoff clusters and time features. Trip volume ranged from 15,002 at 05:00 to 90,600 at 18:00 in the analysed data.

**Which model performs best?** In the reported cross-validation comparison, XGBoost and Random Forest outperformed Linear Regression; XGBoost had the lowest error among the compared models.

**Does weather help?** Removing all seven weather features increased cross-validation log-RMSE by only 0.0013 (approximately 0.38%). The measured improvement was small relative to the added integration complexity.

The [modelling notebook](notebooks/modeling.ipynb) documents model comparison, feature importance, ablation and residual analysis. The [taxi notebook](notebooks/taxi.ipynb) explains leakage avoidance and the [weather notebook](notebooks/weather.ipynb) documents data cleaning and unit conversion.

## Data provenance

**Taxi trips:** [Kaggle NYC Taxi Trip Duration competition](https://www.kaggle.com/competitions/nyc-taxi-trip-duration), sourced from [NYC Taxi & Limousine Commission trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). The inspected `train.csv` contained 1,458,644 trips, with no missing values, duplicated IDs or duplicate rows. Fields include trip ID, vendor, pickup/dropoff times and coordinates, passenger count, forwarding flag and duration in seconds.

**Weather:** [NYC 2016 hourly weather data](https://www.kaggle.com/datasets/pschale/nyc-taxi-wunderground-weather). The analysed period contains 4,392 hours (183 days × 24 hours). Preprocessing addresses textual conditions, trace precipitation, unit conversions, selected interpolations and hourly aggregation. The repository also includes a [supplementary raw weather file](data/raw/weather2_raw.csv). Taxi and weather records are joined on `hour_of_year`.

## Limitations and future work

- **Distance approximation:** The evaluated model uses Haversine distance, which cannot represent actual road routes. Route-based distance is a possible extension but is not included in the reported evaluation.
- **Evaluation scope:** Cross-validation used a 300,000-row subsample, whereas the reported held-out result followed training on the larger dataset. These are different evaluation settings.
- **Model configuration:** Fixed hyperparameters originated in an earlier randomized search. A search helper and the associated parameter spaces remain in [`training.py`](src/nyc_taxi/pipelines/training.py) and [`model_constants.py`](src/nyc_taxi/config/model_constants.py); hyperparameters are not re-tuned on each run.
- **Deployment:** This is an offline academic prediction project, not a live trip-time estimation service.
