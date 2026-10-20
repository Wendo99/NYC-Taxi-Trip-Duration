# NYC Taxi Trip Duration

A coursework in 'Machine Learning' predicting how long a New York City taxi trip will take, from
1.46 million
yellow-cab records covering January–June 2016
([Kaggle competition](https://www.kaggle.com/c/nyc-taxi-trip-duration/overview)),
enriched with hourly weather observations.

Result: With XGBoost, test RMSE 0.3288 on the log target, 54.6 % better than predicting
the mean, with an overfit gap of 0.0045.

## Results

Held-out test set of 288,397 trips, never seen during training or model
selection. The target is `log1p(trip_duration)`, matching the competition's
RMSLE metric, so every RMSE below is on that log scale.

|                                         | RMSE       | vs. baseline |
|-----------------------------------------|------------|--------------|
| Baseline (predict the training mean)    | 0.7247     | —            |
| **XGBoost, trained on all 1.15 M rows** | **0.3288** | **54.6 %**   |

Model selection, 3-fold cross-validation on an identical 300,000-row
subsample:

| Model            | CV RMSE    | ±      | Fit time |
|------------------|------------|--------|----------|
| **XGBoost**      | **0.3404** | 0.0007 | 4 s      |
| RandomForest     | 0.3450     | 0.0008 | 61 s     |
| LinearRegression | 0.4228     | 0.0008 | 1 s      |

The train/test gap of 0.0045 is the number worth noting: on 1.15 M rows a
gradient-boosted ensemble has ample capacity to memorize, and it has not.

## The three questions this project set out to answer

Which characteristics matter most?
Distance dominates, `hav_dist_km_log` alone accounts for 52.96 % of total
model gain. Everything after that is time and place: pickup/dropoff clusters,
`is_early_morning`, `hour_of_day`, `is_night`, `pickup_weekday`. Trip volume
swings six-fold between the 05:00 trough (15,002 trips) and the 18:00 peak (90,600).

Which model predicts most reliably?
XGBoost, on identical folds. The gap between the linear model (~0.42) and the
tree models (~0.34) was expected as the relationship is not linear.

Does weather improve the forecast?
Marginally. Removing all seven weather features costs 0.0013 RMSE, about
0.38 %, or one to two cross-validation standard deviations. Real, consistent,
and far smaller than the effort of building the join. Stated plainly because
"we built it, and it barely helped" is a result, not a failure.

## Notebooks

Read in this order. All three are committed with their outputs, so they render
in full on GitHub without running anything.

| Notebook                                   | What it covers                                                                                                         |
|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| [taxi.ipynb](notebooks/taxi.ipynb)         | Raw trip data: why `dropoff_datetime` is excluded, four data-quality issues, why the target is log-transformed         |
| [weather.ipynb](notebooks/weather.ipynb)   | Hourly weather: unit conversion, the trailing-whitespace defect in `conditions`, which extremes are clipping artefacts |
| [modeling.ipynb](notebooks/modeling.ipynb) | Baseline, model comparison, the weather ablation, feature importance, residual analysis                                |

Two findings from the EDA that shaped everything downstream:

- `trip_duration` is exactly `dropoff_datetime - pickup_datetime`, for all
  1,458,644 rows, to the second. Handing a model `dropoff_datetime` lets it
  recover the target by subtraction, so it sits in `DROPPED_FEATURES`. It is
  also unavailable at prediction time, the trip has not finished.
- Cleaning flags, then excludes. Out-of-range values are clipped to the boundary
    and marked in a `*_invalid` column. 16,660 rows (1.14 %) trip at least one
    check (coordinates outside the configured bounds, passenger count outside its
    limits, or a trip shorter than 60 s or longer than 3 h). The modeling notebook
    removes these rows before the train/test split, so models are trained and
    scored on 1,441,984 trips. The flags are not model features.

## Project structure

```
src/nyc_taxi/
├── config/          # all tunables: feature allow/deny lists, cleaning
│                    # bounds, hyper-parameters, every filesystem path
├── features/        # per-column feature engineering (taxi, weather,
│                    # distance, clustering)
├── pipelines/       # raw -> processed -> merged, plus model building,
│                    # training and feature importance
├── visuals/         # EDA and residual plots
├── data_io.py       # Kaggle download, unpack, pickle cache
├── frames.py        # type-narrowing pandas readers
└── main.py          # build every dataset in dependency order

notebooks/           # the analysis, with committed outputs
tests/               # 69 tests
data/                # gitignored; created on demand
```

Paths are resolved from `pyproject.toml` upward, so every entry point behaves
the same whether it runs from `notebooks/`, `src/` or the repository root.

## Setup, build, test, and run

### 1. Install uv

[uv](https://docs.astral.sh/uv/) manages the virtual environment, the Python
version and the dependencies together.

### 2. Create the environment and install everything

```bash
uv sync --extra xgb        # plus `brew install libomp` on macOS
```

One command does all of it:

- creates `.venv/` in the project root
- downloads Python 3.11 if it is not already present (the version is pinned in
  `.python-version`)
- installs the exact versions from `uv.lock` — not "whatever PyPI ships today"

#### Get Kaggle Credentials

Kaggle credentials are needed only to download the raw data. Create a token at kaggle.com →
Settings → API and save it to ~/.kaggle/access_token

### 3. Run things

```bash
uv run python -m nyc_taxi.main    # build every dataset
```

then run the notebooks manually starting with `taxi.ipynb` followed by `weather.ipynb` and
`modeling.ipynb`

#### Run Tests

If you would like to run the tests you additionally need to sync uv again like:

```bash
uv sync --extra dev
```

and then run the test suite:

```bash
uv run pytest
```

## Known limitations

- No routed distance. Haversine understates real travel distance; OSRM is
  the most obvious route for improvement.
- Model selection used a 300 k subsample. The ranking is fair, every model
  saw identical folds, but the absolute CV figures are slightly pessimistic
  against the full-data result.
- Hyperparameters are fixed, taken from an earlier randomized search (reproducible via
  `search_hyperparameters`; the spaces live in
  `config/model_constants.py`) rather than re-tuned.

## Data

### Taxi trips

[Kaggle competition](https://www.kaggle.com/c/nyc-taxi-trip-duration/overview)
· [NYC TLC source](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

`train.csv` holds 1,458,644 trips with no missing values, no duplicate ids and
no duplicated rows.

| Field                                    | Meaning                                                          |
|------------------------------------------|------------------------------------------------------------------|
| `id`                                     | unique trip identifier                                           |
| `vendor_id`                              | provider associated with the record                              |
| `pickup_datetime` / `dropoff_datetime`   | when the meter was engaged / disengaged                          |
| `passenger_count`                        | passengers, driver-entered                                       |
| `pickup_longitude` / `pickup_latitude`   | where the meter was engaged                                      |
| `dropoff_longitude` / `dropoff_latitude` | where the meter was disengaged                                   |
| `store_and_fwd_flag`                     | `Y` if the record was held in vehicle memory before transmission |
| `trip_duration`                          | **target**, in seconds                                           |

### Weather

[NYC 2016 Jan–June hourly weather](https://www.kaggle.com/datasets/pschale/nyc-taxi-wunderground-weather)

4,392 rows forming a complete hourly grid, 183 days × 24 hours, no gaps,
covering the entire taxi window with a day to spare. Supplied in imperial
units and converted on ingest: temperature (°F → °C), windspeed (mph → km/h),
precipitation and pressure (inches → mm / hPa), plus relative humidity, a
free-text conditions description, daily precipitation and snow totals, and
fog / rain / snow flags.

The two datasets join on `hour_of_year`.
