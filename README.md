# NYC Taxi Trip Duration

A coursework in 'Machine Learning' predicting how long a New York City taxi trip will take, from 1.46 million
yellow-cab records covering January–June 2016
([Kaggle competition](https://www.kaggle.com/c/nyc-taxi-trip-duration/overview)),
enriched with hourly weather observations.

Result: test RMSE 0.3288 on the log target, 54.6 % better than predicting
the mean, with an overfit gap of 0.0045.

---

## Results

Held-out test set of 288,397 trips, never seen during training or model
selection. The target is `log1p(trip_duration)`, matching the competition's
RMSLE metric, so every RMSE below is on that log scale.

| | RMSE | vs. baseline |
|---|---|---|
| Baseline (predict the training mean) | 0.7247 | — |
| **XGBoost, trained on all 1.15 M rows** | **0.3288** | **54.6 %** |

Model selection, 3-fold cross-validation on an identical 300,000-row
subsample:

| Model | CV RMSE | ± | Fit time |
|---|---|---|---|
| **XGBoost** | **0.3404** | 0.0007 | 4 s |
| RandomForest | 0.3450 | 0.0008 | 61 s |
| LinearRegression | 0.4228 | 0.0008 | 1 s |
| Ridge | 0.4228 | 0.0008 | 1 s |
| BayesianRidge | 0.4228 | 0.0008 | 1 s |

The train/test gap of 0.0045 is the number worth noting: on 1.15 M rows a
gradient-boosted ensemble has ample capacity to memorise, and it has not.

## The three questions this project set out to answer

Which characteristics matter most?
Distance dominates, `hav_dist_km_log` alone accounts for 52.96 % of total
model gain. Everything after that is time and place: pickup/dropoff clusters,
`is_early_morning`, `hour_of_day`, `is_night`, `pickup_weekday`. Trip volume
swings six-fold between the 05:00 trough (15,002 trips) and the 18:00 peak
(90,600).

Which model predicts most reliably?
XGBoost, on identical folds. The gap between the linear family (~0.42) and the
tree models (~0.34) is the real finding: the relationship is not linear, and
the interactions, distance × hour, cluster × weekday, are where the signal
lives. The three linear models agree to four decimal places, so regularisation
is doing nothing at this sample size.

Does weather improve the forecast?
Marginally. Removing all seven weather features costs 0.0013 RMSE, about
0.38 %, or one to two cross-validation standard deviations. Real, consistent,
and far smaller than the effort of building the join. Stated plainly because
"we built it and it barely helped" is a result, not a failure.

## Notebooks

Read in this order. All three are committed with their outputs, so they render
in full on GitHub without running anything.

| Notebook | What it covers |
|---|---|
| [taxi.ipynb](notebooks/taxi.ipynb) | Raw trip data: why `dropoff_datetime` is excluded, four data-quality issues, why the target is log-transformed |
| [weather.ipynb](notebooks/weather.ipynb) | Hourly weather: unit conversion, the trailing-whitespace defect in `conditions`, which extremes are clipping artefacts |
| [modelling.ipynb](notebooks/modelling.ipynb) | Baseline, five-model comparison, the weather ablation, feature importance, residual analysis |

Two findings from the EDA that shaped everything downstream:

- `trip_duration` is exactly `dropoff_datetime - pickup_datetime`, for all
  1,458,644 rows, to the second. Handing a model `dropoff_datetime` lets it
  recover the target by subtraction, so it sits in `DROPPED_FEATURES`. It is
  also unavailable at prediction time, the trip has not finished.
- Cleaning flags rather than deletes. 16,660 rows (1.14 %) trip at least
  one validity check; out-of-range values are clipped to the boundary and
  marked in a `*_invalid` column, so the row count stays stable and the flag
  itself becomes a feature.

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

## Setup

### 1. Install uv

[uv](https://docs.astral.sh/uv/) manages the virtual environment, the Python
version and the dependencies together.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, in PowerShell:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Homebrew users can run `brew install uv` instead.

### 2. Create the environment and install everything

```bash
uv sync
```

One command does all of it:

- creates `.venv/` in the project root
- downloads Python 3.11 if it is not already present (the version is pinned in
  `.python-version`)
- installs the exact versions from `uv.lock` — not "whatever PyPI ships today"
- installs this project itself, so `import nyc_taxi` works everywhere without
  setting `PYTHONPATH`

No system libraries are required.

### 3. Run things

`uv run` uses `.venv` automatically, so **activation is optional**:

```bash
uv run python -m nyc_taxi.main    # build every dataset
uv run pytest                     # run the test suite
uv run jupyter lab                # open the notebooks
```

If you would rather activate the environment and use plain `python`:

```bash
source .venv/bin/activate         # macOS / Linux
.venv\Scripts\activate            # Windows
```

Then `python -m nyc_taxi.main`, `pytest`, and so on. `deactivate` exits.

### Using an IDE

Point the interpreter at `.venv/bin/python` (`.venv\Scripts\python.exe` on
Windows). PyCharm and VS Code both detect `.venv/` in the project root
automatically. Because the project is installed into the environment, no
"Sources Root" marking or `PYTHONPATH` entry is needed.

### Without uv

`uv.lock` can be exported for plain pip:

```bash
uv export --format requirements-txt --no-hashes > requirements.txt

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The export begins with `-e .`, so that last command installs the project
itself along with its 59 pinned dependencies — no separate step needed.

### Optional extras

XGBoost is an optional dependency, because its wheel does not bundle the
OpenMP runtime it needs:

```bash
uv sync --extra xgb        # plus `brew install libomp` on macOS
```

Without it the other four models still run; `models_factory` raises a message
saying exactly this rather than failing at import.

Kaggle credentials are needed only to download the raw data. Create a
token at *kaggle.com → Settings → API*, save it to `~/.kaggle/kaggle.json`, or
export `KAGGLE_USERNAME` / `KAGGLE_KEY`. Nothing is read from this repository.

OSRM, `distance_utilities.add_route_distance` calls a locally hosted
routing server for true driven distance. It is disabled in the pipeline;
the model uses straight-line haversine. Enabling it means running OSRM with an
NYC extract on `localhost:5001`.

## Engineering

```bash
uv run pytest          # 69 tests
uv run ruff check src tests
uv run mypy --ignore-missing-imports src
```

The test suite is mostly characterisation tests: they pin the exact
columns, dtypes and values the feature pipeline produces, so a refactor that
changes behaviour fails immediately rather than silently shifting a metric.
There is also a guard that imports every module in its own subprocess, because
a circular import only surfaces when the wrong module is imported first.

## Known limitations

- No routed distance. Haversine understates real travel distance; OSRM is
  the most obvious avenue for improvement.
- Model selection used a 300 k subsample. The ranking is fair, every model
  saw identical folds, but the absolute CV figures are slightly pessimistic
  against the full-data result.
- Hyper-parameters are fixed, taken from an earlier randomised search
  (reproducible via `search_hyperparameters`; the spaces live in
  `config/modell_constants.py`) rather than re-tuned.
- This is a random split of the competition's training file, not Kaggle's
  holdout, so the score is not directly leaderboard-comparable.
- 55 rows lose sky-cover information to trailing whitespace and unmapped
  labels in `conditions`, quantified in `weather.ipynb`, left unfixed as it
  cannot move the metric.

---

## Data

### Taxi trips

[Kaggle competition](https://www.kaggle.com/c/nyc-taxi-trip-duration/overview)
· [NYC TLC source](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

`train.csv` holds 1,458,644 trips with no missing values, no duplicate ids and
no duplicated rows.

| Field | Meaning |
|---|---|
| `id` | unique trip identifier |
| `vendor_id` | provider associated with the record |
| `pickup_datetime` / `dropoff_datetime` | when the meter was engaged / disengaged |
| `passenger_count` | passengers, driver-entered |
| `pickup_longitude` / `pickup_latitude` | where the meter was engaged |
| `dropoff_longitude` / `dropoff_latitude` | where the meter was disengaged |
| `store_and_fwd_flag` | `Y` if the record was held in vehicle memory before transmission |
| `trip_duration` | **target**, in seconds |

### Weather

[NYC 2016 Jan–June hourly weather](https://www.kaggle.com/datasets/pschale/nyc-taxi-wunderground-weather)

4,392 rows forming a complete hourly grid, 183 days × 24 hours, no gaps,
covering the entire taxi window with a day to spare. Supplied in imperial
units and converted on ingest: temperature (°F → °C), windspeed (mph → km/h),
precipitation and pressure (inches → mm / hPa), plus relative humidity, a
free-text conditions description, daily precipitation and snow totals, and
fog / rain / snow flags.

The two datasets join on `hour_of_year`.
