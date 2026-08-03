"""Tests for the feature-importance helpers in pipelines.training."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nyc_taxi.pipelines.training import (get_feature_names_safe, importance_table,
                                         top_generic_features, top_linreg_features,
                                         top_tree_features)

Dataset = Tuple[pd.DataFrame, pd.Series]


@pytest.fixture(scope="module")
def data() -> Dataset:
  """Two informative-ish numeric columns, one categorical, target driven by 'a'."""
  rng = np.random.default_rng(0)
  x = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200),
                    "c": rng.choice(list("xyz"), 200)})
  y = pd.Series(x["a"] * 2 + rng.normal(scale=0.1, size=200))
  return x, y


@pytest.fixture(scope="module")
def preprocessor() -> ColumnTransformer:
  return ColumnTransformer([
    ("nums", StandardScaler(), ["a", "b"]),
    ("geo", OneHotEncoder(sparse_output=False), ["c"]),
  ])


@pytest.fixture(scope="module")
def ridge_pipe(data: Dataset, preprocessor: ColumnTransformer) -> object:
  x, y = data
  return Pipeline([("preprocessor", preprocessor), ("model", Ridge())]).fit(x, y)


@pytest.fixture(scope="module")
def forest_pipe(data: Dataset, preprocessor: ColumnTransformer) -> object:
  x, y = data
  return Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=10, random_state=0)),
  ]).fit(x, y)


# ------------------------------------------------------------ feature names
def test_feature_names_strip_transformer_prefix(ridge_pipe: Pipeline):
  """sklearn emits 'nums__a'; the importance tables want a bare 'a'.

  Guards the rewrite that replaced a hand-rolled walk over the private
  ``ColumnTransformer._iter`` with the public ``get_feature_names_out``.
  """
  names = get_feature_names_safe(ridge_pipe.named_steps["preprocessor"])
  assert [str(n) for n in names] == ["a", "b", "c_x", "c_y", "c_z"]


def test_feature_names_match_transformed_width(ridge_pipe: Pipeline,
    data: Dataset):
  """One name per output column, or the importance mapping is misaligned."""
  x, _ = data
  pre = ridge_pipe.named_steps["preprocessor"]
  assert len(get_feature_names_safe(pre)) == pre.transform(x).shape[1]


# --------------------------------------------------------- importance table
def test_importance_table_percentages_sum_to_100():
  df = importance_table(["a", "b", "c"], [3.0, 1.0, 0.0], top_n=3)
  assert df["feature"].tolist() == ["a", "b", "c"]
  assert df["rel_importance"].tolist() == [75.0, 25.0, 0.0]
  assert df["cum_importance"].iloc[-1] == 100.0


def test_importance_table_handles_all_zero_values():
  """A degenerate all-zero ranking must not divide by zero."""
  df = importance_table(["a", "b"], [0.0, 0.0], top_n=2)
  assert df["rel_importance"].tolist() == [0.0, 0.0]


def test_importance_table_respects_top_n():
  df = importance_table(list("abcde"), [5.0, 4, 3, 2, 1], top_n=2)
  assert df["feature"].tolist() == ["a", "b"]


# ------------------------------------------------------------ top_* rankings
def test_top_linreg_ranks_the_informative_feature_first(ridge_pipe: Pipeline,
    data: Dataset):
  x, _ = data
  df = top_linreg_features(ridge_pipe, x, top_n=3)
  assert df.loc[0, "feature"] == "a"
  assert "abs_coef" in df.columns


def test_top_tree_ranks_the_informative_feature_first(forest_pipe: Pipeline,
    data: Dataset):
  x, _ = data
  df = top_tree_features(forest_pipe, x, top_n=3)
  assert df.loc[0, "feature"] == "a"


def test_top_tree_returns_a_dataframe_of_top_n_rows(forest_pipe: Pipeline,
    data: Dataset):
  """The return type no longer varies with an argument value."""
  x, _ = data
  df = top_tree_features(forest_pipe, x, top_n=2)
  assert isinstance(df, pd.DataFrame)
  assert len(df) == 2
  assert list(df.columns) == [
    "feature", "importance", "rel_importance", "cum_importance"]


def test_top_generic_uses_coef_path_for_linear_models(ridge_pipe: Pipeline,
    data: Dataset):
  x, y = data
  df = top_generic_features(ridge_pipe, x, y, top_n=3)
  assert "abs_coef" in df.columns  # not the permutation fallback
  assert df.loc[0, "feature"] == "a"


# -------------------------------------------------------- pipeline guarding
def test_require_pipeline_rejects_a_bare_estimator(data: Dataset):
  x, _ = data
  # Built outside the block so only one call inside it can raise; otherwise a
  # TypeError from Ridge() itself would pass the test (sonar python:S5778).
  bare_estimator = Ridge()
  with pytest.raises(TypeError, match="must be a sklearn Pipeline"):
    top_linreg_features(bare_estimator, x)


def test_require_pipeline_rejects_a_wrongly_named_step(data: Dataset,
    preprocessor: ColumnTransformer):
  x, y = data
  wrong = Pipeline([("preprocessor", preprocessor),
                    ("modell", Ridge())]).fit(x, y)
  with pytest.raises(ValueError, match="no step named"):
    top_linreg_features(wrong, x)
