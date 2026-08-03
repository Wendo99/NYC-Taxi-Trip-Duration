"""Tests for the type-narrowing pandas reader wrapper."""
from __future__ import annotations

import pandas as pd
import pytest

from nyc_taxi.frames import read_frame


@pytest.fixture
def csv(tmp_path):
  path = tmp_path / "sample.csv"
  path.write_text("a,b\n1,2\n3,4\n")
  return path


def test_returns_a_dataframe(csv):
  """The whole point: never a TextFileReader."""
  frame = read_frame(csv)
  assert isinstance(frame, pd.DataFrame)
  assert frame.shape == (2, 2)
  assert frame["a"].tolist() == [1, 3]
  assert frame.columns.tolist() == ["a", "b"]


def test_accepts_a_string_path(csv):
  assert read_frame(str(csv)).shape == (2, 2)


def test_accepts_a_path_object(csv):
  assert read_frame(csv).shape == (2, 2)


def test_takes_no_reader_options(csv):
  """No **kwargs, so chunksize can never break the DataFrame guarantee."""
  with pytest.raises(TypeError):
    read_frame(csv, chunksize=1)


def test_missing_file_raises(tmp_path):
  with pytest.raises(FileNotFoundError):
    read_frame(tmp_path / "does_not_exist.csv")
