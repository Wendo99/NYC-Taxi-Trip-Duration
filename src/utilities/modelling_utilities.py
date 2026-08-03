"""Feature groupings used when breaking model error down by segment."""
from __future__ import annotations

from constants.features_constants import FEATURES, GEO_DROP, GEO_PICK, \
  RES_TABLE_EXCLUDE_FEATURES


def fill_res_col(features: dict, exclude: set | None = None) -> set:
  """Return the feature keys to group residuals by.

  Starts from *features*, adds the geo-cluster columns, then removes anything
  in *exclude* (features that are either the target's near-duplicate or too
  high-cardinality to make a useful error table).

  Args:
      features: Mapping of feature name -> (group, flag).
      exclude: Feature names to leave out.
  """
  result = set(features.keys())
  result.update(GEO_PICK)
  result.update(GEO_DROP)
  if exclude:
    result.difference_update(exclude)
  return result


RES_COL = fill_res_col(FEATURES, exclude=set(RES_TABLE_EXCLUDE_FEATURES))
