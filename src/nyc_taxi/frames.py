"""Thin, type-narrowing wrapper around the pandas CSV reader.

``pd.read_csv`` is declared through overloads returning
``DataFrame | TextFileReader`` — the reader appears only when ``chunksize`` or
``iterator`` is passed. This project never streams, so that union is always a
DataFrame in practice, but it propagates: into ``pd.concat``, then into every
function that receives the result.

Reading through :func:`read_frame` states that fact once, in one place,
instead of casting at each call site.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd


def read_frame(path: str | Path) -> pd.DataFrame:
  """Read a CSV as a DataFrame.

  Deliberately takes no ``**kwargs``. Forwarding them would mean either
  accepting ``chunksize``/``iterator`` — which really do return a reader and
  would make the annotation a lie — or declaring the argument as pandas'
  private ``_read_shared`` TypedDict, which an untyped ``dict[str, Any]``
  cannot satisfy. No caller needs the passthrough; anything that eventually
  does should gain an explicit, typed parameter here, or call ``pd.read_csv``
  directly.

  Parameters
  ----------
  path
      CSV to read. Coerced to ``Path`` so the ``read_csv`` overloads resolve;
      an untyped argument matches none of them.
  """
  return cast(pd.DataFrame, pd.read_csv(Path(path)))
