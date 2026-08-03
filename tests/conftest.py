"""Make ``src/`` importable during the refactor.

This shim exists only while the project is not yet installable as a package.
Once ``pyproject.toml`` declares a build backend and the code lives in
``src/nyc_taxi/``, this file can be deleted and the tests import the package
directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
  sys.path.insert(0, str(SRC))
