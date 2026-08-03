"""Guard against circular imports.

A cycle only shows up when the module at the *wrong* end is imported first,
so importing everything inside one interpreter can pass by luck of ordering.
Each module therefore gets its own fresh subprocess.

This exists because ``constants.weather_constants`` used to import
``OrdinalScale`` from ``utilities.weather_utilities`` while the latter
imported the former, a cycle papered over by ten function-local imports.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

MODULES = sorted(
    str(p.relative_to(SRC)).removesuffix(".py").replace("/", ".")
    for p in SRC.rglob("*.py")
    # main.py runs the whole pipeline when executed, but importing it is safe
    # thanks to its __main__ guard; it is covered like everything else.
)


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_standalone(module: str):
  """Importing any module first, in a clean interpreter, must succeed."""
  # check=False on purpose: a non-zero exit is the thing under test, and the
  # assertion below reports the child's stderr, which CalledProcessError would
  # swallow.
  result = subprocess.run(
      [sys.executable, "-c", f"import {module}"],
      cwd=SRC, capture_output=True, text=True, timeout=120, check=False,
  )
  assert result.returncode == 0, (
      f"`import {module}` failed in a fresh interpreter:\n{result.stderr}")
