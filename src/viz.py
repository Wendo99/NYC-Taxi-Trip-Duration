"""Model-diagnostic plots: residual distributions, scatters, heatmaps.

Figures are written under ``figures/`` at the repository root. Pass
``save_path=None`` (the default) to display without saving.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import NotFittedError

from constants.path_file_constants import FIGURES_DIR


def unique_path(path) -> Path:
  """Return *path*, or ``name_1.ext``/``name_2.ext``… if it already exists."""
  path = Path(path)
  if not path.exists():
    return path

  for i in range(1, 10_000):
    candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
    if not candidate.exists():
      return candidate
  raise RuntimeError(f"could not find a free filename for {path}")


def _resolve_figure_path(save_path) -> Path:
  """Absolute paths are honoured; bare names land in ``figures/``."""
  save_path = Path(save_path)
  if not save_path.is_absolute():
    save_path = FIGURES_DIR / save_path.name
  save_path.parent.mkdir(parents=True, exist_ok=True)
  return unique_path(save_path)


def _save_if_requested(save_path, what: str) -> None:
  if not save_path:
    return
  target = _resolve_figure_path(save_path)
  plt.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
  print(f"Saved {what} to {target}")


def plot_hist(df, col):
  """Quick distribution check for a single column."""
  plt.figure(figsize=(8, 4))
  df[col].plot.hist(bins=40, edgecolor="black")
  plt.title(f"Distribution of {col.title()}")
  plt.xlabel(col.title())
  plt.ylabel("Frequency")
  plt.grid(True)
  plt.tight_layout()
  plt.show()


def plot_boxplot(df, col: str) -> None:
  """Box plot for *col*, matching case-insensitively if needed."""
  if col not in df.columns:
    matches = [c for c in df.columns if c.lower() == col.lower()]
    if not matches:
      raise KeyError(f"{col!r} not found in DataFrame")
    col = matches[0]

  plt.figure(figsize=(4, 6))
  plt.boxplot(df[col].dropna())
  plt.title(col.replace("_", " ").title())
  plt.xlabel(col.replace("_", " ").title())
  plt.tight_layout()
  plt.show()


def ensure_predictions_and_residuals(df_err, model, X, y, pred_col="y_pred",
    resid_col="residual"):
  """Add prediction and residual columns to *df_err* if they are missing."""
  df = df_err.copy()

  try:
    model.predict(X.iloc[:1] if hasattr(X, "iloc") else X[:1])
  except NotFittedError as exc:
    raise RuntimeError(
        "Model is not fitted. Fit it before computing predictions.") from exc
  except Exception:
    # Any other failure here is incidental; the real call below will surface it.
    pass

  if pred_col not in df.columns:
    preds = np.asarray(model.predict(X)).ravel()
    df[pred_col] = pd.Series(preds, index=X.index)

  if resid_col not in df.columns:
    y_series = y.copy() if isinstance(y, pd.Series) else pd.Series(y,
                                                                   index=X.index)
    df[resid_col] = df[pred_col] - y_series

  return df


def plot_residual_distribution(df_err, model_name, save_path=None):
  """Distribution of residuals, to assess bias and spread."""
  plt.figure(figsize=(8, 5))
  sns.histplot(df_err["residual"], bins=50, kde=True, color="steelblue")

  plt.axvline(0, color="red", linestyle="--", linewidth=1.2,
              label="Zero Residual")
  plt.title(f"Residual Distribution – {model_name}", fontsize=14,
            weight="bold")
  plt.xlabel("Residual (log-seconds)")
  plt.ylabel("Frequency")
  plt.legend()
  plt.grid(alpha=0.3, linestyle="--")
  plt.tight_layout()

  _save_if_requested(save_path, "residual distribution")
  plt.show()


def plot_residual_scatter(df_err, model_name, save_path=None):
  """Residuals against predictions, to detect heteroscedasticity or bias."""
  plt.figure(figsize=(8, 5))
  sns.scatterplot(data=df_err, x="y_pred", y="residual", alpha=0.4,
                  edgecolor=None, color="royalblue")

  plt.axhline(0, color="red", linestyle="--", linewidth=1.2,
              label="Zero Residual")
  plt.title(f"Residuals vs Predicted Values – {model_name}", fontsize=14,
            weight="bold")
  plt.xlabel("Predicted log(Trip Duration)")
  plt.ylabel("Residual (Predicted - Actual)")
  plt.legend()
  plt.grid(alpha=0.3, linestyle="--")
  plt.tight_layout()

  _save_if_requested(save_path, "residual scatter")
  plt.show()


def plot_residual_heatmap(df_err, model_name, x_col="dist_bin",
    y_col="hour_of_day", save_path=None):
  """Mean residual across two binned dimensions."""
  pivot_table = (
    df_err
    .groupby([y_col, x_col])
    .agg(mean_residual=("residual", "mean"))
    .reset_index()
    .pivot(index=y_col, columns=x_col, values="mean_residual")
  )

  plt.figure(figsize=(9, 6))
  sns.heatmap(pivot_table, cmap="RdYlBu_r", center=0, linewidths=0.5,
              cbar_kws={"label": "Mean Residual (log-seconds)"})

  plt.title(f"Residual Heatmap – {model_name}", fontsize=14, weight="bold")
  plt.xlabel("Distance Bin (binned log-distance)")
  plt.ylabel("Hour of Day")
  plt.xticks(rotation=0)
  plt.yticks(rotation=0)
  plt.tight_layout()

  _save_if_requested(save_path, "residual heatmap")
  plt.show()


def show_corr_matrix(mask, corr, save_path=None):
  """Lower-triangle Pearson correlation matrix of the numeric features."""
  sns.set_theme(style="white")
  sns.set_context("talk")
  plt.figure(figsize=(12, 10))
  ax = sns.heatmap(
      corr,
      mask=mask,
      cmap="coolwarm",
      vmin=-1,
      vmax=1,
      center=0,
      annot=True,
      fmt=".2f",
      annot_kws={"size": 9},
      linewidths=0.5,
      linecolor="gray",
      cbar_kws={"shrink": 0.75, "label": "Pearson r"},
      square=False,
  )
  ax.set_title("Correlation Matrix of Numeric Features", fontsize=18,
               fontweight="bold", pad=16)
  ax.set_xlabel("Features", fontsize=12, labelpad=10)
  ax.set_ylabel("Features", fontsize=12, labelpad=10)

  plt.xticks(rotation=45, ha="right", fontsize=10)
  plt.yticks(rotation=0, fontsize=10)
  plt.tight_layout()

  _save_if_requested(save_path, "correlation matrix")
  plt.show()
