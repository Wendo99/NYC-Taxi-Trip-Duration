import folium
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from shapely.geometry.multipoint import MultiPoint
from sklearn.exceptions import NotFittedError

from pipelines.training import rmse


def plot_hist(df, col):
  plt.figure(figsize=(8, 4))
  df[col].plot.hist(bins=40, edgecolor='black')
  plt.title(f'Distribution of {col.title()}')
  plt.xlabel(col.title())
  plt.ylabel("Frequency")
  plt.grid(True)
  plt.tight_layout()
  plt.show()


def plot_boxplot(df, col: str) -> None:
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


import os


def unique_path(path: str) -> str:
  directory, fname = os.path.split(path)
  base, ext = os.path.splitext(fname)

  if not os.path.exists(path):
    return path

  i = 1
  while True:
    new_fname = f"{base}_{i}{ext}"
    new_path = os.path.join(directory, new_fname)
    if not os.path.exists(new_path):
      return new_path
    i += 1


def plot_residuals(df_err, model_name: str) -> None:
  err_nonzero = df_err["abs_res"][df_err["abs_res"] > 0]

  sns.histplot(err_nonzero, bins=50)
  plt.yscale("log")
  plt.xlabel("|Residual| (log-seconds)")
  plt.ylabel("frequency (log-Y)")
  plt.title(model_name + " dist |Residual|, LinReg – Log-Y")

  save_map(plt, model_name, "hist")
  plt.show()


def plot_residual_scatter(df_err, model_name: str) -> None:
  plt.figure(figsize=(6, 4))
  sns.scatterplot(x=df_err["y_pred_log"],
                  y=df_err["residual"],
                  alpha=.15, s=8)
  plt.axhline(0, color="red", lw=1)
  plt.xlabel("ŷ  (log-seconds)")
  plt.ylabel("Residual  (log-seconds)")
  plt.title(model_name + " – residual scatter")
  plt.tight_layout()
  save_map(plt, model_name, "scatter")
  plt.show()


def plot_residual_heatmap(df_err, model_name: str, x_col, y_col) -> None:
  pivot = (df_err
           .groupby([y_col, x_col])
           .apply(lambda g: rmse(g["y_true_log"], g["y_pred_log"]))
           .unstack())

  vmax = pivot.quantile(0.95).max()
  plt.figure(figsize=(12, 6))
  sns.heatmap(pivot, cmap="rocket_r", vmin=0, vmax=vmax)
  plt.title(model_name + " RMSE Heatmap – " + y_col + "×" + x_col)
  plt.xlabel(x_col)
  plt.ylabel(y_col)
  save_map(plt, model_name, "heat", x_col, y_col)
  plt.show()


import pathlib, matplotlib.pyplot as plt, seaborn as sns


def save_map(plt, model_name, map_name, x_col="", y_col=""):
  out_dir = pathlib.Path("../figures")
  out_dir.mkdir(parents=True, exist_ok=True)

  output_path = f"{model_name.lower()}_res_" + map_name + (f""
                                                           f"_{x_col}"
                                                           f"_{y_col}.png")

  safe_path = unique_path(out_dir / output_path)

  plt.savefig(
      safe_path,
      dpi=600,
      bbox_inches="tight",
      facecolor="white",
  )
  print(f"Saved figure to {safe_path}")


def cluster_map(df, cluster_type):
  df_z = df.sample(n=50000, random_state=42)

  m = folium.Map(
      location=[40.75, -73.97],
      zoom_start=11,
      tiles="CartoDB positron"
  )

  cluster_ids = sorted(df_z[cluster_type + "_cluster_hdb"].unique())
  cmap = plt.colormaps["Set2"].resampled(len(cluster_ids))

  palette = [mcolors.to_hex(cmap(i)) for i in range(len(cluster_ids))]
  color_map = {cid: palette[i] for i, cid in enumerate(cluster_ids)}

  for cid in cluster_ids:
    pts = df_z[df_z[cluster_type + "_cluster_hdb"] == cid][
      cluster_type + ["_longitude", cluster_type + "_latitude"]].values
    if len(pts) < 3:
      continue
  hull = MultiPoint(pts).convex_hull
  hull_coords = [(lat, lon) for lon, lat in hull.exterior.coords]
  folium.Polygon(
      locations=hull_coords,
      color=color_map[cid],
      weight=2,
      fill=True,
      fill_color=color_map[cid],
      fill_opacity=0.2,
      popup=f"Cluster {cid}"
  ).add_to(m)

  centroids = df.groupby(cluster_type + "_cluster_hdb")[
    [cluster_type + "_latitude", cluster_type + "_longitude"]].mean()
  for cid, row in centroids.iterrows():
    folium.CircleMarker(
        location=(row[cluster_type + "_latitude"],
                  row[cluster_type + "_longitude"]),
        radius=6,
        color="black",
        fill=True,
        fill_color=color_map[cid],
        fill_opacity=1.0,
        popup=f"Centroid {cid}"
    ).add_to(m)

  m

  m.save("../figures/" + cluster_type + "_clusters_hdb.html")
  png = m._to_png(5)
  with open("../figures/" + cluster_type + "_clusters_hdb.png", "wb") as f:
    f.write(png)

def plot_residual_distribution(df_err, model_name, save_path=None):
  """
  Plots the distribution of residuals to assess bias and variance.
  """
  plt.figure(figsize=(8, 5))
  sns.histplot(df_err['residual'], bins=50, kde=True, color='steelblue')

  plt.axvline(0, color='red', linestyle='--', linewidth=1.2,
              label="Zero Residual")
  plt.title(f"Residual Distribution – {model_name}", fontsize=14,
            weight="bold")
  plt.xlabel("Residual (log-seconds)")
  plt.ylabel("Frequency")
  plt.legend()
  plt.grid(alpha=0.3, linestyle='--')
  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved residual distribution to {save_path}")
  plt.show()

def ensure_predictions_and_residuals(df_err, model, X, y, pred_col='y_pred',
    resid_col='residual'):
  """
  Ensure df_err has prediction and residual columns. Returns updated df_err.
  """
  df = df_err.copy()

  try:
    _ = model.predict(X.iloc[:1]) if hasattr(X, "iloc") else model.predict(
        X[:1])
  except NotFittedError:
    raise RuntimeError(
      "Model is not fitted. Fit it before computing predictions.")
  except Exception:

    pass

  if pred_col not in df.columns:
    preds = model.predict(X)
    preds = np.asarray(preds).ravel()
    df[pred_col] = pd.Series(preds, index=X.index)

  if resid_col not in df.columns:
    # ensure y is a Series aligned to X
    y_series = y.copy() if isinstance(y, pd.Series) else pd.Series(y,
                                                                   index=X.index)
    df[resid_col] = df[pred_col] - y_series

  return df

def plot_residual_scatter(df_err, model_name, save_path=None):
  """
  Plots residuals against predicted values to detect heteroscedasticity or bias.
  """
  plt.figure(figsize=(8, 5))
  sns.scatterplot(
      data=df_err,
      x="y_pred",
      y="residual",
      alpha=0.4,
      edgecolor=None,
      color='royalblue'
  )

  plt.axhline(0, color='red', linestyle='--', linewidth=1.2,
              label="Zero Residual")
  plt.title(f"Residuals vs Predicted Values – {model_name}", fontsize=14,
            weight="bold")
  plt.xlabel("Predicted log(Trip Duration)")
  plt.ylabel("Residual (Predicted - Actual)")
  plt.legend()
  plt.grid(alpha=0.3, linestyle='--')
  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved residual scatter to {save_path}")
  plt.show()

def plot_residual_heatmap(df_err, model_name, x_col='dist_bin',
    y_col='hour_of_day', save_path=None):
  """
  Visualizes mean residuals as a 2D heatmap across distance and time-of-day bins.
  """

  pivot_table = (
    df_err
    .groupby([y_col, x_col])
    .agg(mean_residual=('residual', 'mean'))
    .reset_index()
    .pivot(index=y_col, columns=x_col, values='mean_residual')
  )

  plt.figure(figsize=(9, 6))
  sns.heatmap(
      pivot_table,
      cmap="RdYlBu_r",
      center=0,
      linewidths=0.5,
      cbar_kws={'label': 'Mean Residual (log-seconds)'}
  )

  plt.title(f"Residual Heatmap – {model_name}", fontsize=14, weight="bold")
  plt.xlabel("Distance Bin (binned log-distance)")
  plt.ylabel("Hour of Day")
  plt.xticks(rotation=0)
  plt.yticks(rotation=0)
  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved heatmap to {save_path}")
  plt.show()


def show_corr_matrix(mask,corr ):
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
  plt.show()

