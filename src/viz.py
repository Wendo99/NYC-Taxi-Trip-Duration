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
