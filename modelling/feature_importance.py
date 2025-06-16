import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_lin_feature_importance(pipeline, X_train, y_train,
    model_step='model', pre_step='pre'):
  """
  Fit a linear model pipeline and plot the top 20 features based on absolute coefficient values.

  Parameters:
  - pipeline: a scikit-learn Pipeline object containing a linear model
  - X_train: training feature matrix
  - y_train: target vector
  - model_step: name of the model step in the pipeline (default: 'model')
  - pre_step: name of the preprocessing step in the pipeline (default: 'pre')

  Returns:
  - feat_df: DataFrame with feature names, importance (abs. coef), and sign
  """
  pipeline.fit(X_train, y_train)

  model = pipeline.named_steps.get(model_step)
  if model is None or not hasattr(model, 'coef_') or not isinstance(model.coef_,
                                                                    np.ndarray):
    raise AttributeError(
        f"Model step '{model_step}' does not provide valid 'coef_' attribute.")
  coefs = model.coef_
  if coefs.ndim > 1:
    coefs = coefs[0]

  feature_names = pipeline.named_steps[pre_step].get_feature_names_out()

  feat_df = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(coefs),
    'sign': np.sign(coefs)
  }).sort_values(by='importance', ascending=False)

  plt.figure(figsize=(10, 6))
  sns.barplot(x='importance', y='feature', data=feat_df.head(20))
  plt.title("Top 20 feature coefficients (absolute)")
  plt.tight_layout()
  plt.show()
  return feat_df


# Tree-based feature importance plotting

def plot_tree_feature_importance(pipeline, X_train, y_train,
    model_step='model', pre_step='pre'):
  """
  Fit a tree-based model pipeline and plot the top N features based on feature importance.

  Parameters:
  - pipeline: a scikit-learn Pipeline object with a tree-based model as final step
  - model_step: name of the model step in the pipeline (default: 'model')
  - pre_step: name of the preprocessing step in the pipeline (default: 'pre')
  - top_n: number of top features to display (default: 20)

  Returns:
  - feat_df: DataFrame with feature names and importance scores
  """

  pipeline.fit(X_train, y_train)

  model = pipeline.named_steps.get(model_step)
  if model is None or not hasattr(model, 'feature_importances_'):
    raise AttributeError(
        f"Model step '{model_step}' does not provide valid 'feature_importances_'.")

  feature_names = pipeline.named_steps[pre_step].get_feature_names_out()
  importances = model.feature_importances_

  feat_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
  }).sort_values(by='importance', ascending=False)

  plt.figure(figsize=(10, 6))
  sns.barplot(x='importance', y='feature', data=feat_df.head(20))
  plt.title(f"Top {top_n} feature importances")
  plt.tight_layout()
  plt.show()
  return feat_df
