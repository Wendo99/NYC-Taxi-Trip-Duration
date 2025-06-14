from sklearn.model_selection import train_test_split


def split_train_test(df, test_size, random_state, stratify_col=None):
  """
  Split a DataFrame into training and test sets.

  Parameters:
      df (DataFrame): The input DataFrame.
      test_size (float): Proportion for test split.
      random_state (int): Random seed.
      stratify_col (str, optional): Column name to stratify on.

  Returns:
      DataFrame, DataFrame: train_set, test_set
  """
  return train_test_split(
      df, test_size=test_size, random_state=random_state,
      stratify=df[stratify_col] if stratify_col else None
  )
