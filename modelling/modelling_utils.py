def feature_to_category(df, features):
  for col in features:
    df[col] = df[col].astype('category')
  return df


def feature_as_bool(df, features):
  for col in features:
    df[col] = df[col].astype(bool)
  return df
