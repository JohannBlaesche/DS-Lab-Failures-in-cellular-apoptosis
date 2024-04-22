"""Module for column type conversion."""


def dtype_conversion(X_train):
    """Convert columns types."""
    cat_cols = X_train.select_dtypes(include="object").columns
    binary_cols = [col for col in X_train.columns if col.startswith("has")]
    X_train[binary_cols] = X_train[binary_cols].astype(bool)
    X_train[cat_cols] = X_train[cat_cols].astype("category")
    percentage_cols = [col for col in X_train.columns if col.endswith("percentage")]
    X_train[percentage_cols] = X_train[percentage_cols].astype(float) / 100.0
    id_cols = [col for col in X_train.columns if col.endswith("id")]
    X_train[id_cols] = X_train[id_cols].astype("category")

    return X_train
