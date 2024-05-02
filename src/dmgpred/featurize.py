"""Featurization step in the pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def featurize(*dataframes):
    """Run the featurization.

    Parameters
    ----------
    train_values : pd.DataFrame
        Training features.
    test_values : pd.DataFrame
        Test features.

    Returns
    -------
    list of pd.DataFrame
        DataFrames with featurization applied.
    """
    featurized = []
    for df in dataframes:
        X = df.copy()
        X = featurize_single(X)
        featurized.append(X)

    return featurized


def featurize_single(X: pd.DataFrame) -> pd.DataFrame:
    """Run the featurization on the given dataframe.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe with features.

    Returns
    -------
    pd.DataFrame
        DataFrame with featurization applied.
    """
    # X = encode_categoricals(X)
    return X


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns using one-hot encoding."""
    object_cols = X.select_dtypes(include="category").columns
    # object_cols = np.append(object_cols, "count_families")
    for col in object_cols:
        le = OneHotEncoder()
        X[col] = le.fit_transform(X[col])
    return X


def get_encoder(X: pd.DataFrame):
    """Return the categorical encoder for the pipeline."""
    nominal_cols = X.select_dtypes(include="category").columns
    nominal_cols = nominal_cols.difference(["count_families"])
    return ColumnTransformer(
        transformers=[
            (
                "nominal",
                OneHotEncoder(sparse_output=False),
                make_column_selector(dtype_include="category"),
            ),
            ("ordinal", OrdinalEncoder(), ["count_families"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
