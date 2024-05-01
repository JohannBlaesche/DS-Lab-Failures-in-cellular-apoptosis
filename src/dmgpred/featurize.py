"""Featurization step in the pipeline."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
    X = encode_categoricals(X)
    return X


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns using one-hot encoding."""
    object_cols = X.select_dtypes(include="object").columns.to_numpy()
    object_cols = np.append(object_cols, "count_families")
    for col in object_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    return X
