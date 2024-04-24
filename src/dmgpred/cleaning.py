"""Cleaning step in the pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer


def clean(*dataframes):
    """Run the cleaning.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        DataFrames to clean.

    Returns
    -------
    list of pd.DataFrame
        Cleaned DataFrames.
    """
    cleaned = []
    for df in dataframes:
        X = df.copy()
        X = clean_single(X)
        cleaned.append(X)

    return cleaned


def clean_single(X: pd.DataFrame) -> pd.DataFrame:
    """Clean the given dataframe.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    X = remove_outliers(X)
    return X


def remove_outliers(X: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers from X."""
    # remove outliers in age column by removing rows with age > 99th percentile
    age_threshold = X["age"].quantile(0.99)
    X = X[X["age"] <= age_threshold]
    return X


def get_normalization_pipeline():
    """Get the normalization pipeline."""
    normalizer = ColumnTransformer(
        transformers=[
            (
                "box-cox",
                PowerTransformer(method="box-cox"),
                ["area_percentage", "height_percentage"],
            )
        ],
        remainder="passthrough",
    )
    return normalizer
