"""Cleaning step in the pipeline."""

import pandas as pd


def clean(*dataframes):
    """Run the cleaning.

    Parameters
    ----------
    train_values : pd.DataFrame
        Training features.
    train_labels : pd.DataFrame
        Training labels.

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
    return X
