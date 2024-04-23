"""Featurization step in the pipeline."""

import pandas as pd


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
    return X
