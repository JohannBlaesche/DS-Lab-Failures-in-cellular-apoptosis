"""Featurization step in the pipeline."""

import pandas as pd
from category_encoders.james_stein import JamesSteinEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder


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


def get_encoder(X: pd.DataFrame):
    """Return the categorical encoder for the pipeline."""
    # cannot use make_column_selector because we need to exclude count_families
    nominal_cols = X.select_dtypes(include=["category", "object"]).columns
    ordinal_cols = ["count_families"]
    nominal_cols = nominal_cols.difference(ordinal_cols)
    return ColumnTransformer(
        transformers=[
            (
                "nominal",
                JamesSteinEncoder(),
                nominal_cols,
            ),
            ("ordinal", OrdinalEncoder(), ordinal_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def _reverse_multi_hot_encoding(row, cols: list[str], sep="_", prefix_len=2):
    """Reverse the multi-hot encoding of the given cols."""
    suffixes = []
    for col in cols:
        suffix = col.split(sep)[prefix_len:]
        suffix = sep.join(suffix)
        if row[col] == 1:
            suffixes.append(suffix)
    return ",".join(suffixes)


def reverse_superstructure_encoding(X: pd.DataFrame):
    """Reverse the multi-hot encoding of superstructure columns."""
    prefix = "has_superstructure"
    prefix_len = 2
    sep = "_"
    cols = [col for col in X.columns if col.startswith(prefix) and col != prefix]
    X = X.assign(
        has_superstructure=X.apply(
            _reverse_multi_hot_encoding, axis=1, args=(cols, sep, prefix_len)
        )
    )

    # drop columns
    X = X.drop(cols, axis=1)
    return X
