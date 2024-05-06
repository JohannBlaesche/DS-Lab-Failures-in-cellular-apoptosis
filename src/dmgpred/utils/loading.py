"""Loading utilities."""

import pandas as pd

from dmgpred.cleaning import clean
from dmgpred.featurize import featurize


def load_data(data_dir: str, processed=False) -> dict:
    """Load the data from the given directory.

    Parameters
    ----------
    data_dir : str
        The directory containing the data files.
    processed : bool
        Whether the data should be cleaned and featurized.
    """
    X_train = pd.read_csv(f"{data_dir}/train_values.csv", index_col="building_id")
    X_test = pd.read_csv(f"{data_dir}/test_values.csv", index_col="building_id")

    y_train = pd.read_csv(f"{data_dir}/train_labels.csv", index_col="building_id")
    y_train = y_train["damage_grade"]

    if processed:
        X_train, X_test = clean(X_train, X_test)
        X_train, X_test = featurize(X_train, X_test)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
    }
