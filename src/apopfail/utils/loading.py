"""Data loading utilities."""

import pandas as pd


def load_data(root="."):
    """Load the data.

    Parameters
    ----------
    root : str, optional
        The root directory, by default "."

    """
    TEST_VALUES_PATH = f"{root}/data/test_data_p53_mutant.parquet"
    TRAIN_VALUES_PATH = f"{root}/data/train_set_p53mutant.parquet"
    TRAIN_LABELS_PATH = f"{root}/data/train_labels_p53mutant.csv"

    X_test = pd.read_parquet(TEST_VALUES_PATH)
    X_train = pd.read_parquet(TRAIN_VALUES_PATH)
    y_train = pd.read_csv(TRAIN_LABELS_PATH, index_col=0, names=["target"], skiprows=1)[
        "target"
    ]
    y_train = y_train.map({"inactive": 0, "active": 1}).astype("int8")
    return X_train, X_test, y_train
