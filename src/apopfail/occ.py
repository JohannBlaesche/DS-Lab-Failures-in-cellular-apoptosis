"""One-Class Classification Pipeline."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from apopfail.evaluate import score


def occ(model, X, y, refit=True, random_state=0):
    """Run the one-class classification pipeline.

    Parameters
    ----------
    model: BaseEstimator
        The one-class classification model to use.
    X: array_like
        The input features.
    y: array_like
        The target variable.
    """
    # Split data into normal and abnormal
    X_normal = X.loc[y == 0]
    X_abnormal = X.loc[y == 1]

    train, test, test_labels = split_data_subset(
        X_normal, X_abnormal, random_state=random_state
    )
    model.fit(train)

    scores = score(model, test, test_labels)

    if refit:
        model.fit(X_normal)

    return model, scores


def split_data_contamination(X_normal, X_abnormal, contamination):
    """Split the data into train and test sets with a specified contamination level."""
    # Determine the number of normal samples to match the desired contamination level
    num_abnormal = len(X_abnormal)
    num_normal = int(num_abnormal / contamination)

    # Split normal data into train and test sets
    X_normal_train, X_normal_test = train_test_split(
        X_normal, test_size=num_normal, random_state=0, shuffle=True
    )

    # Create the test set with the specified contamination level
    test = pd.concat([X_normal_test, X_abnormal])
    test_labels = np.concatenate(
        [
            np.zeros(len(X_normal_test), dtype="int8"),
            np.ones(len(X_abnormal), dtype="int8"),
        ],
    )

    return X_normal_train, test, test_labels


def split_data_subset(
    X_normal: pd.DataFrame,
    X_abnormal: pd.DataFrame,
    random_state=0,
    normal_test_size=0.2,
    abnormal_sample_ratio=0.8,
):
    """Split the data into a 80-20 train-test split.

    The test data consists of 80% of all abnormal data and 20%
    of the normal data randomly drawn.
    """
    X_normal_train, X_normal_test = train_test_split(
        X_normal, test_size=normal_test_size, random_state=random_state, shuffle=True
    )
    X_abnormal_test = X_abnormal.sample(
        frac=abnormal_sample_ratio, random_state=random_state
    )

    test = pd.concat([X_normal_test, X_abnormal_test])
    test_labels = np.concatenate(
        [
            np.zeros(len(X_normal_test), dtype="int8"),
            np.ones(len(X_abnormal_test), dtype="int8"),
        ],
    )

    return X_normal_train, test, test_labels
