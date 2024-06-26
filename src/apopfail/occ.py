"""One-Class Classification Pipeline."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from apopfail.evaluate import score


def occ(model, X, y):
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
    X_normal = X.loc[y == 1]
    X_abnormal = X.loc[y == -1]

    # Create train-test split with a specified contamination level
    train, test, test_labels = split_data(X_normal, X_abnormal, contamination=0.01)
    model.fit(train)

    score(model, test, test_labels)

    # Fit model on the full normal data
    return model.fit(X_normal)


def split_data(X_normal, X_abnormal, contamination):
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
            np.ones(len(X_normal_test), dtype="int8"),
            -1 * np.ones(len(X_abnormal), dtype="int8"),
        ],
    )

    return X_normal_train, test, test_labels
