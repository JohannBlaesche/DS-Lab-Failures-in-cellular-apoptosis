"""Evaluation step in the pipeline."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split


def evaluate(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> float:
    """Run the evaluation.

    Parameters
    ----------
    model : scikit-learn like model with fit and predict methods.
        The model to evaluate.
    X : pd.DataFrame
        Features.
    y : pd.Series
        Labels.

    Returns
    -------
    mcc : float
        Matthews Correlation Coefficient.
    """
    # implement evaluation here, e.g. 5-fold Cross validation
    # for now, only a simple train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mcc = matthews_corrcoef(y_test, y_pred)
    return mcc
