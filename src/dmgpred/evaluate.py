"""Evaluation step in the pipeline."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import matthews_corrcoef


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
    y_pred = model.predict(X)
    mcc = matthews_corrcoef(y, y_pred)
    return mcc
