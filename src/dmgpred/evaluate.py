"""Evaluation step in the pipeline."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    scoring: str | list[str] = "f1_micro",
    **kwargs,
) -> float:
    """Run the evaluation.

    Parameters
    ----------
    model : scikit-learn like model with fit and predict methods.
        The model to evaluate.
    X : pd.DataFrame
        Features.
    y : pd.Series
        Labels.
    n_folds : int, default=5
        Number of folds for cross-validation, by default 5.
    scoring : str or list of str, default="f1_micro"
        Scoring method, by default "f1_micro".

    Returns
    -------
    mcc : float
        Matthews Correlation Coefficient.
    """
    cv = StratifiedKFold(n_splits=n_folds)
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, **kwargs)
    scores = results["test_score"]
    return scores
