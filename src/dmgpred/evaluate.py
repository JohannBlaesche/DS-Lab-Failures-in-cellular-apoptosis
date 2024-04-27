"""Evaluation step in the pipeline."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    additional_scoring: dict | None = None,
    verbose=True,
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
    additional_scoring : dict, optional
        Additional scoring metrics to compute, by default None.
        Must be a dict of str, str or str, callable.

    Returns
    -------
    mcc : float
        Matthews Correlation Coefficient.
    """
    cv = StratifiedKFold(n_splits=n_folds)

    if additional_scoring is None:
        additional_scoring = {}

    scoring = {
        "F1 Micro": "f1_micro",
        "MCC": make_scorer(matthews_corrcoef, greater_is_better=True),
        **additional_scoring,
    }

    results = cross_validate(model, X, y, cv=cv, scoring=scoring, **kwargs)

    if verbose:
        for key in scoring:
            scores = results[f"test_{key}"]
            print(f"{key}: {scores.mean(): .4f} (± {scores.std(): .2f})")

    return results
