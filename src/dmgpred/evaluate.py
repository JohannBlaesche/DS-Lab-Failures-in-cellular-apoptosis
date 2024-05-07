"""Evaluation step in the pipeline."""

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    additional_scoring: dict | None = None,
    **kwargs,
) -> float:
    """Evaluate the model using cross-validation.

    The given model is evaluated with cross-validation, using a
    stratified KFold with the given number of folds. The Matthews
    Correlation Coefficient as well as F1 Micro are reported by default.
    This function can also compute additional scoring metrics passed via
    the additional_scoring parameter. The results are printed to the console
    if verbose is set to True.

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
        Must be a dict of str, str or dict of str, callable.
        If a dict of str, str is passed, the value must match a name
        of a sklearn metric. If a dict of str, callable is passed, the
        callable must be a valid scoring function, e.g. one created with
        `make_scorer` from sklearn.metrics.

    Returns
    -------
    results : dict
        The results dict of sklearn.model_selection.cross_validate.
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

    for key in scoring:
        scores = results[f"test_{key}"]
        logger.success(f"{key}: {scores.mean(): .4f} (± {scores.std(): .2f})")

    return results
