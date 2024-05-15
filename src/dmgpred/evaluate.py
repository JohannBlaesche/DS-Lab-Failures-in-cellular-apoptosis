"""Evaluation step in the pipeline."""

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    f1_score,
    get_scorer,
    get_scorer_names,
    make_scorer,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold

from dmgpred.train import train


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
    train_scores = True
    results = cross_validate_custom(
        model, X, y, cv=cv, scoring=scoring, train_scores=train_scores
    )
    for key in scoring:
        scores = results[f"test_{key}"]
        logger.success(f"{key}: {scores.mean(): .4f} (± {scores.std(): .2f})")
        if train_scores:
            scores = results[f"train_{key}"]
        logger.success(f"Train {key}: {scores.mean(): .4f} (± {scores.std(): .2f})")

    return results


def cross_validate_custom(model, X, y, cv, scoring, train_scores=False):
    """
    Perform cross-validation with given scoring metrics.

    Parameters
    ----------
    model : scikit-learn like model with fit and predict methods.
        The model to evaluate.
    X : pd.DataFrame
        Features.
    y : pd.Series
        Labels.
    cv : cross-validation generator
        A cross-validation iterator with `split` method.
    scoring : dict
        A dictionary where keys are strings representing scoring metric names and values
        are either strings representing predefined scorer names in scikit-learn's
        'metrics' module or scorer functions.

    Returns
    -------
    scores : dict
        A dictionary containing the scores for each fold and each scoring metric.
        Keys are strings in the format 'test_<scoring_metric>', where <scoring_metric>
        is the name of the scoring metric. Values are arrays of shape (n_splits,)
        containing the scores for each fold.
    """
    scores = dict()
    for key in scoring:
        scores[f"test_{key}"] = np.zeros(cv.n_splits)
        if train_scores:
            scores[f"train_{key}"] = np.zeros(cv.n_splits)

    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        start = time.perf_counter()
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        for key in scoring:
            scoring_arr = scores[f"test_{key}"]
            if train_scores:
                scoring_arr_train = scores[f"train_{key}"]
            scorer = scoring[key]
            if scorer in get_scorer_names():
                scorer_func = get_scorer(scorer)
                score = scorer_func(model, X_test, y_test)
                if train_scores:
                    score_train = scorer_func(model, X_train, y_train)
            else:
                score = scorer(model, X_test, y_test)
                if train_scores:
                    score_train = scorer(model, X_train, y_train)
            scoring_arr[i] = score
            if train_scores:
                scoring_arr_train[i] = score_train
        end = time.perf_counter()
        logger.info(f"Split {i} trained in{end - start: .2f} seconds.")

    return scores


def parallel_cv(X, y, cv):
    """
    Attempt to parallelize the cross validation loop.

    Not yet working.
    """
    scores = dict()

    def train_model(train_index, test_index):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # model.fit(X_train, y_train) #this is where an error occurs
        model = train(X_train, y_train)  # alternative, but same error
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="micro")
        mcc = matthews_corrcoef(y_test, y_pred)

        return dict(f1=f1, mcc=mcc)

    out = Parallel(n_jobs=5)(
        delayed(train_model)(train_index, test_index)
        for i, (train_index, test_index) in enumerate(cv.split(X, y))
    )

    f1_scores = [d["f1"] for d in out]
    mcc = [d["mcc"] for d in out]
    scores["test_MCC"] = f1_scores
    scores["test_F1 Micro"] = mcc
