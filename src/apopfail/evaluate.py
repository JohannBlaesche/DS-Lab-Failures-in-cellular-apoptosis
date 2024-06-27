"""Evaluation step in the pipeline."""

import time

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    get_scorer,
    get_scorer_names,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


def score(model, X, y, pos_label=-1):
    """Score the model on the test set."""
    scoring = _check_scoring(
        {
            "ROC AUC": "roc_auc",
            "Average Precision": make_scorer(
                average_precision_score, pos_label=pos_label
            ),
            "Recall (Sensitivity)": make_scorer(recall_score, pos_label=pos_label),
            "Precision": make_scorer(precision_score, pos_label=pos_label),
            "F2 Score": make_scorer(fbeta_score, beta=2, pos_label=pos_label),
        },
    )
    scores = {}
    for key, scorer in scoring.items():
        score = scorer(model, X, y)
        scores[key] = score
        logger.info(f"{key}: {score:.4f}")
    return scores


def evaluate(
    model: BaseEstimator, X: pd.DataFrame, y: pd.DataFrame, n_folds: int = 5
) -> dict:
    """
    Evaluate the performance of the model using cross-validation or train-test split.

    Parameters
    ----------
        model (BaseEstimator): The model to evaluate.
        X (pd.DataFrame): The input features.
        y (pd.DataFrame): The target variable.
        n_folds (int, optional): The number of folds for cross-validation.
                            Defaults to 5. If the value is 1, train-test split is used.

    Returns
    -------
        dict: A dictionary containing the evaluation results.

    Raises
    ------
        ValueError: If the number of folds is less than 0.

    """
    metrics = {
        "ROC AUC": "roc_auc",
        "Average Precision": make_scorer(average_precision_score, pos_label=1),
        "Recall (Sensitivity)": make_scorer(recall_score, pos_label=1),
        "F2 Score": make_scorer(fbeta_score, beta=2, pos_label=1),
    }
    # for IsolationForest evaluation we need to change the y
    # y = y.map({0: 1, 1: -1})

    scoring = _check_scoring(metrics)

    result_dict = {}
    if n_folds > 1:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        results = cross_validate_custom(model, X, y, cv=cv, scoring=scoring)
        for key in scoring:
            scores = results[f"test_{key}"]
            logger.success(f"{key}: {scores.mean(): .4f} (± {scores.std(): .4f})")
            scores = results[f"train_{key}"]
            logger.success(f"Train {key}: {scores.mean(): .4f} (± {scores.std(): .4f})")
    else:
        results = custom_train_test_split(model, X, y, scoring, train_size=0.8)
        for key in scoring:
            scores = results[f"test_{key}"]
            logger.success(f"{key}: {scores: .4f}")
            scores = results[f"train_{key}"]
            logger.success(f"Train {key}: {scores: .4f}")

    return result_dict


def custom_train_test_split(model, X, y, scoring, train_size=0.8):
    """Perform evaluation with train-test-split."""
    logger.info("Running evaluation with train-test-split.")
    scores = dict()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, stratify=y
    )

    for key, scorer in scoring.items():
        score = scorer(model, X_test, y_test)
        scores[f"test_{key}"] = score
        score_train = scorer(model, X_train, y_train)
        scores[f"train_{key}"] = score_train

    return scores


def cross_validate_custom(model, X, y, cv, scoring):
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
    logger.info("Running evaluation with cross-validation.")
    scores = dict()
    scoring = _check_scoring(scoring)
    for key in scoring:
        scores[f"test_{key}"] = np.zeros(cv.n_splits)
        scores[f"train_{key}"] = np.zeros(cv.n_splits)

    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        start = time.perf_counter()
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)

        for key, scorer in scoring.items():
            score = scorer(model, X_test, y_test)
            scores[f"test_{key}"][i] = score
            score_train = scorer(model, X_train, y_train)
            scores[f"train_{key}"][i] = score_train

        end = time.perf_counter()
        logger.info(
            f"ROC_AUC in Fold {i+1}: {scores['test_ROC AUC'][i]:.4f} (took {end - start:.2f} seconds)"  # noqa: E501
        )

    return scores


def _check_scoring(scoring: dict) -> dict:
    """Check the scoring dict and return a dict with valid scorers."""
    scoring_ = scoring.copy()
    valid_scorers = get_scorer_names()
    for key, value in scoring.items():
        if callable(value):
            continue
        elif isinstance(value, str):
            if value not in valid_scorers:
                raise ValueError(f"Scorer {value} is not a valid scorer.")
            else:
                scoring_[key] = get_scorer(value)
    return scoring_
