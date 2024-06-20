"""Evaluation step in the pipeline."""

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer, get_scorer_names
from sklearn.model_selection import train_test_split


def evaluate(model: BaseEstimator, X: pd.DataFrame, y: pd.DataFrame) -> dict:
    """
    Evaluate the performance of a machine learning model using train-test-split.

    Args:
        model (BaseEstimator): The machine learning model to evaluate.
        X (pd.DataFrame): The input features for evaluation.
        y (pd.DataFrame): The target variable for evaluation.

    Returns
    -------
        dict: A dictionary containing the evaluation scores for train and test sets.

    """
    logger.info("Running evaluation with train-test-split.")
    metrics = {"Average Precision": "average_precision", "ROC AUC": "roc_auc"}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, stratify=y
    )
    scoring = _check_scoring(metrics)

    result_dict = {}

    for key, scorer in scoring.items():
        train_score = scorer(model, X_train, y_train)
        test_score = scorer(model, X_test, y_test)
        logger.success(f"Train-{key}: {train_score}")
        logger.success(f"Test-{key}: {test_score}")
        result_dict[f"Train-{key}"] = train_score
        result_dict[f"Test-{key}"] = test_score

    return result_dict


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
