"""Evaluation utilities."""

from loguru import logger
from sklearn.metrics import make_scorer, matthews_corrcoef

from dmgpred.evaluate import _check_scoring
from dmgpred.train import get_pipeline


def evaluate_clf(
    clf, X_train, y_train, X_test, y_test, additional_scoring=None, **fit_kwargs
):
    """Evaluate the classifier on the given test data."""
    if additional_scoring is None:
        additional_scoring = {}
    scoring = {
        "F1 Micro": "f1_micro",
        "MCC": make_scorer(matthews_corrcoef, greater_is_better=True),
        **additional_scoring,
    }
    scoring = _check_scoring(scoring)
    model = get_pipeline(X_test, clf)
    model.fit(X_train, y_train, **fit_kwargs)
    scores = {}
    for name, scorer in scoring.items():
        scores[name] = scorer(model, X_test, y_test)
        logger.info(f"{name}: {scores[name]:.4f}")
    return clf, scores
