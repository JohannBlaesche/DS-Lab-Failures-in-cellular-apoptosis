"""Training step in the pipeline."""

import pandas as pd
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from dmgpred.cleaning import get_normalizer
from dmgpred.featurize import get_encoder


def get_pipeline(X: pd.DataFrame, clf=None):
    """Return the training pipeline."""
    if clf is None:
        clf = DummyClassifier(strategy="most_frequent")
    normalizer = get_normalizer()
    encoder = get_encoder(X)
    return Pipeline(
        [
            ("normalizer", normalizer),
            ("encoder", encoder),
            ("clf", clf),
        ],
        verbose=False,
    )


def train(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Train the model on the full dataset.

    This model is used to predict the damage grade of the test data.
    A seperate evaluation is done using cross-validation.
    """
    clf = VotingClassifier(
        estimators=[
            ("xgb", XGBClassifier()),
            ("lgbm", LGBMClassifier()),
            ("knn", KNeighborsClassifier()),
        ],
        voting="soft",
    )

    pipe = get_pipeline(X_train, clf=clf)
    pipe.fit(X_train, y_train)
    return pipe
