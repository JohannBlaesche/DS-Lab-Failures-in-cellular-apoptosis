"""Training step in the pipeline."""

import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
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
            ("oversampler", SMOTE()),
            ("clf", clf),
        ],
        verbose=False,
    )


def get_classifier(X_train: pd.DataFrame, use_gpu=True):
    """Return the classifier used in the pipeline."""
    if use_gpu:
        task_type = "GPU"
        device = "cuda"
    else:
        task_type = "CPU"
        device = "cpu"

    return VotingClassifier(
        estimators=[
            (
                "xgb",
                XGBClassifier(
                    enable_categorical=True,
                    n_estimators=1000,
                    tree_method="hist",
                    device=device,
                ),
            ),
            (
                "catboost",
                CatBoostClassifier(
                    n_estimators=1000,
                    task_type=task_type,
                    verbose=False,
                ),
            ),
        ],
        voting="soft",
    )


def train(X_train: pd.DataFrame, y_train: pd.DataFrame, use_gpu=True):
    """Train the model on the full dataset.

    This model is used to predict the damage grade of the test data.
    A seperate evaluation is done using cross-validation.
    """
    clf = get_classifier(X_train, use_gpu=use_gpu)
    pipe = get_pipeline(X_train, clf=clf)
    pipe.fit(X_train, y_train)
    return pipe
