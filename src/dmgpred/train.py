"""Training step in the pipeline."""

import json

import pandas as pd
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
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

    with open("./output/lgbm_best_params.json") as f:
        lgbm_params = json.load(f)

    return VotingClassifier(
        estimators=[
            (
                "xgb",
                XGBClassifier(
                    enable_categorical=True,
                    n_estimators=1000,
                    tree_method="hist",
                    device=device,
                    random_state=0,
                ),
            ),
            (
                "catboost",
                CatBoostClassifier(
                    n_estimators=1000,
                    task_type=task_type,
                    verbose=False,
                    random_state=0,
                ),
            ),
            (
                "lgbm",
                LGBMClassifier(**lgbm_params),
            ),
        ],
        voting="soft",
    )


def train(X_train: pd.DataFrame, y_train: pd.DataFrame, use_gpu=True, clf=None):
    """Train the model on the full dataset.

    This model is used to predict the damage grade of the test data.
    A seperate evaluation is done using cross-validation.
    """
    if clf is None:
        clf = get_classifier(X_train, use_gpu=use_gpu)
    pipe = get_pipeline(X_train, clf=clf)
    pipe.fit(X_train, y_train)
    return pipe
