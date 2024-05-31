"""Training step in the pipeline."""

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
        lgbm_device = "gpu"
    else:
        task_type = "CPU"
        device = "cpu"
        lgbm_device = "cpu"

    xgb_params = {
        "objective": "multi:softprob",
        "learning_rate": 0.0906481523921039,
        "n_estimators": 1850,
        "max_depth": 5,
        "subsample": 0.7296057237367,
        "colsample_bytree": 0.7504806008221163,
        "min_child_weight": 6,
        "reg_lambda": 103,
        "random_state": 0,
        "tree_method": "hist",
        "seed": 0,
        "device": device,
    }

    return VotingClassifier(
        estimators=[
            (
                "xgb",
                XGBClassifier(**xgb_params),
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
                LGBMClassifier(
                    n_estimators=1000,
                    random_state=0,
                    class_weight="balanced",
                    device=lgbm_device,
                    verbose=0,
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
