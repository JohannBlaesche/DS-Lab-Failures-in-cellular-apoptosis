"""Training step in the pipeline."""

import pandas as pd
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

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
        device = "cuda"  # noqa: F841
        lgbm_device = "gpu"  # noqa: F841
    else:
        task_type = "CPU"
        device = "cpu"  # noqa: F841
        lgbm_device = "cpu"  # noqa: F841

    # return VotingClassifier(
    #     estimators=[
    #         (
    #             "xgb",
    #             XGBClassifier(
    #                 enable_categorical=True,
    #                 n_estimators=1000,
    #                 tree_method="hist",
    #                 device=device,
    #                 random_state=0,
    #             ),
    #         ),
    #         (
    #             "catboost",
    #             CatBoostClassifier(
    #                 n_estimators=1000,
    #                 task_type=task_type,
    #                 verbose=False,
    #                 random_state=0,
    #             ),
    #         ),
    #         (
    #             "lgbm",
    #             LGBMClassifier(
    #                 n_estimators=1000,
    #                 random_state=0,
    #                 class_weight="balanced",
    #                 device=lgbm_device,
    #                 verbose=0,
    #             ),
    #         ),
    #     ],
    #     voting="soft",
    # )
    return CatBoostClassifier(
        n_estimators=1000,
        task_type=task_type,
        verbose=False,
        random_state=0,
        # cat_features=X_train.select_dtypes(include=["category", "object"])
        # .columns.to_numpy()
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
