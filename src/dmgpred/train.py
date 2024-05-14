"""Training step in the pipeline."""

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

from dmgpred.cleaning import get_normalization_pipeline
from dmgpred.featurize import get_encoder


def get_preprocessor(X: pd.DataFrame):
    """Return the preprocessor for the pipeline."""
    normalizer = get_normalization_pipeline()
    encoder = get_encoder(X)  # noqa: F841
    selector = SelectKBest(k=20, score_func=mutual_info_classif)  # noqa: F841
    return Pipeline(
        [
            ("normalizer", normalizer),
            # ("encoder", encoder),
            # ("selector", selector),
        ],
        verbose=False,
    )


def get_pipeline(X: pd.DataFrame, clf=None):
    """Return the training pipeline."""
    if clf is None:
        clf = DummyClassifier(strategy="most_frequent")
    preprocessor = get_preprocessor(X)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clf", clf),
        ],
        verbose=False,
    )


def train(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Train the model on the full dataset.

    This model is used to predict the damage grade of the test data.
    A seperate evaluation is done using cross-validation.
    """
    cat_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    clf = VotingClassifier(
        estimators=[
            # (
            #     "xgb",
            #     XGBClassifier(
            #         enable_categorical=True,
            #         n_estimators=500,
            #         tree_method="hist",
            #         device="cuda",
            #         colsample_bytree=0.7,
            #     ),
            # ),
            (
                "catboost",
                CatBoostClassifier(
                    n_estimators=1500,
                    cat_features=cat_features,
                    task_type="GPU",
                    auto_class_weights="Balanced",
                ),
            ),
        ],
        voting="soft",
    )
    pipe = get_pipeline(X_train, clf=clf)
    pipe.fit(X_train, y_train)
    return pipe
