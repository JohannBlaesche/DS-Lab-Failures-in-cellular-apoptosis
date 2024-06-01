"""Training step in the pipeline."""

import json

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from dmgpred.cleaning import get_normalizer
from dmgpred.featurize import get_encoder


def get_pipeline(X: pd.DataFrame, clf=None):
    """Return the training pipeline."""
    normalizer = get_normalizer()
    encoder = get_encoder(X)
    preprocessor = Pipeline(
        [
            ("normalizer", normalizer),
            ("encoder", encoder),
        ],
    )
    if clf is None:
        return preprocessor
    else:
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("clf", clf),
            ],
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

    with open("./output/xgb_best_params.json") as f:
        xgb_params = json.load(f)

    lgbm_params = {
        "max_depth": 63,
        "num_leaves": 825,
        "objective": "multiclass",
        "num_class": 3,
        "class_weight": "balanced",
        "random_state": 0,
        "verbose": -1,
        "subsample_freq": 1,
        "n_estimators": 155,
        "learning_rate": 0.1,
        "subsample": 0.9369635107813565,
        "colsample_bytree": 0.4901640047162536,
        "reg_alpha": 0.016921023196097174,
        "reg_lambda": 0.3670664208192765,
        "min_split_gain": 0.1711538139218646,
        "min_child_samples": 26,
    }

    xgb_params = {
        "objective": "multi:softmax",
        "random_state": 0,
        "verbosity": 0,
        "n_jobs": 1,
        "max_depth": 28,
        "num_leaves": 3282,
        "n_estimators": 289,
        "learning_rate": 0.02275038606221722,
        "subsample": 0.9696010831998462,
        "colsample_bytree": 0.40043798194255403,
        "reg_alpha": 0.016473331281367257,
        "reg_lambda": 3.853385806806193,
        "min_split_loss": 0.003404333309597038,
    }

    lgbm_params["device"] = device
    xgb_params["device"] = device

    return VotingClassifier(
        estimators=[
            (
                "xgb",
                XGBClassifier(
                    **xgb_params,
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
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    if clf is None:
        clf = get_classifier(X_train, use_gpu=use_gpu)
    pipe = get_pipeline(X_train, clf=clf)
    pipe.fit(X_train, y_train, clf__xgb__sample_weight=sample_weight)
    return pipe
