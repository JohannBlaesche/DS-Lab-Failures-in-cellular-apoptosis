"""Hyperparameter tuning for CatBoost using hyperopt."""

import warnings

import joblib
import numpy as np
from catboost import CatBoostClassifier
from hyperopt import Trials, fmin, hp, tpe
from loguru import logger
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OutputCodeClassifier

from dmgpred.train import get_pipeline

# workaround instead of changing numpy version to 1.24.0
np.warnings = warnings


def tune(X, y, n_trials=300, random_state=0):
    """Tune the model using hyperopt."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    def objective(params):
        clf = OutputCodeClassifier(CatBoostClassifier(**params, task_type="GPU"))
        model = get_pipeline(X_train, clf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = matthews_corrcoef(y_test, y_pred)
        return -score

    space = {
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
        "depth": hp.choice("depth", [3, 4, 5, 6, 7, 8, 9, 10]),
        "l2_leaf_reg": hp.uniform("l2_leaf_reg", 1, 10),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
        "random_strength": hp.uniform("random_strength", 0, 1),
        "border_count": hp.choice("border_count", [32, 64, 128, 254]),
    }

    trials = Trials()
    best_params = fmin(
        objective, space=space, algo=tpe.suggest, max_evals=n_trials, trials=trials
    )
    joblib.dump(trials, "./output/trials_catboost.pkl")

    logger.info(f"Study completed with best score: {best_params}")

    return OutputCodeClassifier(CatBoostClassifier(**best_params, task_type="GPU"))
