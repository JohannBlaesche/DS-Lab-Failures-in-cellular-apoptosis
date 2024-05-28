"""Hyperparameter tuning for the model."""

import json

import optuna
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

from dmgpred.train import get_pipeline


def tune(X, y, n_trials=100, random_state=0, study_name="lgbm", objective=None) -> dict:
    """Tune the model with optuna."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    if objective is None:
        objective = get_lgbm_objective(X_train, y_train, X_test, y_test)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage_name = f"sqlite:///./output/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage_name,
        load_if_exists=True,
    )

    logger.info(f"Starting Hyperparameter Optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    constants = get_lgbm_constant_params(random_state)
    best_params.update(constants)
    logger.info(f"Study completed with best score: {study.best_value:.4f}")

    with open(f"./output/{study_name}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    return best_params


def get_lgbm_constant_params(random_state=0):
    """Return the constant parameters for LGBM."""
    constant_params = {
        "objective": "multiclass",
        "num_class": 3,
        "class_weight": "balanced",
        "random_state": random_state,
        "verbose": -1,
        "device": "gpu",
        "subsample_freq": 1,
        "n_estimators": 1250,
        "learning_rate": 0.0215,
    }
    return constant_params


def get_lgbm_param_space(trial, constant_params=None):
    """Return the parameter space for LGBM to tune."""
    constant_params = constant_params or get_lgbm_constant_params()
    max_depth = trial.suggest_int("max_depth", 6, 12)
    num_leaves = trial.suggest_int("num_leaves", 50, 2**max_depth)
    param_space = {
        **constant_params,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        # "n_estimators": trial.suggest_int("n_estimators", 1000, 1500),
        # "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 2),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
    }
    return param_space


def get_lgbm_objective(X_train, y_train, X_test, y_test):
    """Return the objective function for LGBM."""

    def objective(trial):
        """Tune LGBM."""
        param_space = get_lgbm_param_space(trial)
        clf = LGBMClassifier(**param_space)
        model = get_pipeline(X_train, clf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = matthews_corrcoef(y_test, y_pred)
        return score

    return objective
