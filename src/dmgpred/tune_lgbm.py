"""Hyperparameter tuning for the model."""

import json

import optuna
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

from dmgpred.train import get_pipeline


def run_optimization(
    X, y, n_trials=100, random_state=0, study_name="lgbm", objective=None, use_gpu=True
) -> dict:
    """Tune the model with optuna."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    if objective is None:
        objective = get_lgbm_objective(
            X_train, y_train, X_test, y_test, use_gpu=use_gpu
        )

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
    logger.info(f"Study completed with best score: {study.best_value:.4f}")

    with open(f"./output/{study_name}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    return best_params


def get_lgbm_param_space(trial, use_gpu=True, random_state=0):
    """Return the parameter space for LGBM to tune."""
    max_depth = trial.suggest_int("max_depth", 6, 12)
    num_leaves = trial.suggest_int("num_leaves", 50, 2**max_depth)
    param_space = {
        "objective": trial.suggest_categorical("objective", ["multiclass"]),
        "num_class": trial.suggest_categorical("num_class", [3]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced"]),
        "random_state": trial.suggest_categorical("random_state", [random_state]),
        "verbose": trial.suggest_categorical("verbose", [-1]),
        "device": trial.suggest_categorical("device", ["gpu" if use_gpu else "cpu"]),
        "subsample_freq": trial.suggest_categorical("subsample_freq", [1]),
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "n_estimators": trial.suggest_int("n_estimators", 1000, 1500),
        # "n_estimators": trial.suggest_categorical("n_estimators", [1250]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=False),
        # "learning_rate": trial.suggest_categorical("learning_rate", [0.05]),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 2e-2, 4.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 2e-2, 4.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 2),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
    }
    return param_space


def get_lgbm_objective(X_train, y_train, X_test, y_test, use_gpu=True):
    """Return the objective function for LGBM."""

    def objective(trial):
        """Tune LGBM."""
        param_space = get_lgbm_param_space(trial, use_gpu=use_gpu)
        clf = LGBMClassifier(**param_space)
        model = get_pipeline(X_train, clf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = matthews_corrcoef(y_test, y_pred)
        return score

    return objective
