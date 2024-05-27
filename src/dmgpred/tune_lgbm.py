"""Hyperparameter tuning for the model."""

import joblib
import optuna
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

from dmgpred.train import get_pipeline


def tune(X, y, n_trials=100, random_state=0):
    """Tune the model with optuna."""
    # create train-val-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    def objective(trial):
        """Tune LGBM."""
        max_depth = trial.suggest_int("max_depth", 6, 12)
        num_leaves = trial.suggest_int("num_leaves", 50, 2**max_depth)
        param_space = {
            "objective": "multiclass",
            "num_class": 3,
            "class_weight": "balanced",
            "random_state": random_state,
            "verbose": -1,
            "device": "gpu",
            "subsample_freq": 1,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "n_estimators": trial.suggest_int("n_estimators", 1000, 1700),
            "subsample": trial.suggest_float("subsample", 0.4, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 2),
            "min_child_samples": trial.suggest_int("min_samples_leaf", 10, 80),
        }
        clf = LGBMClassifier(**param_space)
        model = get_pipeline(X_train, clf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = matthews_corrcoef(y_test, y_pred)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name="optimize_lgbm", direction="maximize")
    logger.info(f"Starting Hyperparameter Optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    logger.info(f"Study completed with best score: {study.best_value}")
    # persist the study
    joblib.dump(study, "./output/lgbm_study.pkl")
    return LGBMClassifier(**best_params)
