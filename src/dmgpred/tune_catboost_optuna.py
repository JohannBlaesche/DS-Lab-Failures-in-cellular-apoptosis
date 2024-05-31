"""Hyperparameter tuning for the model."""

import joblib
import optuna
from catboost import CatBoostClassifier
from loguru import logger
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

from dmgpred.train import get_pipeline


def tune_optuna(X, y, n_trials=100, random_state=0):
    """Tune the model with optuna."""
    # create train-val-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    def objective(trial):
        """Tune Catboost."""
        space = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("depth", 1, 10),
            # "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        }

        clf = CatBoostClassifier(**space, logging_level="Silent", task_type="GPU")
        model = get_pipeline(X_train, clf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = matthews_corrcoef(y_test, y_pred)
        return score

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        study_name="optimize_lgbm",
        direction="maximize",
        pruner=SuccessiveHalvingPruner(),
    )
    logger.info(f"Starting Hyperparameter Optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"Study completed with best score: {study.best_value}")

    joblib.dump(study, "./output/catboost_study.pkl")
    return CatBoostClassifier(**best_params)
