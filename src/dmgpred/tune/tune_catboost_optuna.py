"""Hyperparameter tuning for the model."""

import json

import joblib
import optuna
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline
from loguru import logger
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

from dmgpred.cleaning import get_normalizer
from dmgpred.featurize import get_encoder
from dmgpred.train import get_pipeline


def tune_optuna(X, y, n_trials=100, random_state=0):
    """Tune the model with optuna."""
    # create train-val-test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train_full,
    )

    def objective(trial):
        """Tune Catboost."""
        space = {
            "iterations": trial.suggest_int("iterations", 100, 10000, log=True),
            "logging_level": trial.suggest_categorical("logging_level", ["Silent"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-1, 0.3),
            "task_type": trial.suggest_categorical("task_type", ["GPU"]),
            "depth": trial.suggest_int("depth", 2, 16),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            # "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0), #noqa: E501
            # "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            # "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            # "border_count": trial.suggest_int("border_count", 1, 128, log=True),
            # "model_size_reg": trial.suggest_float("model_size_reg", 0.0, 100.0),
            "od_type": trial.suggest_categorical("od_type", ["IncToDec"]),
            "od_wait": trial.suggest_categorical("od_wait", [30]),
            "od_pval": trial.suggest_categorical("od_pval", [1e-10]),
            "eval_metric": trial.suggest_categorical("eval_metric", ["MCC"]),
            # "cat_features": trial.suggest_categorical("cat_features", [cat_features])
        }

        pre_Pipeline = Pipeline(
            [
                ("normalizer", get_normalizer()),
                ("encoder", get_encoder(X_train)),
            ],
            verbose=False,
        )

        pre_Pipeline.fit(X_train, y_train)
        X_val_fitted = pre_Pipeline.transform(X_val)
        clf = CatBoostClassifier(**space)
        model = get_pipeline(X_train, clf)
        model.fit(X_train, y_train, clf__eval_set=(X_val_fitted, y_val))
        y_pred = model.predict(X_val)
        score = matthews_corrcoef(y_val, y_pred)
        logger.info(f"Final score on validation set: {score:.4f}")
        return score

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        study_name="optimize_catboost",
        direction="maximize",
    )
    logger.info(f"Starting Hyperparameter Optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"Study completed with best score: {study.best_value}")

    with open("./output/catboost_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    joblib.dump(study, "./output/catboost_study.pkl")

    logger.info("Evaluating parameters on hold-out test set...")
    clf = CatBoostClassifier(**best_params)
    model = get_pipeline(X_train, clf)
    model.fit(X_train_full, y_train_full)
    y_pred = model.predict(X_test)
    score = matthews_corrcoef(y_test, y_pred)
    logger.info(f"Final score on test set: {score:.4f}")

    return CatBoostClassifier(**best_params)
