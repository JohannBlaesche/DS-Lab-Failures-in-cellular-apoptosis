"""Hyperparameter tuning for the model."""

import json
import logging
from typing import Literal

import joblib
import lightgbm as lgb
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from optuna_integration import XGBoostPruningCallback
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from dmgpred.train import get_pipeline
from dmgpred.utils.eval import evaluate_clf
from dmgpred.utils.logging import InterceptHandler


def run_optimization(
    X,
    y,
    n_trials=100,
    random_state=0,
    study_name=Literal["lgbm", "xgb", "catboost"],
    use_gpu=True,
) -> dict:
    """Tune the model with optuna."""
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

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

    if study_name == "xgb":
        objective, X_val_processed = get_xgb_objective(
            X_train, y_train, X_val, y_val, use_gpu=use_gpu
        )
    elif study_name == "catboost":
        objective, X_val_processed = get_catboost_objective(
            X_train, y_train, X_val, y_val, use_gpu=use_gpu
        )
    else:
        objective, X_val_processed = get_lgbm_objective(
            X_train, y_train, X_val, y_val, use_gpu=use_gpu
        )

    storage_name = f"sqlite:///./output/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage_name,
        load_if_exists=True,
        # pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
    )

    logger.info(f"Starting Hyperparameter Optimization with {n_trials} trials...")
    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=True, timeout=1 * 60 * 60
    )

    best_params = study.best_params
    logger.info(f"Study completed with best score: {study.best_value:.4f}")

    logger.info("Evaluating parameters on hold-out test set...")

    fit_kwargs = {"clf__eval_set": [(X_val_processed, y_val)]}

    if study_name == "xgb":
        clf = XGBClassifier(**best_params)
        fit_kwargs["clf__sample_weight"] = compute_sample_weight(
            class_weight="balanced", y=y_train
        )
    elif study_name == "catboost":
        clf = CatBoostClassifier(**best_params)
    else:
        clf = LGBMClassifier(**best_params)
        fit_kwargs["clf__callbacks"] = [lgb.early_stopping(50, verbose=False)]

    clf, _ = evaluate_clf(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        **fit_kwargs,
    )
    with open(f"./output/{study_name}_best_clf.pkl", "wb") as f:
        joblib.dump(clf, f)
    best_iter = clf.best_iteration_
    best_params["n_estimators"] = best_iter
    best_params.pop("early_stopping_rounds", None)
    with open(f"./output/{study_name}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    return best_params


def get_lgbm_param_space(trial, use_gpu=True, random_state=0):
    """Return the parameter space for LGBM to tune."""
    max_depth = trial.suggest_int("max_depth", 8, 63)
    num_leaves = trial.suggest_int("num_leaves", 7, 3000)
    device = "gpu" if use_gpu else "cpu"
    n_jobs = -1
    param_space = {
        "objective": trial.suggest_categorical("objective", ["multiclass"]),
        "num_class": trial.suggest_categorical("num_class", [3]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced"]),
        "random_state": trial.suggest_categorical("random_state", [random_state]),
        "verbose": trial.suggest_categorical("verbose", [-1]),
        "n_jobs": trial.suggest_categorical("n_jobs", [n_jobs]),
        "device": trial.suggest_categorical("device", [device]),
        "subsample_freq": trial.suggest_categorical("subsample_freq", [1]),
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "n_estimators": trial.suggest_int("n_estimators", 100_000, 100_000),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.05),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 2),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
    }
    return param_space


def get_lgbm_objective(X_train, y_train, X_val, y_val, use_gpu=True):
    """Return the objective function for LGBM."""
    preprocessor = get_pipeline(X_train)
    preprocessor.fit(X_train, y_train)
    X_val_processed = preprocessor.transform(X_val)

    def objective(trial):
        """Tune LGBM."""
        param_space = get_lgbm_param_space(trial, use_gpu=use_gpu)
        clf = LGBMClassifier(**param_space)
        model = get_pipeline(X_train, clf)
        model.fit(
            X_train,
            y_train,
            clf__eval_set=[(X_val_processed, y_val)],
            clf__callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        y_pred = model.predict(X_val)
        score = matthews_corrcoef(y_val, y_pred)
        return score

    return objective, X_val_processed


def get_xgb_objective(X_train, y_train, X_val, y_val, use_gpu=True):
    """Return the objective function for LGBM."""
    preprocessor = get_pipeline(X_train)
    preprocessor.fit(X_train, y_train)
    X_val_processed = preprocessor.transform(X_val)

    def objective(trial):
        """Tune LGBM."""
        param_space = get_xgb_param_space(trial, use_gpu=use_gpu)
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        pruning_callback = XGBoostPruningCallback(trial, "validation_0-mlogloss")
        clf = XGBClassifier(**param_space, callbacks=[pruning_callback])
        model = get_pipeline(X_train, clf)
        model.fit(
            X_train,
            y_train,
            clf__eval_set=[(X_val_processed, y_val)],
            clf__sample_weight=sample_weight,
            clf__verbose=False,
        )
        y_pred = model.predict(X_val)
        score = matthews_corrcoef(y_val, y_pred)
        return score

    return objective, X_val_processed


def get_xgb_param_space(trial, use_gpu=True):
    """Return the parameter space for XGBoost to tune."""
    device = "gpu" if use_gpu else "cpu"
    n_jobs = 1 if use_gpu else -1
    param_space = {
        "objective": trial.suggest_categorical("objective", ["multi:softmax"]),
        "random_state": trial.suggest_categorical("random_state", [0]),
        "verbosity": trial.suggest_categorical("verbosity", [0]),
        "n_jobs": trial.suggest_categorical("n_jobs", [n_jobs]),
        "device": trial.suggest_categorical("device", [device]),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 50),
        "max_depth": trial.suggest_int("max_depth", 10, 63),
        "num_leaves": trial.suggest_int("num_leaves", 200, 3300),
        "n_estimators": trial.suggest_int("n_estimators", 100_000, 100_000),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.04),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        "min_split_loss": trial.suggest_float("min_split_loss", 0, 2),
    }
    return param_space


def get_catboost_param_space(trial, use_gpu=True):
    """Return the parameter space for CatBoost to tune."""
    param_space = {
        "iterations": trial.suggest_int("iterations", 100_000, 100_000),
        "logging_level": trial.suggest_categorical("logging_level", ["Silent"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.05),
        "auto_class_weights": trial.suggest_categorical(
            "auto_class_weights", ["SqrtBalanced"]
        ),
        "random_seed": trial.suggest_categorical("random_seed", [0]),
        "use_best_model": trial.suggest_categorical("use_best_model", [True]),
        "task_type": trial.suggest_categorical("task_type", ["GPU"]),
        "depth": trial.suggest_int("depth", 4, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 5.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.5, 2.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        # "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "border_count": trial.suggest_int("border_count", 32, 128, log=True),
        "od_type": trial.suggest_categorical("od_type", ["IncToDec"]),
        "od_wait": trial.suggest_categorical("od_wait", [50]),
        "od_pval": trial.suggest_categorical("od_pval", [1e-6]),
        "eval_metric": trial.suggest_categorical("eval_metric", ["MCC"]),
    }
    return param_space


def get_catboost_objective(X_train, y_train, X_val, y_val, use_gpu=True):
    """Return the objective function for LGBM."""
    preprocessor = get_pipeline(X_train)
    preprocessor.fit(X_train, y_train)
    X_val_processed = preprocessor.transform(X_val)

    def objective(trial):
        """Tune LGBM."""
        param_space = get_catboost_param_space(trial, use_gpu=use_gpu)
        clf = CatBoostClassifier(**param_space)
        model = get_pipeline(X_train, clf)
        model.fit(
            X_train,
            y_train,
            clf__eval_set=[(X_val_processed, y_val)],
        )
        y_pred = model.predict(X_val)
        score = matthews_corrcoef(y_val, y_pred)
        return score

    return objective, X_val_processed
